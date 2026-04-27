import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import build_data_reference, clean_dye_id, deltaE_CMC, detect_dye_columns, transform_bag_of_dyes


class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)


MODEL_LABELS = {
    'current_ensemble': '目前模型 (Voting Ensemble)',
    'automl_lite': 'AutoML-lite (RandomizedSearchCV)',
}

MODEL_ARTIFACT_PATH = Path('saved_model_artifact.joblib')


def _build_preprocessor(known_dyes):
    return ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否', '色系名稱']),
            (
                'num',
                StandardScaler(),
                ['標準樣L', '標準樣a', '標準樣b', 'DPF', '色系編號', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes],
            ),
        ]
    )


def _build_current_model(pre, model_params):
    hgb = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=int(model_params['hgb_max_iter']),
        learning_rate=float(model_params['hgb_learning_rate']),
        random_state=42,
    )
    et = ExtraTreesRegressor(
        n_estimators=int(model_params['et_n_estimators']),
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=int(model_params['rf_n_estimators']),
        random_state=42,
    )

    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([('hgb', hgb), ('et', et), ('rf', rf)]))),
        ]
    )


def _build_automl_pipeline(pre):
    return Pipeline(
        [
            ('pre', pre),
            ('reg', MultiOutputRegressor(RandomForestRegressor(random_state=42))),
        ]
    )


def _build_de_model(pre, model_type, model_params):
    if model_type == 'automl_lite':
        de_reg = ExtraTreesRegressor(
            n_estimators=int(model_params.get('de_n_estimators', 500)),
            max_depth=None,
            random_state=42,
        )
    else:
        de_reg = ExplicitWeightedVotingRegressor(
            [
                (
                    'hgb',
                    HistGradientBoostingRegressor(
                        loss='absolute_error',
                        max_iter=int(model_params['hgb_max_iter']),
                        learning_rate=float(model_params['hgb_learning_rate']),
                        random_state=42,
                    ),
                ),
                ('et', ExtraTreesRegressor(n_estimators=int(model_params['et_n_estimators']), random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=int(model_params['rf_n_estimators']), random_state=42)),
            ]
        )
    return Pipeline([('pre', pre), ('reg', de_reg)])


def _fit_automl_lite(model, X_train, Y_train, sample_weights, model_params):
    search_space = [
        {
            'reg__estimator': [RandomForestRegressor(random_state=42)],
            'reg__estimator__n_estimators': [200, 300, 400, 500, 700],
            'reg__estimator__max_depth': [None, 8, 12, 20],
            'reg__estimator__min_samples_split': [2, 4, 8],
        },
        {
            'reg__estimator': [ExtraTreesRegressor(random_state=42)],
            'reg__estimator__n_estimators': [200, 300, 400, 500, 700],
            'reg__estimator__max_depth': [None, 8, 12, 20],
            'reg__estimator__min_samples_split': [2, 4, 8],
        },
        {
            'reg__estimator': [HistGradientBoostingRegressor(loss='absolute_error', random_state=42)],
            'reg__estimator__max_iter': [300, 500, 800, 1000],
            'reg__estimator__learning_rate': [0.01, 0.03, 0.05, 0.08],
            'reg__estimator__max_depth': [None, 4, 8],
        },
    ]

    automl = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=int(model_params['n_iter']),
        cv=int(model_params['cv_folds']),
        scoring=model_params['scoring'],
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    automl.fit(X_train, Y_train, reg__sample_weight=sample_weights)
    return automl.best_estimator_, automl.best_params_, automl.best_score_


def train_model(df_raw, dye_cols, model_type='current_ensemble', model_params=None):
    model_params = model_params or {}

    df_train = df_raw.copy().dropna(
        subset=['標準樣L', '標準樣a', '標準樣b', 'L', 'a', 'b', 'DPF', 'OP否', 'CMC_DE']
    )

    df_c = df_train.copy()
    for dc in dye_cols:
        df_c[dc] = df_c[dc].apply(clean_dye_id)

    X_bag, known_dyes = transform_bag_of_dyes(df_c, dye_cols)
    Y = df_train[['CIE_DL', 'CIE_Da', 'CIE_Db']]

    X_train, X_test, Y_train, Y_test = train_test_split(X_bag, Y, test_size=0.2, random_state=42)
    actual_de_train = df_train.loc[X_train.index, 'CMC_DE'].values
    sample_weights = np.where(actual_de_train > float(model_params.get('high_de_threshold', 0.8)), float(model_params.get('high_de_weight', 5.0)), 1.0)

    pre = _build_preprocessor(known_dyes)

    automl_info = None
    if model_type == 'automl_lite':
        base_pipeline = _build_automl_pipeline(pre)
        model, best_params, best_score = _fit_automl_lite(base_pipeline, X_train, Y_train, sample_weights, model_params)
        automl_info = {
            'best_params': best_params,
            'best_cv_score': best_score,
        }
    else:
        model = _build_current_model(pre, model_params)
        model.fit(X_train, Y_train, reg__sample_weight=sample_weights)

    de_model = _build_de_model(pre, model_type, model_params)
    de_model.fit(X_train, actual_de_train, reg__sample_weight=sample_weights)

    preds = model.predict(X_test)
    de_preds_direct = de_model.predict(X_test)

    df_val_raw = df_train.loc[X_test.index].copy()
    df_fb = df_val_raw[
        ['成品布號', 'L', 'a', 'b', 'CIE_DL', 'CIE_Da', 'CIE_Db', 'CMC_DE', '標準樣L', '標準樣a', '標準樣b']
    ].copy().reset_index(drop=True)
    df_fb.columns = ['布號', '實際L', '實際a', '實際b', '實際DL', '實際Da', '實際Db', '實際DE', 'stdL', 'stda', 'stdb']

    df_fb['預測DL'], df_fb['預測Da'], df_fb['預測Db'] = preds[:, 0], preds[:, 1], preds[:, 2]
    df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
    df_fb['預測a'] = df_fb['stda'] + df_fb['預測Da']
    df_fb['預測b'] = df_fb['stdb'] + df_fb['預測Db']
    df_fb['預測DE_向量推導'] = [
        deltaE_CMC(
            (df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']),
            (df_fb.loc[i, '預測L'], df_fb.loc[i, '預測a'], df_fb.loc[i, '預測b']),
        )
        for i in range(len(df_fb))
    ]
    de_blend_weight = float(model_params.get('de_blend_weight', 0.7))
    df_fb['預測DE_直接模型'] = de_preds_direct
    df_fb['預測DE'] = de_blend_weight * df_fb['預測DE_直接模型'] + (1.0 - de_blend_weight) * df_fb['預測DE_向量推導']

    return {
        'model': model,
        'de_model': de_model,
        'kd': known_dyes,
        'fb': df_fb,
        'dc': dye_cols,
        'data_reference': build_data_reference(df_train, dye_cols, known_dyes),
        'df_raw': df_raw,
        'model_type': model_type,
        'model_label': MODEL_LABELS.get(model_type, model_type),
        'model_params': model_params,
        'automl_info': automl_info,
    }


def save_trained_artifact(state, save_path=MODEL_ARTIFACT_PATH):
    artifact = {
        'model': state['model'],
        'de_model': state.get('de_model'),
        'kd': state['kd'],
        'dc': state['dc'],
        'data_reference': state.get('data_reference', {}),
        'model_type': state.get('model_type'),
        'model_label': state.get('model_label'),
        'model_params': state.get('model_params', {}),
        'automl_info': state.get('automl_info'),
        'trained_df_meta': {
            '色系名稱': sorted(state['df_raw']['色系名稱'].astype(str).unique()),
            '色系編號': sorted(state['df_raw']['色系編號'].astype(str).unique()),
        },
    }
    joblib.dump(artifact, save_path)
    return save_path


def load_trained_artifact(save_path=MODEL_ARTIFACT_PATH):
    if not Path(save_path).exists():
        return None
    return joblib.load(save_path)


def _render_model_params(container, model_type):
    params = {}

    container.markdown('### 🧩 模型參數設定')

    params['high_de_threshold'] = container.number_input('高誤差閾值 (CMC_DE)', min_value=0.0, max_value=10.0, value=0.8, step=0.1)
    params['high_de_weight'] = container.number_input('高誤差樣本權重', min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    params['de_blend_weight'] = container.slider('DE直接模型權重', min_value=0.0, max_value=1.0, value=0.7, step=0.05)

    if model_type == 'current_ensemble':
        container.caption('目前模型：Voting Ensemble')
        params['hgb_max_iter'] = container.slider('HGB max_iter', min_value=200, max_value=2000, value=1000, step=100)
        params['hgb_learning_rate'] = container.number_input('HGB learning_rate', min_value=0.001, max_value=0.3, value=0.03, step=0.005, format='%.3f')
        params['et_n_estimators'] = container.slider('ExtraTrees n_estimators', min_value=100, max_value=1000, value=400, step=50)
        params['rf_n_estimators'] = container.slider('RandomForest n_estimators', min_value=100, max_value=1000, value=400, step=50)
    elif model_type == 'automl_lite':
        container.caption('AutoML-lite：對多個模型做隨機搜尋')
        params['n_iter'] = container.slider('搜尋次數 (n_iter)', min_value=5, max_value=60, value=20, step=5)
        params['cv_folds'] = container.selectbox('交叉驗證折數', [3, 4, 5], index=0)
        params['scoring'] = container.selectbox('評分指標', ['neg_mean_absolute_error', 'neg_root_mean_squared_error'], index=0)
        params['de_n_estimators'] = container.slider('DE模型樹數', min_value=200, max_value=1200, value=500, step=100)

    return params


def render_training_button(df_raw=None, dye_cols=None):
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 🤖 訓練模型選擇')

    if df_raw is not None and dye_cols is not None:
        form = st.sidebar.form('training_form')
        with form:
            model_type = st.selectbox('模型類型', options=list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])
            model_params = _render_model_params(form, model_type)
            train_clicked = st.form_submit_button('🚀 啟動模型訓練')

        if train_clicked:
            with st.spinner('AI 運算中 (模型訓練中)...'):
                state = train_model(df_raw, dye_cols, model_type=model_type, model_params=model_params)
                st.session_state.update(state)
                st.session_state['trained_df_meta'] = {
                    '色系名稱': sorted(df_raw['色系名稱'].astype(str).unique()),
                    '色系編號': sorted(df_raw['色系編號'].astype(str).unique()),
                }
                save_path = save_trained_artifact(state)
                st.sidebar.success(f"訓練完成：{state['model_label']}")
                st.sidebar.caption(f"已儲存模型：{save_path}")
                if state.get('automl_info'):
                    st.sidebar.info(f"最佳CV分數: {state['automl_info']['best_cv_score']:.4f}")
    else:
        st.sidebar.caption('若只需預測，可直接載入先前已儲存模型。')

    if st.sidebar.button('📥 載入已儲存模型'):
        artifact = load_trained_artifact()
        if artifact is None:
            st.sidebar.warning('尚未找到已儲存模型，請先完成一次訓練。')
        else:
            st.session_state['model'] = artifact['model']
            st.session_state['de_model'] = artifact.get('de_model')
            st.session_state['kd'] = artifact['kd']
            st.session_state['dc'] = artifact['dc']
            data_reference = artifact.get('data_reference', {})
            if not data_reference and 'df_raw' in st.session_state:
                fallback_dye_cols = artifact.get('dc') or detect_dye_columns(st.session_state['df_raw'])
                data_reference = build_data_reference(st.session_state['df_raw'], fallback_dye_cols, artifact['kd'])
            st.session_state['data_reference'] = data_reference
            st.session_state['model_type'] = artifact.get('model_type')
            st.session_state['model_label'] = artifact.get('model_label')
            st.session_state['model_params'] = artifact.get('model_params', {})
            st.session_state['automl_info'] = artifact.get('automl_info')
            st.session_state['trained_df_meta'] = artifact.get('trained_df_meta', {})
            st.sidebar.success(f"已載入模型：{artifact.get('model_label', '已儲存模型')}")
