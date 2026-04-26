import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

from model_utils import clean_dye_id, deltaE_CMC, transform_bag_of_dyes


class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)


def train_model(df_raw, dye_cols):
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
    sample_weights = np.where(actual_de_train > 0.8, 5.0, 1.0)

    pre = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否', '色系名稱']),
            (
                'num',
                StandardScaler(),
                ['標準樣L', '標準樣a', '標準樣b', 'DPF', '色系編號', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes],
            ),
        ]
    )

    model = Pipeline(
        [
            ('pre', pre),
            (
                'reg',
                MultiOutputRegressor(
                    ExplicitWeightedVotingRegressor(
                        [
                            ('hgb', HistGradientBoostingRegressor(loss='absolute_error', max_iter=1000, learning_rate=0.03, random_state=42)),
                            ('et', ExtraTreesRegressor(n_estimators=400, random_state=42)),
                            ('rf', RandomForestRegressor(n_estimators=400, random_state=42)),
                        ]
                    )
                ),
            ),
        ]
    )

    model.fit(X_train, Y_train, reg__sample_weight=sample_weights)
    preds = model.predict(X_test)

    df_val_raw = df_train.loc[X_test.index].copy()
    df_fb = df_val_raw[
        ['成品布號', 'L', 'a', 'b', 'CIE_DL', 'CIE_Da', 'CIE_Db', 'CMC_DE', '標準樣L', '標準樣a', '標準樣b']
    ].copy().reset_index(drop=True)
    df_fb.columns = ['布號', '實際L', '實際a', '實際b', '實際DL', '實際Da', '實際Db', '實際DE', 'stdL', 'stda', 'stdb']

    df_fb['預測DL'], df_fb['預測Da'], df_fb['預測Db'] = preds[:, 0], preds[:, 1], preds[:, 2]
    df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
    df_fb['預測a'] = df_fb['stda'] + df_fb['預測Da']
    df_fb['預測b'] = df_fb['stdb'] + df_fb['預測Db']
    df_fb['預測DE'] = [
        deltaE_CMC(
            (df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']),
            (df_fb.loc[i, '預測L'], df_fb.loc[i, '預測a'], df_fb.loc[i, '預測b']),
        )
        for i in range(len(df_fb))
    ]

    return {
        'model': model,
        'kd': known_dyes,
        'fb': df_fb,
        'dc': dye_cols,
        'df_raw': df_raw,
    }


def render_training_button(df_raw, dye_cols):
    if st.sidebar.button("🚀 啟動模型訓練"):
        with st.spinner("AI 運算中 (平衡物理特徵學習模式)..."):
            state = train_model(df_raw, dye_cols)
            st.session_state.update(state)
            st.sidebar.success("訓練完成！")
