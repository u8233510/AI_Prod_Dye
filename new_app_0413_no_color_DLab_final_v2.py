import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split 

# ==========================================
# 0. 核心包裝：確保權重支援
# ==========================================
class ExplicitWeightedVotingRegressor(VotingRegressor):
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=sample_weight)

# ==========================================
# 1. 核心數學：CMC (2:1) 色差計算
# ==========================================
def deltaE_CMC(lab_std, lab_smpl, l=2, c=1):
    L1, a1, b1 = lab_std
    L2, a2, b2 = lab_smpl
    dL, da, db = L2 - L1, a2 - a1, b2 - b1
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C2 - C1
    dH_sq = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH_sq))
    SL = 0.040975 * L1 / (1 + 0.01765 * L1) if L1 >= 16 else 0.511
    SC = (0.0638 * C1 / (1 + 0.0131 * C1)) + 0.638
    H1 = np.degrees(np.arctan2(b1, a1)) % 360
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    T = 0.56 + abs(0.2 * np.cos(np.radians(H1 + 168))) if 164 <= H1 <= 345 else 0.36 + abs(0.4 * np.cos(np.radians(H1 + 35)))
    SH = SC * (F * T + 1 - F)
    return np.sqrt((dL/(l*SL))**2 + (dC/(c*SC))**2 + (dH/SH)**2)

# ==========================================
# 2. 數據處理工具 (物理特性模式)
# ==========================================
def clean_dye_id(val):
    s = str(val).strip()
    if s.endswith('.0'): s = s[:-2]
    if s.lower() in ['nan', 'none', '', 'null'] or pd.isna(val): s = '無'
    return s

def detect_dye_columns(df):
    return sorted([c for c in df.columns if c.startswith('配方料號')], key=lambda x: int(x.replace('配方料號', '')))

def transform_bag_of_dyes(df_clean, dye_cols, known_dyes=None):
    df_out = df_clean.copy()
    if known_dyes is None:
        raw_values = df_out[dye_cols].values.flatten()
        known_dyes = sorted(list(set([clean_dye_id(d) for d in raw_values if clean_dye_id(d) != '無'])))
    dye_f_cols = [f'Dye_{d}' for d in known_dyes]
    for dfc in dye_f_cols: df_out[dfc] = 0.0
    for d_col in dye_cols:
        n_col = d_col.replace('料號', '濃度')
        if n_col in df_out.columns:
            for d in known_dyes:
                mask = (df_out[d_col] == d)
                df_out.loc[mask, f'Dye_{d}'] += df_out[n_col][mask]
    df_out['Total_Conc'] = df_out[dye_f_cols].sum(axis=1)
    df_out['Log_Total_Conc'] = np.log1p(df_out['Total_Conc'])
    # 物理參考特徵：標樣 LAB + DPF + OP
    base_cols = ['標準樣L', '標準樣a', '標準樣b', '色系名稱', '色系編號', 'DPF', 'OP否', 'Total_Conc', 'Log_Total_Conc']
    return df_out[base_cols + dye_f_cols], known_dyes

# ==========================================
# 3. Streamlit 介面佈局
# ==========================================
st.set_page_config(page_title="AI 專業打色系統 v20.0", layout="wide")
st.sidebar.title("⚙️ AI 系統配置")
uploaded_file = st.sidebar.file_uploader("1. 上傳打色資料", type=["csv", "xlsx"])

tab_ana, tab_feedback, tab_val = st.tabs(["📊 數據分布分析", "📈 訓練回測回饋", "🔍 單筆輸入預測"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    dye_cols = detect_dye_columns(df_raw)

    # --- 修正 1: 數據分布分析 (新增染料與色系名稱統計) ---
    with tab_ana:
        st.header("📊 全資料分布分析")
        df_ana = df_raw.copy()
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(px.histogram(df_ana, x='L', title="L* (亮/淺分布)", color_discrete_sequence=['#555555']), use_container_width=True)
        with c2: 
            df_ana['a_type'] = df_ana['a'].apply(lambda x: '紅' if x > 0 else '綠')
            st.plotly_chart(px.histogram(df_ana, x='a', color='a_type', title="a* (紅/綠分布)", color_discrete_map={'紅':'#EF553B','綠':'#00CC96'}), use_container_width=True)
        with c3:
            df_ana['b_type'] = df_ana['b'].apply(lambda x: '黃' if x > 0 else '藍')
            st.plotly_chart(px.histogram(df_ana, x='b', color='b_type', title="b* (黃/藍分布)", color_discrete_map={'黃':'#FECB52','藍':'#636EFA'}), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 染料料號對色系名稱統計表 (Dye vs. Shade Name)")
        
        # 提取所有不為「無」的染料料號
        unique_d = sorted(list(set([clean_dye_id(d) for d in df_ana[dye_cols].values.flatten() if clean_dye_id(d) != '無'])))
        shade_names = sorted(df_ana['色系名稱'].astype(str).unique())
        
        dist_data = []
        for d in unique_d:
            row = {'染料料號': d}
            for sn in shade_names:
                # 統計該染料在特定色系名稱中出現的筆數
                count = ((df_ana['色系名稱'] == sn) & (df_ana[dye_cols].apply(lambda x: d in [clean_dye_id(i) for i in x], axis=1))).sum()
                row[sn] = count
            dist_data.append(row)
        
        dist_df = pd.DataFrame(dist_data).set_index('染料料號')
        st.dataframe(dist_df.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)

        # --- 新增：染劑組合次數統計 ---
        st.markdown("---")
        st.subheader("🧪 染劑組合出現次數統計 (Dye Combinations Frequency)")
        
        # 1. 取得所有配方料號欄位名稱
        dye_id_cols = detect_dye_columns(df_ana)
        
        # 2. 定義一個函數來提取並整理每行的染劑組合
        def get_sorted_combination(row):
            # 取得該行所有染劑編號，清理掉「無」或空值
            dyes = [clean_dye_id(row[c]) for c in dye_id_cols if clean_dye_id(row[c]) != '無']
            # 排序染劑，確保組合的一致性 (例如 1191+1588 跟 1588+1191 是同一組)
            dyes.sort()
            # 用 "+" 號連接
            return "+".join(dyes) if dyes else "僅基礎藥劑(無染料)"

        # 3. 產生組合欄位並計算次數
        df_ana['染劑組合'] = df_ana.apply(get_sorted_combination, axis=1)
        comb_df = df_ana['染劑組合'].value_counts().reset_index()
        comb_df.columns = ['染劑組合', '出現次數 (筆)']
        
        # 4. 顯示統計表格，並加入百分比參考
        comb_df['佔比 (%)'] = (comb_df['出現次數 (筆)'] / len(df_ana) * 100).round(2)
        
        st.write(f"📊 總計偵測到 `{len(comb_df)}` 種不同的染劑組合。")
        st.dataframe(
            comb_df.style.background_gradient(subset=['出現次數 (筆)'], cmap='Blues'),
            use_container_width=True
        ) 


    # --- 2. 訓練核心：精準平衡加權 ---
    if st.sidebar.button("🚀 啟動模型訓練"):
        with st.spinner("AI 運算中 (平衡物理特徵學習模式)..."):
            df_train = df_raw.copy()
            # 剔除空值
            df_train = df_train.dropna(subset=['標準樣L', '標準樣a', '標準樣b', 'L', 'a', 'b', 'DPF', 'OP否', 'CMC_DE'])
            df_c = df_train.copy()
            for dc in dye_cols: df_c[dc] = df_c[dc].apply(clean_dye_id)
            X_bag, known_dyes = transform_bag_of_dyes(df_c, dye_cols)
            #Y = df_train[['L', 'a', 'b']]
            Y = df_train[['CIE_DL', 'CIE_Da', 'CIE_Db']]
            
            X_train, X_test, Y_train, Y_test = train_test_split(X_bag, Y, test_size=0.2, random_state=42)
            
            actual_de_train = df_train.loc[X_train.index, 'CMC_DE'].values
            
            sample_weights = np.where(actual_de_train > 0.8, 5.0, 1.0) 
            #sample_weights = np.ones(len(actual_de_train))            
            
            pre = ColumnTransformer([
                #('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否']), 
                #('num', StandardScaler(), ['標準樣L', '標準樣a', '標準樣b','DPF', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes])
                
                # 【修改點】加入 '色系名稱' 到類別處理
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['OP否', '色系名稱']), 
    
                # 【修改點】加入 '色系編號' 到數值縮放
                ('num', StandardScaler(), ['標準樣L', '標準樣a', '標準樣b', 'DPF', '色系編號', 'Total_Conc', 'Log_Total_Conc'] + [f'Dye_{d}' for d in known_dyes])
            ])
            
            #model = Pipeline([
            #    ('pre', pre), 
            #    ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([
            #        ('hgb', HistGradientBoostingRegressor(loss='squared_error', max_iter=1000, learning_rate=0.03, random_state=42)), 
            #        ('et', ExtraTreesRegressor(n_estimators=400, random_state=42)),
            #        ('rf', RandomForestRegressor(n_estimators=400, random_state=42))
            #    ]))) 
            #])
            
            model = Pipeline([
                ('pre', pre), 
                ('reg', MultiOutputRegressor(ExplicitWeightedVotingRegressor([
                    # 【修改點】loss 改為 'absolute_error'
                    ('hgb', HistGradientBoostingRegressor(
                       loss='absolute_error',  # 關鍵修改
                       max_iter=1000, 
                       learning_rate=0.03, 
                       random_state=42
                    )), 
                   ('et', ExtraTreesRegressor(n_estimators=400, random_state=42)),
                   ('rf', RandomForestRegressor(n_estimators=400, random_state=42))
                ]))) 
            ])
            
            model.fit(X_train, Y_train, reg__sample_weight=sample_weights) 
            preds = model.predict(X_test) 
            
            df_val_raw = df_train.loc[X_test.index].copy()
            df_fb = df_val_raw[['成品布號', 'L', 'a', 'b', 'CIE_DL', 'CIE_Da', 'CIE_Db', 'CMC_DE', '標準樣L', '標準樣a', '標準樣b']].copy().reset_index(drop=True)
            df_fb.columns = ['布號', '實際L', '實際a', '實際b', '實際DL', '實際Da', '實際Db', '實際DE', 'stdL', 'stda', 'stdb']
            #df_fb['預測L'], df_fb['預測a'], df_fb['預測b'] = preds[:,0], preds[:,1], preds[:,2]
            # 先接收預測的 Delta 值
            df_fb['預測DL'], df_fb['預測Da'], df_fb['預測Db'] = preds[:,0], preds[:,1], preds[:,2]

            # 執行還原：預測絕對值 = 標準樣 + 預測差距
            df_fb['預測L'] = df_fb['stdL'] + df_fb['預測DL']
            df_fb['預測a'] = df_fb['stda'] + df_fb['預測Da']
            df_fb['預測b'] = df_fb['stdb'] + df_fb['預測Db']
                       
            
            # 補全指標
            #df_fb['預測DL'] = df_fb['預測L'] - df_fb['stdL']
            #df_fb['預測Da'] = df_fb['預測a'] - df_fb['stda']
            #df_fb['預測Db'] = df_fb['預測b'] - df_fb['stdb']
            #df_fb['預測DE'] = [deltaE_CMC((df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']), (preds[i,0], preds[i,1], preds[i,2])) for i in range(len(df_fb))]
            df_fb['預測DE'] = [deltaE_CMC((df_fb.loc[i, 'stdL'], df_fb.loc[i, 'stda'], df_fb.loc[i, 'stdb']), (df_fb.loc[i, '預測L'], df_fb.loc[i, '預測a'], df_fb.loc[i, '預測b'])) for i in range(len(df_fb))]
            
            st.session_state.update({'model': model, 'kd': known_dyes, 'fb': df_fb, 'dc': dye_cols, 'df_raw': df_raw})
            st.sidebar.success(f"訓練完成！")

# --- 修正 2: 訓練回測回饋 (詳細顯示 7 項指標比對) ---
if 'fb' in st.session_state:
    with tab_feedback:
        df_fb = st.session_state['fb']
        st.header("📈 判定一致性回饋 (驗證集結果)")
        
        tp = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb['預測DE'] <= 0.8)])
        tn = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb['預測DE'] > 0.8)])
        fp = len(df_fb[(df_fb['實際DE'] > 0.8) & (df_fb['預測DE'] <= 0.8)])
        fn = len(df_fb[(df_fb['實際DE'] <= 0.8) & (df_fb['預測DE'] > 0.8)])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ 雙重通過", f"{tp}")
        c2.metric("❌ 雙重失敗", f"{tn}")
        c3.metric("⚠️ 誤判通過 (危險)", f"{fp}")
        c4.metric("🔍 誤判失敗 (保守)", f"{fn}")
        
        st.info(f"### 總體判定一致性比率：{(tp + tn) / len(df_fb):.2%}")

        st.markdown("### 🔍 完整 7 項指標比對清單 (L, a, b, DL, Da, Db, DE)")
        
        # 建立展示用表格，包含實際、預測、差異
        display_df = pd.DataFrame({'布號': df_fb['布號']})
        
        metrics_map = [
            ('L', '實際L', '預測L'),
            ('a', '實際a', '預測a'),
            ('b', '實際b', '預測b'),
            ('DL', '實際DL', '預測DL'),
            ('Da', '實際Da', '預測Da'),
            ('Db', '實際Db', '預測Db'),
            ('DE', '實際DE', '預測DE')
        ]
        
        for label, act, pre in metrics_map:
            display_df[f'實際{label}'] = df_fb[act]
            display_df[f'預測{label}'] = df_fb[pre]
            display_df[f'Δ{label}差異'] = df_fb[pre] - df_fb[act]
            
        st.dataframe(display_df.round(3).style.background_gradient(subset=[f'Δ{m}差異' for m in ['L','a','b','DL','Da','Db','DE']], cmap='RdBu_r'), use_container_width=True)

        # 保留原有的圖表供深度參考
        st.markdown("---")
        st.write("#### 指標分佈圖表")
        for label, act, pre in metrics_map:
            with st.expander(f"📊 {label} 指標詳情"):
                fig = px.scatter(df_fb, x=act, y=pre, title=f"{label}：預測 vs 實際")
                fig.add_shape(type="line", x0=df_fb[act].min(), y0=df_fb[act].min(), x1=df_fb[act].max(), y1=df_fb[act].max(), line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

# --- 分頁 3: 通用預測 ---# --- 分頁 3: 通用預測 ---
with tab_val:
    if 'model' in st.session_state:
        st.header("🔍 通用 AI 配方預測")
        col_in, col_res = st.columns([2, 1])
        
        with col_in:
            # 【關鍵修改】加入 st.form，把所有的輸入框打包起來
            with st.form("prediction_form"):
                st.write("#### 1. 物理參數 (標樣數值作為核心輸入)")
                c1, c2, c3, c4 = st.columns(4)
                v_name = c1.selectbox("色系名稱 (記錄用)", sorted(st.session_state['df_raw']['色系名稱'].astype(str).unique()))
                v_id = c2.selectbox("色系編號 (記錄用)", sorted(st.session_state['df_raw']['色系編號'].astype(str).unique()))
                v_dpf = c3.number_input("DPF", value=1.0)
                v_op = c4.selectbox("OP否", ['Y', 'N'])
                
                csL, csa, csb = st.columns(3)
                std_L_val = csL.number_input("標準樣 L*", value=50.0)
                std_a_val = csa.number_input("標準樣 a*", value=0.0)
                std_b_val = csb.number_input("標準樣 b*", value=0.0)
                
                manual_input = {}
                for i in range(1, 7):
                    cc1, cc2 = st.columns(2)
                    manual_input[f'配方料號{i}'] = cc1.selectbox(f"配方料號 {i}", ['無'] + st.session_state['kd'], key=f"p{i}")
                    manual_input[f'配方濃度{i}'] = cc2.number_input(f"配方濃度 {i}", value=0.0, format="%.4f", key=f"c{i}")

                # 【關鍵修改】按鈕必須改成 form_submit_button
                predict_btn = st.form_submit_button("🔮 執行通用預測", type="primary")

        # 只有在按下表單內的送出按鈕後，才會執行這裡的預測邏輯
        if predict_btn:
            #df_m = pd.DataFrame([{'標準樣L': std_L_val, '標準樣a': std_a_val, '標準樣b': std_b_val, 'DPF': v_dpf, 'OP否': v_op, **manual_input}])
            df_m = pd.DataFrame([{'標準樣L': std_L_val,'標準樣a': std_a_val,'標準樣b': std_b_val,'DPF': v_dpf,'OP否': v_op,'色系名稱': v_name, '色系編號': v_id, **manual_input}])
        
            
            X_m, _ = transform_bag_of_dyes(df_m, st.session_state['dc'], known_dyes=st.session_state['kd'])
            m_pred = st.session_state['model'].predict(X_m)[0]
            #p_L, p_a, p_b = m_pred[0], m_pred[1], m_pred[2]
            # 接收 Delta
            p_DL, p_Da, p_Db = m_pred[0], m_pred[1], m_pred[2]
            # 還原成座標
            p_L = std_L_val + p_DL
            p_a = std_a_val + p_Da
            p_b = std_b_val + p_Db
            
            
            de_val = deltaE_CMC((std_L_val, std_a_val, std_b_val), (p_L, p_a, p_b))
            
            with col_res:
                st.write("### 📊 預測結果")
                st.metric("預測 L*", f"{p_L:.2f}")
                st.metric("預測 a*", f"{p_a:.2f}")
                st.metric("預測 b*", f"{p_b:.2f}")
                st.write(f"預測 CMC DE: `{de_val:.3f}`")
                if de_val <= 0.8: 
                    st.success("✅ 合格 (DE <= 0.8)")
                else: 
                    st.error("❌ 不合格 (DE > 0.8)")
    else:
        st.warning("請先完成模型訓練。")