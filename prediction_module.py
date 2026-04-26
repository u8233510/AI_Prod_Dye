import pandas as pd
import streamlit as st

from model_utils import assess_input_data_confidence, deltaE_CMC, transform_bag_of_dyes


def render_prediction(tab_val):
    with tab_val:
        if 'model' not in st.session_state:
            st.warning("請先完成模型訓練。")
            return

        st.header("🔍 通用 AI 配方預測")
        col_in, col_res = st.columns([2, 1])

        with col_in:
            with st.form("prediction_form"):
                st.write("#### 1. 物理參數 (標樣數值作為核心輸入)")
                c1, c2, c3, c4 = st.columns(4)
                shade_name_options = sorted(st.session_state.get('trained_df_meta', {}).get('色系名稱', []))
                shade_id_options = sorted(st.session_state.get('trained_df_meta', {}).get('色系編號', []))
                if not shade_name_options and 'df_raw' in st.session_state:
                    shade_name_options = sorted(st.session_state['df_raw']['色系名稱'].astype(str).unique())
                if not shade_id_options and 'df_raw' in st.session_state:
                    shade_id_options = sorted(st.session_state['df_raw']['色系編號'].astype(str).unique())
                shade_name_options = shade_name_options or ['未知']
                shade_id_options = shade_id_options or ['未知']

                v_name = c1.selectbox("色系名稱 (記錄用)", shade_name_options)
                v_id = c2.selectbox("色系編號 (記錄用)", shade_id_options)
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

                predict_btn = st.form_submit_button("🔮 執行通用預測", type="primary")

        if predict_btn:
            df_m = pd.DataFrame(
                [
                    {
                        '標準樣L': std_L_val,
                        '標準樣a': std_a_val,
                        '標準樣b': std_b_val,
                        'DPF': v_dpf,
                        'OP否': v_op,
                        '色系名稱': v_name,
                        '色系編號': v_id,
                        **manual_input,
                    }
                ]
            )

            confidence_result = None
            if st.session_state.get('data_reference'):
                confidence_result = assess_input_data_confidence(
                    df_m,
                    st.session_state['dc'],
                    st.session_state['kd'],
                    st.session_state['data_reference'],
                )

            X_m, _ = transform_bag_of_dyes(df_m, st.session_state['dc'], known_dyes=st.session_state['kd'])
            m_pred = st.session_state['model'].predict(X_m)[0]
            p_DL, p_Da, p_Db = m_pred[0], m_pred[1], m_pred[2]
            p_L = std_L_val + p_DL
            p_a = std_a_val + p_Da
            p_b = std_b_val + p_Db
            de_val = deltaE_CMC((std_L_val, std_a_val, std_b_val), (p_L, p_a, p_b))

            with col_res:
                st.write("### 📊 預測結果")
                if confidence_result is not None:
                    st.write("### 🧪 資料信心分析 (預測前)")
                    st.metric('資料信心指數', f"{confidence_result['score']:.1f}/100", confidence_result['level'])
                    c_conf1, c_conf2, c_conf3 = st.columns(3)
                    c_conf1.metric('類別匹配', f"{confidence_result['category']['score'] * 100:.1f}%")
                    c_conf2.metric('數值範圍匹配', f"{confidence_result['numeric']['score'] * 100:.1f}%")
                    c_conf3.metric('染劑組合匹配', f"{confidence_result['combo']['score'] * 100:.1f}%")

                    if confidence_result['score'] < 60:
                        st.warning('⚠️ 輸入資料與訓練資料分布差異較大，建議先檢查後再採信預測結果。')

                    with st.expander('查看資料比對細節'):
                        st.write('**類別欄位是否出現在訓練資料**')
                        st.json(confidence_result['category'])
                        st.write('**數值欄位範圍比對**')
                        st.dataframe(pd.DataFrame(confidence_result['numeric']['details']))
                        st.write('**染劑組合比對**')
                        st.json(confidence_result['combo'])

                st.metric("預測 L*", f"{p_L:.2f}")
                st.metric("預測 a*", f"{p_a:.2f}")
                st.metric("預測 b*", f"{p_b:.2f}")
                st.write(f"預測 CMC DE: `{de_val:.3f}`")
                if de_val <= 0.8:
                    st.success("✅ 合格 (DE <= 0.8)")
                else:
                    st.error("❌ 不合格 (DE > 0.8)")
