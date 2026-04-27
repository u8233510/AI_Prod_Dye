import pandas as pd
import streamlit as st

from model_utils import assess_input_data_confidence, build_data_reference, deltaE_CMC, transform_bag_of_dyes


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
            if not st.session_state.get('data_reference') and 'df_raw' in st.session_state:
                st.session_state['data_reference'] = build_data_reference(
                    st.session_state['df_raw'],
                    st.session_state['dc'],
                    st.session_state['kd'],
                )

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
            de_from_vector = deltaE_CMC((std_L_val, std_a_val, std_b_val), (p_L, p_a, p_b))
            de_direct = None
            de_val = de_from_vector
            if st.session_state.get('de_model') is not None:
                de_direct = float(st.session_state['de_model'].predict(X_m)[0])
                blend_w = float(st.session_state.get('model_params', {}).get('de_blend_weight', 0.7))
                de_val = blend_w * de_direct + (1.0 - blend_w) * de_from_vector

            with col_res:
                st.write("### 📊 預測結果")
                st.write("### 🧪 資料信心分析 (預測前)")
                if confidence_result is not None:
                    st.metric('訓練資料覆蓋指數', f"{confidence_result['score']:.1f}/100", confidence_result['level'])
                    st.metric('預估預測正確率(參考)', f"{confidence_result['estimated_correctness']:.1f}%")
                    st.write("#### 支持度明細")
                    st.write(f"類別組合支持度：{confidence_result['category']['score'] * 100:.1f}%")
                    st.write(f"數值範圍支持度：{confidence_result['numeric']['score'] * 100:.1f}%")
                    st.write(f"染劑組合支持度：{confidence_result['combo']['score'] * 100:.1f}%")
                    st.write(f"Lab 鄰域支持度：{confidence_result['lab_region']['score'] * 100:.1f}%")

                    st.write("#### 訓練資料比對數量")
                    st.write(f"此染劑組合在訓練資料出現次數：{confidence_result['combo']['出現次數']}")
                    st.write(f"此類別三欄位組合出現次數：{confidence_result['category']['三欄位組合出現次數']}")
                    st.write(f"Lab ±3 鄰域樣本數：{confidence_result['lab_region']['count_within_3_0']}")

                    if confidence_result['score'] < 60:
                        st.warning('⚠️ 輸入資料在訓練資料中支持度偏低，預測風險高，建議先補資料或人工覆核。')

                    with st.expander('查看資料比對細節'):
                        st.write('**類別欄位/組合在訓練資料的支持程度**')
                        st.json(confidence_result['category'])
                        st.write('**數值欄位範圍比對**')
                        st.dataframe(pd.DataFrame(confidence_result['numeric']['details']))
                        st.write('**染劑組合比對**')
                        st.json(confidence_result['combo'])
                        st.write('**Lab 鄰域比對 (標準樣 L*a*b*)**')
                        st.json(confidence_result['lab_region'])
                else:
                    st.metric('訓練資料覆蓋指數', 'N/A')
                    st.warning('⚠️ 目前缺少訓練資料參考分布，無法計算支持度。請先上傳訓練資料並重新訓練/載入模型。')

                st.metric("預測 L*", f"{p_L:.2f}")
                st.metric("預測 a*", f"{p_a:.2f}")
                st.metric("預測 b*", f"{p_b:.2f}")
                st.write(f"DE(向量推導): `{de_from_vector:.3f}`")
                if de_direct is not None:
                    st.write(f"DE(直接模型): `{de_direct:.3f}`")
                st.write(f"預測 CMC DE: `{de_val:.3f}`")
                if de_val <= 0.8:
                    st.success("✅ 合格 (DE <= 0.8)")
                else:
                    st.error("❌ 不合格 (DE > 0.8)")
