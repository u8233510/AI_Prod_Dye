import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import clean_dye_id, detect_dye_columns


def render_data_distribution(tab_ana, df_raw, dye_cols):
    with tab_ana:
        st.header("📊 全資料分布分析")
        df_ana = df_raw.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                px.histogram(df_ana, x='L', title="L* (亮/淺分布)", color_discrete_sequence=['#555555']),
                use_container_width=True,
            )
        with c2:
            df_ana['a_type'] = df_ana['a'].apply(lambda x: '紅' if x > 0 else '綠')
            st.plotly_chart(
                px.histogram(df_ana, x='a', color='a_type', title="a* (紅/綠分布)", color_discrete_map={'紅': '#EF553B', '綠': '#00CC96'}),
                use_container_width=True,
            )
        with c3:
            df_ana['b_type'] = df_ana['b'].apply(lambda x: '黃' if x > 0 else '藍')
            st.plotly_chart(
                px.histogram(df_ana, x='b', color='b_type', title="b* (黃/藍分布)", color_discrete_map={'黃': '#FECB52', '藍': '#636EFA'}),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("📋 染料料號對色系名稱統計表 (Dye vs. Shade Name)")

        unique_d = sorted(list(set([clean_dye_id(d) for d in df_ana[dye_cols].values.flatten() if clean_dye_id(d) != '無'])))
        shade_names = sorted(df_ana['色系名稱'].astype(str).unique())

        dist_data = []
        for d in unique_d:
            row = {'染料料號': d}
            for sn in shade_names:
                count = ((df_ana['色系名稱'] == sn) & (df_ana[dye_cols].apply(lambda x: d in [clean_dye_id(i) for i in x], axis=1))).sum()
                row[sn] = count
            dist_data.append(row)

        dist_df = pd.DataFrame(dist_data).set_index('染料料號')
        st.dataframe(dist_df.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)

        st.markdown("---")
        st.subheader("🧪 染劑組合出現次數統計 (Dye Combinations Frequency)")

        dye_id_cols = detect_dye_columns(df_ana)

        def get_sorted_combination(row):
            dyes = [clean_dye_id(row[c]) for c in dye_id_cols if clean_dye_id(row[c]) != '無']
            dyes.sort()
            return "+".join(dyes) if dyes else "僅基礎藥劑(無染料)"

        df_ana['染劑組合'] = df_ana.apply(get_sorted_combination, axis=1)
        comb_df = df_ana['染劑組合'].value_counts().reset_index()
        comb_df.columns = ['染劑組合', '出現次數 (筆)']
        comb_df['佔比 (%)'] = (comb_df['出現次數 (筆)'] / len(df_ana) * 100).round(2)

        st.write(f"📊 總計偵測到 `{len(comb_df)}` 種不同的染劑組合。")
        st.dataframe(
            comb_df.style.background_gradient(subset=['出現次數 (筆)'], cmap='Blues'),
            use_container_width=True,
        )
