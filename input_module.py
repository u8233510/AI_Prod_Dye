import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _load_uploaded_data_from_bytes(file_name, file_bytes):
    if file_name.endswith('.csv'):
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return pd.read_excel(pd.io.common.BytesIO(file_bytes))


def render_input_sidebar():
    st.sidebar.title("⚙️ AI 系統配置")
    return st.sidebar.file_uploader("1. 上傳打色資料", type=["csv", "xlsx"])


def load_uploaded_data(uploaded_file):
    return _load_uploaded_data_from_bytes(uploaded_file.name, uploaded_file.getvalue())
