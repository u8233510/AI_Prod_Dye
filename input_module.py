import pandas as pd
import streamlit as st


def render_input_sidebar():
    st.sidebar.title("⚙️ AI 系統配置")
    return st.sidebar.file_uploader("1. 上傳打色資料", type=["csv", "xlsx"])


def load_uploaded_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)
