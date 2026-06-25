import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title = "Agentic AI Research Journal",
    page_icon = "🤖",
    layout = "wide"
)

st.title("🤖 Agentic AI Research Tracker Dashboard")
st.markdown("Real-time visual monitoring layer for the underlying LangGraph automation pipeline.")

# API Endpoint definition
API_URL = "http://127.0.0.1:8000/journal"