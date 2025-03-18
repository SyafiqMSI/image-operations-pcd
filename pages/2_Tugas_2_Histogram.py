import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_2 import histogram_dashboard

st.set_page_config(page_title="Histogram Analysis - PCD Kelompok 2", layout="wide")
histogram_dashboard()