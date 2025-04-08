import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_3 import edge_detection_dashboard

st.set_page_config(page_title="Deteksi Tepi - PCD Kelompok 2", layout="wide")
edge_detection_dashboard()