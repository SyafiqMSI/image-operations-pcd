import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dashboard import image_operations_dashboard

st.set_page_config(page_title="Image Operations - PCD Kelompok 2", layout="wide")
image_operations_dashboard()