"""
Logistics Control Tower - Main Application

Streamlit dashboard acting as a unified entry point for all ML projects.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Logistics AI Control Tower",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #ff4b4b;
            color: white;
        }
        .reportview-container {
            background: #0e1117;
        }
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            color: #fafafa;
        }
        .metric-card {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #363945;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🚛 Control Tower")
st.sidebar.markdown("---")

# Main Page Content
st.title("🌐 Supply Chain Logistics AI Platform")

st.markdown("""
<div style='background-color: #262730; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='margin:0'>Operational Overview</h3>
    <p style='color: #a3a8b4;'>Real-time AI monitoring of global supply chain operations.</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #a3a8b4;">On-Time Delivery Rate</div>
            <div style="font-size: 28px; font-weight: bold; color: #4CAF50;">92.4%</div>
            <div style="font-size: 12px; color: #4CAF50;">+1.2% vs last week</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #a3a8b4;">High Risk Shipments</div>
            <div style="font-size: 28px; font-weight: bold; color: #ff4b4b;">14</div>
            <div style="font-size: 12px; color: #ff4b4b;">⚠️ Action Required</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #a3a8b4;">Forecasted Demand</div>
            <div style="font-size: 28px; font-weight: bold; color: #2196F3;">2.4k</div>
            <div style="font-size: 12px; color: #a3a8b4;">Units (Next 24h)</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 14px; color: #a3a8b4;">System Health</div>
            <div style="font-size: 28px; font-weight: bold; color: #4CAF50;">Active</div>
            <div style="font-size: 12px; color: #a3a8b4;">No Anomalies</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("🤖 AI Modules")

c1, c2 = st.columns(2)
with c1:
    st.info("📦 **Delivery Delay Prediction**\n\nPredict shipment delays and estimate arrival times using XGBoost.")
    st.info("🎯 **Risk Classification**\n\nClassify shipments into Low, Moderate, and High risk categories.")

with c2:
    st.info("📈 **Demand Forecasting**\n\nForecast inventory levels and demand using LSTM and Prophet models.")
    st.info("🔍 **Anomaly Detection**\n\nDetect unusual patterns in sensor data and logistics operations.")

st.markdown("---")
st.markdown("*Select a module from the sidebar to begin.*")
