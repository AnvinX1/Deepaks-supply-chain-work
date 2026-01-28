"""
Anomaly Detection Dashboard Module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dashboard.utils.data_loader import load_model, load_main_dataset

def show():
    st.title("🔍 Anomaly Detection")
    
    df = load_main_dataset()
    
    # Mock anomaly scores for visualization if model not loaded
    # In prod: scores = model.score_samples(df)
    if 'anomaly_score' not in df.columns:
        import numpy as np
        df['anomaly_score'] = np.random.uniform(0, 1, len(df))
        df['is_anomaly'] = df['anomaly_score'] > 0.95
        
    anomalies = df[df['is_anomaly']]
    
    # Alerts
    st.subheader("🚨 Active Alerts")
    
    if len(anomalies) > 0:
        recent_anomalies = anomalies.tail(5)
        for _, row in recent_anomalies.iterrows():
            st.error(f"Anomaly detected at {row.get('timestamp', 'Unknown Time')} - Score: {row.get('anomaly_score', 0):.2f}")
    else:
        st.success("No active anomalies detected.")
        
    # Visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Anomaly Timeline")
        fig = px.scatter(
            df.tail(1000), 
            x='timestamp', 
            y='anomaly_score',
            color='is_anomaly',
            color_discrete_map={True: 'red', False: 'blue'},
            title="Real-time Anomaly Scoring"
        )
        fig.add_hline(y=0.95, line_dash="dot", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Stats")
        st.metric("Total Anomalies", len(anomalies))
        st.metric("Anomaly Rate", f"{(len(anomalies)/len(df))*100:.2f}%")
        
    # Deep Dive
    st.subheader("Anomaly Analysis")
    selected_metric = st.selectbox("Select Metric to Analyze", [
        'iot_temperature', 
        'fuel_consumption_rate', 
        'traffic_congestion_level'
    ])
    
    fig2 = px.box(df, x='is_anomaly', y=selected_metric, color='is_anomaly', 
                  title=f"{selected_metric} Distribution (Normal vs Anomaly)")
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    show()
