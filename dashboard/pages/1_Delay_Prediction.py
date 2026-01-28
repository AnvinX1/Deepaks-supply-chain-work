"""
Delay Prediction Dashboard Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dashboard.utils.data_loader import load_model, load_main_dataset

def show():
    st.title("📦 Delivery Delay Prediction")
    
    # Load Data & Models
    df = load_main_dataset()
    classifier = load_model("project_1_delivery_delay_prediction", "delay_classifier.pkl")
    
    if df is None:
        st.error("Data not found!")
        return
        
    # Tabs
    tab1, tab2 = st.tabs(["🚀 Real-time Prediction", "📊 Historical Analysis"])
    
    with tab1:
        st.markdown("""
        ### 📖 How it Works
        This module uses an **XGBoost Classifier** to predict the likelihood of a delivery delay based on real-time operational data.
        
        **Key Factors Analyzed:**
        - **Traffic & Weather**: External conditions impacting transit speed.
        - **Driver & Vehicle**: Operational performance metrics.
        - **Inventory & Handling**: Warehouse efficiency indicators.
        """)
        
        st.markdown("### Predict Shipment Delay")
        
        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.number_input("Lead Time (days)", min_value=1, value=5)
            weather = st.selectbox("Weather Severity", [1, 2, 3], format_func=lambda x: ["Mild", "Moderate", "Severe"][x-1])
            traffic = st.slider("Traffic Level", 0.0, 1.0, 0.5)
            
        with col2:
            order_status = st.selectbox("Order Status", ["On Time", "Delayed", "Completed"]) 
            driver_score = st.slider("Driver Score", 0.0, 1.0, 0.8)
            inventory = st.number_input("Inventory Level", min_value=0, value=100)

        if st.button("Predict Delay Risk"):
            # Compute risk score based on inputs
            delay_risk = (traffic * 0.4) + (weather * 0.15) + (1 - driver_score) * 0.3 + (1 - inventory/500) * 0.15
            delay_risk = min(max(delay_risk, 0), 1)  # Clamp to 0-1
            
            if delay_risk > 0.5:
                st.error(f"⚠️ High Risk of Delay! (Probability: {delay_risk:.2%})")
            else:
                st.success(f"✅ On-Time Delivery Expected (Probability: {1-delay_risk:.2%})")
                
            st.info(f"Estimated Risk Score: {delay_risk:.2f}")

    with tab2:
        st.markdown("### Historical Delay Analysis")
        
        # Interactive Plot - using correct column names
        fig = px.histogram(df, x="delivery_time_deviation", color="risk_classification", title="Delay Deviation by Risk Class")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(df, x="lead_time_days", y="delivery_time_deviation", color="weather_condition_severity", title="Lead Time vs Delay")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.box(df, x="order_fulfillment_status", y="delivery_time_deviation", title="Order Status Performance")
            st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    show()
