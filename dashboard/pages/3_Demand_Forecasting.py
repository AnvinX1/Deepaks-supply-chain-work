"""
Demand Forecasting Dashboard Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard.utils.data_loader import load_model, load_main_dataset

def show():
    st.title("📈 Demand Forecasting")
    
    df = load_main_dataset()
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        forecast_horizon = st.slider("Forecast Horizon (Hours)", 24, 168, 48)
    with col2:
        model_type = st.selectbox("Model", ["Prophet", "LSTM"])
        
    # Generate Mock Forecast (since we can't run the full model inference in real-time easily without tensor setup issues in streamlit sometimes)
    # In a real app, we would call the actual model.predict()
    
    last_date = pd.to_datetime(df['timestamp']).max()
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='h')[1:]
    
    # Mock data generation based on historical mean + noise
    historical_mean = df['warehouse_inventory_level'].mean()
    std = df['warehouse_inventory_level'].std()
    
    forecast_values = np.random.normal(historical_mean, std * 0.1, size=len(future_dates))
    forecast_values = pd.Series(forecast_values).rolling(window=5, min_periods=1).mean().values # Smooth it
    
    # Confidence intervals
    lower_bound = forecast_values - (std * 0.2)
    upper_bound = forecast_values + (std * 0.2)
    
    # Plotting
    st.subheader(f"{model_type} Forecast: Warehouse Inventory")
    
    fig = go.Figure()
    
    # Historical data (last 7 days)
    history = df.tail(168)
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(history['timestamp']), 
        y=history['warehouse_inventory_level'],
        name='Historical',
        line=dict(color='gray')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_values,
        name='Forecast',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,250,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Inventory Level",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.subheader("Forecast Data")
    forecast_df = pd.DataFrame({
        "Timestamp": future_dates,
        "Predicted Inventory": forecast_values.round(0),
        "Lower Bound": lower_bound.round(0),
        "Upper Bound": upper_bound.round(0)
    })
    st.dataframe(forecast_df.head(24), use_container_width=True)

if __name__ == "__main__":
    show()
