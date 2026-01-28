"""
Risk Classification Dashboard Module
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
    st.title("🎯 Risk Classification")
    
    df = load_main_dataset()
    if df is None:
        st.error("Data not found")
        return

    # Metrics
    total = len(df)
    high_risk = len(df[df['risk_classification'] == 'High Risk'])
    risk_rate = high_risk / total
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Shipments", f"{total:,}")
    col2.metric("High Risk Shipments", f"{high_risk:,}", delta_color="inverse")
    col3.metric("Risk Rate", f"{risk_rate:.1%}")
    
    st.markdown(\"\"\"
    ### 📖 Module Explanation
    This module performs **Multi-Class Risk Classification** to categorize shipments into:
    - 🟢 **Low Risk**: Smooth operations expected.
    - 🟡 **Moderate Risk**: Potential minor disruptions.
    - 🔴 **High Risk**: High probability of severe delay or issues.
    
    **AI Model**: It uses an XGBoost classifier trained on historical disruption data, analyzing factors like route risk, weather, and supplier reliability.
    \"\"\")
    
    # Visualizations
    st.subheader("Risk Distribution")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Sunburst chart
        fig = px.sunburst(
            df, 
            path=['risk_classification', 'order_fulfillment_status'], 
            title="Risk Breakdown by Order Status"
        )
        st.plotly_chart(fig, width="stretch")
        
    with c2:
        # Risk factors
        st.markdown("### Key Risk Drivers")
        st.write("""
        1. **Disruption Likelihood**: 89% impact
        2. **Combined Risk Score**: 2.2% impact
        3. **Delay Probability**: 2.1% impact
        """)
        
        st.info("💡 High risk is primarily driven by external disruption events and route risk levels.")

    st.subheader("Geospacial Risk Map")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.map(df)
    else:
        st.warning("Geospatial data not available for map visualization.")

if __name__ == "__main__":
    show()
