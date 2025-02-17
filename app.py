import streamlit as st
import pandas as pd
import numpy as np

from src.data.synthetic_grid_data import get_training_data
from src.utils.preprocessing import GridDataPreprocessor
from src.models.grid_stabilizer import GridStabilizer
from src.visualization.plots import create_time_series_plot, create_stability_gauge

# Page config
st.set_page_config(page_title="Microgrid Stabilization System",
                   layout="wide")

# Title and description
st.title("🔋 Microgrid Voltage & Frequency Stabilization")
st.markdown("""
This system monitors and predicts voltage and frequency variations in microgrids,
providing real-time stabilization recommendations.
""")

# Load and process data
@st.cache_data
def load_data():
    return get_training_data()

data = load_data()
preprocessor = GridDataPreprocessor()
stabilizer = GridStabilizer()

# Training section
st.header("Model Training")
with st.expander("Train Models"):
    if st.button("Train New Models"):
        with st.spinner("Training models..."):
            # Prepare and train voltage model
            (X_train_v, y_train_v), (X_test_v, y_test_v) = \
                preprocessor.prepare_data(data, 'voltage_scaled')
            stabilizer.train(X_train_v, y_train_v, 'voltage')
            
            # Prepare and train frequency model
            (X_train_f, y_train_f), (X_test_f, y_test_f) = \
                preprocessor.prepare_data(data, 'frequency_scaled')
            stabilizer.train(X_train_f, y_train_f, 'frequency')
            
            # Make predictions
            v_pred = stabilizer.predict(X_test_v, 'voltage')
            f_pred = stabilizer.predict(X_test_f, 'frequency')
            
            # Calculate metrics
            v_metrics = stabilizer.evaluate(y_test_v, v_pred)
            f_metrics = stabilizer.evaluate(y_test_f, f_pred)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Voltage Model Metrics")
                st.write(f"MSE: {v_metrics['mse']:.4f}")
                st.write(f"MAE: {v_metrics['mae']:.4f}")
                st.write(f"R²: {v_metrics['r2']:.4f}")
            
            with col2:
                st.subheader("Frequency Model Metrics")
                st.write(f"MSE: {f_metrics['mse']:.4f}")
                st.write(f"MAE: {f_metrics['mae']:.4f}")
                st.write(f"R²: {f_metrics['r2']:.4f}")

# Monitoring section
st.header("Real-time Monitoring")
col1, col2 = st.columns(2)

# Latest readings
latest_voltage = data['voltage'].iloc[-1]
latest_frequency = data['frequency'].iloc[-1]

with col1:
    st.plotly_chart(create_stability_gauge(
        latest_voltage, 230, 210, 250, "Voltage (V)"))
    
with col2:
    st.plotly_chart(create_stability_gauge(
        latest_frequency, 50, 49, 51, "Frequency (Hz)"))

# Time series visualization
st.header("Historical Data Analysis")
st.plotly_chart(create_time_series_plot(data), use_container_width=True)

# Stabilization recommendations
st.header("Stabilization Recommendations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Voltage Stability")
    voltage_rec = stabilizer.get_stabilization_recommendations(
        latest_voltage, latest_voltage, 'voltage')
    st.write(voltage_rec)

with col2:
    st.subheader("Frequency Stability")
    frequency_rec = stabilizer.get_stabilization_recommendations(
        latest_frequency, latest_frequency, 'frequency')
    st.write(frequency_rec)

# Footer
st.markdown("---")
st.markdown("""
### About this System
This AI-powered system uses machine learning to predict and stabilize voltage and
frequency variations in microgrids. It provides real-time monitoring, predictive
analytics, and actionable recommendations for grid stability maintenance.
""")
