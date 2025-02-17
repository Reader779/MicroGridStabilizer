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

# Input Section
st.header("📊 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    input_voltage = st.number_input(
        "Enter Voltage (V)",
        min_value=200.0,
        max_value=260.0,
        value=230.0,
        step=0.1,
        help="Normal range: 220V-240V"
    )

with col2:
    input_frequency = st.number_input(
        "Enter Frequency (Hz)",
        min_value=48.0,
        max_value=52.0,
        value=50.0,
        step=0.1,
        help="Normal range: 49.5Hz-50.5Hz"
    )

analyze_button = st.button("Analyze Stability")

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
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.subheader("Voltage Model Metrics")
                st.write(f"MSE: {v_metrics['mse']:.4f}")
                st.write(f"MAE: {v_metrics['mae']:.4f}")
                st.write(f"R²: {v_metrics['r2']:.4f}")

            with metric_col2:
                st.subheader("Frequency Model Metrics")
                st.write(f"MSE: {f_metrics['mse']:.4f}")
                st.write(f"MAE: {f_metrics['mae']:.4f}")
                st.write(f"R²: {f_metrics['r2']:.4f}")

# Monitoring section
st.header("Real-time Monitoring")
monitor_col1, monitor_col2 = st.columns(2)

# Use input values if analyze button is clicked, otherwise use latest readings
display_voltage = input_voltage if analyze_button else data['voltage'].iloc[-1]
display_frequency = input_frequency if analyze_button else data['frequency'].iloc[-1]

with monitor_col1:
    st.plotly_chart(create_stability_gauge(
        display_voltage, 230, 210, 250, "Voltage (V)"))

with monitor_col2:
    st.plotly_chart(create_stability_gauge(
        display_frequency, 50, 49, 51, "Frequency (Hz)"))

# Time series visualization
st.header("Historical Data Analysis")
st.plotly_chart(create_time_series_plot(data), use_container_width=True)

# Stabilization recommendations
st.header("Stabilization Recommendations")

if analyze_button:
    st.success("Analysis completed based on input values!")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader("Voltage Stability")
        voltage_rec = stabilizer.get_stabilization_recommendations(
            input_voltage, input_voltage, 'voltage')
        st.write(voltage_rec)

    with rec_col2:
        st.subheader("Frequency Stability")
        frequency_rec = stabilizer.get_stabilization_recommendations(
            input_frequency, input_frequency, 'frequency')
        st.write(frequency_rec)
else:
    st.info("👆 Enter values and click 'Analyze Stability' to get recommendations")

# Footer
st.markdown("---")
st.markdown("""
### About this System
This AI-powered system uses machine learning to predict and stabilize voltage and
frequency variations in microgrids. It provides real-time monitoring, predictive
analytics, and actionable recommendations for grid stability maintenance.

#### How to Use:
1. Enter voltage and frequency values in the input fields
2. Click 'Analyze Stability' to get recommendations
3. View real-time monitoring gauges and historical data
4. Train new models as needed using the training section
""")