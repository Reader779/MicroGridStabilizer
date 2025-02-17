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

# Add system explanation
with st.expander("ℹ️ How the Stabilization System Works", expanded=False):
    st.markdown("""
    ### AI-Powered Grid Stabilization

    This system uses advanced machine learning to maintain grid stability:

    1. **Data Collection & Analysis**
       - Continuous monitoring of voltage and frequency
       - Real-time data processing and anomaly detection
       - Historical data analysis for pattern recognition

    2. **Stabilization Process**
       - Predictive modeling of grid behavior
       - Early warning system for potential instabilities
       - Automated recommendation generation

    3. **Decision Making**
       - Analysis of current grid state
       - Comparison with historical patterns
       - Risk assessment and mitigation strategies
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
    st.markdown("""
    ### Model Training Process
    The system uses Random Forest models for both voltage and frequency prediction:
    - **Feature Engineering**: Rolling means and standard deviations
    - **Training Data**: Historical grid measurements
    - **Validation**: Cross-validation with multiple metrics
    """)

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
    st.markdown("""
    #### Voltage Stability Indicators
    - 🟢 **220V-240V**: Optimal Range
    - 🟡 **215V-220V / 240V-245V**: Warning Range
    - 🔴 **<215V / >245V**: Critical Range
    """)

with monitor_col2:
    st.plotly_chart(create_stability_gauge(
        display_frequency, 50, 49, 51, "Frequency (Hz)"))
    st.markdown("""
    #### Frequency Stability Indicators
    - 🟢 **49.8Hz-50.2Hz**: Optimal Range
    - 🟡 **49.5Hz-49.8Hz / 50.2Hz-50.5Hz**: Warning Range
    - 🔴 **<49.5Hz / >50.5Hz**: Critical Range
    """)

# Time series visualization
st.header("Historical Data Analysis")
st.markdown("""
### Understanding the Patterns
The time series graph shows:
- **Blue Line**: Actual voltage measurements
- **Green Line**: Actual frequency measurements
- **Dotted Lines**: Predicted values (when available)
- **Shaded Areas**: Normal operating ranges
""")
st.plotly_chart(create_time_series_plot(data), use_container_width=True)

# Stabilization recommendations
st.header("Stabilization Recommendations")

if analyze_button:
    st.success("Analysis completed based on input values!")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader("Voltage Stability Analysis")
        voltage_rec = stabilizer.get_stabilization_recommendations(
            input_voltage, input_voltage, 'voltage')
        st.write(voltage_rec)

        # Add detailed voltage analysis
        v_deviation = abs(input_voltage - 230)
        st.markdown(f"""
        #### Detailed Analysis
        - **Deviation from Nominal**: {v_deviation:.2f}V
        - **Stability Status**: {
            "✅ Stable" if v_deviation < 5 else
            "⚠️ Marginal" if v_deviation < 10 else
            "❌ Unstable"
        }
        - **Required Action**: {
            "No action needed" if v_deviation < 5 else
            "Monitor closely" if v_deviation < 10 else
            "Immediate intervention required"
        }
        """)

    with rec_col2:
        st.subheader("Frequency Stability Analysis")
        frequency_rec = stabilizer.get_stabilization_recommendations(
            input_frequency, input_frequency, 'frequency')
        st.write(frequency_rec)

        # Add detailed frequency analysis
        f_deviation = abs(input_frequency - 50)
        st.markdown(f"""
        #### Detailed Analysis
        - **Deviation from Nominal**: {f_deviation:.3f}Hz
        - **Stability Status**: {
            "✅ Stable" if f_deviation < 0.2 else
            "⚠️ Marginal" if f_deviation < 0.5 else
            "❌ Unstable"
        }
        - **Required Action**: {
            "No action needed" if f_deviation < 0.2 else
            "Monitor closely" if f_deviation < 0.5 else
            "Immediate intervention required"
        }
        """)
else:
    st.info("👆 Enter values and click 'Analyze Stability' to get recommendations")

# Footer with technical details
st.markdown("---")
st.markdown("""
### Technical Details

#### System Components
1. **Data Processing**
   - Real-time data acquisition
   - Signal processing and filtering
   - Feature engineering

2. **AI Model**
   - Random Forest Regressor
   - Sliding window analysis
   - Anomaly detection

3. **Stabilization Logic**
   - Rule-based threshold monitoring
   - Predictive analytics
   - Risk assessment algorithms

#### How to Use:
1. Enter voltage and frequency values in the input fields
2. Click 'Analyze Stability' to get recommendations
3. View real-time monitoring gauges and historical data
4. Train new models as needed using the training section

#### Safety Features:
- Continuous monitoring of grid parameters
- Early warning system for potential issues
- Automated recommendations for stability maintenance
- Historical data analysis for pattern recognition
""")