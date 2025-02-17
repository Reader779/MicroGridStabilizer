import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_time_series_plot(df, predictions_v=None, predictions_f=None):
    """Create interactive time series plot for voltage and frequency."""
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Voltage Over Time', 
                                      'Frequency Over Time'))

    # Voltage plot
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['voltage'],
                  name='Actual Voltage',
                  line=dict(color='blue')),
        row=1, col=1
    )

    if predictions_v is not None:
        fig.add_trace(
            go.Scatter(x=df['timestamp'][-len(predictions_v):],
                      y=predictions_v,
                      name='Predicted Voltage',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )

    # Frequency plot
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['frequency'],
                  name='Actual Frequency',
                  line=dict(color='green')),
        row=2, col=1
    )

    if predictions_f is not None:
        fig.add_trace(
            go.Scatter(x=df['timestamp'][-len(predictions_f):],
                      y=predictions_f,
                      name='Predicted Frequency',
                      line=dict(color='orange', dash='dash')),
            row=2, col=1
        )

    fig.update_layout(height=800, showlegend=True,
                     title_text="Microgrid Stability Monitoring")
    return fig

def create_stability_gauge(value, nominal, lower, upper, title):
    """Create a gauge chart for voltage or frequency with detailed ranges."""
    # Calculate warning thresholds
    if "Voltage" in title:
        critical_lower = lower  # 210V
        warning_lower = nominal - nominal * 0.05  # 218.5V
        warning_upper = nominal + nominal * 0.05  # 241.5V
        critical_upper = upper  # 250V
        steps = [
            {'range': [critical_lower, warning_lower], 'color': "red"},
            {'range': [warning_lower, nominal-nominal*0.02], 'color': "yellow"},
            {'range': [nominal-nominal*0.02, nominal+nominal*0.02], 'color': "green"},
            {'range': [nominal+nominal*0.02, warning_upper], 'color': "yellow"},
            {'range': [warning_upper, critical_upper], 'color': "red"}
        ]
        min_text = f"Min: {critical_lower}V"
        max_text = f"Max: {critical_upper}V"
        nominal_text = f"Nominal: {nominal}V"
    else:  # Frequency gauge
        critical_lower = lower  # 49Hz
        warning_lower = nominal - 0.3  # 49.7Hz
        warning_upper = nominal + 0.3  # 50.3Hz
        critical_upper = upper  # 51Hz
        steps = [
            {'range': [critical_lower, warning_lower], 'color': "red"},
            {'range': [warning_lower, nominal-0.1], 'color': "yellow"},
            {'range': [nominal-0.1, nominal+0.1], 'color': "green"},
            {'range': [nominal+0.1, warning_upper], 'color': "yellow"},
            {'range': [warning_upper, critical_upper], 'color': "red"}
        ]
        min_text = f"Min: {critical_lower}Hz"
        max_text = f"Max: {critical_upper}Hz"
        nominal_text = f"Nominal: {nominal}Hz"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [lower, upper],
                    'tickmode': 'array',
                    'ticktext': [str(lower), str(nominal), str(upper)],
                    'tickvals': [lower, nominal, upper]},
            'bar': {'color': "darkblue"},
            'steps': steps,
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    # Add min/max/nominal annotations
    fig.add_annotation(
        x=0.1, y=0.8,
        text=min_text,
        showarrow=False,
        font=dict(color="red")
    )
    fig.add_annotation(
        x=0.9, y=0.8,
        text=max_text,
        showarrow=False,
        font=dict(color="red")
    )
    fig.add_annotation(
        x=0.5, y=0.9,
        text=nominal_text,
        showarrow=False,
        font=dict(color="green")
    )

    # Add reference line for nominal value
    fig.add_shape(
        type="line",
        x0=0.5,
        x1=0.5,
        y0=0.45,
        y1=0.55,
        line=dict(color="white", width=2)
    )

    # Update layout for better annotation visibility
    fig.update_layout(
        margin=dict(t=100, b=0, l=0, r=0),
        height=300
    )

    return fig