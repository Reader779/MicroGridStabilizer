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
    """Create a gauge chart for voltage or frequency."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [lower, upper]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [lower, nominal-nominal*0.02], 'color': "red"},
                {'range': [nominal-nominal*0.02, nominal+nominal*0.02], 
                 'color': "green"},
                {'range': [nominal+nominal*0.02, upper], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    return fig
