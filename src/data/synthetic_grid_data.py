import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic microgrid data for voltage and frequency."""
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime.now()
    timestamps = [base_timestamp + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate voltage data (nominal: 230V ± 10%)
    voltage_nominal = 230
    voltage = voltage_nominal + np.random.normal(0, 5, n_samples)
    voltage_anomalies = np.random.randint(0, n_samples, 20)
    voltage[voltage_anomalies] += np.random.normal(0, 15, 20)
    
    # Generate frequency data (nominal: 50Hz ± 0.5Hz)
    frequency_nominal = 50
    frequency = frequency_nominal + np.random.normal(0, 0.1, n_samples)
    frequency_anomalies = np.random.randint(0, n_samples, 20)
    frequency[frequency_anomalies] += np.random.normal(0, 0.3, 20)
    
    # Create load variations (0-100%)
    load = np.abs(np.sin(np.linspace(0, 10, n_samples)) * 70 + \
           np.random.normal(0, 10, n_samples))
    load = np.clip(load, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'voltage': voltage,
        'frequency': frequency,
        'load_percentage': load
    })
    
    return df

def get_training_data():
    """Get synthetic data for training and testing."""
    df = generate_synthetic_data()
    
    # Add engineered features
    df['voltage_rolling_mean'] = df['voltage'].rolling(window=5).mean()
    df['frequency_rolling_mean'] = df['frequency'].rolling(window=5).mean()
    df['voltage_rolling_std'] = df['voltage'].rolling(window=5).std()
    df['frequency_rolling_std'] = df['frequency'].rolling(window=5).std()
    
    # Forward fill NaN values created by rolling operations
    df = df.fillna(method='ffill')
    
    return df
