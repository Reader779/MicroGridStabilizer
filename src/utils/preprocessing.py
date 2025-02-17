import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GridDataPreprocessor:
    def __init__(self):
        self.voltage_scaler = MinMaxScaler()
        self.frequency_scaler = MinMaxScaler()
        self.sequence_length = 10
        
    def create_sequences(self, data, target_col):
        """Create sequences for time series prediction."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length][target_col]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, df, target_col):
        """Prepare data for model training."""
        # Scale the data
        df['voltage_scaled'] = self.voltage_scaler.fit_transform(
            df[['voltage']].values
        )
        df['frequency_scaled'] = self.frequency_scaler.fit_transform(
            df[['frequency']].values
        )
        
        # Create feature matrix
        features = ['voltage_scaled', 'frequency_scaled', 
                   'load_percentage']
        
        # Create sequences
        X, y = self.create_sequences(df[features].values, 
                                   features.index(target_col))
        
        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return (X_train, y_train), (X_test, y_test)
    
    def inverse_transform_voltage(self, scaled_values):
        """Convert scaled values back to original voltage values."""
        return self.voltage_scaler.inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()
    
    def inverse_transform_frequency(self, scaled_values):
        """Convert scaled values back to original frequency values."""
        return self.frequency_scaler.inverse_transform(
            scaled_values.reshape(-1, 1)
        ).flatten()
