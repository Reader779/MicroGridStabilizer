import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class GridStabilizer:
    def __init__(self):
        self.voltage_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.frequency_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        
    def train(self, X_train, y_train, target_type='voltage'):
        """Train the model for either voltage or frequency prediction."""
        if target_type == 'voltage':
            self.voltage_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        else:
            self.frequency_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    def predict(self, X, target_type='voltage'):
        """Make predictions for either voltage or frequency."""
        X_reshaped = X.reshape(X.shape[0], -1)
        if target_type == 'voltage':
            return self.voltage_model.predict(X_reshaped)
        return self.frequency_model.predict(X_reshaped)
    
    def evaluate(self, y_true, y_pred):
        """Calculate performance metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def get_stabilization_recommendations(self, current_value, predicted_value, 
                                        target_type='voltage'):
        """Generate stabilization recommendations."""
        if target_type == 'voltage':
            nominal = 230
            lower_threshold = 220
            upper_threshold = 240
        else:
            nominal = 50
            lower_threshold = 49.5
            upper_threshold = 50.5
            
        deviation = predicted_value - nominal
        
        if predicted_value < lower_threshold:
            return f"Warning: {target_type.capitalize()} dropping below threshold. " \
                   f"Increase needed: {lower_threshold - predicted_value:.2f}"
        elif predicted_value > upper_threshold:
            return f"Warning: {target_type.capitalize()} exceeding threshold. " \
                   f"Decrease needed: {predicted_value - upper_threshold:.2f}"
        return f"{target_type.capitalize()} within normal range. " \
               f"Deviation from nominal: {deviation:.2f}"
