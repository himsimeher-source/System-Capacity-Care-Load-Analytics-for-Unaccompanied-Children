"""
Forecasting Module for UAC Care Analytics
Uses time series forecasting methods to predict future care load
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class UACForecaster:
    def __init__(self, data_processor):
        """Initialize with processed data"""
        self.df = data_processor.get_processed_data()
        self.data_processor = data_processor
        self.models = {}
        self.forecasts = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, target='hhs_care', forecast_horizon=30):
        """Prepare features for forecasting"""
        df = self.df.copy()
        
        # Create lag features
        for lag in [1, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = df[target].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[target].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target].rolling(window).std()
        
        # Date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        
        # Add other relevant features
        df['apprehended_lag1'] = df['apprehended'].shift(1)
        df['transferred_lag1'] = df['transferred_to_hhs'].shift(1)
        df['discharged_lag1'] = df['discharged'].shift(1)
        df['cbp_lag1'] = df['cbp_custody'].shift(1)
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Define features
        feature_cols = [col for col in df.columns if col not in ['date', target, 'cumulative_backlog']]
        
        return df, feature_cols
    
    def train_random_forest(self, target='hhs_care', horizon=30):
        """Train Random Forest model for forecasting"""
        df, feature_cols = self.prepare_features(target=target)
        
        # Split data
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target]
        X_test = test_data[feature_cols]
        y_test = test_data[target]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = rf_model.predict(X_train_scaled)
        test_pred = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['random_forest'] = {
            'model': rf_model,
            'scaler': self.scaler,
            'feature_cols': feature_cols,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"✅ Random Forest Model Trained")
        print(f"   • Train R²: {metrics['train_r2']:.4f}")
        print(f"   • Test R²: {metrics['test_r2']:.4f}")
        print(f"   • Test MAE: {metrics['test_mae']:.2f}")
        
        return self.models['random_forest']
    
    def train_arima(self, target='hhs_care'):
        """Train ARIMA model"""
        data = self.df[target].dropna()
        
        # Use last 60 days for testing
        train_data = data[:-30]
        test_data = data
