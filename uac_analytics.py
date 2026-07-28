"""
UAC Care Analytics - Complete Implementation
System Capacity & Care Load Analytics for Unaccompanied Children
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

class UACDataProcessor:
    def __init__(self, file_path):
        """Initialize the data processor with the CSV file path"""
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        self.column_mapping = {}
        
    def load_data(self):
        """Load and parse the CSV data"""
        try:
            # First, read the CSV without any conversions to see the actual column names
            self.df = pd.read_csv(self.file_path)
            
            # Print original column names for debugging
            print("📋 Original column names found:")
            for i, col in enumerate(self.df.columns):
                print(f"   {i+1}. '{col}'")
            
            # Strip whitespace from column names
            self.df.columns = self.df.columns.str.strip()
            
            # Define expected column names (with various possible formats)
            possible_names = {
                'Date': ['Date', 'date', 'DATE', 'Reporting Date', 'Reporting date'],
                'apprehended': [
                    'Children apprehended and placed in CBP custody',
                    'Children Apprehended and Placed in CBP Custody',
                    'Apprehended',
                    'apprehended'
                ],
                'cbp_custody': [
                    'Children in CBP custody',
                    'Children in CBP Custody',
                    'CBP Custody',
                    'cbp_custody'
                ],
                'transferred': [
                    'Children transferred out of CBP custody',
                    'Children Transferred out of CBP Custody',
                    'Transferred out of CBP Custody',
                    'transferred'
                ],
                'hhs_care': [
                    'Children in HHS Care',
                    'Children in HHS care',
                    'HHS Care',
                    'hhs_care'
                ],
                'discharged': [
                    'Children discharged from HHS Care',
                    'Children Discharged from HHS Care',
                    'Discharged from HHS Care',
                    'discharged'
                ]
            }
            
            # Find actual column names
            actual_columns = {}
            for key, possible in possible_names.items():
                for col in self.df.columns:
                    if col in possible or col.lower() in [p.lower() for p in possible]:
                        actual_columns[key] = col
                        break
            
            # If we couldn't find all columns, try fuzzy matching
            if len(actual_columns) < 6:
                print("⚠️  Some columns not found exactly. Trying fuzzy matching...")
                for key, possible in possible_names.items():
                    if key not in actual_columns:
                        for col in self.df.columns:
                            col_lower = col.lower()
                            for p in possible:
                                p_lower = p.lower()
                                # Check if the column name contains the key phrase
                                if p_lower in col_lower or col_lower in p_lower:
                                    actual_columns[key] = col
                                    break
                            if key in actual_columns:
                                break
            
            # Store the column mapping
            self.column_mapping = actual_columns
            
            print("\n📋 Column mapping found:")
            for key, value in actual_columns.items():
                print(f"   {key} -> '{value}'")
            
            # Rename columns for easier access
            rename_dict = {}
            for key, value in actual_columns.items():
                if value:
                    rename_dict[value] = key
            
            if rename_dict:
                self.df = self.df.rename(columns=rename_dict)
            
            # Now convert numeric columns
            numeric_columns = ['apprehended', 'cbp_custody', 'transferred', 'hhs_care', 'discharged']
            for col in numeric_columns:
                if col in self.df.columns:
                    # Remove commas and convert to numeric
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Convert Date column to datetime
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], format='%B %d, %Y', errors='coerce')
            else:
                # Try to find a date column
                for col in self.df.columns:
                    if 'date' in col.lower():
                        self.df['Date'] = pd.to_datetime(self.df[col], errors='coerce')
                        break
            
            # Remove rows with all NaN values
            self.df = self.df.dropna(how='all')
            
            # Remove rows where Date is NaN
            if 'Date' in self.df.columns:
                self.df = self.df.dropna(subset=['Date'])
            
            # Sort by date
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
            print(f"\n✅ Data loaded successfully: {len(self.df)} records")
            print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f"📊 Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_data(self):
        """Validate data quality and logical constraints"""
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        validation_results = {
            'missing_dates': [],
            'duplicate_dates': [],
            'constraint_violations': [],
            'anomalies': []
        }
        
        # Check for missing dates
        if 'Date' in self.df.columns:
            date_range = pd.date_range(
                start=self.df['Date'].min(), 
                end=self.df['Date'].max()
            )
            missing_dates = date_range[~date_range.isin(self.df['Date'])]
            validation_results['missing_dates'] = missing_dates.tolist()
            
            # Check for duplicate dates
            duplicates = self.df[self.df['Date'].duplicated()]
            if not duplicates.empty:
                validation_results['duplicate_dates'] = duplicates['Date'].tolist()
        
        # Validate logical constraints
        if 'transferred' in self.df.columns and 'cbp_custody' in self.df.columns:
            invalid_transfers = self.df[self.df['transferred'] > self.df['cbp_custody']]
            if not invalid_transfers.empty:
                validation_results['constraint_violations'].append({
                    'type': 'transfers_exceed_cbp',
                    'dates': invalid_transfers['Date'].tolist() if 'Date' in self.df.columns else []
                })
        
        if 'discharged' in self.df.columns and 'hhs_care' in self.df.columns:
            invalid_discharges = self.df[self.df['discharged'] > self.df['hhs_care']]
            if not invalid_discharges.empty:
                validation_results['constraint_violations'].append({
                    'type': 'discharges_exceed_hhs',
                    'dates': invalid_discharges['Date'].tolist() if 'Date' in self.df.columns else []
                })
        
        # Check for zero values
        cols_to_check = ['apprehended', 'cbp_custody', 'transferred', 'hhs_care', 'discharged']
        cols_present = [col for col in cols_to_check if col in self.df.columns]
        if cols_present:
            zero_mask = True
            for col in cols_present:
                zero_mask = zero_mask & (self.df[col] == 0)
            zero_children = self.df[zero_mask]
            if not zero_children.empty:
                validation_results['anomalies'].append({
                    'type': 'all_zero_values',
                    'dates': zero_children['Date'].tolist() if 'Date' in self.df.columns else []
                })
        
        # Print validation summary
        print("\n📊 Data Validation Summary:")
        print(f"  • Missing dates: {len(validation_results['missing_dates'])}")
        print(f"  • Duplicate dates: {len(validation_results['duplicate_dates'])}")
        print(f"  • Constraint violations: {len(validation_results['constraint_violations'])}")
        print(f"  • Anomalies detected: {len(validation_results['anomalies'])}")
        
        return validation_results
    
    def create_features(self):
        """Create derived features for analysis"""
        if self.df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        df = self.df.copy()
        
        # Ensure we have the required columns with correct names
        required_cols = ['apprehended', 'cbp_custody', 'transferred', 'hhs_care', 'discharged']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  Column '{col}' not found. Please check the data.")
                # Try to find alternative names
                for df_col in df.columns:
                    if col.lower() in df_col.lower():
                        df[col] = df[df_col]
                        print(f"   Mapped '{df_col}' to '{col}'")
                        break
        
        # Fill NaN values with 0 for calculations
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 1. Total System Load
        if 'cbp_custody' in df.columns and 'hhs_care' in df.columns:
            df['total_system_load'] = df['cbp_custody'] + df['hhs_care']
        
        # 2. Net Daily Intake
        if 'transferred' in df.columns and 'discharged' in df.columns:
            df['net_intake'] = df['transferred'] - df['discharged']
        
        # 3. Care Load Growth Rate (day-over-day)
        if 'hhs_care' in df.columns:
            df['hhs_growth_rate'] = df['hhs_care'].pct_change() * 100
        
        # 4. CBP Growth Rate
        if 'cbp_custody' in df.columns:
            df['cbp_growth_rate'] = df['cbp_custody'].pct_change() * 100
        
        # 5. Backlog Indicator
        if 'net_intake' in df.columns:
            df['backlog_indicator'] = (df['net_intake'] > 0).astype(int)
            df['cumulative_backlog'] = df['net_intake'].cumsum()
        
        # 6. Rolling averages
        if 'hhs_care' in df.columns:
            df['hhs_7d_avg'] = df['hhs_care'].rolling(window=7, min_periods=1).mean()
            df['hhs_14d_avg'] = df['hhs_care'].rolling(window=14, min_periods=1).mean()
        
        if 'total_system_load' in df.columns:
            df['total_7d_avg'] = df['total_system_load'].rolling(window=7, min_periods=1).mean()
            df['total_14d_avg'] = df['total_system_load'].rolling(window=14, min_periods=1).mean()
        
        # 7. Discharge Offset Ratio
        if 'discharged' in df.columns and 'transferred' in df.columns:
            df['discharge_offset_ratio'] = df['discharged'] / df['transferred']
            df['discharge_offset_ratio'] = df['discharge_offset_ratio'].replace([np.inf, -np.inf], np.nan)
        
        # 8. Care Load Volatility Index (rolling std)
        if 'hhs_care' in df.columns:
            df['hhs_volatility'] = df['hhs_care'].rolling(window=7, min_periods=2).std()
        
        if 'total_system_load' in df.columns:
            df['total_volatility'] = df['total_system_load'].rolling(window=7, min_periods=2).std()
        
        # 9. Day of week, month, year features
        if 'Date' in df.columns:
            df['day_of_week'] = df['Date'].dt.day_name()
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year
            df['quarter'] = df['Date'].dt.quarter
        
        # 10. Pressure identification
        if 'hhs_care' in df.columns:
            hhs_std = df['hhs_care'].std()
            df['hhs_pressure'] = (df['hhs_care'] > df['hhs_care'].mean() + hhs_std).astype(int)
        
        # 11. Cumulative metrics
        if 'apprehended' in df.columns:
            df['cumulative_apprehended'] = df['apprehended'].cumsum()
        if 'transferred' in df.columns:
            df['cumulative_transferred'] = df['transferred'].cumsum()
        if 'discharged' in df.columns:
            df['cumulative_discharged'] = df['discharged'].cumsum()
        
        # 12. Flow metrics
        if 'transferred' in df.columns and 'cbp_custody' in df.columns:
            df['transfer_to_cbp_ratio'] = df['transferred'] / df['cbp_custody']
            df['transfer_to_cbp_ratio'] = df['transfer_to_cbp_ratio'].replace([np.inf, -np.inf], np.nan)
        
        self.processed_df = df
        print(f"✅ Feature engineering complete: {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def get_processed_data(self):
        """Return the processed DataFrame"""
        if self.processed_df is None:
            print("⚠️  Data not processed. Running create_features()...")
            self.create_features()
        return self.processed_df
    
    def get_summary_statistics(self):
        """Generate summary statistics for KPIs"""
        df = self.get_processed_data()
        
        summary = {
            'Total Records': len(df),
            'Date Range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            'Avg CBP Custody': df['cbp_custody'].mean() if 'cbp_custody' in df.columns else 0,
            'Avg HHS Care': df['hhs_care'].mean() if 'hhs_care' in df.columns else 0,
            'Avg Total System Load': df['total_system_load'].mean() if 'total_system_load' in df.columns else 0,
            'Max Total System Load': df['total_system_load'].max() if 'total_system_load' in df.columns else 0,
            'Avg Net Intake': df['net_intake'].mean() if 'net_intake' in df.columns else 0,
            'Total Apprehended': df['apprehended'].sum() if 'apprehended' in df.columns else 0,
            'Total Transferred': df['transferred'].sum() if 'transferred' in df.columns else 0,
            'Total Discharged': df['discharged'].sum() if 'discharged' in df.columns else 0,
            'Avg Discharge Offset Ratio': df['discharge_offset_ratio'].mean() if 'discharge_offset_ratio' in df.columns else 0,
            'Current HHS Load': df['hhs_care'].iloc[-1] if 'hhs_care' in df.columns and len(df) > 0 else 0,
            'Current CBP Load': df['cbp_custody'].iloc[-1] if 'cbp_custody' in df.columns and len(df) > 0 else 0,
            'Current System Load': df['total_system_load'].iloc[-1] if 'total_system_load' in df.columns and len(df) > 0 else 0,
        }
        
        return summary


class UACAnalytics:
    def __init__(self, data_processor):
        """Initialize with processed data"""
        self.df = data_processor.get_processed_data()
        self.data_processor = data_processor
        
    def calculate_kpis(self):
        """Calculate all Key Performance Indicators"""
        df = self.df
        
        kpis = {
            'Total Children Under Care': {
                'current': df['total_system_load'].iloc[-1] if 'total_system_load' in df.columns and len(df) > 0 else 0,
                'avg': df['total_system_load'].mean() if 'total_system_load' in df.columns else 0,
                'max': df['total_system_load'].max() if 'total_system_load' in df.columns else 0,
                'min': df['total_system_load'].min() if 'total_system_load' in df.columns else 0
            },
            'Net Intake Pressure': {
                'current': df['net_intake'].iloc[-1] if 'net_intake' in df.columns and len(df) > 0 else 0,
                'avg': df['net_intake'].mean() if 'net_intake' in df.columns else 0,
                'max': df['net_intake'].max() if 'net_intake' in df.columns else 0,
                'min': df['net_intake'].min() if 'net_intake' in df.columns else 0,
                'positive_days_pct': (df['net_intake'] > 0).mean() * 100 if 'net_intake' in df.columns else 0
            },
            'Volatility Index': {
                'hhs_volatility': df['hhs_volatility'].mean() if 'hhs_volatility' in df.columns else 0,
                'total_volatility': df['total_volatility'].mean() if 'total_volatility' in df.columns else 0,
            },
            'Backlog Accumulation': {
                'cumulative_backlog': df['cumulative_backlog'].iloc[-1] if 'cumulative_backlog' in df.columns and len(df) > 0 else 0,
                'avg_daily_backlog': df['net_intake'].mean() if 'net_intake' in df.columns else 0,
                'backlog_days': (df['backlog_indicator'] == 1).sum() if 'backlog_indicator' in df.columns else 0
            },
            'Discharge Offset Ratio': {
                'current': df['discharge_offset_ratio'].iloc[-1] if 'discharge_offset_ratio' in df.columns and len(df) > 0 else 0,
                'avg': df['discharge_offset_ratio'].mean() if 'discharge_offset_ratio' in df.columns else 0,
                'min': df['discharge_offset_ratio'].min() if 'discharge_offset_ratio' in df.columns else 0,
                'max': df['discharge_offset_ratio'].max() if 'discharge_offset_ratio' in df.columns else 0
            }
        }
        
        return kpis
    
    def analyze_trends(self):
        """Analyze temporal trends"""
        df = self.df
        
        trends = {}
        for col in ['hhs_care', 'cbp_custody', 'total_system_load', 'apprehended', 'discharged', 'transferred']:
            if col in df.columns:
                trends[col] = self._compute_trend_metrics(df, col)
        
        return trends
    
    def _compute_trend_metrics(self, df, column):
        """Compute trend metrics for a specific column"""
        data = df[column].dropna()
        
        if len(data) < 2:
            return {'error': 'Insufficient data'}
        
        # Linear trend
        x = np.arange(len(data))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        except:
            return {'error': 'Regression failed'}
        
        # Recent trend (last 30 days)
        recent_data = data.iloc[-30:] if len(data) >= 30 else data
        if len(recent_data) >= 2:
            recent_x = np.arange(len(recent_data))
            try:
                recent_slope, recent_intercept, recent_r, recent_p, recent_std = stats.linregress(recent_x, recent_data)
                recent_r2 = recent_r**2
            except:
                recent_slope = 0
                recent_r2 = 0
        else:
            recent_slope = 0
            recent_r2 = 0
        
        # Calculate percentage changes
        pct_change = data.pct_change().dropna()
        
        return {
            'overall_trend_slope': slope,
            'overall_trend_r2': r_value**2,
            'overall_trend_p_value': p_value,
            'recent_trend_slope': recent_slope,
            'recent_trend_r2': recent_r2,
            'avg_daily_change': pct_change.mean() if len(pct_change) > 0 else 0,
            'max_daily_change': pct_change.max() if len(pct_change) > 0 else 0,
            'min_daily_change': pct_change.min() if len(pct_change) > 0 else 0,
            'volatility': pct_change.std() if len(pct_change) > 0 else 0,
        }
    
    def identify_pressure_periods(self):
        """Identify periods of high system pressure"""
        df = self.df
        
        if 'hhs_care' not in df.columns:
            return {'error': 'hhs_care column not found'}
        
        # Define pressure thresholds
        hhs_mean = df['hhs_care'].mean()
        hhs_std = df['hhs_care'].std()
        
        # High pressure: > 1 std above mean
        df['hhs_high_pressure'] = df['hhs_care'] > (hhs_mean + hhs_std)
        df['extreme_pressure'] = df['hhs_care'] > (hhs_mean + 2 * hhs_std)
        
        # Identify pressure periods
        pressure_periods = []
        current_period = None
        
        for idx, row in df.iterrows():
            if row['hhs_high_pressure']:
                if current_period is None:
                    current_period = {'start': row['Date'], 'end': row['Date']}
                else:
                    current_period['end'] = row['Date']
            else:
                if current_period is not None:
                    current_period['duration_days'] = (current_period['end'] - current_period['start']).days + 1
                    pressure_periods.append(current_period)
                    current_period = None
        
        # Add last period if still ongoing
        if current_period is not None:
            current_period['duration_days'] = (current_period['end'] - current_period['start']).days + 1
            pressure_periods.append(current_period)
        
        # Calculate pressure statistics
        pressure_stats = {
            'total_pressure_days': df['hhs_high_pressure'].sum(),
            'pressure_percentage': (df['hhs_high_pressure'].sum() / len(df)) * 100,
            'extreme_pressure_days': df['extreme_pressure'].sum(),
            'num_pressure_periods': len(pressure_periods),
            'avg_pressure_duration': np.mean([p['duration_days'] for p in pressure_periods]) if pressure_periods else 0,
            'max_pressure_duration': max([p['duration_days'] for p in pressure_periods]) if pressure_periods else 0,
            'pressure_periods': pressure_periods
        }
        
        return pressure_stats
    
    def create_visualizations(self):
        """Create key visualizations"""
        df = self.df
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top-left: Total System Load
        if 'total_system_load' in df.columns and 'Date' in df.columns:
            axes[0, 0].fill_between(df['Date'], df['total_system_load'], alpha=0.3, color='blue')
            axes[0, 0].plot(df['Date'], df['total_system_load'], color='blue', linewidth=1.5)
            axes[0, 0].axhline(df['total_system_load'].mean(), color='red', linestyle='--', label='Mean')
            axes[0, 0].set_title('Total System Load Over Time', fontsize=14)
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Number of Children')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Top-right: CBP vs HHS Load
        if 'cbp_custody' in df.columns and 'hhs_care' in df.columns and 'Date' in df.columns:
            axes[0, 1].plot(df['Date'], df['cbp_custody'], label='CBP Custody', color='orange', linewidth=1.5)
            axes[0, 1].plot(df['Date'], df['hhs_care'], label='HHS Care', color='green', linewidth=1.5)
            axes[0, 1].set_title('CBP vs HHS Care Load', fontsize=14)
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Number of Children')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Bottom-left: Net Intake
        if 'net_intake' in df.columns and 'Date' in df.columns:
            colors = ['green' if x > 0 else 'red' for x in df['net_intake']]
            axes[1, 0].bar(df['Date'], df['net_intake'], color=colors, alpha=0.7)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 0].set_title('Net Daily Intake (Inflow - Outflow)', fontsize=14)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Net Intake')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Bottom-right: Cumulative Backlog
        if 'cumulative_backlog' in df.columns and 'Date' in df.columns:
            axes[1, 1].plot(df['Date'], df['cumulative_backlog'], color='purple', linewidth=1.5)
            axes[1, 1].fill_between(df['Date'], 0, df['cumulative_backlog'], alpha=0.3, color='purple')
            axes[1, 1].set_title('Cumulative Backlog', fontsize=14)
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Cumulative Backlog')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Second set of visualizations
        fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))
        
        # Top: HHS Care with rolling averages
        if 'hhs_care' in df.columns and 'Date' in df.columns:
            axes2[0].plot(df['Date'], df['hhs_care'], label='Daily HHS Care', alpha=0.5, color='blue')
            if 'hhs_7d_avg' in df.columns:
                axes2[0].plot(df['Date'], df['hhs_7d_avg'], label='7-Day Moving Average', color='red', linewidth=2)
            if 'hhs_14d_avg' in df.columns:
                axes2[0].plot(df['Date'], df['hhs_14d_avg'], label='14-Day Moving Average', color='green', linewidth=2)
            axes2[0].set_title('HHS Care Load with Rolling Averages', fontsize=14)
            axes2[0].set_xlabel('Date')
            axes2[0].set_ylabel('Number of Children')
            axes2[0].legend()
            axes2[0].tick_params(axis='x', rotation=45)
        
        # Bottom: Pressure Indicators
        if 'hhs_care' in df.columns and 'Date' in df.columns and 'hhs_pressure' in df.columns:
            pressure_colors = ['red' if x == 1 else 'green' for x in df['hhs_pressure']]
            axes2[1].scatter(df['Date'], df['hhs_care'], c=pressure_colors, alpha=0.6, s=20)
            if 'hhs_14d_avg' in df.columns:
                axes2[1].plot(df['Date'], df['hhs_14d_avg'], label='14-Day Average', color='blue', linewidth=1.5)
            axes2[1].axhline(y=df['hhs_care'].mean() + df['hhs_care'].std(), 
                           color='red', linestyle='--', label='High Pressure Threshold')
            axes2[1].set_title('Pressure Indicators (Red = High Pressure)', fontsize=14)
            axes2[1].set_xlabel('Date')
            axes2[1].set_ylabel('HHS Care Load')
            axes2[1].legend()
            axes2[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig, fig2


class UACForecaster:
    def __init__(self, data_processor):
        """Initialize with processed data"""
        self.df = data_processor.get_processed_data()
        self.data_processor = data_processor
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, target='hhs_care'):
        """Prepare features for forecasting"""
        df = self.df.copy()
        
        # Create lag features
        for lag in [1, 3, 7, 14]:
            if target in df.columns:
                df[f'lag_{lag}'] = df[target].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14]:
            if target in df.columns:
                df[f'rolling_mean_{window}'] = df[target].rolling(window).mean()
                df[f'rolling_std_{window}'] = df[target].rolling(window).std()
        
        # Date features
        if 'Date' in df.columns:
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter
        
        # Add other relevant features
        for col in ['apprehended', 'transferred', 'discharged', 'cbp_custody']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
        
        # Remove rows with NaN
        df = df.dropna()
        
        # Define features
        feature_cols = [col for col in df.columns if col not in ['Date', target, 'cumulative_backlog']]
        
        return df, feature_cols
    
    def train_random_forest(self, target='hhs_care'):
        """Train Random Forest model for forecasting"""
        df, feature_cols = self.prepare_features(target=target)
        
        if len(df) < 30:
            print("❌ Insufficient data for training")
            return None
        
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
    
    def forecast_future(self, target='hhs_care', horizon=30):
        """Generate future forecasts"""
        if 'random_forest' not in self.models:
            print("❌ No trained model found. Train a model first.")
            return None
        
        if target not in self.df.columns or 'Date' not in self.df.columns:
            print("❌ Required columns not found")
            return None
        
        # Prepare features for future dates
        future_dates = pd.date_range(
            start=self.df['Date'].iloc[-1] + timedelta(days=1),
            periods=horizon
        )
        
        forecasts = []
        
        for i, date in enumerate(future_dates):
            # Create features for this date
            feature_dict = {}
            
            # Get previous values
            prev_idx = len(self.df) - 1 + i - 1
            if prev_idx >= len(self.df):
                prev_idx = len(self.df) - 1
            
            # Add lag features
            for lag in [1, 3, 7, 14]:
                lag_idx = prev_idx - lag + 1
                if lag_idx < 0:
                    lag_idx = 0
                if lag_idx < len(self.df):
                    feature_dict[f'lag_{lag}'] = self.df.iloc[lag_idx][target]
                else:
                    feature_dict[f'lag_{lag}'] = self.df.iloc[-1][target]
            
            # Add rolling statistics
            for window in [3, 7, 14]:
                start_idx = max(0, prev_idx - window + 1)
                window_data = self.df.iloc[start_idx:prev_idx+1][target]
                feature_dict[f'rolling_mean_{window}'] = window_data.mean()
                feature_dict[f'rolling_std_{window}'] = window_data.std()
            
            # Add date features
            feature_dict['day_of_week'] = date.dayofweek
            feature_dict['month'] = date.month
            feature_dict['quarter'] = date.quarter
            
            # Add other features
            for col in ['apprehended', 'transferred', 'discharged', 'cbp_custody']:
                if col in self.df.columns:
                    feature_dict[f'{col}_lag1'] = self.df.iloc[-1][col]
            
            # Create DataFrame for prediction
            features_df = pd.DataFrame([feature_dict])
            
            # Ensure all feature columns exist
            for col in self.models['random_forest']['feature_cols']:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Select only the required features
            features_df = features_df[self.models['random_forest']['feature_cols']]
            
            # Scale features
            features_scaled = self.models['random_forest']['scaler'].transform(features_df)
            
            # Make prediction
            prediction = self.models['random_forest']['model'].predict(features_scaled)[0]
            
            forecasts.append({
                'date': date,
                'predicted_value': max(0, prediction)
            })
        
        forecasts_df = pd.DataFrame(forecasts)
        return forecasts_df


def main():
    """Main execution function"""
    print("=" * 60)
    print("UAC CARE ANALYTICS - SYSTEM CAPACITY ANALYSIS")
    print("=" * 60)
    
    # File path - UPDATE THIS TO YOUR FILE LOCATION
    file_path = r"C:\Users\Lenovo\HHS_Unaccompanied_Alien_Children_Program (1).csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please update the file path in the code.")
        return
    
    # 1. Load and Process Data
    print("\n📁 Loading Data...")
    processor = UACDataProcessor(file_path)
    df = processor.load_data()
    
    if df is None or len(df) == 0:
        print("❌ Failed to load data. Please check the file path and format.")
        return
    
    # 2. Validate Data
    print("\n🔍 Validating Data...")
    validation = processor.validate_data()
    
    # 3. Feature Engineering
    print("\n⚙️ Creating Features...")
    processed_df = processor.create_features()
    
    # 4. Summary Statistics
    print("\n📊 Summary Statistics:")
    summary = processor.get_summary_statistics()
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  • {key}: {value:,.2f}" if isinstance(value, float) else f"  • {key}: {value:,}")
        else:
            print(f"  • {key}: {value}")
    
    # 5. Analytics
    print("\n📈 Running Analytics...")
    analytics = UACAnalytics(processor)
    
    # Calculate KPIs
    kpis = analytics.calculate_kpis()
    print("\n🎯 Key Performance Indicators:")
    for kpi_name, kpi_values in kpis.items():
        print(f"\n  {kpi_name}:")
        for metric, value in kpi_values.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value > 1000:
                    print(f"    • {metric}: {value:,.2f}")
                elif isinstance(value, float):
                    print(f"    • {metric}: {value:.2f}")
                else:
                    print(f"    • {metric}: {value:,}")
            else:
                print(f"    • {metric}: {value}")
    
    # Analyze Trends
    trends = analytics.analyze_trends()
    print("\n📉 Trend Analysis:")
    for metric, values in trends.items():
        if 'error' not in values:
            print(f"\n  {metric}:")
            print(f"    • Overall trend slope: {values['overall_trend_slope']:.2f}")
            print(f"    • Recent trend slope: {values['recent_trend_slope']:.2f}")
            print(f"    • Volatility: {values['volatility']:.2f}%")
    
    # Identify Pressure Periods
    pressure = analytics.identify_pressure_periods()
    if 'error' not in pressure:
        print(f"\n🔴 Pressure Analysis:")
        print(f"  • Pressure days: {pressure['total_pressure_days']} ({pressure['pressure_percentage']:.1f}%)")
        print(f"  • Extreme pressure days: {pressure['extreme_pressure_days']}")
        print(f"  • Number of pressure periods: {pressure['num_pressure_periods']}")
        print(f"  • Average pressure duration: {pressure['avg_pressure_duration']:.1f} days")
        print(f"  • Max pressure duration: {pressure['max_pressure_duration']:.0f} days")
    else:
        print(f"\n⚠️  Pressure analysis: {pressure['error']}")
    
    # 6. Forecasting
    print("\n🤖 Training Forecasting Model...")
    forecaster = UACForecaster(processor)
    rf_model = forecaster.train_random_forest()
    
    if rf_model:
        print("\n📊 Feature Importance (Top 10):")
        print(rf_model['feature_importance'].head(10))
        
        # Generate future forecast
        print("\n🔮 Generating 30-day Forecast...")
        forecast = forecaster.forecast_future(horizon=30)
        if forecast is not None:
            print("\n  Forecasted Values (Next 7 days):")
            for i in range(min(7, len(forecast))):
                row = forecast.iloc[i]
                print(f"    • {row['date'].strftime('%Y-%m-%d')}: {row['predicted_value']:.0f} children")
            print("  ...")
            print(f"  • ... and {len(forecast) - 7} more days")
    
    # 7. Visualizations
    print("\n📊 Creating Visualizations...")
    analytics.create_visualizations()
    
    print("\n✅ Analysis Complete!")
    return processor, analytics, forecaster


if __name__ == "__main__":
    main()
