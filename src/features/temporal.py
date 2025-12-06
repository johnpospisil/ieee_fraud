"""
Advanced Temporal Feature Engineering for IEEE-CIS Fraud Detection

This module creates sophisticated time-based features including:
- Time since last transaction (by card/email/device)
- Transaction velocity in rolling windows
- RFM (Recency, Frequency, Monetary) features
- Time-of-day and day-of-week risk patterns

These features capture temporal patterns in user behavior that are critical
for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import timedelta


class TemporalFeatureEngine:
    """
    Creates advanced temporal features for fraud detection.
    
    Features include:
    - Time differences between consecutive transactions
    - Transaction velocity (count in rolling windows)
    - RFM (Recency, Frequency, Monetary) metrics
    - Time-of-day risk patterns
    """
    
    def __init__(self):
        """Initialize the TemporalFeatureEngine."""
        self.reference_time = None
        
    def create_time_since_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features based on time since last transaction.
        
        For each grouping key (card, email, device), calculate:
        - Time since previous transaction (seconds)
        - Time since first transaction (days)
        - Time since last transaction (minutes)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with TransactionDT column
            
        Returns:
        --------
        tuple
            (DataFrame with time-since features, list of feature names)
        """
        print("Creating time-since-last-transaction features...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionDT' not in df.columns:
            print("  Warning: TransactionDT not found, skipping time-since features")
            return df, feature_names
        
        # Sort by time to ensure correct ordering
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        
        # Grouping keys
        group_keys = ['card1', 'card2', 'P_emaildomain', 'R_emaildomain', 
                      'DeviceInfo', 'DeviceType', 'addr1']
        
        for key in group_keys:
            if key not in df.columns:
                continue
            
            # Time since previous transaction for this key
            feat_name = f'{key}_time_since_prev'
            df[feat_name] = df.groupby(key)['TransactionDT'].diff()
            
            # Convert to hours for better scale
            df[feat_name] = df[feat_name] / 3600
            feature_names.append(feat_name)
            
            # Time since first transaction for this key
            feat_name_first = f'{key}_time_since_first'
            first_time = df.groupby(key)['TransactionDT'].transform('first')
            df[feat_name_first] = (df['TransactionDT'] - first_time) / 3600
            feature_names.append(feat_name_first)
            
            # Number of transactions so far for this key
            feat_name_count = f'{key}_txn_count_so_far'
            df[feat_name_count] = df.groupby(key).cumcount() + 1
            feature_names.append(feat_name_count)
        
        print(f"  Created {len(feature_names)} time-since features")
        return df, feature_names
    
    def create_velocity_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create transaction velocity features in rolling time windows.
        
        For each grouping key, calculate:
        - Number of transactions in last 1 hour, 6 hours, 24 hours, 7 days
        - Mean amount in rolling windows
        - Transaction frequency patterns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with TransactionDT and TransactionAmt
            
        Returns:
        --------
        tuple
            (DataFrame with velocity features, list of feature names)
        """
        print("Creating transaction velocity features...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionDT' not in df.columns:
            print("  Warning: TransactionDT not found, skipping velocity features")
            return df, feature_names
        
        # Sort by time
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        
        # Time windows in seconds
        windows = {
            '1h': 3600,
            '6h': 6 * 3600,
            '24h': 24 * 3600,
            '7d': 7 * 24 * 3600
        }
        
        # Grouping keys
        group_keys = ['card1', 'P_emaildomain', 'DeviceInfo', 'addr1']
        
        for key in group_keys:
            if key not in df.columns:
                continue
            
            for window_name, window_seconds in windows.items():
                # Count transactions in window
                feat_name = f'{key}_velocity_{window_name}'
                
                # For each transaction, count how many transactions from same key in the window
                velocities = []
                for idx, row in df.iterrows():
                    current_time = row['TransactionDT']
                    current_key_val = row[key]
                    
                    # Count transactions in window before current transaction
                    mask = (
                        (df[key] == current_key_val) &
                        (df['TransactionDT'] < current_time) &
                        (df['TransactionDT'] >= current_time - window_seconds)
                    )
                    count = mask.sum()
                    velocities.append(count)
                
                df[feat_name] = velocities
                feature_names.append(feat_name)
        
        print(f"  Created {len(feature_names)} velocity features")
        return df, feature_names
    
    def create_rfm_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create RFM (Recency, Frequency, Monetary) features.
        
        For each grouping key, calculate:
        - Recency: Days since last transaction
        - Frequency: Total number of transactions
        - Monetary: Total and average transaction amounts
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with RFM features, list of feature names)
        """
        print("Creating RFM features...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionDT' not in df.columns or 'TransactionAmt' not in df.columns:
            print("  Warning: Required columns not found, skipping RFM features")
            return df, feature_names
        
        # Reference time (latest transaction)
        max_time = df['TransactionDT'].max()
        
        # Grouping keys
        group_keys = ['card1', 'card2', 'P_emaildomain', 'addr1', 'DeviceInfo']
        
        for key in group_keys:
            if key not in df.columns:
                continue
            
            # Recency: Time since last transaction (in hours)
            recency_name = f'{key}_recency'
            last_time = df.groupby(key)['TransactionDT'].transform('max')
            df[recency_name] = (max_time - last_time) / 3600
            feature_names.append(recency_name)
            
            # Frequency: Total transaction count
            freq_name = f'{key}_frequency'
            df[freq_name] = df.groupby(key)['TransactionDT'].transform('count')
            feature_names.append(freq_name)
            
            # Monetary: Total amount
            monetary_total = f'{key}_monetary_total'
            df[monetary_total] = df.groupby(key)['TransactionAmt'].transform('sum')
            feature_names.append(monetary_total)
            
            # Monetary: Average amount
            monetary_mean = f'{key}_monetary_mean'
            df[monetary_mean] = df.groupby(key)['TransactionAmt'].transform('mean')
            feature_names.append(monetary_mean)
            
            # Monetary: Std amount
            monetary_std = f'{key}_monetary_std'
            df[monetary_std] = df.groupby(key)['TransactionAmt'].transform('std')
            feature_names.append(monetary_std)
            
            # RFM Score (simple combination)
            rfm_score = f'{key}_rfm_score'
            # Normalize recency (lower is better), frequency (higher is better), monetary (higher is better)
            recency_norm = 1 / (df[recency_name] + 1)
            freq_norm = df[freq_name] / (df[freq_name].max() + 1)
            monetary_norm = df[monetary_total] / (df[monetary_total].max() + 1)
            df[rfm_score] = recency_norm + freq_norm + monetary_norm
            feature_names.append(rfm_score)
        
        print(f"  Created {len(feature_names)} RFM features")
        return df, feature_names
    
    def create_time_pattern_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features based on time-of-day and day-of-week patterns.
        
        Features include:
        - Hour of day, day of week (if not already present)
        - Is weekend, is business hours
        - Time-based fraud rate patterns
        - Cyclical encoding of time features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with TransactionDT
            
        Returns:
        --------
        tuple
            (DataFrame with time pattern features, list of feature names)
        """
        print("Creating time pattern features...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionDT' not in df.columns:
            print("  Warning: TransactionDT not found, skipping time pattern features")
            return df, feature_names
        
        # Convert TransactionDT to datetime if needed
        # Assuming TransactionDT is seconds since reference
        reference_date = pd.Timestamp('2017-12-01')
        df['datetime'] = reference_date + pd.to_timedelta(df['TransactionDT'], unit='s')
        
        # Extract time components if not present
        if 'DT_hour' not in df.columns:
            df['DT_hour'] = df['datetime'].dt.hour
            feature_names.append('DT_hour')
        
        if 'DT_day_of_week' not in df.columns:
            df['DT_day_of_week'] = df['datetime'].dt.dayofweek
            feature_names.append('DT_day_of_week')
        
        # Business hours (9-17)
        df['is_business_hours'] = ((df['DT_hour'] >= 9) & (df['DT_hour'] < 17)).astype(int)
        feature_names.append('is_business_hours')
        
        # Late night (0-6)
        df['is_late_night'] = (df['DT_hour'] < 6).astype(int)
        feature_names.append('is_late_night')
        
        # Cyclical encoding of hour (captures circular nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['DT_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['DT_hour'] / 24)
        feature_names.extend(['hour_sin', 'hour_cos'])
        
        # Cyclical encoding of day of week
        df['day_sin'] = np.sin(2 * np.pi * df['DT_day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['DT_day_of_week'] / 7)
        feature_names.extend(['day_sin', 'day_cos'])
        
        # Time of day bins
        df['time_of_day'] = pd.cut(
            df['DT_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        feature_names.append('time_of_day')
        
        # Drop temporary datetime column
        df = df.drop('datetime', axis=1)
        
        print(f"  Created {len(feature_names)} time pattern features")
        return df, feature_names
    
    def create_transaction_gap_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features based on gaps between transactions.
        
        Features include:
        - Average gap between transactions
        - Std of gaps
        - Min/max gaps
        - Unusual gap indicator (gap > 2*std from mean)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with gap features, list of feature names)
        """
        print("Creating transaction gap features...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionDT' not in df.columns:
            print("  Warning: TransactionDT not found, skipping gap features")
            return df, feature_names
        
        # Sort by time
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        
        # Grouping keys
        group_keys = ['card1', 'P_emaildomain', 'DeviceInfo']
        
        for key in group_keys:
            if key not in df.columns:
                continue
            
            # Calculate time gaps
            gap_col = f'{key}_gap'
            df[gap_col] = df.groupby(key)['TransactionDT'].diff() / 3600  # hours
            
            # Mean gap
            mean_gap = f'{key}_mean_gap'
            df[mean_gap] = df.groupby(key)[gap_col].transform('mean')
            feature_names.append(mean_gap)
            
            # Std of gap
            std_gap = f'{key}_std_gap'
            df[std_gap] = df.groupby(key)[gap_col].transform('std')
            feature_names.append(std_gap)
            
            # Min gap
            min_gap = f'{key}_min_gap'
            df[min_gap] = df.groupby(key)[gap_col].transform('min')
            feature_names.append(min_gap)
            
            # Max gap
            max_gap = f'{key}_max_gap'
            df[max_gap] = df.groupby(key)[gap_col].transform('max')
            feature_names.append(max_gap)
            
            # Unusual gap indicator
            unusual_gap = f'{key}_unusual_gap'
            df[unusual_gap] = (
                (df[gap_col] > df[mean_gap] + 2 * df[std_gap]) |
                (df[gap_col] < df[mean_gap] - 2 * df[std_gap])
            ).astype(int)
            feature_names.append(unusual_gap)
            
            # Remove temporary gap column
            df = df.drop(gap_col, axis=1)
        
        print(f"  Created {len(feature_names)} gap features")
        return df, feature_names
    
    def create_all_temporal_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create all temporal features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with all temporal features, list of feature names)
        """
        print("\n" + "="*60)
        print("CREATING ADVANCED TEMPORAL FEATURES")
        print("="*60 + "\n")
        
        all_feature_names = []
        
        # Time since last transaction
        df, features = self.create_time_since_features(df)
        all_feature_names.extend(features)
        
        # Transaction velocity (WARNING: This can be slow for large datasets)
        print("\nNote: Velocity features may take several minutes for large datasets...")
        df, features = self.create_velocity_features(df)
        all_feature_names.extend(features)
        
        # RFM features
        df, features = self.create_rfm_features(df)
        all_feature_names.extend(features)
        
        # Time pattern features
        df, features = self.create_time_pattern_features(df)
        all_feature_names.extend(features)
        
        # Transaction gap features
        df, features = self.create_transaction_gap_features(df)
        all_feature_names.extend(features)
        
        print("\n" + "="*60)
        print(f"TOTAL TEMPORAL FEATURES CREATED: {len(all_feature_names)}")
        print("="*60 + "\n")
        
        return df, all_feature_names


def create_temporal_features(df: pd.DataFrame, 
                            include_velocity: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all temporal features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with TransactionDT and TransactionAmt
    include_velocity : bool, default=True
        Whether to include velocity features (can be slow for large datasets)
        
    Returns:
    --------
    tuple
        (DataFrame with temporal features, list of feature names)
        
    Example:
    --------
    >>> df_with_temporal, temporal_features = create_temporal_features(train_df)
    >>> print(f"Created {len(temporal_features)} temporal features")
    """
    engine = TemporalFeatureEngine()
    
    if not include_velocity:
        # Skip velocity features for faster processing
        print("\n" + "="*60)
        print("CREATING TEMPORAL FEATURES (without velocity)")
        print("="*60 + "\n")
        
        all_feature_names = []
        
        df, features = engine.create_time_since_features(df)
        all_feature_names.extend(features)
        
        df, features = engine.create_rfm_features(df)
        all_feature_names.extend(features)
        
        df, features = engine.create_time_pattern_features(df)
        all_feature_names.extend(features)
        
        df, features = engine.create_transaction_gap_features(df)
        all_feature_names.extend(features)
        
        print("\n" + "="*60)
        print(f"TOTAL TEMPORAL FEATURES CREATED: {len(all_feature_names)}")
        print("="*60 + "\n")
        
        return df, all_feature_names
    
    return engine.create_all_temporal_features(df)
