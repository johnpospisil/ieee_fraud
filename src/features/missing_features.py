"""
Missing Value Feature Engineering for IEEE-CIS Fraud Detection

This module creates features based on missing value patterns, which can be highly
informative for fraud detection as fraudsters may deliberately omit information.

Key Features:
- Missing value indicators for each feature
- Missing value counts by feature groups (V, C, D, M)
- Co-occurrence patterns of missing values
- Missing value ratios and statistics
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


class MissingValueFeatureEngine:
    """
    Creates features based on missing value patterns.
    
    Features capture:
    - Which features are missing
    - How many features are missing in each group
    - Co-occurrence of missing values
    - Unusual missingness patterns
    """
    
    def __init__(self):
        """Initialize the MissingValueFeatureEngine."""
        pass
    
    def create_basic_missing_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create basic missing value indicators.
        
        For each column with missing values, create a binary indicator.
        Also create total missing count and percentage.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with missing indicators, list of feature names)
        """
        print("Creating basic missing value indicators...")
        
        df = df.copy()
        feature_names = []
        
        # Overall missing statistics
        df['total_missing_count'] = df.isnull().sum(axis=1)
        feature_names.append('total_missing_count')
        
        df['total_missing_pct'] = df['total_missing_count'] / len(df.columns)
        feature_names.append('total_missing_pct')
        
        # Individual missing indicators for columns with high missing rates
        missing_rates = df.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > 0.05].index.tolist()
        
        # Exclude target and ID columns
        exclude_cols = ['isFraud', 'TransactionID']
        high_missing_cols = [c for c in high_missing_cols if c not in exclude_cols]
        
        print(f"  Creating indicators for {len(high_missing_cols)} columns with >5% missing")
        
        for col in high_missing_cols[:50]:  # Limit to avoid too many features
            indicator_name = f'{col}_ismissing'
            df[indicator_name] = df[col].isnull().astype(int)
            feature_names.append(indicator_name)
        
        print(f"  Created {len(feature_names)} basic missing indicators")
        return df, feature_names
    
    def create_group_missing_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create missing value features grouped by column prefixes.
        
        Groups include:
        - V columns (Vesta engineered features)
        - C columns (Counting features)
        - D columns (Timedelta features)
        - M columns (Match features)
        - id columns (Identity features)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with group missing features, list of feature names)
        """
        print("Creating group-level missing features...")
        
        df = df.copy()
        feature_names = []
        
        # Define column groups
        groups = {
            'V': [c for c in df.columns if c.startswith('V')],
            'C': [c for c in df.columns if c.startswith('C')],
            'D': [c for c in df.columns if c.startswith('D')],
            'M': [c for c in df.columns if c.startswith('M')],
            'id': [c for c in df.columns if c.startswith('id_')]
        }
        
        for group_name, group_cols in groups.items():
            if not group_cols:
                continue
            
            # Count of missing values in group
            count_name = f'{group_name}_missing_count'
            df[count_name] = df[group_cols].isnull().sum(axis=1)
            feature_names.append(count_name)
            
            # Percentage of missing values in group
            pct_name = f'{group_name}_missing_pct'
            df[pct_name] = df[count_name] / len(group_cols)
            feature_names.append(pct_name)
            
            # All missing indicator
            all_missing_name = f'{group_name}_all_missing'
            df[all_missing_name] = (df[count_name] == len(group_cols)).astype(int)
            feature_names.append(all_missing_name)
            
            # None missing indicator
            none_missing_name = f'{group_name}_none_missing'
            df[none_missing_name] = (df[count_name] == 0).astype(int)
            feature_names.append(none_missing_name)
        
        print(f"  Created {len(feature_names)} group missing features")
        return df, feature_names
    
    def create_cooccurrence_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features based on co-occurrence of missing values.
        
        Identifies which features tend to be missing together, which can
        indicate specific data collection patterns or fraud behaviors.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with co-occurrence features, list of feature names)
        """
        print("Creating missing value co-occurrence features...")
        
        df = df.copy()
        feature_names = []
        
        # Key columns to check for co-occurrence
        key_cols = {
            'email': ['P_emaildomain', 'R_emaildomain'],
            'address': ['addr1', 'addr2'],
            'device': ['DeviceType', 'DeviceInfo'],
            'identity': ['id_30', 'id_31', 'id_33']
        }
        
        for group_name, cols in key_cols.items():
            existing_cols = [c for c in cols if c in df.columns]
            if len(existing_cols) < 2:
                continue
            
            # Both missing
            both_missing_name = f'{group_name}_both_missing'
            df[both_missing_name] = df[existing_cols].isnull().all(axis=1).astype(int)
            feature_names.append(both_missing_name)
            
            # Exactly one missing
            one_missing_name = f'{group_name}_one_missing'
            missing_count = df[existing_cols].isnull().sum(axis=1)
            df[one_missing_name] = (missing_count == 1).astype(int)
            feature_names.append(one_missing_name)
            
            # All present
            all_present_name = f'{group_name}_all_present'
            df[all_present_name] = (~df[existing_cols].isnull()).all(axis=1).astype(int)
            feature_names.append(all_present_name)
        
        print(f"  Created {len(feature_names)} co-occurrence features")
        return df, feature_names
    
    def create_card_missing_patterns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features based on missing patterns in card-related columns.
        
        Card columns (card1-6) have specific patterns that may indicate
        different payment methods or fraud attempts.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with card missing pattern features, list of feature names)
        """
        print("Creating card-related missing pattern features...")
        
        df = df.copy()
        feature_names = []
        
        card_cols = [c for c in df.columns if c.startswith('card') and c[4:].isdigit()]
        
        if not card_cols:
            print("  No card columns found")
            return df, feature_names
        
        # Count of missing card features
        df['card_missing_count'] = df[card_cols].isnull().sum(axis=1)
        feature_names.append('card_missing_count')
        
        # Percentage of missing card features
        df['card_missing_pct'] = df['card_missing_count'] / len(card_cols)
        feature_names.append('card_missing_pct')
        
        # Specific card missing indicators
        for col in card_cols:
            indicator_name = f'{col}_missing'
            df[indicator_name] = df[col].isnull().astype(int)
            feature_names.append(indicator_name)
        
        print(f"  Created {len(feature_names)} card missing features")
        return df, feature_names
    
    def create_unusual_missing_patterns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features identifying unusual missing patterns.
        
        Features include:
        - Unusually high/low missing rates compared to average
        - Specific combinations that are rare
        - Missing in critical vs non-critical features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with unusual pattern features, list of feature names)
        """
        print("Creating unusual missing pattern features...")
        
        df = df.copy()
        feature_names = []
        
        # Critical features (should rarely be missing)
        critical_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2']
        critical_existing = [c for c in critical_features if c in df.columns]
        
        if critical_existing:
            df['critical_missing_count'] = df[critical_existing].isnull().sum(axis=1)
            feature_names.append('critical_missing_count')
            
            df['has_critical_missing'] = (df['critical_missing_count'] > 0).astype(int)
            feature_names.append('has_critical_missing')
        
        # Device info missing but transaction present pattern
        if 'DeviceInfo' in df.columns and 'DeviceType' in df.columns:
            df['device_info_anomaly'] = (
                df['DeviceInfo'].isnull() & df['DeviceType'].notnull()
            ).astype(int)
            feature_names.append('device_info_anomaly')
        
        # Email mismatch pattern
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_mismatch_pattern'] = (
                (df['P_emaildomain'].isnull() != df['R_emaildomain'].isnull())
            ).astype(int)
            feature_names.append('email_mismatch_pattern')
        
        # Address mismatch pattern
        if 'addr1' in df.columns and 'addr2' in df.columns:
            df['addr_mismatch_pattern'] = (
                (df['addr1'].isnull() != df['addr2'].isnull())
            ).astype(int)
            feature_names.append('addr_mismatch_pattern')
        
        # High V features missing (unusual for legitimate transactions)
        v_cols = [c for c in df.columns if c.startswith('V')]
        if len(v_cols) > 50:
            df['high_V_missing'] = (df[v_cols].isnull().sum(axis=1) > len(v_cols) * 0.7).astype(int)
            feature_names.append('high_V_missing')
        
        # Missing rate compared to overall average
        if 'total_missing_pct' in df.columns:
            avg_missing = df['total_missing_pct'].mean()
            df['missing_above_avg'] = (df['total_missing_pct'] > avg_missing * 1.5).astype(int)
            feature_names.append('missing_above_avg')
            
            df['missing_below_avg'] = (df['total_missing_pct'] < avg_missing * 0.5).astype(int)
            feature_names.append('missing_below_avg')
        
        print(f"  Created {len(feature_names)} unusual pattern features")
        return df, feature_names
    
    def create_missing_interaction_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create interaction features between missing patterns and other variables.
        
        For example, missing patterns may differ by ProductCD or card type.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with missing interaction features, list of feature names)
        """
        print("Creating missing value interaction features...")
        
        df = df.copy()
        feature_names = []
        
        # Missing count by ProductCD
        if 'ProductCD' in df.columns and 'total_missing_count' in df.columns:
            product_missing = df.groupby('ProductCD')['total_missing_count'].transform('mean')
            df['ProductCD_avg_missing'] = product_missing
            feature_names.append('ProductCD_avg_missing')
            
            df['missing_vs_product_avg'] = df['total_missing_count'] - product_missing
            feature_names.append('missing_vs_product_avg')
        
        # Missing count by card1
        if 'card1' in df.columns and 'total_missing_count' in df.columns:
            card1_missing = df.groupby('card1')['total_missing_count'].transform('mean')
            df['card1_avg_missing'] = card1_missing
            feature_names.append('card1_avg_missing')
        
        # Missing pattern differs from email domain average
        if 'P_emaildomain' in df.columns and 'total_missing_count' in df.columns:
            email_missing = df.groupby('P_emaildomain')['total_missing_count'].transform('mean')
            df['email_avg_missing'] = email_missing
            feature_names.append('email_avg_missing')
        
        print(f"  Created {len(feature_names)} missing interaction features")
        return df, feature_names
    
    def create_all_missing_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create all missing value features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with all missing features, list of feature names)
        """
        print("\n" + "="*60)
        print("CREATING MISSING VALUE FEATURES")
        print("="*60 + "\n")
        
        all_feature_names = []
        
        # Basic missing indicators
        df, features = self.create_basic_missing_indicators(df)
        all_feature_names.extend(features)
        
        # Group-level missing features
        df, features = self.create_group_missing_features(df)
        all_feature_names.extend(features)
        
        # Co-occurrence features
        df, features = self.create_cooccurrence_features(df)
        all_feature_names.extend(features)
        
        # Card missing patterns
        df, features = self.create_card_missing_patterns(df)
        all_feature_names.extend(features)
        
        # Unusual patterns
        df, features = self.create_unusual_missing_patterns(df)
        all_feature_names.extend(features)
        
        # Interaction features
        df, features = self.create_missing_interaction_features(df)
        all_feature_names.extend(features)
        
        print("\n" + "="*60)
        print(f"TOTAL MISSING VALUE FEATURES CREATED: {len(all_feature_names)}")
        print("="*60 + "\n")
        
        return df, all_feature_names


def create_missing_value_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all missing value features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple
        (DataFrame with missing value features, list of feature names)
        
    Example:
    --------
    >>> df_with_missing, missing_features = create_missing_value_features(train_df)
    >>> print(f"Created {len(missing_features)} missing value features")
    """
    engine = MissingValueFeatureEngine()
    return engine.create_all_missing_features(df)
