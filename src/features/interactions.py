"""
Interaction Feature Engineering for IEEE-CIS Fraud Detection

This module creates interaction features by combining different categorical variables
and calculating risk scores based on historical fraud rates.

Key Features:
- Card × Address combinations
- Card × Email combinations
- Device × Browser combinations
- Amount Bin × Product combinations
- Card × Product combinations
- Fraud rate-based risk scores
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class InteractionFeatureEngine:
    """
    Creates interaction features by combining categorical variables
    and computing fraud-based risk scores.
    """
    
    def __init__(self, min_samples: int = 10):
        """
        Initialize the InteractionFeatureEngine.
        
        Parameters:
        -----------
        min_samples : int, default=10
            Minimum number of samples required for a combination to compute fraud rate.
            Combinations with fewer samples will use the global fraud rate.
        """
        self.min_samples = min_samples
        self.global_fraud_rate = None
        
    def _compute_fraud_rate(self, df: pd.DataFrame, group_cols: List[str], 
                           target_col: str = 'isFraud') -> pd.DataFrame:
        """
        Compute fraud rate for each combination of group columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        group_cols : list
            List of columns to group by
        target_col : str, default='isFraud'
            Name of the target column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with fraud rate statistics
        """
        # Compute global fraud rate as fallback
        if self.global_fraud_rate is None:
            self.global_fraud_rate = df[target_col].mean()
        
        # Group and calculate stats
        grouped = df.groupby(group_cols).agg({
            target_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        grouped.columns = group_cols + ['fraud_count', 'total_count', 'fraud_rate']
        
        # Use global fraud rate for combinations with few samples
        grouped['fraud_rate'] = np.where(
            grouped['total_count'] >= self.min_samples,
            grouped['fraud_rate'],
            self.global_fraud_rate
        )
        
        # Add smoothed fraud rate (Bayesian average)
        alpha = 10  # Smoothing parameter
        grouped['fraud_rate_smoothed'] = (
            (grouped['fraud_count'] + alpha * self.global_fraud_rate) / 
            (grouped['total_count'] + alpha)
        )
        
        return grouped
    
    def create_card_address_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between card and address variables.
        
        Features created:
        - card1 × addr1, card1 × addr2
        - card2 × addr1, card2 × addr2
        - Fraud rate for each combination
        - Transaction count for each combination
        """
        print("Creating card × address interactions...")
        
        df = df.copy()
        feature_names = []
        
        # Card1 × Address combinations
        for card_col in ['card1', 'card2']:
            if card_col not in df.columns:
                continue
                
            for addr_col in ['addr1', 'addr2']:
                if addr_col not in df.columns:
                    continue
                
                # Create combination column
                combo_name = f'{card_col}_{addr_col}'
                df[combo_name] = df[card_col].astype(str) + '_' + df[addr_col].astype(str)
                feature_names.append(combo_name)
                
                # Count transactions per combination
                count_name = f'{combo_name}_count'
                combo_counts = df.groupby(combo_name).size()
                df[count_name] = df[combo_name].map(combo_counts)
                feature_names.append(count_name)
                
                # Fraud rate per combination (if target available)
                if 'isFraud' in df.columns:
                    fraud_stats = self._compute_fraud_rate(df, [combo_name])
                    
                    rate_name = f'{combo_name}_fraud_rate'
                    df[rate_name] = df[combo_name].map(
                        fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                    ).fillna(self.global_fraud_rate)
                    feature_names.append(rate_name)
        
        print(f"  Created {len(feature_names)} card × address features")
        return df, feature_names
    
    def create_card_email_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between card and email variables.
        
        Features created:
        - card1 × P_emaildomain, card1 × R_emaildomain
        - card2 × P_emaildomain, card2 × R_emaildomain
        - Fraud rate for each combination
        """
        print("Creating card × email interactions...")
        
        df = df.copy()
        feature_names = []
        
        # Card × Email combinations
        for card_col in ['card1', 'card2', 'card4', 'card6']:
            if card_col not in df.columns:
                continue
                
            for email_col in ['P_emaildomain', 'R_emaildomain']:
                if email_col not in df.columns:
                    continue
                
                # Create combination column
                combo_name = f'{card_col}_{email_col}'
                df[combo_name] = df[card_col].astype(str) + '_' + df[email_col].astype(str)
                feature_names.append(combo_name)
                
                # Count transactions per combination
                count_name = f'{combo_name}_count'
                combo_counts = df.groupby(combo_name).size()
                df[count_name] = df[combo_name].map(combo_counts)
                feature_names.append(count_name)
                
                # Fraud rate per combination
                if 'isFraud' in df.columns:
                    fraud_stats = self._compute_fraud_rate(df, [combo_name])
                    
                    rate_name = f'{combo_name}_fraud_rate'
                    df[rate_name] = df[combo_name].map(
                        fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                    ).fillna(self.global_fraud_rate)
                    feature_names.append(rate_name)
        
        print(f"  Created {len(feature_names)} card × email features")
        return df, feature_names
    
    def create_device_browser_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between device and browser variables.
        
        Features created:
        - DeviceType × id_31 (browser)
        - DeviceInfo × id_31
        - id_30 (OS) × id_31 (browser)
        - Fraud rates for each combination
        """
        print("Creating device × browser interactions...")
        
        df = df.copy()
        feature_names = []
        
        # Device × Browser combinations
        device_cols = ['DeviceType', 'DeviceInfo', 'id_30']
        browser_col = 'id_31'
        
        if browser_col in df.columns:
            for device_col in device_cols:
                if device_col not in df.columns:
                    continue
                
                # Create combination column
                combo_name = f'{device_col}_{browser_col}'
                df[combo_name] = df[device_col].astype(str) + '_' + df[browser_col].astype(str)
                feature_names.append(combo_name)
                
                # Count transactions per combination
                count_name = f'{combo_name}_count'
                combo_counts = df.groupby(combo_name).size()
                df[count_name] = df[combo_name].map(combo_counts)
                feature_names.append(count_name)
                
                # Fraud rate per combination
                if 'isFraud' in df.columns:
                    fraud_stats = self._compute_fraud_rate(df, [combo_name])
                    
                    rate_name = f'{combo_name}_fraud_rate'
                    df[rate_name] = df[combo_name].map(
                        fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                    ).fillna(self.global_fraud_rate)
                    feature_names.append(rate_name)
        
        print(f"  Created {len(feature_names)} device × browser features")
        return df, feature_names
    
    def create_amount_product_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between amount bins and product.
        
        Features created:
        - Amount bins (low, medium, high, very_high)
        - Amount bin × ProductCD
        - Fraud rates for each combination
        """
        print("Creating amount × product interactions...")
        
        df = df.copy()
        feature_names = []
        
        if 'TransactionAmt' not in df.columns or 'ProductCD' not in df.columns:
            print("  Skipping: TransactionAmt or ProductCD not found")
            return df, feature_names
        
        # Create amount bins
        df['amt_bin'] = pd.cut(
            df['TransactionAmt'],
            bins=[0, 50, 150, 500, np.inf],
            labels=['low', 'medium', 'high', 'very_high']
        ).astype(str)
        feature_names.append('amt_bin')
        
        # Amount bin × ProductCD
        combo_name = 'amt_bin_ProductCD'
        df[combo_name] = df['amt_bin'] + '_' + df['ProductCD'].astype(str)
        feature_names.append(combo_name)
        
        # Count transactions per combination
        count_name = f'{combo_name}_count'
        combo_counts = df.groupby(combo_name).size()
        df[count_name] = df[combo_name].map(combo_counts)
        feature_names.append(count_name)
        
        # Fraud rate per combination
        if 'isFraud' in df.columns:
            fraud_stats = self._compute_fraud_rate(df, [combo_name])
            
            rate_name = f'{combo_name}_fraud_rate'
            df[rate_name] = df[combo_name].map(
                fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
            ).fillna(self.global_fraud_rate)
            feature_names.append(rate_name)
        
        print(f"  Created {len(feature_names)} amount × product features")
        return df, feature_names
    
    def create_card_product_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between card and product variables.
        
        Features created:
        - card1 × ProductCD, card2 × ProductCD
        - card4 × ProductCD, card6 × ProductCD
        - Fraud rates for each combination
        """
        print("Creating card × product interactions...")
        
        df = df.copy()
        feature_names = []
        
        if 'ProductCD' not in df.columns:
            print("  Skipping: ProductCD not found")
            return df, feature_names
        
        # Card × Product combinations
        for card_col in ['card1', 'card2', 'card4', 'card6']:
            if card_col not in df.columns:
                continue
            
            # Create combination column
            combo_name = f'{card_col}_ProductCD'
            df[combo_name] = df[card_col].astype(str) + '_' + df['ProductCD'].astype(str)
            feature_names.append(combo_name)
            
            # Count transactions per combination
            count_name = f'{combo_name}_count'
            combo_counts = df.groupby(combo_name).size()
            df[count_name] = df[combo_name].map(combo_counts)
            feature_names.append(count_name)
            
            # Fraud rate per combination
            if 'isFraud' in df.columns:
                fraud_stats = self._compute_fraud_rate(df, [combo_name])
                
                rate_name = f'{combo_name}_fraud_rate'
                df[rate_name] = df[combo_name].map(
                    fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                ).fillna(self.global_fraud_rate)
                feature_names.append(rate_name)
        
        print(f"  Created {len(feature_names)} card × product features")
        return df, feature_names
    
    def create_complex_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complex 3-way interaction features.
        
        Features created:
        - card1 × addr1 × ProductCD
        - card1 × P_emaildomain × ProductCD
        - DeviceType × id_31 × ProductCD
        """
        print("Creating complex 3-way interactions...")
        
        df = df.copy()
        feature_names = []
        
        # Card × Address × Product
        if all(col in df.columns for col in ['card1', 'addr1', 'ProductCD']):
            combo_name = 'card1_addr1_ProductCD'
            df[combo_name] = (df['card1'].astype(str) + '_' + 
                             df['addr1'].astype(str) + '_' + 
                             df['ProductCD'].astype(str))
            feature_names.append(combo_name)
            
            # Count and fraud rate
            combo_counts = df.groupby(combo_name).size()
            df[f'{combo_name}_count'] = df[combo_name].map(combo_counts)
            feature_names.append(f'{combo_name}_count')
            
            if 'isFraud' in df.columns:
                fraud_stats = self._compute_fraud_rate(df, [combo_name])
                df[f'{combo_name}_fraud_rate'] = df[combo_name].map(
                    fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                ).fillna(self.global_fraud_rate)
                feature_names.append(f'{combo_name}_fraud_rate')
        
        # Card × Email × Product
        if all(col in df.columns for col in ['card1', 'P_emaildomain', 'ProductCD']):
            combo_name = 'card1_email_ProductCD'
            df[combo_name] = (df['card1'].astype(str) + '_' + 
                             df['P_emaildomain'].astype(str) + '_' + 
                             df['ProductCD'].astype(str))
            feature_names.append(combo_name)
            
            combo_counts = df.groupby(combo_name).size()
            df[f'{combo_name}_count'] = df[combo_name].map(combo_counts)
            feature_names.append(f'{combo_name}_count')
            
            if 'isFraud' in df.columns:
                fraud_stats = self._compute_fraud_rate(df, [combo_name])
                df[f'{combo_name}_fraud_rate'] = df[combo_name].map(
                    fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                ).fillna(self.global_fraud_rate)
                feature_names.append(f'{combo_name}_fraud_rate')
        
        # Device × Browser × Product
        if all(col in df.columns for col in ['DeviceType', 'id_31', 'ProductCD']):
            combo_name = 'device_browser_ProductCD'
            df[combo_name] = (df['DeviceType'].astype(str) + '_' + 
                             df['id_31'].astype(str) + '_' + 
                             df['ProductCD'].astype(str))
            feature_names.append(combo_name)
            
            combo_counts = df.groupby(combo_name).size()
            df[f'{combo_name}_count'] = df[combo_name].map(combo_counts)
            feature_names.append(f'{combo_name}_count')
            
            if 'isFraud' in df.columns:
                fraud_stats = self._compute_fraud_rate(df, [combo_name])
                df[f'{combo_name}_fraud_rate'] = df[combo_name].map(
                    fraud_stats.set_index(combo_name)['fraud_rate_smoothed']
                ).fillna(self.global_fraud_rate)
                feature_names.append(f'{combo_name}_fraud_rate')
        
        print(f"  Created {len(feature_names)} complex interaction features")
        return df, feature_names
    
    def create_all_interactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create all interaction features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            (DataFrame with interaction features, list of feature names)
        """
        print("\n" + "="*60)
        print("CREATING INTERACTION FEATURES")
        print("="*60 + "\n")
        
        all_feature_names = []
        
        # Card × Address
        df, features = self.create_card_address_interactions(df)
        all_feature_names.extend(features)
        
        # Card × Email
        df, features = self.create_card_email_interactions(df)
        all_feature_names.extend(features)
        
        # Device × Browser
        df, features = self.create_device_browser_interactions(df)
        all_feature_names.extend(features)
        
        # Amount × Product
        df, features = self.create_amount_product_interactions(df)
        all_feature_names.extend(features)
        
        # Card × Product
        df, features = self.create_card_product_interactions(df)
        all_feature_names.extend(features)
        
        # Complex 3-way interactions
        df, features = self.create_complex_interactions(df)
        all_feature_names.extend(features)
        
        print("\n" + "="*60)
        print(f"TOTAL INTERACTION FEATURES CREATED: {len(all_feature_names)}")
        print("="*60 + "\n")
        
        return df, all_feature_names


def create_interaction_features(df: pd.DataFrame, 
                                min_samples: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all interaction features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    min_samples : int, default=10
        Minimum samples required for computing fraud rates
        
    Returns:
    --------
    tuple
        (DataFrame with interaction features, list of feature names)
        
    Example:
    --------
    >>> df_with_interactions, interaction_features = create_interaction_features(train_df)
    >>> print(f"Created {len(interaction_features)} interaction features")
    """
    engine = InteractionFeatureEngine(min_samples=min_samples)
    return engine.create_all_interactions(df)
