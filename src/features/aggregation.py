"""
Aggregation Feature Engineering Module
Creates powerful aggregation features for fraud detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


class AggregationFeatureEngine:
    """
    Generate aggregation features based on card, email, device, and product groupings.
    """
    
    def __init__(self):
        """Initialize the aggregation feature engine."""
        self.aggregation_stats = {}
        self.feature_names = []
        
    def create_card_aggregations(self, 
                                  df: pd.DataFrame,
                                  target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create aggregation features based on card information.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate (default: TransactionAmt)
            
        Returns:
            DataFrame with new aggregation features
        """
        print("Creating card-based aggregation features...")
        
        card_features = []
        
        # Card1 aggregations (primary card identifier)
        if 'card1' in df.columns:
            print("  • card1 aggregations...")
            card1_agg = df.groupby('card1')[target_col].agg([
                'count', 'mean', 'std', 'min', 'max', 'median', 'sum'
            ]).add_prefix('card1_').reset_index()
            
            # Add derived features
            card1_agg['card1_range'] = card1_agg['card1_max'] - card1_agg['card1_min']
            card1_agg['card1_mean_to_std_ratio'] = card1_agg['card1_mean'] / (card1_agg['card1_std'] + 1e-5)
            
            df = df.merge(card1_agg, on='card1', how='left')
            card_features.extend([c for c in card1_agg.columns if c != 'card1'])
            
            # Deviation features
            df['card1_amt_deviation'] = df[target_col] - df['card1_mean']
            df['card1_amt_deviation_ratio'] = df[target_col] / (df['card1_mean'] + 1)
            df['card1_amt_zscore'] = (df[target_col] - df['card1_mean']) / (df['card1_std'] + 1e-5)
            card_features.extend(['card1_amt_deviation', 'card1_amt_deviation_ratio', 'card1_amt_zscore'])
        
        # Card2 aggregations
        if 'card2' in df.columns:
            print("  • card2 aggregations...")
            card2_agg = df.groupby('card2')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('card2_').reset_index()
            
            df = df.merge(card2_agg, on='card2', how='left')
            card_features.extend([c for c in card2_agg.columns if c != 'card2'])
        
        # Card3 aggregations
        if 'card3' in df.columns:
            print("  • card3 aggregations...")
            card3_agg = df.groupby('card3')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('card3_').reset_index()
            
            df = df.merge(card3_agg, on='card3', how='left')
            card_features.extend([c for c in card3_agg.columns if c != 'card3'])
        
        # Card4 aggregations (card type)
        if 'card4' in df.columns:
            print("  • card4 aggregations...")
            card4_agg = df.groupby('card4')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('card4_').reset_index()
            
            df = df.merge(card4_agg, on='card4', how='left')
            card_features.extend([c for c in card4_agg.columns if c != 'card4'])
        
        # Card5 aggregations
        if 'card5' in df.columns:
            print("  • card5 aggregations...")
            card5_agg = df.groupby('card5')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('card5_').reset_index()
            
            df = df.merge(card5_agg, on='card5', how='left')
            card_features.extend([c for c in card5_agg.columns if c != 'card5'])
        
        # Card6 aggregations (card category)
        if 'card6' in df.columns:
            print("  • card6 aggregations...")
            card6_agg = df.groupby('card6')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('card6_').reset_index()
            
            df = df.merge(card6_agg, on='card6', how='left')
            card_features.extend([c for c in card6_agg.columns if c != 'card6'])
        
        # Card combinations
        if 'card1' in df.columns and 'card2' in df.columns:
            print("  • card1_card2 combination aggregations...")
            df['card1_card2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
            card12_agg = df.groupby('card1_card2')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('card12_').reset_index()
            
            df = df.merge(card12_agg, on='card1_card2', how='left')
            card_features.extend([c for c in card12_agg.columns if c != 'card1_card2'])
        
        print(f"  ✓ Created {len(card_features)} card aggregation features")
        self.feature_names.extend(card_features)
        
        return df
    
    def create_email_aggregations(self, 
                                   df: pd.DataFrame,
                                   target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create aggregation features based on email domains.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            
        Returns:
            DataFrame with new email aggregation features
        """
        print("Creating email-based aggregation features...")
        
        email_features = []
        
        # P_emaildomain aggregations (purchaser email)
        if 'P_emaildomain' in df.columns:
            print("  • P_emaildomain aggregations...")
            p_email_agg = df.groupby('P_emaildomain')[target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).add_prefix('P_email_').reset_index()
            
            df = df.merge(p_email_agg, on='P_emaildomain', how='left')
            email_features.extend([c for c in p_email_agg.columns if c != 'P_emaildomain'])
            
            # Email domain transaction count (risk indicator)
            df['P_email_risk_score'] = 1 / (df['P_email_count'] + 1)
            email_features.append('P_email_risk_score')
        
        # R_emaildomain aggregations (recipient email)
        if 'R_emaildomain' in df.columns:
            print("  • R_emaildomain aggregations...")
            r_email_agg = df.groupby('R_emaildomain')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('R_email_').reset_index()
            
            df = df.merge(r_email_agg, on='R_emaildomain', how='left')
            email_features.extend([c for c in r_email_agg.columns if c != 'R_emaildomain'])
            
            df['R_email_risk_score'] = 1 / (df['R_email_count'] + 1)
            email_features.append('R_email_risk_score')
        
        # Email combination features
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            print("  • P_email + R_email combination aggregations...")
            df['email_combination'] = df['P_emaildomain'].astype(str) + '_' + df['R_emaildomain'].astype(str)
            email_comb_agg = df.groupby('email_combination')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('email_comb_').reset_index()
            
            df = df.merge(email_comb_agg, on='email_combination', how='left')
            email_features.extend([c for c in email_comb_agg.columns if c != 'email_combination'])
        
        print(f"  ✓ Created {len(email_features)} email aggregation features")
        self.feature_names.extend(email_features)
        
        return df
    
    def create_device_aggregations(self, 
                                    df: pd.DataFrame,
                                    target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create aggregation features based on device information.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            
        Returns:
            DataFrame with new device aggregation features
        """
        print("Creating device-based aggregation features...")
        
        device_features = []
        
        # DeviceType aggregations
        if 'DeviceType' in df.columns:
            print("  • DeviceType aggregations...")
            device_type_agg = df.groupby('DeviceType')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('DeviceType_').reset_index()
            
            df = df.merge(device_type_agg, on='DeviceType', how='left')
            device_features.extend([c for c in device_type_agg.columns if c != 'DeviceType'])
        
        # DeviceInfo aggregations
        if 'DeviceInfo' in df.columns:
            print("  • DeviceInfo aggregations...")
            device_info_agg = df.groupby('DeviceInfo')[target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).add_prefix('DeviceInfo_').reset_index()
            
            df = df.merge(device_info_agg, on='DeviceInfo', how='left')
            device_features.extend([c for c in device_info_agg.columns if c != 'DeviceInfo'])
            
            # Device risk score
            df['DeviceInfo_risk_score'] = 1 / (df['DeviceInfo_count'] + 1)
            device_features.append('DeviceInfo_risk_score')
        
        # id_30 (device OS) aggregations
        if 'id_30' in df.columns:
            print("  • id_30 (OS) aggregations...")
            os_agg = df.groupby('id_30')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('OS_').reset_index()
            
            df = df.merge(os_agg, on='id_30', how='left')
            device_features.extend([c for c in os_agg.columns if c != 'id_30'])
        
        # id_31 (browser) aggregations
        if 'id_31' in df.columns:
            print("  • id_31 (browser) aggregations...")
            browser_agg = df.groupby('id_31')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('browser_').reset_index()
            
            df = df.merge(browser_agg, on='id_31', how='left')
            device_features.extend([c for c in browser_agg.columns if c != 'id_31'])
        
        print(f"  ✓ Created {len(device_features)} device aggregation features")
        self.feature_names.extend(device_features)
        
        return df
    
    def create_product_aggregations(self, 
                                     df: pd.DataFrame,
                                     target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create aggregation features based on product information.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            
        Returns:
            DataFrame with new product aggregation features
        """
        print("Creating product-based aggregation features...")
        
        product_features = []
        
        # ProductCD aggregations
        if 'ProductCD' in df.columns:
            print("  • ProductCD aggregations...")
            product_agg = df.groupby('ProductCD')[target_col].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).add_prefix('ProductCD_').reset_index()
            
            df = df.merge(product_agg, on='ProductCD', how='left')
            product_features.extend([c for c in product_agg.columns if c != 'ProductCD'])
            
            # Product-specific deviations
            df['ProductCD_amt_deviation'] = df[target_col] - df['ProductCD_mean']
            df['ProductCD_amt_ratio'] = df[target_col] / (df['ProductCD_mean'] + 1)
            product_features.extend(['ProductCD_amt_deviation', 'ProductCD_amt_ratio'])
        
        print(f"  ✓ Created {len(product_features)} product aggregation features")
        self.feature_names.extend(product_features)
        
        return df
    
    def create_address_aggregations(self, 
                                     df: pd.DataFrame,
                                     target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create aggregation features based on address information.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            
        Returns:
            DataFrame with new address aggregation features
        """
        print("Creating address-based aggregation features...")
        
        address_features = []
        
        # addr1 aggregations
        if 'addr1' in df.columns:
            print("  • addr1 aggregations...")
            addr1_agg = df.groupby('addr1')[target_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).add_prefix('addr1_').reset_index()
            
            df = df.merge(addr1_agg, on='addr1', how='left')
            address_features.extend([c for c in addr1_agg.columns if c != 'addr1'])
            
            df['addr1_risk_score'] = 1 / (df['addr1_count'] + 1)
            address_features.append('addr1_risk_score')
        
        # addr2 aggregations
        if 'addr2' in df.columns:
            print("  • addr2 aggregations...")
            addr2_agg = df.groupby('addr2')[target_col].agg([
                'count', 'mean', 'std'
            ]).add_prefix('addr2_').reset_index()
            
            df = df.merge(addr2_agg, on='addr2', how='left')
            address_features.extend([c for c in addr2_agg.columns if c != 'addr2'])
        
        # Address combination
        if 'addr1' in df.columns and 'addr2' in df.columns:
            print("  • addr1 + addr2 combination aggregations...")
            df['addr_combined'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
            addr_comb_agg = df.groupby('addr_combined')[target_col].agg([
                'count', 'mean'
            ]).add_prefix('addr_comb_').reset_index()
            
            df = df.merge(addr_comb_agg, on='addr_combined', how='left')
            address_features.extend([c for c in addr_comb_agg.columns if c != 'addr_combined'])
        
        print(f"  ✓ Created {len(address_features)} address aggregation features")
        self.feature_names.extend(address_features)
        
        return df
    
    def create_all_aggregations(self, 
                                df: pd.DataFrame,
                                target_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create all aggregation features at once.
        
        Args:
            df: Input DataFrame
            target_col: Column to aggregate
            
        Returns:
            DataFrame with all aggregation features
        """
        print("="*60)
        print("CREATING ALL AGGREGATION FEATURES")
        print("="*60)
        print()
        
        initial_shape = df.shape
        
        # Create all aggregations
        df = self.create_card_aggregations(df, target_col)
        df = self.create_email_aggregations(df, target_col)
        df = self.create_device_aggregations(df, target_col)
        df = self.create_product_aggregations(df, target_col)
        df = self.create_address_aggregations(df, target_col)
        
        print()
        print("="*60)
        print("AGGREGATION FEATURE SUMMARY")
        print("="*60)
        print(f"\nOriginal shape: {initial_shape}")
        print(f"New shape: {df.shape}")
        print(f"Total new features: {len(self.feature_names)}")
        print(f"Features added: {df.shape[1] - initial_shape[1]}")
        print("="*60)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        return self.feature_names


# Convenience function
def create_aggregation_features(df: pd.DataFrame, 
                                target_col: str = 'TransactionAmt') -> Tuple[pd.DataFrame, List[str]]:
    """
    Quick function to create all aggregation features.
    
    Args:
        df: Input DataFrame
        target_col: Column to aggregate
        
    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    engine = AggregationFeatureEngine()
    df_transformed = engine.create_all_aggregations(df, target_col)
    feature_names = engine.get_feature_names()
    
    return df_transformed, feature_names


if __name__ == "__main__":
    print("Aggregation Feature Engineering Module")
    print("\nUsage:")
    print("  from src.features.aggregation import create_aggregation_features")
    print("  df_with_features, feature_names = create_aggregation_features(df)")
