"""
Data Preprocessing Pipeline for IEEE-CIS Fraud Detection

This module provides functions for:
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features
- Merging transaction and identity datasets
- Creating train/validation splits using time-based splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class FraudPreprocessor:
    """
    Main preprocessing class for fraud detection data.
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize preprocessor.
        
        Args:
            test_size: Proportion of data to use for validation (time-based split)
        """
        self.test_size = test_size
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, 
                  train_transaction_path: str,
                  train_identity_path: str,
                  test_transaction_path: Optional[str] = None,
                  test_identity_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load transaction and identity datasets.
        
        Args:
            train_transaction_path: Path to train transaction CSV
            train_identity_path: Path to train identity CSV
            test_transaction_path: Path to test transaction CSV (optional)
            test_identity_path: Path to test identity CSV (optional)
            
        Returns:
            Tuple of (train_df, test_df) where test_df can be None
        """
        print("Loading datasets...")
        
        # Load training data
        train_trans = pd.read_csv(train_transaction_path)
        train_ident = pd.read_csv(train_identity_path)
        
        print(f"  Train transaction: {train_trans.shape}")
        print(f"  Train identity: {train_ident.shape}")
        
        # Merge transaction and identity
        train_df = train_trans.merge(train_ident, on='TransactionID', how='left')
        print(f"  Merged train: {train_df.shape}")
        
        test_df = None
        if test_transaction_path and test_identity_path:
            test_trans = pd.read_csv(test_transaction_path)
            test_ident = pd.read_csv(test_identity_path)
            test_df = test_trans.merge(test_ident, on='TransactionID', how='left')
            print(f"  Merged test: {test_df.shape}")
        
        print("✓ Data loaded successfully\n")
        return train_df, test_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from TransactionDT.
        
        Args:
            df: DataFrame with TransactionDT column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        SECONDS_IN_DAY = 86400
        SECONDS_IN_WEEK = 604800
        
        # Day of week (0-6)
        df['DT_day_of_week'] = ((df['TransactionDT'] / SECONDS_IN_DAY) % 7).astype(int)
        
        # Hour of day (0-23)
        df['DT_hour'] = ((df['TransactionDT'] % SECONDS_IN_DAY) / 3600).astype(int)
        
        # Day since start
        df['DT_day'] = (df['TransactionDT'] / SECONDS_IN_DAY).astype(int)
        
        # Week since start
        df['DT_week'] = (df['TransactionDT'] / SECONDS_IN_WEEK).astype(int)
        
        # Is weekend
        df['DT_is_weekend'] = (df['DT_day_of_week'] >= 5).astype(int)
        
        # Part of day (encoded as integers)
        hour = df['DT_hour']
        df['DT_is_night'] = ((hour >= 0) & (hour < 6)).astype(int)
        df['DT_is_morning'] = ((hour >= 6) & (hour < 12)).astype(int)
        df['DT_is_afternoon'] = ((hour >= 12) & (hour < 18)).astype(int)
        df['DT_is_evening'] = ((hour >= 18) & (hour < 24)).astype(int)
        
        return df
    
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        # Exclude ID and target
        exclude_cols = ['TransactionID', 'isFraud']
        
        # Numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        # Categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Also treat low-cardinality numeric as categorical (for specific columns)
        m_cols = [col for col in df.columns if col.startswith('M')]
        categorical_features.extend(m_cols)
        numeric_features = [col for col in numeric_features if col not in m_cols]
        
        print(f"Identified {len(numeric_features)} numeric features")
        print(f"Identified {len(categorical_features)} categorical features")
        
        return numeric_features, categorical_features
    
    def handle_missing_values(self, 
                              df: pd.DataFrame, 
                              numeric_strategy: str = 'median',
                              categorical_strategy: str = 'missing') -> pd.DataFrame:
        """
        Handle missing values in numeric and categorical features.
        
        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric features ('mean', 'median', or 'zero')
            categorical_strategy: Strategy for categorical features ('mode' or 'missing')
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        print(f"Handling missing values...")
        print(f"  Numeric strategy: {numeric_strategy}")
        print(f"  Categorical strategy: {categorical_strategy}")
        
        # Numeric features
        if self.numeric_features:
            if numeric_strategy == 'mean':
                df[self.numeric_features] = df[self.numeric_features].fillna(df[self.numeric_features].mean())
            elif numeric_strategy == 'median':
                df[self.numeric_features] = df[self.numeric_features].fillna(df[self.numeric_features].median())
            elif numeric_strategy == 'zero':
                df[self.numeric_features] = df[self.numeric_features].fillna(0)
        
        # Categorical features
        if self.categorical_features:
            if categorical_strategy == 'mode':
                for col in self.categorical_features:
                    if col in df.columns and df[col].isnull().any():
                        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing'
                        df.loc[:, col] = df[col].fillna(mode_val)
            elif categorical_strategy == 'missing':
                for col in self.categorical_features:
                    if col in df.columns and df[col].isnull().any():
                        df.loc[:, col] = df[col].fillna('missing')
        
        remaining_nulls = df.isnull().sum().sum()
        print(f"✓ Missing values handled. Remaining nulls: {remaining_nulls}\n")
        
        return df
    
    def encode_categorical(self, 
                          train_df: pd.DataFrame, 
                          test_df: Optional[pd.DataFrame] = None,
                          method: str = 'label') -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Encode categorical variables.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            method: Encoding method ('label' or 'onehot')
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df)
        """
        print(f"Encoding categorical features using {method} encoding...")
        
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
        
        if method == 'label':
            for col in self.categorical_features:
                if col in train_df.columns:
                    # Fit encoder on train data
                    le = LabelEncoder()
                    train_df[col] = train_df[col].astype(str)
                    le.fit(train_df[col])
                    train_df[col] = le.transform(train_df[col])
                    
                    # Store encoder
                    self.label_encoders[col] = le
                    
                    # Transform test data
                    if test_df is not None and col in test_df.columns:
                        test_df[col] = test_df[col].astype(str)
                        # Handle unseen categories
                        test_df[col] = test_df[col].apply(
                            lambda x: x if x in le.classes_ else 'missing'
                        )
                        # Add missing to classes if needed
                        if 'missing' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'missing')
                        test_df[col] = le.transform(test_df[col])
        
        print(f"✓ Categorical encoding complete\n")
        return train_df, test_df
    
    def normalize_features(self, 
                          train_df: pd.DataFrame, 
                          test_df: Optional[pd.DataFrame] = None,
                          features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Normalize numeric features using StandardScaler.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            features: List of features to normalize (if None, uses all numeric features)
            
        Returns:
            Tuple of (normalized_train_df, normalized_test_df)
        """
        print("Normalizing features...")
        
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
        
        # Select features to normalize
        if features is None:
            features = self.numeric_features
        
        features = [f for f in features if f in train_df.columns]
        
        if features:
            # Fit on train, transform both
            self.scaler.fit(train_df[features])
            train_df[features] = self.scaler.transform(train_df[features])
            
            if test_df is not None:
                test_features = [f for f in features if f in test_df.columns]
                if test_features:
                    test_df[test_features] = self.scaler.transform(test_df[test_features])
        
        print(f"✓ Normalized {len(features)} features\n")
        return train_df, test_df
    
    def time_based_split(self, 
                        df: pd.DataFrame, 
                        test_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation split.
        
        Args:
            df: DataFrame with TransactionDT column
            test_size: Proportion for validation (uses self.test_size if None)
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if test_size is None:
            test_size = self.test_size
        
        print(f"Creating time-based split (validation size: {test_size*100:.0f}%)...")
        
        # Sort by time
        df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        val_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"  Train size: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val size: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        
        if 'isFraud' in df.columns:
            train_fraud_rate = train_df['isFraud'].mean() * 100
            val_fraud_rate = val_df['isFraud'].mean() * 100
            print(f"  Train fraud rate: {train_fraud_rate:.2f}%")
            print(f"  Val fraud rate: {val_fraud_rate:.2f}%")
        
        print("✓ Time-based split complete\n")
        return train_df, val_df
    
    def prepare_features(self, 
                        train_df: pd.DataFrame,
                        test_df: Optional[pd.DataFrame] = None,
                        create_time_features: bool = True,
                        handle_missing: bool = True,
                        encode_categorical: bool = True,
                        normalize: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            create_time_features: Whether to create time-based features
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical variables
            normalize: Whether to normalize numeric features
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        print("="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        print()
        
        # Step 1: Create time features
        if create_time_features:
            print("[1/5] Creating time-based features...")
            train_df = self.create_time_features(train_df)
            if test_df is not None:
                test_df = self.create_time_features(test_df)
            print("✓ Time features created\n")
        
        # Step 2: Identify feature types
        print("[2/5] Identifying feature types...")
        self.numeric_features, self.categorical_features = self.identify_feature_types(train_df)
        print()
        
        # Step 3: Handle missing values
        if handle_missing:
            print("[3/5] Handling missing values...")
            train_df = self.handle_missing_values(train_df)
            if test_df is not None:
                test_df = self.handle_missing_values(test_df)
        
        # Step 4: Encode categorical
        if encode_categorical:
            print("[4/5] Encoding categorical variables...")
            train_df, test_df = self.encode_categorical(train_df, test_df)
        
        # Step 5: Normalize features (optional)
        if normalize:
            print("[5/5] Normalizing features...")
            train_df, test_df = self.normalize_features(train_df, test_df)
        else:
            print("[5/5] Skipping normalization (tree-based models don't require it)")
            print()
        
        print("="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final train shape: {train_df.shape}")
        if test_df is not None:
            print(f"Final test shape: {test_df.shape}")
        print()
        
        return train_df, test_df


def quick_preprocess(train_transaction_path: str,
                     train_identity_path: str,
                     test_size: float = 0.2,
                     normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Quick preprocessing function for rapid prototyping.
    
    Args:
        train_transaction_path: Path to train transaction CSV
        train_identity_path: Path to train identity CSV
        test_size: Proportion for validation split
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Initialize preprocessor
    preprocessor = FraudPreprocessor(test_size=test_size)
    
    # Load data
    train_df, _ = preprocessor.load_data(train_transaction_path, train_identity_path)
    
    # Preprocess
    train_df, _ = preprocessor.prepare_features(
        train_df, 
        create_time_features=True,
        handle_missing=True,
        encode_categorical=True,
        normalize=normalize
    )
    
    # Time-based split
    train_split, val_split = preprocessor.time_based_split(train_df, test_size=test_size)
    
    # Separate features and target
    feature_cols = [col for col in train_split.columns if col not in ['TransactionID', 'isFraud']]
    
    X_train = train_split[feature_cols]
    y_train = train_split['isFraud']
    X_val = val_split[feature_cols]
    y_val = val_split['isFraud']
    
    print("✓ Quick preprocessing complete!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_train fraud rate: {y_train.mean()*100:.2f}%")
    print(f"  y_val fraud rate: {y_val.mean()*100:.2f}%")
    
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully!")
    print("\nExample usage:")
    print("  from src.preprocessing import FraudPreprocessor, quick_preprocess")
    print("  X_train, X_val, y_train, y_val = quick_preprocess('data/train_transaction.csv', 'data/train_identity.csv')")
