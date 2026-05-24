import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTETomek

def load_data(filepath):
    """
    Loading dataset from CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Performing initial data cleaning:
    - Dropping the customerID column.
    - Handling missing values in TotalCharges.
    """
    df_clean = df.copy()
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    if 'TotalCharges' in df_clean.columns:
        # Changing spaces to NaN
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
        df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(float)
        # Imputing with median (for modular convenience, calculated from the entire dataset)
        median_total_charges = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_total_charges)
        
    return df_clean

def encode_features(df):
    """
    Performing categorical encoding:
    - Label Encoding for binary columns.
    - One-Hot Encoding for multi-category columns.
    """
    df_encoded = df.copy()
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    binary_cols = []
    categorical_cols = []
    
    # Identifying column types
    for col in df_encoded.columns:
        if col not in numerical_cols and col != 'Churn':
            if df_encoded[col].nunique() == 2:
                binary_cols.append(col)
            else:
                categorical_cols.append(col)
                
    # Label Encoding for binary columns
    le = LabelEncoder()
    for col in binary_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    # Label Encoding for target 'Churn' if there is one
    if 'Churn' in df_encoded.columns:
        df_encoded['Churn'] = le.fit_transform(df_encoded['Churn'])
        
    # One-Hot Encoding for multi-category columns
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    # Converting boolean columns resulting from dummy to integer
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    return df_encoded, binary_cols, categorical_cols

def split_and_scale(df_processed, test_size=0.2, random_state=42):
    """
    Splitting data into training & testing set, as well as performing scaling on numerical features.
    """
    if 'Churn' not in df_processed.columns:
        raise ValueError("Target column 'Churn' not found in dataframe.")
        
    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']
    
    # Split dataset with stratification to balance the ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardization of numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    # Only performing scaling on numerical columns that exist in X_train
    numerical_in_X = [col for col in numerical_cols if col in X_train.columns]
    
    if numerical_in_X:
        X_train[numerical_in_X] = scaler.fit_transform(X_train[numerical_in_X])
        X_test[numerical_in_X] = scaler.transform(X_test[numerical_in_X])
        
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """
    Applying SMOTE-Tomek on training data to handle class imbalance.
    """
    smote = SMOTETomek(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def save_processed_data(X_train, X_test, y_train, y_test, output_dir, X_train_smote=None, y_train_smote=None):
    """
    Saving processed data (and SMOTE results if provided) to separate CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    if X_train_smote is not None and y_train_smote is not None:
        X_train_smote.to_csv(os.path.join(output_dir, 'X_train_smote.csv'), index=False)
        y_train_smote.to_csv(os.path.join(output_dir, 'y_train_smote.csv'), index=False)
        print(f"Data preprocessed & SMOTE successfully saved to {output_dir}")
    else:
        print(f"Data preprocessed successfully saved to {output_dir}")
