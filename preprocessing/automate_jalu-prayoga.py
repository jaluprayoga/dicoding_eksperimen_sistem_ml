import os
import sys
import subprocess
import customer_churn_preprocessing as prep

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def main():
    # Setting path to the root folder
    root_dir = os.path.dirname(current_dir)
    raw_dir = os.path.join(root_dir, "customer_churn_raw")
    dataset_file = os.path.join(raw_dir, "customer_churn.csv")
    output_dir = os.path.join(root_dir, "customer_churn_preprocessed")
    
    # Checking existence of raw dataset
    if not os.path.exists(dataset_file):
        print(f"Raw dataset not found at {dataset_file}. Trying to download...")
        download_script = os.path.join(raw_dir, "download_dataset_script.py")
        if os.path.exists(download_script):
            try:
                # Running the download script in the root folder so that the internal script's relative path is correct
                subprocess.run([sys.executable, download_script], cwd=root_dir, check=True)
                print("Dataset successfully downloaded.")
            except Exception as e:
                print(f"Failed to download dataset: {e}")
                sys.exit(1)
        else:
            print(f"Download script not found at {download_script}. Please provide the dataset manually.")
            sys.exit(1)
            
    print("Starting automatic data preprocessing pipeline...")
    
    # Load Data
    print("Loading dataset...")
    df = prep.load_data(dataset_file)
    print(f"   Number of rows: {len(df)}, Number of columns: {len(df.columns)}")
    
    # Clean Data
    print("Cleaning data...")
    df_clean = prep.clean_data(df)
    
    # Encode Features
    print("Performing categorical encoding...")
    df_encoded, binary_cols, categorical_cols = prep.encode_features(df_clean)
    print(f"   Binary columns: {len(binary_cols)}, Categorical columns: {len(categorical_cols)}")
    
    # Split and Scale
    print("Splitting and scaling dataset...")
    X_train, X_test, y_train, y_test = prep.split_and_scale(df_encoded, test_size=0.2, random_state=42)
    print(f"   Training Set Size (Before SMOTE): {X_train.shape}")
    print(f"   Testing Set Size: {X_test.shape}")
    
    # Applying SMOTE on training data
    print("Applying SMOTE on training data...")
    X_train_smote, y_train_smote = prep.apply_smote(X_train, y_train, random_state=42)
    print(f"   Training Set Size (After SMOTE): {X_train_smote.shape}")
    
    # Saving the processed datasets
    print("Saving processed datasets...")
    prep.save_processed_data(
        X_train, X_test, y_train, y_test, 
        output_dir, 
        X_train_smote=X_train_smote, 
        y_train_smote=y_train_smote
    )
    
    # Returning the processed dataset
    print("\nAutomation completed! Preprocessing pipeline executed successfully.")
    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote

if __name__ == "__main__":
    main()
