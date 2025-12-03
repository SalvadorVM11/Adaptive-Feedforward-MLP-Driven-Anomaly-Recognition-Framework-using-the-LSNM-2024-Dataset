import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
BENIGN_FILE = os.path.join(DATASET_DIR, 'Benign', 'normal_data.csv')
MALICIOUS_DIR = os.path.join(DATASET_DIR, 'Malicious')

# --- Output Paths ---
PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'preprocessed')
SCALER_PATH = os.path.join(PREPROCESSED_DIR, 'scaler.pkl')
LE_PATH = os.path.join(PREPROCESSED_DIR, 'label_encoder.pkl')

X_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'X_train.npy')
Y_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, 'y_train.npy')
X_VAL_PATH = os.path.join(PREPROCESSED_DIR, 'X_val.npy')
Y_VAL_PATH = os.path.join(PREPROCESSED_DIR, 'y_val.npy')
X_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'X_test.npy')
Y_TEST_PATH = os.path.join(PREPROCESSED_DIR, 'y_test.npy')

MAX_SAMPLES_PER_CLASS = 2000  # max samples per class to ensure balance
# --------------------------

def load_data():
    # Load benign
    df_benign = pd.read_csv(BENIGN_FILE)
    df_benign['label'] = 'Benign'
    
    # Load malicious
    df_malicious_list = []
    for root, dirs, files in os.walk(MALICIOUS_DIR):
        for f in files:
            if f.endswith('.csv'):
                try:
                    df_mal = pd.read_csv(os.path.join(root, f))
                    if not df_mal.empty:
                        df_mal['label'] = os.path.basename(os.path.dirname(os.path.join(root,f)))
                        df_malicious_list.append(df_mal)
                except Exception as e:
                    print(f"Warning: Could not read {f}. Error: {e}")
    
    df_malicious = pd.concat(df_malicious_list, ignore_index=True)
    df = pd.concat([df_benign, df_malicious], ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def preprocess_and_save(df):
    # Create output directory if it doesn't exist
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    # Keep only numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'label' in numeric_cols:
        numeric_cols.remove('label')
    
    # Limit samples per class
    balanced_df_list = []
    for label, group in df.groupby('label'):
        group_sampled = group.sample(n=min(len(group), MAX_SAMPLES_PER_CLASS), random_state=42)
        balanced_df_list.append(group_sampled)
    
    df_balanced = pd.concat(balanced_df_list, ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split features and labels
    X = df_balanced[numeric_cols].values
    y = df_balanced['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, LE_PATH)
    print(f"Label encoder saved at {LE_PATH}")
    
    # --- FIX: SPLIT DATA *BEFORE* SCALING ---
    
    # Stratified split: 70% train, 10% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )  # 0.125 * 0.8 = 0.1
    
    # --- FIX: FIT SCALER ON TRAIN DATA ONLY ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Fit and transform train
    
    # --- FIX: TRANSFORM VAL AND TEST WITH FITTED SCALER ---
    X_val = scaler.transform(X_val)       # Only transform val
    X_test = scaler.transform(X_test)      # Only transform test
    
    # Save the fitted scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved at {SCALER_PATH}")
    
    # --- SAVE PREPROCESSED DATA SPLITS ---
    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_VAL_PATH, y_val)
    np.save(X_TEST_PATH, X_test)
    np.save(Y_TEST_PATH, y_test)
    
    print("Preprocessed data splits saved to .npy files.")
    
    return X_train.shape, X_val.shape, X_test.shape

if __name__ == "__main__":
    print("Loading raw data...")
    df = load_data()
    print("Data loading complete. Starting preprocessing...")
    train_shape, val_shape, test_shape = preprocess_and_save(df)
    
    print("\nPreprocessing complete.")
    print(f"Train shape: {train_shape}")
    print(f"Val shape:   {val_shape}")
    print(f"Test shape:  {test_shape}")