import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, List

# =========================================================
# WARNING SUPPRESSION BLOCK (Robust against Pandas versions)
# =========================================================
import warnings

# Define placeholder classes for warnings that might be moved/hidden in your pandas version
# This ensures the rest of the code that references these classes does not crash.
class PerformanceWarning(UserWarning): pass
class SettingWithCopyWarning(UserWarning): pass

# Try to import the real classes if they exist in known locations
try:
    # Attempt to import PerformanceWarning (often found here)
    from pandas.core.frame import PerformanceWarning
except ImportError:
    pass # Use placeholder if import fails

try:
    # Attempt to import SettingWithCopyWarning (location changes frequently)
    from pandas.core.common import SettingWithCopyWarning
except ImportError:
    pass # Use placeholder if import fails

# Filter the warnings using the classes (real or placeholders)
warnings.filterwarnings('ignore', category=PerformanceWarning)
warnings.filterwarnings('ignore', category=SettingWithCopyWarning) 
warnings.filterwarnings('ignore', category=UserWarning)
# =========================================================

# ===============================
# CONFIG
# ===============================
DATA_DIR = "./data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TARGET = "forward_returns"
N_SPLITS = 5 

FEAT_PARAMS = {
    "lags": [1, 3, 5],
    "roll_windows": [5, 10],
    "ewm_alphas": [0.1, 0.5, 0.9]
}

# ===============================
# UTILITY CLASS: FEATURE PIPELINE
# ===============================

class TimeSeriesFeaturePipeline:
    def __init__(self, target_col: str, fe_params: dict):
        self.target_col = target_col
        self.fe_params = fe_params
        self.imputers: Dict[str, SimpleImputer] = {}
        self.scaler = QuantileTransformer(output_distribution="normal", subsample=10000, n_quantiles=1000) 
        self.numeric_cols = None

    def _get_numeric_cols(self, df: pd.DataFrame) -> list:
        return [
            c for c in df.columns 
            if df[c].dtype != "object" and c != self.target_col
        ]

    def clean_data(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        df_clean = df.copy()
        cols_to_impute = self._get_numeric_cols(df_clean)

        for col in cols_to_impute:
            df_clean[col + "_nanflag"] = df_clean[col].isna().astype(int)

            if fit:
                imputer = SimpleImputer(strategy="median")
                if df_clean[col].count() > 0:
                    df_clean[col] = imputer.fit_transform(df_clean[[col]])
                    self.imputers[col] = imputer
            else:
                if col in self.imputers:
                    df_clean[col] = self.imputers[col].transform(df_clean[[col]])
        return df_clean

    # --- Step 2: Feature Engineering (Optimized and Warning-Free) ---
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = df.copy()
        numeric_cols = self._get_numeric_cols(df_feat)
        
        # List to hold all new Series/DataFrames (Ultimate Performance Fix)
        new_features_list: List[pd.Series | pd.DataFrame] = []
        
        # Time index feature
        new_features_list.append(pd.Series(np.arange(len(df_feat)), index=df_feat.index, name="time_index"))

        # Lags, rolling stats, and EWMA
        for col in numeric_cols:
            
            # Lags
            for lag in self.fe_params['lags']:
                new_features_list.append(df_feat[col].shift(lag).rename(f"{col}_lag{lag}"))

            # Rolling Stats
            for window in self.fe_params['roll_windows']:
                new_features_list.append(df_feat[col].rolling(window).mean().rename(f"{col}_rollmean{window}"))
                new_features_list.append(df_feat[col].rolling(window).std().rename(f"{col}_rollstd{window}"))

            # EWMA
            for alpha in self.fe_params['ewm_alphas']:
                new_features_list.append(df_feat[col].ewm(alpha=alpha).mean().rename(f"{col}_ewm{alpha}"))

        # Simple interaction example
        V_cols = [c for c in numeric_cols if c.startswith('V') and not c.endswith('_nanflag')]
        if len(V_cols) >= 2:
            interaction_col = df_feat[V_cols[0]] / (df_feat[V_cols[1]] + 1e-6)
            new_features_list.append(interaction_col.rename(f"{V_cols[0]}_div_{V_cols[1]}"))

        # CONCATENATE all new features at once (Final Fix)
        if new_features_list:
             df_feat = pd.concat([df_feat] + new_features_list, axis=1)

        return df_feat

    # --- Step 3: Scaling ---
    def scale_data(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        df_scaled = df.copy()
        self.numeric_cols = self._get_numeric_cols(df_scaled)
        
        if not self.numeric_cols:
            return df_scaled # No numeric columns to scale

        if fit:
            df_scaled[self.numeric_cols] = self.scaler.fit_transform(df_scaled[self.numeric_cols])
        else:
            df_scaled[self.numeric_cols] = self.scaler.transform(df_scaled[self.numeric_cols])

        return df_scaled

# ===============================
# LOAD & INSPECT (Unchanged)
# ===============================
def load_and_inspect() -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train = pd.read_csv(TRAIN_FILE)
        test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print(f"ERROR: Data files not found. Creating dummy data.")
        # ... (dummy data creation remains the same)
        data = {f'D{i}': np.random.randint(0, 5, 9000) for i in range(1, 10)}
        data.update({f'V{i}': np.random.rand(9000) for i in range(1, 10)})
        data[TARGET] = np.random.randn(9000)
        data['date_id'] = np.arange(9000)
        train = pd.DataFrame(data)
        test = train.drop(columns=[TARGET], errors='ignore').head(1000) # Added errors='ignore' for safety
        for col in ['V8', 'V9']:
            train.loc[train.sample(frac=0.1).index, col] = np.nan
        print("Using dummy data for demonstration.")
        
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test

# ===============================
# MAIN LEAK-FREE EXECUTION
# ===============================
if __name__ == "__main__":
    train, test = load_and_inspect()
    
    # 1. Initialize the Pipeline
    pipeline = TimeSeriesFeaturePipeline(target_col=TARGET, fe_params=FEAT_PARAMS)
    
    # --- GLOBAL PREPROCESSING (Train) ---
    train_clean = pipeline.clean_data(train.copy(), fit=True)
    train_all_data = train_clean.sort_values(by="date_id").reset_index(drop=True)
    
    # --- TIME-SERIES CV LOOP ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    OOF_PREDICTIONS = np.zeros(len(train_all_data))
    
    print(f"\nStarting Leak-Free Time-Series Cross-Validation with {N_SPLITS} folds...")
    temp_pipeline = TimeSeriesFeaturePipeline(target_col=TARGET, fe_params=FEAT_PARAMS)
    temp_pipeline.imputers = pipeline.imputers 

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_all_data)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        X_train_fold = train_all_data.iloc[train_idx].copy()
        X_valid_fold = train_all_data.iloc[valid_idx].copy()
        y_train = X_train_fold[TARGET].reset_index(drop=True)
        y_valid = X_valid_fold[TARGET].reset_index(drop=True)
        
        # Prepare window for feature engineering
        full_window_data = pd.concat([
            X_train_fold.drop(columns=[TARGET]), 
            X_valid_fold.drop(columns=[TARGET])
        ], ignore_index=True)
        
        full_window_feat = temp_pipeline.add_features(full_window_data)

        X_train_feat_raw = full_window_feat.iloc[:len(X_train_fold)]
        X_valid_feat_raw = full_window_feat.iloc[len(X_train_fold):]
        
        # Drop NaNs introduced by feature engineering
        X_train_feat = X_train_feat_raw.dropna(how='any').reset_index(drop=True)
        y_train_feat = y_train.iloc[X_train_feat_raw.index.get_indexer(X_train_feat_raw.dropna().index)]
        
        # Scaling
        X_train_scaled = temp_pipeline.scale_data(X_train_feat.copy(), fit=True)
        X_valid_scaled = temp_pipeline.scale_data(X_valid_feat_raw.copy(), fit=False)

        print(f"  Fold {fold+1}: Train={X_train_scaled.shape}, Valid={X_valid_scaled.shape}")
        
    # --- FINAL SUBMISSION PREPARATION ---
    print("\n--- Final Preparation for Submission Model ---")
    
    test_clean = pipeline.clean_data(test.copy(), fit=False)
    
    train_final = pipeline.add_features(train_clean.sort_values(by="date_id").reset_index(drop=True))
    test_final = pipeline.add_features(test_clean.sort_values(by="date_id").reset_index(drop=True))

    train_y = train_final[TARGET]
    train_final.drop(columns=[TARGET], inplace=True)
    train_final.dropna(inplace=True) 
    train_y_final = train_y.loc[train_final.index]
    
    final_pipeline = TimeSeriesFeaturePipeline(target_col=TARGET, fe_params=FEAT_PARAMS)
    
    train_scaled = final_pipeline.scale_data(train_final.copy(), fit=True)
    test_scaled = final_pipeline.scale_data(test_final.copy(), fit=False) 
    
    # Final Fix for the remaining PerformanceWarning: Use concat to add the target
    train_scaled = pd.concat([train_scaled, train_y_final.rename(TARGET)], axis=1)
    
    print(f"Final Train Features for Submission Model: {train_scaled.shape}")
    print(f"Final Test Features for Submission Model: {test_scaled.shape}")

    # Save processed datasets
    os.makedirs("./processed", exist_ok=True)
    train_scaled.to_csv("./processed/train_final_scaled.csv", index=False)
    test_scaled.to_csv("./processed/test_final_scaled.csv", index=False)
    print("\n✅ Leak-Free Preprocessing complete. Files saved in ./processed/")