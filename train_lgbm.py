import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import random

# ===============================
# 1. REPRODUCIBILITY & SEEDS
# ===============================
SEED = 42
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    if 'LGBM_RANDOM_SEED' not in os.environ:
        os.environ['LGBM_RANDOM_SEED'] = str(seed)
set_seed(SEED)

# ===============================
# CONFIG & SETUP
# ===============================
TARGET = "forward_returns"
N_SPLITS = 5
PROCESSED_DIR = "./processed"
OOF_FILE = "oof_lgbm.npy"
TEST_PRED_FILE = "test_lgbm.npy"
MODEL_FILE = "lgbm_model"

# MLflow Setup
mlflow.set_experiment("aria_kaggle_timeseries_ensemble")

# ===============================
# METRIC FUNCTION
# ===============================
def rms_error(y_true, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# 6. CONFIG FILE FOR PARAMETERS (Simulated internal config)
# ===============================
LGBM_PARAMS = {
    'objective': 'regression_l1', # Use L1 loss (MAE) for robustness
    # 5. Early Stopping Safety: Eval metrics include both L1 and RMSE
    'metric': ['rmse', 'l1'],
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'num_leaves': 64,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED, # 1. Pass seed consistently
    'n_jobs': -1,
    'boosting_type': 'gbdt',
    'verbose': -1,
}
EARLY_STOPPING_ROUNDS = 100

# ===============================
# MAIN TRAINING FUNCTION
# ===============================
def train_lgbm():
    print("Starting LightGBM Training...")

    # --- 1. Load Data ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_final_scaled.csv")
    
    if not os.path.exists(train_file):
        print(f"ERROR: Training file not found at {train_file}. Please run data_prep.py first.")
        return

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Separate features and target
    X = train_data.drop(columns=[TARGET])
    y = train_data[TARGET]
    X_test = test_data

    # Initialize OOF array and Model List
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    feature_importances = pd.DataFrame(index=X.columns)
    
    # 1. Pass seed consistently to TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS) 
    
    with mlflow.start_run(run_name="LGBM_Baseline_TSCV"):
        # Log parameters
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)

        print(f"Total features: {len(X.columns)}")

        # --- 2. TimeSeries Cross-Validation ---
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Initialize and Train Model
            lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
            
            # 5. Early Stopping Safety: Log multiple metrics
            lgbm.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=LGBM_PARAMS['metric'], # Use metrics defined in params
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )

            # Predict OOF
            oof_pred = lgbm.predict(X_valid)
            oof_predictions[valid_idx] = oof_pred
            
            # Log Fold Metric
            fold_rmse = rms_error(y_valid, oof_pred)
            print(f"  Fold RMSE: {fold_rmse:.4f}")
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
            
            # 2. Better Cross-Validation Logging: Save per-fold predictions
            fold_oof_path = os.path.join(PROCESSED_DIR, f"oof_fold_{fold+1}.csv")
            pd.DataFrame({
                "idx": valid_idx,
                "y_true": y_valid,
                "y_pred": oof_pred
            }).to_csv(fold_oof_path, index=False)
            mlflow.log_artifact(fold_oof_path)
            
            # Save Feature Importance (for later aggregation)
            fold_fi = pd.Series(lgbm.feature_importances_, index=X.columns).rename(f"Fold_{fold+1}")
            feature_importances = pd.concat([feature_importances, fold_fi], axis=1)
            
        # --- 3. Final OOF Evaluation ---
        final_oof_rmse = rms_error(y, oof_predictions)
        print(f"\n===== Final OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_oof_rmse", final_oof_rmse)

        # --- 4. Final Model Training (for Test Predictions) ---
        print("\nTraining final model on ALL data...")
        
        # Train a final model on all data
        final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        final_model.fit(X, y) 

        # --- 5. Predict Test Set ---
        test_predictions = final_model.predict(X_test)
        
        # --- 6. Log Artifacts ---
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # 4. Ensemble Hooks: Save OOF and Test Predictions (Numpy)
        np.save(os.path.join(PROCESSED_DIR, OOF_FILE), oof_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, OOF_FILE))
        
        np.save(os.path.join(PROCESSED_DIR, TEST_PRED_FILE), test_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, TEST_PRED_FILE))
        print(f"OOF and Test predictions saved as numpy arrays.")

        # Save Final Model
        mlflow.lightgbm.log_model(final_model, MODEL_FILE)
        print(f"Final LGBM model logged to MLflow under artifact path: {MODEL_FILE}")

        # Save Feature Importance CSV
        fi_csv_path = os.path.join(PROCESSED_DIR, "lgbm_feature_importance.csv")
        feature_importances['Average'] = feature_importances.mean(axis=1)
        feature_importances.sort_values(by='Average', ascending=False, inplace=True)
        feature_importances.to_csv(fi_csv_path)
        mlflow.log_artifact(fi_csv_path)

        # 3. Feature Importance Visualization: Save PNG
        plt.figure(figsize=(12, 8))
        top_feats = feature_importances['Average'].nlargest(30)
        top_feats.sort_values(ascending=True).plot(kind='barh') # Horizontal bar chart is often better for features
        plt.title("Top 30 Features (LGBM Importance)")
        plt.tight_layout()
        plot_path = os.path.join(PROCESSED_DIR, "feature_importance.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close() # Close plot to free memory

        # --- 7. Create Submission File ---
        submission_data = pd.DataFrame({
            'date_id': X_test['date_id'], 
            'prediction': test_predictions
        })
        submission_path = os.path.join("./processed", "submission_lgbm_baseline.csv")
        submission_data.to_csv(submission_path, index=False)
        mlflow.log_artifact(submission_path)
        print(f"Baseline submission file saved to {submission_path}")

    print("\n✅ LightGBM Baseline training complete and results logged to MLflow.")

if __name__ == "__main__":
    train_lgbm()