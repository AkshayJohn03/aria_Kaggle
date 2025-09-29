import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import random

# ===============================
# CONFIG & SETUP
# ===============================
TARGET = "forward_returns"
PROCESSED_DIR = "./processed"
BLENDER_MODEL_RIDGE_FILE = "ridge_blender_3_model"

# List of all models to include in the stack
MODELS_TO_STACK = {
    "lgbm": "lgbm",
    "tcn": "tcn",
    "lstm": "lstm",  # NEW MODEL INCLUDED
}

MLFLOW_RUN_NAME = "Blender_Ridge_3_Model_Stack"

# Seeds
SEED = 42
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
set_seed(SEED)

mlflow.set_experiment("aria_kaggle_timeseries_ensemble")

# ===============================
# METRIC FUNCTION
# ===============================
def rms_error(y_true, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# MAIN BLENDING FUNCTION
# ===============================
def stack_and_blend():
    print("Starting 3-Model Stacking and Blending with Ridge...")
    
    # --- 1. Load Data (Targets & OOF/Test Predictions) ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    train_data = pd.read_csv(train_file)
    y_full = train_data[TARGET].values

    # Load OOF predictions and find min length for alignment
    oof_preds_list = []
    min_oof_len = len(y_full) 
    
    print("Loading OOF predictions...")
    for model_name, suffix in MODELS_TO_STACK.items():
        oof_path = os.path.join(PROCESSED_DIR, f"oof_{suffix}.npy")
        if not os.path.exists(oof_path):
            print(f"Error: OOF file for {model_name} not found at {oof_path}. Cannot blend.")
            return
        oof = np.load(oof_path)
        oof_preds_list.append((model_name, oof))
        min_oof_len = min(min_oof_len, len(oof))
        print(f"Loaded {model_name} OOF (Length: {len(oof)})")

    # Align OOF and Targets
    y_aligned = y_full[-min_oof_len:]
    X_blend = np.stack([oof[-min_oof_len:] for _, oof in oof_preds_list], axis=1)
    
    model_names = [name for name, _ in oof_preds_list]
    print(f"\nBlending {len(model_names)} models: {', '.join(model_names)}")
    print(f"X_blend shape: {X_blend.shape} | y_aligned shape: {y_aligned.shape}")

    # --- 2. Train the Ridge Blender Model ---
    
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        
        # LINEAR RIDGE BLENDER
        blender = Ridge(alpha=0.1, fit_intercept=True)
        blender.fit(X_blend, y_aligned)
        oof_blended_preds = blender.predict(X_blend)
        
        # Log Weights
        weights = dict(zip(model_names, blender.coef_))
        intercept = blender.intercept_
        print("\n--- Ridge Blender Weights ---")
        print(f"Intercept: {intercept:.4f}")
        for model, weight in weights.items():
            print(f"{model}: {weight:.4f}")
            mlflow.log_metric(f"weight_{model}", weight)
        
        mlflow.log_param("blender_model", "Ridge")
        mlflow.sklearn.log_model(blender, BLENDER_MODEL_RIDGE_FILE)

        # --- 3. Final OOF Evaluation ---
        final_oof_rmse = rms_error(y_aligned, oof_blended_preds)
        print(f"\n===== Final Blended OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_blended_oof_rmse", final_oof_rmse)

        # --- 4. Predict Final Test Set ---
        test_preds_list = []
        min_test_len = 10000 
        
        print("\nLoading Test predictions...")
        for model_name, suffix in MODELS_TO_STACK.items():
            test_path = os.path.join(PROCESSED_DIR, f"test_{suffix}.npy")
            test_pred = np.load(test_path)
            test_preds_list.append(test_pred)
            min_test_len = min(min_test_len, len(test_pred))
            print(f"Loaded {model_name} Test Preds (Length: {len(test_pred)})")
            
        # Align Test Predictions and create matrix
        X_test_blend = np.stack([pred[-min_test_len:] for pred in test_preds_list], axis=1)
        
        # Apply Blender to Test Predictions
        final_test_predictions = blender.predict(X_test_blend).flatten()
        
        # --- 5. Create Final Submission File ---
        test_index_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_final_scaled.csv"))
        submission_index = test_index_data.iloc[-min_test_len:]['date_id'].values
        
        submission_data = pd.DataFrame({
            'date_id': submission_index,
            'prediction': final_test_predictions
        })
        # Note: Overwriting the general submission file with the best version
        submission_path = os.path.join("./processed", f"submission_final_blended.csv")
        submission_data.to_csv(submission_path, index=False)
        mlflow.log_artifact(submission_path)
        print(f"\n✅ Final Blended Submission file saved to {submission_path}")

    print("\nBlending process complete.")

if __name__ == "__main__":
    stack_and_blend()