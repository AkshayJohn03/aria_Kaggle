import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# ===============================
# CONFIG & SETUP
# ===============================
TARGET = "forward_returns"
PROCESSED_DIR = "./processed"
BLENDER_MODEL_FILE = "ridge_blender"

# List of all models to include in the stack
# Key: Model Name, Value: Target Filename Suffix
MODELS_TO_STACK = {
    "lgbm": "lgbm",
    "tcn": "tcn",
    # Add 'lstm': 'lstm' here if you decide to run the LSTM model later
}

# MLflow Setup
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
    print("Starting Stacking and Blending...")
    
    # --- 1. Load Data (Targets & OOF/Test Predictions) ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    
    # Load targets (y_true)
    train_data = pd.read_csv(train_file)
    y_full = train_data[TARGET].values

    # Determine the shortest prediction length (usually TCN/LSTM due to sequence lookback)
    # This ensures alignment. We must use the minimum length across all OOF files.
    oof_preds_list = []
    min_oof_len = len(y_full) 
    
    print("Loading OOF predictions...")
    
    # Iterate through models to load OOF and find min length
    for model_name, suffix in MODELS_TO_STACK.items():
        oof_path = os.path.join(PROCESSED_DIR, f"oof_{suffix}.npy")
        
        if not os.path.exists(oof_path):
            print(f"Warning: OOF file for {model_name} not found at {oof_path}. Skipping.")
            continue
        
        oof = np.load(oof_path)
        oof_preds_list.append((model_name, oof))
        min_oof_len = min(min_oof_len, len(oof))
        print(f"Loaded {model_name} OOF (Length: {len(oof)})")

    if not oof_preds_list:
        print("Error: No OOF predictions loaded. Cannot blend.")
        return

    # --- 2. Align OOF and Targets ---
    
    # Filter the true targets to match the length of the shortest OOF array
    y_aligned = y_full[-min_oof_len:]
    
    # Create the OOF stacking matrix (X_blend)
    X_blend = np.stack([oof[-min_oof_len:] for _, oof in oof_preds_list], axis=1)
    
    model_names = [name for name, _ in oof_preds_list]
    print(f"\nBlending {len(model_names)} models: {', '.join(model_names)}")
    print(f"X_blend shape: {X_blend.shape} | y_aligned shape: {y_aligned.shape}")

    # --- 3. Train the Blender Model (Ridge Regression) ---
    
    # Ridge is preferred for blending as it encourages stable, non-zero weights
    # Alpha controls regularization (0.1 is a good starting point)
    blender = Ridge(alpha=0.1, fit_intercept=True)
    
    with mlflow.start_run(run_name="Blender_Ridge_Stack"):
        # Log parameters
        mlflow.log_param("blender_model", "Ridge")
        mlflow.log_param("blender_alpha", blender.alpha)
        mlflow.log_param("base_models", ",".join(model_names))
        
        # Fit the blender on OOF predictions against true targets
        blender.fit(X_blend, y_aligned)
        
        # Predict blended OOF score
        oof_blended_preds = blender.predict(X_blend)
        final_oof_rmse = rms_error(y_aligned, oof_blended_preds)
        
        # --- 4. Log Results ---
        
        # Log Blender Weights
        weights = dict(zip(model_names, blender.coef_))
        intercept = blender.intercept_
        print("\n--- Blender Weights ---")
        print(f"Intercept: {intercept:.4f}")
        for model, weight in weights.items():
            print(f"{model}: {weight:.4f}")
            mlflow.log_metric(f"weight_{model}", weight)

        print(f"\n===== Final Blended OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_blended_oof_rmse", final_oof_rmse)
        
        # Save Blender Model
        mlflow.sklearn.log_model(blender, BLENDER_MODEL_FILE)
        print(f"Blender model logged to MLflow.")

        # --- 5. Predict Final Test Set ---
        
        # Load Test Predictions and Align
        test_preds_list = []
        min_test_len = 10000 # Use a large number initially

        print("\nLoading Test predictions...")
        for model_name, suffix in MODELS_TO_STACK.items():
            test_path = os.path.join(PROCESSED_DIR, f"test_{suffix}.npy")
            
            if not os.path.exists(test_path):
                # Ensure we only use models that contributed to the blend
                if model_name in model_names:
                    print(f"Error: Test file for {model_name} not found. Cannot create submission.")
                    return
                continue
                
            test_pred = np.load(test_path)
            test_preds_list.append(test_pred)
            min_test_len = min(min_test_len, len(test_pred))
            print(f"Loaded {model_name} Test Preds (Length: {len(test_pred)})")
            
        # Align Test Predictions and create matrix
        X_test_blend = np.stack([pred[-min_test_len:] for pred in test_preds_list], axis=1)
        
        # Apply Blender to Test Predictions
        final_test_predictions = blender.predict(X_test_blend)
        
        # --- 6. Create Final Submission File ---
        
        # Load the test_final_scaled.csv to get the original index (date_id)
        test_index_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_final_scaled.csv"))
        
        # Ensure submission index aligns with the shortest test prediction length
        submission_index = test_index_data.iloc[-min_test_len:]['date_id'].values
        
        submission_data = pd.DataFrame({
            'date_id': submission_index,
            'prediction': final_test_predictions
        })
        submission_path = os.path.join("./processed", "submission_final_blended.csv")
        submission_data.to_csv(submission_path, index=False)
        mlflow.log_artifact(submission_path)
        print(f"\n✅ Final Blended Submission file saved to {submission_path}")

    print("\nBlending process complete.")

if __name__ == "__main__":
    stack_and_blend()