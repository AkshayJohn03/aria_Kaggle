import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
import mlflow
import mlflow.lightgbm
import random

# ===============================
# CONFIG & SEEDS
# ===============================
TARGET = "forward_returns"
N_SPLITS = 5
PROCESSED_DIR = "./processed"
SEED = 42

def set_seed(seed):
    """Sets seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(SEED)

mlflow.set_experiment("aria_kaggle_timeseries_ensemble")

# ===============================
# METRIC
# ===============================
def rms_error(y_true, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# OBJECTIVE FUNCTION FOR OPTUNA
# ===============================
def objective(trial, X, y, X_test):
    """
    Optuna objective function for tuning LightGBM hyperparameters 
    via TimeSeries Cross-Validation (TSCV).
    """
    
    # 1. Define Search Space (Modified from user input for better Optuna practice)
    params = {
        "objective": "regression_l1",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        # Increase the search space for complexity to drive divergence from TCN
        "num_leaves": trial.suggest_int("num_leaves", 64, 512), 
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_uniform("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.7, 1.0),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-2, 10.0), # Stronger regularization search
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-2, 10.0),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds = np.zeros(len(X))
    rmse_scores = []

    # Use nested run for each trial to track parameters and CV results
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        EARLY_STOPPING_ROUNDS = 100

        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )

            preds = model.predict(X_valid)
            oof_preds[valid_idx] = preds
            fold_rmse = rms_error(y_valid, preds)
            rmse_scores.append(fold_rmse)

            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
            
        final_rmse = rms_error(y, oof_preds)
        mlflow.log_metric("final_oof_rmse", final_rmse)
        
    return final_rmse

# ===============================
# FINAL TRAINING AND ARTIFACT SAVING
# ===============================

def final_train_and_save(best_params, X, y, X_test):
    """Retrains the best model using the full TSCV and saves OOF/Test artifacts."""
    
    OOF_FILE = "oof_lgbm.npy"
    TEST_PRED_FILE = "test_lgbm.npy"
    MODEL_FILE = "lgbm_optimized_model"
    
    EARLY_STOPPING_ROUNDS = 100

    print("\n\n===== Final Training of Optimized LGBM =====")
    
    # Initialize containers
    oof_predictions = np.zeros(len(X))
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Use a non-nested MLflow run for the final, best model
    with mlflow.start_run(run_name="LGBM_OPTIMIZED_FINAL_MODEL"):
        mlflow.log_params(best_params)

        # 1. Generate Final OOF Predictions using best params
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMRegressor(**best_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof_predictions[valid_idx] = model.predict(X_valid)

        final_oof_rmse = rms_error(y, oof_predictions)
        print(f"Optimized LGBM Final OOF RMSE: {final_oof_rmse:.4f}")
        mlflow.log_metric("final_oof_rmse", final_oof_rmse)
        
        # 2. Train Model on ALL Data and Predict Test Set
        print("Training final model on ALL data...")
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X, y)
        test_predictions = final_model.predict(X_test)
        
        # 3. Save and Log Artifacts (overwriting previous files)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        np.save(os.path.join(PROCESSED_DIR, OOF_FILE), oof_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, OOF_FILE))
        
        np.save(os.path.join(PROCESSED_DIR, TEST_PRED_FILE), test_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, TEST_PRED_FILE))
        
        mlflow.lightgbm.log_model(final_model, MODEL_FILE)
        print("✅ Optimized OOF and Test Predictions saved, ready for re-blending.")


# ===============================
# MAIN ENTRY POINT
# ===============================
if __name__ == "__main__":
    
    # --- Data Loading (Once) ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_final_scaled.csv")
    
    if not os.path.exists(train_file):
        print(f"ERROR: Training file not found at {train_file}. Cannot proceed with HPO.")
        exit()

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_full = train_data.drop(columns=[TARGET])
    y_full = train_data[TARGET]
    X_test_full = test_data

    # --- Optuna Optimization ---
    print("Starting Optuna Hyperparameter Optimization (30 trials)...")
    study = optuna.create_study(direction="minimize", study_name="LGBM_Optuna")
    
    # Use lambda to pass necessary data to the objective function
    study.optimize(
        lambda trial: objective(trial, X_full, y_full, X_test_full), 
        n_trials=30
    )

    print("\n===== Optimization Complete =====")
    print(f"Best Trial RMSE: {study.best_value:.4f}")
    print("\nBest Parameters:")
    print(study.best_trial.params)

    # --- Final Training and Artifact Saving ---
    final_train_and_save(study.best_trial.params, X_full, y_full, X_test_full)

    # --- Post-Optimization Action ---
    print("\nNEXT STEP: The optimized LGBM predictions have overwritten the old files.")
    print("Run python stack_and_blend.py to recompute the ensemble with the improved LGBM model.")