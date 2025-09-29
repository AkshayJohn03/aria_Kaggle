import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Dropout, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.tensorflow
import random

# ===============================
# 1. REPRODUCIBILITY & SEEDS
# ===============================
SEED = 42
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

# ===============================
# CONFIG & SETUP
# ===============================
TARGET = "forward_returns"
N_SPLITS = 5
PROCESSED_DIR = "./processed"
OOF_FILE = "oof_tcn.npy"
TEST_PRED_FILE = "test_tcn.npy"
MODEL_FILE = "tcn_model"

# TCN Hyperparameters
SEQUENCE_LENGTH = 30  # Look-back window size (experiment with 20-40)
TCN_FILTERS = 32
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1
NUM_TCN_BLOCKS = 3

# MLflow Setup
mlflow.set_experiment("aria_kaggle_timeseries_ensemble")

# ===============================
# METRIC FUNCTION
# ===============================
def rms_error(y_true, y_pred):
    """Root Mean Squared Error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# DATA TRANSFORMATION
# ===============================

def create_sequences(X, y=None, seq_len=SEQUENCE_LENGTH):
    """
    Transforms 2D tabular data (samples, features) into 3D sequential data 
    (samples, seq_len, features) for RNN/CNN/TCN models.
    """
    X_seq, y_seq = [], []
    num_samples = len(X)
    
    # We start from seq_len to ensure every sample has a full look-back history
    for i in range(seq_len, num_samples):
        X_seq.append(X[i - seq_len:i, :])
        if y is not None:
            # The target is the value *at* the end of the window (i)
            y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq) if y is not None else None
    
    return X_seq, y_seq

# ===============================
# TCN MODEL DEFINITION (Using Residual Blocks)
# ===============================

def tcn_block(input_layer, filters, kernel_size, dilation_rate, dropout):
    """Defines a residual Temporal Convolutional Network block."""
    
    # 1. Causal Convolution Layer
    conv_out = Conv1D(
        filters, 
        kernel_size, 
        dilation_rate=dilation_rate, 
        padding='causal', 
        name=f"conv_{dilation_rate}"
    )(input_layer)
    conv_out = Activation('relu')(conv_out)
    conv_out = Dropout(dropout)(conv_out)

    # 2. Skip Connection (Residual)
    # Match dimensions if needed (1x1 convolution)
    if input_layer.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', name=f"shortcut_{dilation_rate}")(input_layer)
    else:
        shortcut = input_layer

    # 3. Add Residual connection
    return Add()([shortcut, conv_out])

def build_tcn_model(input_shape, filters, kernel_size, dropout, num_blocks):
    """Builds the final TCN model."""
    inputs = Input(shape=input_shape)
    
    # Initial 1D convolution
    x = Conv1D(filters, 1, padding='causal')(inputs)

    # TCN Stack with increasing dilation rates (1, 2, 4, 8, ...)
    for i in range(num_blocks):
        # Dilation rate increases exponentially
        dilation_rate = 2 ** i 
        x = tcn_block(x, filters * (2**i), kernel_size, dilation_rate, dropout)

    # Global Averaging across the time dimension
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Final Dense Layers for Regression
    x = Dense(filters, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='mse')
    return model

# ===============================
# MAIN TRAINING FUNCTION
# ===============================
def train_tcn():
    print("Starting TCN Training...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- 1. Load Data ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_final_scaled.csv")
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Separate features and target (get underlying NumPy arrays for sequence creation)
    X_full_array = train_data.drop(columns=[TARGET]).values
    y_full_array = train_data[TARGET].values
    X_test_array = test_data.values 
    
    num_features = X_full_array.shape[1]
    
    # --- 2. Sequence Transformation (Full Data) ---
    X_seq_full, y_seq_full = create_sequences(X_full_array, y_full_array)
    X_seq_test, _ = create_sequences(X_test_array)
    
    # Initialize OOF array based on the length of the sequence data
    oof_predictions = np.zeros(len(y_seq_full)) 
    test_predictions = np.zeros(len(X_seq_test))

    print(f"Original Train shape: {X_full_array.shape} | Sequence Train shape: {X_seq_full.shape}")
    print(f"Original Test shape: {X_test_array.shape} | Sequence Test shape: {X_seq_test.shape}")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    # Index adjustments for sequence data. The fold indices need to be shifted by SEQUENCE_LENGTH
    # because the first SEQUENCE_LENGTH rows are lost in the sequence transformation.
    base_indices = np.arange(len(X_full_array))
    
    with mlflow.start_run(run_name="TCN_Deep_Temporal"):
        # Log parameters
        tcn_params = {
            'seq_len': SEQUENCE_LENGTH,
            'filters': TCN_FILTERS,
            'kernel_size': TCN_KERNEL_SIZE,
            'dropout': TCN_DROPOUT,
            'num_blocks': NUM_TCN_BLOCKS,
            'n_splits': N_SPLITS,
            'learning_rate': 3e-4
        }
        mlflow.log_params(tcn_params)

        
        # --- 3. TimeSeries Cross-Validation ---
        for fold, (train_idx_raw, valid_idx_raw) in enumerate(tscv.split(base_indices)):
            
            # Adjust indices: Shift by SEQUENCE_LENGTH
            # We are interested in indices *after* the initial look-back period (30)
            train_idx = train_idx_raw[train_idx_raw >= SEQUENCE_LENGTH] - SEQUENCE_LENGTH
            valid_idx = valid_idx_raw[valid_idx_raw >= SEQUENCE_LENGTH] - SEQUENCE_LENGTH
            
            # The indices for the OOF array (valid_idx_oof) correspond to the indices *after* the shift
            valid_idx_oof = valid_idx # Since OOF array is len(y_seq_full), these indices work

            print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")
            print(f"  Train/Valid Size (Seq): {len(train_idx)} / {len(valid_idx)}")

            X_train, X_valid = X_seq_full[train_idx], X_seq_full[valid_idx]
            y_train, y_valid = y_seq_full[train_idx], y_seq_full[valid_idx]

            # Initialize and Train Model
            tcn_model = build_tcn_model(X_train.shape[1:], TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DROPOUT, NUM_TCN_BLOCKS)
            
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = tcn_model.fit(
                X_train, y_train,
                validation_data=(X_valid, y_valid),
                epochs=50,
                batch_size=64,
                callbacks=[early_stopping],
                verbose=0,
            )

            # Predict OOF
            oof_pred = tcn_model.predict(X_valid, verbose=0).flatten()
            oof_predictions[valid_idx_oof] = oof_pred
            
            # Log Fold Metric
            fold_rmse = rms_error(y_valid, oof_pred)
            print(f"  Fold RMSE: {fold_rmse:.4f} | Best Epoch: {np.argmin(history.history['val_loss']) + 1}")
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
            
        # --- 4. Final OOF Evaluation ---
        # FIX APPLIED: Compare OOF array (8961) against y_seq_full (8961)
        final_oof_rmse = rms_error(y_seq_full, oof_predictions) 
        
        print(f"\n===== Final TCN OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_oof_rmse", final_oof_rmse)

        # --- 5. Final Model Training (for Test Predictions) ---
        print("\nTraining final TCN model on ALL sequence data...")
        
        # Use a slightly more aggressive epoch count for the final training, 
        # as it uses the entire dataset for generalization.
        final_model = build_tcn_model(X_seq_full.shape[1:], TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DROPOUT, NUM_TCN_BLOCKS)
        final_model.fit(X_seq_full, y_seq_full, epochs=30, batch_size=64, verbose=0) 

        # --- 6. Predict Test Set ---
        test_predictions = final_model.predict(X_seq_test, verbose=0).flatten()
        
        # --- 7. Log Artifacts ---
        
        # Save OOF and Test Predictions (Numpy)
        np.save(os.path.join(PROCESSED_DIR, OOF_FILE), oof_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, OOF_FILE))
        
        np.save(os.path.join(PROCESSED_DIR, TEST_PRED_FILE), test_predictions)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, TEST_PRED_FILE))
        print(f"TCN OOF and Test predictions saved as numpy arrays.")

        # Save Final Model
        mlflow.tensorflow.log_model(final_model, MODEL_FILE)
        print(f"Final TCN model logged to MLflow under artifact path: {MODEL_FILE}")

    print("\n✅ TCN training complete and results logged to MLflow.")

if __name__ == "__main__":
    train_tcn()