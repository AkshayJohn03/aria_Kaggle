import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.tensorflow
import random

# ===============================
# CONFIG & SEEDS
# ===============================
TARGET = "forward_returns"
N_SPLITS = 5
PROCESSED_DIR = "./processed"

# TCN Parameters (Aggressive Settings)
SEQUENCE_LENGTH = 30
TCN_FILTERS = 128      # INCREASED CAPACITY
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]
LEARNING_RATE = 0.0005 # SMALLER LR
BATCH_SIZE = 16        # SMALLER BATCH SIZE
EPOCHS = 50

# Seeds
SEED = 42
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

mlflow.set_experiment("aria_kaggle_timeseries_ensemble")

# ===============================
# METRIC
# ===============================
def rms_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===============================
# TCN BLOCK DEFINITION
# ===============================

def residual_block(x, dilation_rate, filters, kernel_size, name):
    """Temporal Convolutional Network (TCN) residual block."""
    original_x = x
    
    # Dilated Convolution
    conv = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        name=f'{name}_conv'
    )(x)
    
    # Normalization and Activation
    norm = tf.keras.layers.BatchNormalization(name=f'{name}_norm')(conv)
    act = Activation('relu', name=f'{name}_act')(norm)
    
    # 1x1 Conv for Skip Connection (if input/output shapes don't match)
    if original_x.shape[-1] != filters:
        original_x = tf.keras.layers.Conv1D(
            filters,
            1,
            padding='same',
            name=f'{name}_skip_conv'
        )(original_x)

    # Residual Connection
    return Add(name=f'{name}_add')([original_x, act])

def build_tcn_model(input_shape):
    """Builds the complete TCN sequence model."""
    
    input_layer = Input(shape=input_shape)
    x = input_layer

    # Initial Convolution (optional, but good for feature mixing)
    x = tf.keras.layers.Conv1D(TCN_FILTERS // 2, 1, padding='same', name='initial_conv')(x)
    
    # TCN Blocks
    for i, dilation in enumerate(TCN_DILATIONS):
        x = residual_block(
            x, 
            dilation_rate=dilation, 
            filters=TCN_FILTERS, 
            kernel_size=TCN_KERNEL_SIZE, 
            name=f'resblock_{i+1}'
        )
    
    # Global Average Pooling to reduce sequence to a single vector
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # Final Dense layers for regression
    x = Dense(64, activation='relu', name='dense_1')(x)
    output_layer = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# ===============================
# DATA PREPARATION
# ===============================

def create_sequences(data, sequence_length):
    """Creates time series sequences for TCN/RNN input."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length].drop(columns=[TARGET]).values)
        y.append(data.iloc[i + sequence_length][TARGET])
    return np.array(X), np.array(y)

# ===============================
# MAIN TRAINING FUNCTION
# ===============================

def train_tcn():
    print("Starting TCN Training (Aggressive Optimization)...")
    
    # --- Load Data ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_final_scaled.csv")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # --- Sequence Creation ---
    X_seq_full, y_seq_full = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test_seq, _ = create_sequences(test_data.assign(**{TARGET: 0}), SEQUENCE_LENGTH) # Placeholder target for test data
    
    print(f"Original Train shape: {train_data.shape} | Sequence Train shape: {X_seq_full.shape}")
    print(f"Original Test shape: {test_data.shape} | Sequence Test shape: {X_test_seq.shape}")
    
    input_shape = X_seq_full.shape[1:]
    
    # --- TimeSeries Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds = np.zeros(len(X_seq_full))
    
    # MLflow Setup
    with mlflow.start_run(run_name="TCN_Optimized_Aggressive"):
        mlflow.log_params({
            'sequence_length': SEQUENCE_LENGTH,
            'tcn_filters': TCN_FILTERS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS
        })
        
        test_predictions_list = []
        oof_rmse_scores = []
        
        print(f"\n--- Training {N_SPLITS} Folds ---")
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_seq_full)):
            print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")
            
            X_train, X_valid = X_seq_full[train_idx], X_seq_full[valid_idx]
            y_train, y_valid = y_seq_full[train_idx], y_seq_full[valid_idx]
            
            print(f"  Train/Valid Size (Seq): {len(X_train)} / {len(X_valid)}")
            
            model = build_tcn_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
                ModelCheckpoint(f"./tcn_fold_{fold+1}.keras", monitor='val_loss', save_best_only=True, verbose=0)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_valid, y_valid),
                callbacks=callbacks,
                verbose=0
            )

            # Load best weights and predict
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_val_loss = history.history['val_loss'][best_epoch-1]
            
            best_model = tf.keras.models.load_model(f"./tcn_fold_{fold+1}.keras")
            preds = best_model.predict(X_valid, verbose=0).flatten()
            oof_preds[valid_idx] = preds
            
            fold_rmse = rms_error(y_valid, preds)
            oof_rmse_scores.append(fold_rmse)
            
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
            mlflow.log_metric(f"fold_{fold+1}_best_epoch", best_epoch)
            
            print(f"  Fold RMSE: {fold_rmse:.4f} | Best Epoch: {best_epoch} | Best Loss: {best_val_loss:.6f}")
            
            # Clean up checkpoint file
            os.remove(f"./tcn_fold_{fold+1}.keras")

        # --- Final OOF Score ---
        final_oof_rmse = rms_error(y_seq_full, oof_preds)
        print(f"\n===== Final TCN OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_oof_rmse", final_oof_rmse)
        
        # --- Train Final Model on ALL sequence data ---
        print("\nTraining final TCN model on ALL sequence data...")
        final_tcn_model = build_tcn_model(input_shape)
        
        final_callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]

        final_tcn_model.fit(
            X_seq_full, y_seq_full,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=final_callbacks,
            verbose=0
        )
        
        # --- Predict Test Set ---
        test_predictions = final_tcn_model.predict(X_test_seq, verbose=0).flatten()
        
        # --- Save Artifacts ---
        np.save(os.path.join(PROCESSED_DIR, "oof_tcn.npy"), oof_preds)
        np.save(os.path.join(PROCESSED_DIR, "test_tcn.npy"), test_predictions)
        
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "oof_tcn.npy"))
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "test_tcn.npy"))
        
        mlflow.tensorflow.log_model(final_tcn_model, "tcn_optimized_model")
        
        print("\nTCN OOF and Test predictions saved as numpy arrays (overwritten).")
        print(f"Final TCN model logged to MLflow under artifact path: tcn_optimized_model")
        print("\n✅ TCN aggressive optimization complete and results logged to MLflow.")


if __name__ == "__main__":
    train_tcn()