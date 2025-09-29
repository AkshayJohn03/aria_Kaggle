import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
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

# LSTM Parameters
SEQUENCE_LENGTH = 30
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 40 

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
# MODEL DEFINITION
# ===============================

def build_lstm_model(input_shape):
    """Builds a sequence model using LSTM layers."""
    
    input_layer = Input(shape=input_shape)
    
    # First LSTM layer (return_sequences=True to feed to the next LSTM)
    x = LSTM(
        LSTM_UNITS, 
        return_sequences=True, 
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE,
        name='lstm_1'
    )(input_layer)
    
    # Second LSTM layer (return_sequences=False for single output vector)
    x = LSTM(
        LSTM_UNITS // 2, 
        return_sequences=False, 
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE,
        name='lstm_2'
    )(x)
    
    # Dense layers for final prediction
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# ===============================
# DATA PREPARATION (Uses TCN's helper, but defined here for self-containment)
# ===============================

def create_sequences(data, sequence_length):
    """Creates time series sequences for LSTM input."""
    X, y = [], []
    # Note: We skip the first 'sequence_length' rows since we need the previous data
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length].drop(columns=[TARGET]).values)
        y.append(data.iloc[i + sequence_length][TARGET])
    return np.array(X), np.array(y)

# ===============================
# MAIN TRAINING FUNCTION
# ===============================

def train_lstm():
    print("Starting LSTM Training (Phase C Integration)...")
    
    # --- Load Data ---
    train_file = os.path.join(PROCESSED_DIR, "train_final_scaled.csv")
    test_file = os.path.join(PROCESSED_DIR, "test_final_scaled.csv")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # --- Sequence Creation ---
    X_seq_full, y_seq_full = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test_seq, _ = create_sequences(test_data.assign(**{TARGET: 0}), SEQUENCE_LENGTH) 
    
    print(f"Original Train shape: {train_data.shape} | Sequence Train shape: {X_seq_full.shape}")
    
    input_shape = X_seq_full.shape[1:]
    
    # --- TimeSeries Cross-Validation ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds = np.zeros(len(X_seq_full))
    
    # MLflow Setup
    with mlflow.start_run(run_name="LSTM_Base_Model"):
        mlflow.log_params({
            'sequence_length': SEQUENCE_LENGTH,
            'lstm_units': LSTM_UNITS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS
        })
        
        print(f"\n--- Training {N_SPLITS} Folds ---")
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_seq_full)):
            print(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")
            
            X_train, X_valid = X_seq_full[train_idx], X_seq_full[valid_idx]
            y_train, y_valid = y_seq_full[train_idx], y_seq_full[valid_idx]
            
            print(f"  Train/Valid Size (Seq): {len(X_train)} / {len(X_valid)}")
            
            model = build_lstm_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
                ModelCheckpoint(f"./lstm_fold_{fold+1}.keras", monitor='val_loss', save_best_only=True, verbose=0)
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
            best_model = tf.keras.models.load_model(f"./lstm_fold_{fold+1}.keras")
            preds = best_model.predict(X_valid, verbose=0).flatten()
            oof_preds[valid_idx] = preds
            
            fold_rmse = rms_error(y_valid, preds)
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
            
            print(f"  Fold RMSE: {fold_rmse:.4f}")
            
            # Clean up checkpoint file
            os.remove(f"./lstm_fold_{fold+1}.keras")

        # --- Final OOF Score ---
        final_oof_rmse = rms_error(y_seq_full, oof_preds)
        print(f"\n===== Final LSTM OOF RMSE: {final_oof_rmse:.4f} =====")
        mlflow.log_metric("final_oof_rmse", final_oof_rmse)
        
        # --- Train Final Model on ALL sequence data ---
        print("\nTraining final LSTM model on ALL sequence data...")
        final_lstm_model = build_lstm_model(input_shape)
        
        final_callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]

        final_lstm_model.fit(
            X_seq_full, y_seq_full,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=final_callbacks,
            verbose=0
        )
        
        # --- Predict Test Set ---
        test_predictions = final_lstm_model.predict(X_test_seq, verbose=0).flatten()
        
        # --- Save Artifacts (Note: Save as oof_lstm.npy for blending) ---
        np.save(os.path.join(PROCESSED_DIR, "oof_lstm.npy"), oof_preds)
        np.save(os.path.join(PROCESSED_DIR, "test_lstm.npy"), test_predictions)
        
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "oof_lstm.npy"))
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "test_lstm.npy"))
        
        mlflow.tensorflow.log_model(final_lstm_model, "lstm_model")
        
        print("\nLSTM OOF and Test predictions saved as numpy arrays.")
        print("\nNEXT STEP: Update stack_and_blend.py to include the 'lstm' model and re-run to create a 3-model blend.")

if __name__ == "__main__":
    train_lstm()