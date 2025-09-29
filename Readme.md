# Aria Kaggle Ensemble (Time Series Forecasting)

## 🚀 Project Overview

This project is aimed at building a robust, high-performance ensemble model for a time series forecasting challenge, following best practices for competitive data science (Kaggle). The core strategy is **modular ensemble stacking** built upon leak-free **Time-Series Cross-Validation (TSCV)**.

### **Key Components Implemented:**

1.  **Leak-Free Data Preparation:** Feature engineering that strictly adheres to the temporal order.
2.  **Modular Ensemble:** Training diverse models (Tree-based and Deep Learning) independently.
3.  **MLOps:** Comprehensive experiment tracking using **MLflow**.

## 📊 Data Preparation (`data_prep.py`)

The initial data (assumed to be from a financial or similar time-series domain) was processed to convert a time-series forecasting task into a supervised machine learning regression task.

| Feature Type | Description |
| :--- | :--- |
| **Lags** | Past values of the target variable and core features (e.g., $t-1, t-5, t-10$). |
| **Rolling Windows** | Moving averages, standard deviations, min/max over defined time windows (e.g., 10-day mean, 20-day std). |
| **EWMA** | Exponentially Weighted Moving Averages to give higher weight to recent observations. |
| **Date/Time** | Features derived from the date column (day of week, month, etc.). |

**Crucial Step:** The entire training set was prepared using a **Leak-Free Time-Series Split (5 Folds)**, ensuring that no future information influenced past feature calculations or predictions.

## 🧠 Modeling Strategy

We employ a **stacking ensemble** approach to leverage the strengths of different model types and mitigate the risk of a single model's failure.

1.  **Independent Base Models:** Train each model (LGBM, TCN, LSTM) separately.
2.  **Out-Of-Fold (OOF) Predictions:** Generate a prediction for *every* training sample, ensuring the prediction was made only by models trained on *prior* data. These OOF predictions are the key input for the final Blender model.
3.  **Blender/Stacker:** A simple linear model (or small MLP) will be trained on the OOF predictions to find the optimal weighting/combination of the base models.

## ✅ Models Implemented So Far

### 1. LightGBM Baseline (`train_lgbm.py`)

* **Type:** Gradient Boosted Decision Tree (GBDT).
* **Role:** The robust, feature-focused component. Excellent at finding non-linear patterns in the 420 engineered features.
* **Status:** **Complete**. Generated `oof_lgbm.npy` and `test_lgbm.npy`.

### 2. Temporal Convolutional Network (TCN) (`train_tcn.py`)

* **Type:** Deep Learning (Sequential CNN).
* **Role:** The deep temporal component. Efficiently captures short-to-long term dependencies using causal and dilated convolutions, providing ensemble diversity against the tree-based model.
* **Status:** **In Progress**. Generated `oof_tcn.npy` and `test_tcn.npy`.

## 🛠️ MLOps and Reproducibility

* **MLflow:** Used to track all experiments, hyperparameters, and resulting metrics.
* **Artifacts:** For every model trained, the following key files are logged as artifacts:
    * `oof_<model>.npy`: OOF predictions for stacking.
    * `test_<model>.npy`: Test set predictions for final submission blending.
    * `feature_importance.png/csv` (for tree models).
    * Final model checkpoints.
* **Seeding:** Universal seeds are set across NumPy, Python, and TensorFlow/LightGBM to ensure full reproducibility of results.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This project successfully implemented a **Modular Ensemble Stacking** strategy for the time series forecasting challenge. We successfully trained two diverse base models (LightGBM and TCN) using leak-free Time-Series Cross-Validation (TSCV) and created a final submission using a Ridge Regression Blender.

| Metric | Model Type | OOF RMSE Score | Status |
| :--- | :--- | :--- | :--- |
| Baseline | LightGBM (GBDT) | $\approx 1.0082$ | ✅ Complete |
| Sequence | TCN (Deep Learning) | $1.0078$ | ✅ Complete |
| **Final Ensemble** | **Ridge Blender** | **$1.0076$** | ✅ Complete |

The final submission (`submission_final_blended.csv`) has been generated.

---

## ✅ Models Implemented & Key Findings

### 1. LightGBM Baseline (`train_lgbm.py`)

* **Role:** Robust, feature-focused component.
* **Result:** Established a strong RMSE baseline of **$1.0082$**.
* **Key Artifacts:** `oof_lgbm.npy`, `test_lgbm.npy`, `lgbm_feature_importance.png` logged to MLflow.

### 2. Temporal Convolutional Network (TCN) (`train_tcn.py`)

* **Role:** Deep temporal component, capturing local sequence patterns.
* **Result:** Outperformed the tree-based model slightly with an OOF RMSE of **$1.0078$**. Confirmed the necessity of data transformation to 3D sequences `(samples, time_steps, features)`.
* **Key Artifacts:** `oof_tcn.npy`, `test_tcn.npy` logged to MLflow.

### 3. Stacking and Blending (`stack_and_blend.py`)

* **Blender Model:** Ridge Regression (linear combination).
* **Final Result:** Achieved a small but meaningful lift, dropping the RMSE to **$1.0076$**.
* **Blender Weights Interpretation:**
    * **TCN Weight:** $\mathbf{1.8746}$ (Dominant component).
    * **LGBM Weight:** $\mathbf{-0.1486}$ (Slightly detrimental or highly redundant signal).
    * **Conclusion:** The TCN model is the primary driver of performance, while the current LGBM model is contributing little unique, valuable signal to the ensemble.

---

## 📈 Plans for Competitive Improvement (Next Steps)

The next phase of the project focuses on boosting the ensemble diversity and overall performance, addressing the observed weakness of the LightGBM component.

| Priority | Strategy | Action Plan | Rationale |
| :--- | :--- | :--- | :--- |
| **1 (High)** | **Improve LGBM Diversity** | Rerun `train_lgbm.py` with **new feature engineering** (e.g., deeper interaction terms, different EWMA decay periods) or **major hyperparameter changes** (e.g., higher `num_leaves`) to force it to learn different patterns than the TCN. | The low negative weight indicates the LGBM is not providing unique signal; this must be corrected for a stronger blend. |
| **2 (High)** | **Non-Linear Blending** | **Implement and train a small MLP** (1 hidden layer, 16-32 units) as the blender on `oof_lgbm.npy` and `oof_tcn.npy`. | A non-linear blender may discover complex relationships between the two model predictions that the linear Ridge model missed, potentially leading to a higher lift. |
| **3 (Medium)** | **Implement LSTM Component** | Create and train `train_lstm.py` (Phase C) as a third independent model. | An RNN introduces a third, different modeling bias (long-range dependency focus) compared to the TCN (local focus) and LGBM (feature focus), maximizing diversity. |
| **4 (Low)** | **TCN Hyperparameter Tuning** | Systematically tune TCN parameters like `SEQUENCE_LENGTH` (e.g., 20, 40, 60) and `TCN_FILTERS` to maximize the performance of the best base model. | Optimizing the strongest component (TCN) directly is a low-risk way to gain marginal performance. |

Our immediate focus will be on **Refinement 1 and 2** as they offer the fastest path to a performance gain using existing framework.