import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------
# Core Feature Engineering
# ---------------------------

def add_lags(df, cols, lags=[1,2,3]):
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_diffs(df, cols, periods=[1]):
    for col in cols:
        for p in periods:
            df[f"{col}_diff{p}"] = df[col].diff(p)
    return df

def add_pct_change(df, cols, periods=[1]):
    for col in cols:
        for p in periods:
            df[f"{col}_pctchg{p}"] = df[col].pct_change(p)
    return df

def add_rolling(df, cols, windows=[3,5,10]):
    for col in cols:
        for w in windows:
            df[f"{col}_rollmean{w}"] = df[col].rolling(w).mean()
            df[f"{col}_rollstd{w}"] = df[col].rolling(w).std()
    return df

def add_ewma(df, cols, alphas=[0.1,0.3,0.7]):
    for col in cols:
        for a in alphas:
            df[f"{col}_ewm{a}"] = df[col].ewm(alpha=a, adjust=False).mean()
    return df

def add_missing_flags(df, cols):
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["row_missing_count"] = df[cols].isna().sum(axis=1)
    return df

def add_cross_features(df, cols, topk=10):
    top_cols = cols[:topk]  # restrict to avoid explosion
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            c1, c2 = top_cols[i], top_cols[j]
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-9)
    return df

def add_pca(df, cols, n_components=3, prefix="pca"):
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(df[cols].fillna(0))
    for i in range(n_components):
        df[f"{prefix}_{i+1}"] = comps[:, i]
    return df

# ---------------------------
# Master function (per fold)
# ---------------------------

def create_features(df, feature_cols):
    """
    Apply leak-safe feature creation to a dataframe.
    Should be called separately for each CV fold split (train+valid).
    """
    df = add_missing_flags(df, feature_cols)
    df = add_lags(df, feature_cols)
    df = add_diffs(df, feature_cols)
    df = add_pct_change(df, feature_cols)
    df = add_rolling(df, feature_cols)
    df = add_ewma(df, feature_cols)
    df = add_cross_features(df, feature_cols, topk=10)

    # Grouped PCA (example: all 'V*' columns)
    v_cols = [c for c in feature_cols if c.startswith("V")]
    if v_cols:
        df = add_pca(df, v_cols, n_components=3, prefix="pca_V")
    return df
