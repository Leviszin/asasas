# src/outlier_handler.py
import numpy as np
import pandas as pd

def cap_outliers_iqr(df, cols):
    """
    Faz capping (winsorizing) usando limites Q1-1.5*IQR e Q3+1.5*IQR.
    Retorna df modificado.
    """
    df_copy = df.copy()
    for col in cols:
        if col not in df_copy.columns:
            continue
        if not np.issubdtype(df_copy[col].dtype, np.number):
            continue
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_copy[col] = np.where(df_copy[col] < lower, lower,
                          np.where(df_copy[col] > upper, upper, df_copy[col]))
    return df_copy

def remove_extreme_rows(df, col, lower=None, upper=None):
    """Remove linhas onde col < lower ou col > upper (quando fornecidos)."""
    df_copy = df.copy()
    if lower is not None:
        df_copy = df_copy[df_copy[col] >= lower]
    if upper is not None:
        df_copy = df_copy[df_copy[col] <= upper]
    return df_copy
