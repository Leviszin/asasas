# src/deployment.py
import joblib
import pandas as pd
import os
import numpy as np

def load_scaler(path='models/scaler.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler não encontrado em {path}")
    return joblib.load(path)

def load_model(path='models/modelo_final.joblib'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em {path}")
    return joblib.load(path)

def predict_df(model, scaler, df, numeric_cols):
    """
    Recebe DataFrame com features (já codificadas), escala numeric_cols com scaler.transform
    e retorna array de predições.
    """
    df_copy = df.copy()
    # escala numerics
    df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])
    preds = model.predict(df_copy)
    return preds
