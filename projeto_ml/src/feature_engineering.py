# src/feature_engineering.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_features(df):
    """
    Cria features derivadas:
     - avg_speed_kmh = distance_km / delivery_time_hours (quando possível)
     - time_per_km = delivery_time_hours / distance_km
    Cuida de divisões por zero / NA.
    """
    df_copy = df.copy()

    # Normalize common column names (optional)
    if 'distance_km' in df_copy.columns and 'delivery_time_hours' in df_copy.columns:
        # evitar divisão por zero ou NaN
        df_copy['avg_speed_kmh'] = df_copy.apply(
            lambda r: r['distance_km'] / r['delivery_time_hours']
            if pd.notnull(r['distance_km']) and pd.notnull(r['delivery_time_hours']) and r['delivery_time_hours'] != 0
            else np.nan, axis=1
        )
        df_copy['time_per_km_h'] = df_copy.apply(
            lambda r: r['delivery_time_hours'] / r['distance_km']
            if pd.notnull(r['distance_km']) and pd.notnull(r['delivery_time_hours']) and r['distance_km'] != 0
            else np.nan, axis=1
        )
    return df_copy

def one_hot_encode(df, cat_cols, drop_first=True):
    """
    Retorna df com one-hot-encoding aplicado nas colunas cat_cols.
    Usa pandas.get_dummies internamente (mais simples).
    """
    existing = [c for c in cat_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing, drop_first=drop_first)
    return df_encoded

def scale_numeric(df, numeric_cols, save_path='models/scaler.pkl'):
    """
    Escala numeric_cols com StandardScaler, salva scaler em save_path.
    Retorna df_scaled e o scaler.
    """
    numeric_exist = [c for c in numeric_cols if c in df.columns]
    scaler = StandardScaler()
    df_copy = df.copy()
    if len(numeric_exist) > 0:
        df_copy[numeric_exist] = scaler.fit_transform(df_copy[numeric_exist])
    # salvar
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    joblib.dump(scaler, save_path)
    return df_copy, scaler
