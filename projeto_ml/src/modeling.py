# src/modeling.py
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_baseline(X, y, save_path='models/modelo_final.joblib', cv=5):
    """
    Treina um modelo baseline (LinearRegression), avalia por CV e salva.
    Retorna modelo treinado e dicionário de métricas.
    """
    model = LinearRegression()
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    rmse_cv = -scores.mean()
    # Treina no dataset todo
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    joblib.dump(model, save_path)

    # Métricas no treino
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)

    metrics = {'rmse_cv': rmse_cv, 'mae_train': mae, 'rmse_train': rmse, 'r2_train': r2}
    return model, metrics
