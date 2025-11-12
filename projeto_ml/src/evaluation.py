# src/evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_residuals(y_true, y_pred, figsize=(6,4)):
    resid = y_true - y_pred
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, resid, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predito')
    plt.ylabel('Resíduo (y_true - y_pred)')
    plt.title('Resíduos vs Predito')
    plt.show()

def plot_actual_vs_pred(y_true, y_pred, figsize=(6,4)):
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.4)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn,mx], [mn,mx], color='red', linestyle='--')
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.title('Real vs Predito')
    plt.show()
