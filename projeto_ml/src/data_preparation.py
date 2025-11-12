# src/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# ======================================================
# Função para carregar dataset
# ======================================================
def load_data(path: str) -> pd.DataFrame:
    """Carrega um arquivo CSV e retorna um DataFrame pandas."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset carregado com sucesso: {path}")
        return df
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado em: {path}")
        raise


# ======================================================
# Padronizar texto/categorias
# ======================================================
def standardize_text_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza textos para minúsculas e remove espaços extras."""
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df


# ======================================================
# Imputar valores faltantes
# ======================================================
def impute_missing_values(df: pd.DataFrame, num_cols: list = None, cat_cols: list = None) -> pd.DataFrame:
    """
    Preenche valores faltantes:
    - Numéricos com a mediana
    - Categóricos com a moda
    """
    df = df.copy()

    # Numéricos → mediana
    if num_cols is not None and len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Categóricos → moda
    if cat_cols is not None and len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df
