import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_basic_info(df):
    """Mostra informações gerais e valores nulos"""
    print("Dimensões do dataset:", df.shape)
    print("\nValores nulos por coluna:\n", df.isnull().sum())
    print("\nTipos de dados:\n", df.dtypes)

def plot_top_graphs(df, target='delivery_time_hours'):
    """Gera apenas 4 gráficos principais e bem escolhidos"""

    plt.figure(figsize=(12, 4))
    sns.histplot(df[target], kde=True)
    plt.title("Distribuição do Tempo de Entrega (Target)")
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.boxplot(x=df['distance_km'])
    plt.title("Boxplot da Distância")
    plt.show()

    plt.figure(figsize=(12, 4))
    sns.scatterplot(x='distance_km', y=target, data=df)
    plt.title("Relação entre Distância e Tempo de Entrega")
    plt.show()

    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Mapa de Correlação")
    plt.show()
