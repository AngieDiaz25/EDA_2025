# utils.py
"""
Archivo de utilidades para el EDA de Fast Fashion.
Incluye funciones para:
- Carga y limpieza de datos
- Análisis de hipótesis ambientales, económicas y de sostenibilidad
- Visualización y estadística
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# -----------------------------
# Carga y exploración de datos
# -----------------------------
def cargar_csv(ruta):
    """Carga un CSV y devuelve un DataFrame"""
    return pd.read_csv(ruta)

def info_general(df):
    """Muestra información general del DataFrame"""
    print("="*80)
    print("INFORMACIÓN GENERAL DEL DATASET")
    print("="*80)
    print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head())
    print("\nInformación de columnas:")
    print(df.info())
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nValores duplicados:", df.duplicated().sum())

# -----------------------------
# Limpieza de datos
# -----------------------------
def limpiar_nulos(df):
    """Elimina filas con valores nulos"""
    return df.dropna()

def eliminar_duplicados(df):
    """Elimina filas duplicadas"""
    return df.drop_duplicates()

# -----------------------------
# Funciones generales
# -----------------------------
def limpiar_nombres_columnas(df):
    """Convierte nombres de columnas a minúsculas y reemplaza espacios por guiones bajos"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def convertir_a_mayusculas(texto):
    """Convierte un texto a mayúsculas"""
    return str(texto).upper()

def normalizar_columna(df, columna):
    """Normaliza una columna numérica entre 0 y 1"""
    min_val = df[columna].min()
    max_val = df[columna].max()
    df[columna + "_norm"] = (df[columna] - min_val) / (max_val - min_val)
    return df

# -----------------------------
# Análisis Hipótesis 1: Ambiental
# -----------------------------
def top_paises_por_produccion(df, produccion_col='production_volume', pais_col='country', top=10):
    return df.groupby(pais_col)[produccion_col].sum().sort_values(ascending=False).head(top)

def top_paises_por_co2(df, co2_col='co2_emissions', pais_col='country', top=10):
    return df.groupby(pais_col)[co2_col].sum().sort_values(ascending=False).head(top)

def matriz_correlacion(df, columnas):
    corr_matrix = df[columnas].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación')
    plt.show()
    return corr_matrix

def test_correlacion_pearson(df, col1, col2):
    correlation, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
    return correlation, p_value

# -----------------------------
# Análisis Hipótesis 2: Económico-Social
# -----------------------------
def promedio_por_marca(df):
    return df.groupby('brand').agg({col:'mean' for col in df.select_dtypes(include=[np.number]).columns}).reset_index()

def categorizar_precio(df, price_col):
    df['Price_category'] = pd.qcut(df[price_col], q=3, labels=['Bajo', 'Medio', 'Alto'])
    return df

def comparacion_salario_por_categoria(df, wage_col):
    return df.groupby('Price_category')[wage_col].mean()

def test_anova_salarios(df, wage_col):
    price_groups = [df[df['Price_category']==cat][wage_col].dropna() for cat in ['Bajo','Medio','Alto']]
    f_stat, p_value = stats.f_oneway(*price_groups)
    return f_stat, p_value

# -----------------------------
# Análisis Hipótesis 3: Ética-Imagen
# -----------------------------
def categorizar_sostenibilidad(df, sustainability_col):
    df['Sustainability_category'] = pd.qcut(df[sustainability_col], q=3, labels=['Baja','Media','Alta'])
    return df

def percepcion_por_sostenibilidad(df, sentiment_col):
    return df.groupby('Sustainability_category')[sentiment_col].mean()

# -----------------------------
# Visualizaciones
# -----------------------------
def plot_histograma(df, columna, bins=10, color='skyblue'):
    plt.figure(figsize=(8,5))
    sns.histplot(df[columna], bins=bins, color=color, kde=True)
    plt.title(f'Histograma de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.show()

def plot_scatter(df, x_col, y_col, color='purple'):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=x_col, y=y_col, color=color)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_box(df, columna, categoria_col, color='lightgreen'):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=categoria_col, y=columna, data=df, palette=[color])
    plt.title(f'Boxplot de {columna} por {categoria_col}')
    plt.show()

def plot_correlation_matrix(df, columnas=None):
    if columnas:
        corr = df[columnas].corr()
    else:
        corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación')
    plt.show()
