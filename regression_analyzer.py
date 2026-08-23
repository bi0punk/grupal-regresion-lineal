# Librerías para manipulación de datos
import numpy as np
import pandas as pd

# Modelos de regresión permitidos
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

# Herramientas de preprocesamiento y división de datos
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# Métricas de evaluación
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', None)

# Ignorar advertencias para mejor legibilidad
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('historia_climatica.csv')
print(f"Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
df.head()

print("### Tipos de datos ###")
print(df.dtypes)

print(df.describe().T)


#FALTANTES
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    print(f"\nColumna: {col}")
    print(df[col].unique())


#Extra
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlación entre Variables Numéricas")
plt.show()


# Su código aquí - Identificación de valores faltantes
# Identificación de valores faltantes por columna
df.isna().sum()
print(df.isna().sum())

print("\n")
#porcentaje
# Identificación de valores faltantes por columna
df.isna().sum()
print((df.isna().mean()*100).round(2))

#visualizacion de datos faltantes
plt.figure(figsize=(10,4))
sns.heatmap(df.isna(), cbar=False, cmap="Reds")
plt.title("Mapa de Valores Faltantes en el Dataset")
plt.show()


# Su código aquí - Estrategia de imputación
from sklearn.impute import KNNImputer

# Separar columnas numéricas y categóricas
num_cols = df.select_dtypes(include=['float64','int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Copias para no modificar el original
df_imputed = df.copy()

# Imputación de variables numéricas por mediana
for col in num_cols:
    df_imputed[col].fillna(df_imputed[col].median(), inplace=True)

# Imputación de variables categóricas por moda
for col in cat_cols:
    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)

df_imputed.isna().sum()


# Su código aquí - Visualización de distribuciones
# Visualización de la distribución de variables numéricas
import matplotlib.pyplot as plt
import seaborn as sns

num_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns

# Histogramas
df_imputed[num_cols].hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_imputed[num_cols])
plt.xticks(rotation=45)
plt.title("Boxplots de variables numéricas")
plt.show()


# Su código aquí - Detección de outliers
# Detección de outliers usando la regla de 1.5 * IQR
outlier_info = []

for col in num_cols:
    Q1 = df_imputed[col].quantile(0.25)
    Q3 = df_imputed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    n_outliers = ((df_imputed[col] < lower) | (df_imputed[col] > upper)).sum()
    outlier_info.append([col, lower, upper, n_outliers])

outliers_df = pd.DataFrame(outlier_info,
                           columns=['variable', 'lim_inf', 'lim_sup', 'n_outliers'])

outliers_df


# Su código aquí - Tratamiento de outliers
#Aquí no eliminamos filas, sino que “recortamos” los valores extremos al límite permitido por la regla del IQR
# Tratamiento de outliers: clipping a los límites [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
df_clean = df_imputed.copy()

for col in num_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

# Comprobación rápida
df_clean.describe().T


#outliers por variable
plt.figure(figsize=(8, 4))
sns.barplot(data=outliers_df, x='variable', y='n_outliers')
plt.xticks(rotation=45)
plt.title("Número de outliers por variable numérica")
plt.ylabel("Cantidad de outliers")
plt.show()


# Su código aquí - Análisis de variables categóricas
# Análisis de variables categóricas
# --------------------------------
# Identificamos las columnas categóricas del dataset
cat_cols = df_clean.select_dtypes(include='object').columns
print(cat_cols)

# Mostrar el número de categorías por variable categórica
for col in cat_cols:
    print(f"\nVariable categórica: {col}")
    print("Número de categorías:", df_clean[col].nunique())
    print("Categorías:", df_clean[col].unique()[:10], "...")  # Mostramos solo las primeras 5

# Su código aquí - Codificación de variables categóricas
# Codificación One-Hot Encoding
# --------------------------------
# Convertimos variables categóricas a variables dummy (0/1)
# drop_first=True evita la multicolinealidad (mantiene k-1 columnas)
#quitamos coñumna daily summary porque se hace pesado y se romope, aparte no aporta muchoo
# ==============================
# 1) Aseguramos tipo datetime para Fecha
# ==============================
df_clean['Fecha'] = pd.to_datetime(df_clean['Fecha'], errors='coerce')
# ==============================
# 2) Identificar variables categóricas
#    (ya NO incluye 'Fecha' porque ahora es datetime)
# ==============================
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
print("Categóricas originales:", cat_cols)

# ==============================
# 3) Quitamos 'Daily Summary' porque:
#    - Tiene muchísimas categorías
#    - Aporta poco al modelo
# ==============================
if "Daily Summary" in cat_cols:
    cat_cols.remove("Daily Summary")

print("Categóricas que codificaremos:", cat_cols)

# ==============================
# 4) One-Hot Encoding sobre Summary y Precip Type
#    drop_first=True evita multicolinealidad
# ==============================
df_encoded = pd.get_dummies(
    df_clean,
    columns=cat_cols,
    drop_first=True
)

df_encoded.head()


# Su código aquí - Extracción de atributos temporales
# Extracción de atributos temporales
# ----------------------------------
# Aseguramos que 'Fecha' sea tipo datetime
df_encoded['Fecha'] = pd.to_datetime(df_encoded['Fecha'], errors='coerce')

# Creación de nuevas columnas temporales
df_encoded['año'] = df_encoded['Fecha'].dt.year      # Año de la medición
df_encoded['mes'] = df_encoded['Fecha'].dt.month     # Mes (estacionalidad)
df_encoded['dia'] = df_encoded['Fecha'].dt.day       # Día del mes
df_encoded['hora'] = df_encoded['Fecha'].dt.hour     # Hora del día (ciclos diurnos)
df_encoded['dia_semana'] = df_encoded['Fecha'].dt.dayofweek  # 0 = Lunes ... 6 = Domingo

df_encoded.head()
# Eliminar definitivamente Daily Summary del dataset modelo
if "Daily Summary" in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=["Daily Summary"])
    print("Daily Summary eliminada de df_encoded")
else:
    print("Daily Summary ya no está en df_encoded")



# Su código aquí - Análisis de correlación y selección de variables
# ==========================================
# 4.3 Selección final de variables predictoras
# ==========================================

# 1) Definimos el nombre de la variable objetivo
target_col = 'Temperature (C)'

# 2) Obtenemos solo las columnas numéricas para el análisis de correlación
num_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns

# 3) Calculamos la matriz de correlación entre variables numéricas
corr_matrix = df_encoded[num_cols].corr()

# 4) Extraemos la correlación de cada variable con la temperatura (variable objetivo)
corr_with_target = corr_matrix[target_col].sort_values(ascending=False)

# 5) Mostramos las correlaciones ordenadas (de mayor a menor relación)
print("Correlación de cada variable numérica con la temperatura:\n")
print(corr_with_target)


# 6) Definimos el conjunto final de variables predictoras (features)
#    - Excluimos la variable objetivo
#    - Excluimos 'Fecha' porque ya extrajimos atributos temporales
#    - (Opcional) Excluimos variables con varianza cero como 'Loud Cover'

feature_cols = df_encoded.columns.drop([
    'Temperature (C)',  # variable objetivo
    'Fecha',            # variable temporal original
    'Loud Cover'        # sin variación (siempre 0), no aporta información
])

# 7) Creamos el dataset final que se usará para modelar
df_model = df_encoded[feature_cols.union([target_col])]

df_model.head()


# Su código aquí - Definir X (predictores) e y (variable objetivo)
# ==========================================
# 5.1 Definición de variables X (predictoras) e y (objetivo)
# ==========================================

# Variable objetivo (temperatura)
y = df_model[target_col]

# Variables predictoras (todas las columnas definidas en 'feature_cols')
X = df_model[feature_cols]

# Verificación de dimensiones
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


# Su código aquí - División train/test
# ==========================================
# 5.2 División en conjuntos de entrenamiento y prueba
# ==========================================

from sklearn.model_selection import train_test_split

# Dividimos los datos en train y test
# test_size=0.2 -> 20% de los datos se reserva para prueba
# random_state fija la aleatoriedad para obtener siempre la misma partición
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Tamaño X_train:", X_train.shape)
print("Tamaño X_test:", X_test.shape)
print("Tamaño y_train:", y_train.shape)
print("Tamaño y_test:", y_test.shape)
# Comprobamos si X_train tiene columnas de texto (object)
print("Columnas tipo object en X_train:")
print(X_train.select_dtypes(include='object').columns.tolist())


# Su código aquí - Estandarización/Normalización
# ==========================================
# 5.3 Estandarización / Normalización de variables
# ==========================================

from sklearn.preprocessing import StandardScaler

# Creamos el objeto escalador
scaler = StandardScaler()

# Ajustamos el escalador con los datos de entrenamiento
# y transformamos X_train
X_train_scaled = scaler.fit_transform(X_train)

# Usamos la misma transformación (sin re-ajustar) en el conjunto de prueba
X_test_scaled = scaler.transform(X_test)

# (Opcional) Convertimos nuevamente a DataFrame para legibilidad
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train_scaled.head()
# Comprobamos si X_train tiene columnas de texto (object)
print("Columnas tipo object en X_train:")
print(X_train.select_dtypes(include='object').columns.tolist())


# ==========================================
# Generar plantilla de entrada para el modelo
# Archivo: plantilla_input_modelo.json
# ==========================================

import json

# 1) Usamos las columnas que realmente usa el modelo (X_train)
columnas_modelo = list(X_train.columns)

# 2) Creamos un diccionario con todas las columnas inicializadas en 0
#    - Para numéricas: 0 es un valor neutro razonable.
#    - Para dummies: 0 significa "categoría no activa".
plantilla = {col: 0 for col in columnas_modelo}

# 3) Guardamos la plantilla en un archivo JSON
with open("plantilla_input_modelo.json", "w") as f:
    json.dump(plantilla, f, indent=4)

print("Plantilla guardada como 'plantilla_input_modelo.json'")


# Su código aquí - Entrenamiento del modelo
# ==========================================
# 6.2 Entrenamiento del modelo - Parte 1
# Modelo base: LinearRegression
# ==========================================

from sklearn.linear_model import LinearRegression

# Creamos el modelo lineal
# Lo usamos como baseline porque:
# - Es sencillo de interpretar.
# - Nos permite ver la relación lineal entre variables y temperatura.
lin_model = LinearRegression()

# Entrenamos el modelo con los datos escalados de entrenamiento
lin_model.fit(X_train_scaled, y_train)

# Comprobamos que el modelo quedó entrenado
print("Modelo LinearRegression entrenado correctamente.")



from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
pred_ridge = ridge_model.predict(X_test_scaled)


print("Modelo Ridge Regression entrenado correctamente")

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.001)
lasso_model.fit(X_train_scaled, y_train)
pred_lasso = lasso_model.predict(X_test_scaled)


from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(alpha=0.001, l1_ratio=0.5)
elastic_model.fit(X_train_scaled, y_train)
pred_elastic = elastic_model.predict(X_test_scaled)


# ==========================================
# 6.2 Entrenamiento del modelo - Parte 2
# Modelo no lineal: DecisionTreeRegressor
# ==========================================

from sklearn.tree import DecisionTreeRegressor

# Creamos el árbol de decisión
# Lo usamos como comparación porque:
# - Puede capturar relaciones no lineales.
# - Considera interacciones entre variables (por ejemplo, humedad + hora).
# max_depth se fija para evitar sobreajuste (controla la complejidad del árbol).
tree_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

# Entrenamos el árbol con los mismos datos escalados de entrenamiento
tree_model.fit(X_train_scaled, y_train)

print("Modelo DecisionTreeRegressor entrenado correctamente.")


# ==========================================
# 6.3 Evaluación del Modelo
# Predicciones y métricas para varios modelos
# ==========================================

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd

# ------------------------------
# 1) Predicciones en el conjunto de prueba
# ------------------------------
y_pred_lin    = lin_model.predict(X_test_scaled)       # modelo lineal básico
y_pred_ridge  = ridge_model.predict(X_test_scaled)     # Ridge
y_pred_lasso  = lasso_model.predict(X_test_scaled)     # Lasso
y_pred_elastic = elastic_model.predict(X_test_scaled)  # ElasticNet
y_pred_tree   = tree_model.predict(X_test_scaled)      # árbol de decisión

# ------------------------------
# 2) Función para calcular métricas
#    Métricas: MAE, R² y RMSE
# ------------------------------
def compute_metrics(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'Modelo': model_name,
        'MAE (°C)': mae,
        'R²': r2,
        'RMSE (°C)': rmse
    }

# Calculamos métricas para cada modelo
metrics_lin     = compute_metrics(y_test, y_pred_lin,     'LinearRegression')
metrics_ridge   = compute_metrics(y_test, y_pred_ridge,   'Ridge')
metrics_lasso   = compute_metrics(y_test, y_pred_lasso,   'Lasso')
metrics_elastic = compute_metrics(y_test, y_pred_elastic, 'ElasticNet')
metrics_tree    = compute_metrics(y_test, y_pred_tree,    'DecisionTreeRegressor')

# ------------------------------
# 3) Tabla comparativa de métricas
# ------------------------------
results_df = pd.DataFrame([
    metrics_lin,
    metrics_ridge,
    metrics_lasso,
    metrics_elastic,
    metrics_tree
])

results_df



# Su código aquí - Visualizaciones
# Suponemos que usaremos el modelo lineal como referencia
# ==========================================
# 6.5 Visualización de Resultados
# 1) Gráfico de valores reales vs predichos
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))

# Dibujamos los puntos reales vs predichos
plt.scatter(y_test, y_pred_lin, alpha=0.3, color='blue', label='Predicciones')

# Línea de referencia (donde real = predicho)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Línea ideal')

plt.xlabel("Temperatura real (°C)")
plt.ylabel("Temperatura predicha (°C)")
plt.title("Modelo LinearRegression - Temperatura real vs. predicha")
plt.legend()
plt.grid(True)
plt.show()
##Si los puntos siguen de cerca la línea roja, el modelo predice correctamente.
#Dispersión grande = modelo con errores altos


# ==========================================
# 6.5 Visualización de Resultados
# Gráfico real vs predicho para cada modelo
# ==========================================

import matplotlib.pyplot as plt

# Diccionario con predicciones y nombres
model_predictions = {
    "LinearRegression": y_pred_lin,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "DecisionTreeRegressor": y_pred_tree
}

# Función para graficar cada modelo
def plot_real_vs_pred(y_true, y_pred, model_name):
    plt.figure(figsize=(7, 7))

    # Dispersión real vs predicho
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue', label='Predicciones')

    # Línea ideal donde real = predicho
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', linewidth=2, label='Línea ideal')

    plt.xlabel("Temperatura real (°C)")
    plt.ylabel("Temperatura predicha (°C)")
    plt.title(f"{model_name} - Temperatura real vs. predicha")
    plt.legend()
    plt.grid(True)
    plt.show()

# Generar un gráfico por modelo
for model_name, y_pred in model_predictions.items():
    plot_real_vs_pred(y_test, y_pred, model_name)


plt.figure(figsize=(8, 8))

plt.scatter(y_test, y_pred_lin, alpha=0.3, label='LinearRegression', color='blue')
plt.scatter(y_test, y_pred_tree, alpha=0.3, label='DecisionTreeRegressor', color='green')

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Línea ideal')

plt.xlabel("Temperatura real (°C)")
plt.ylabel("Temperatura predicha (°C)")
plt.title("Comparación: LinearRegression vs DecisionTreeRegressor")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 2) Distribución de residuos
# ==========================================

# Los residuos son: real - predicho
residuals = y_test - y_pred_lin

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=40, color='purple')

plt.title("Distribución de residuos del modelo LinearRegression")
plt.xlabel("Residuo (°C)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()
#Los residuos deberían aproximarse a una distribución normal centrada en 0.
#Si hay asimetría o colas largas, el modelo tiene sesgo o errores altos en ciertos rangos.


# ==========================================
# 3) Importancia de variables del modelo DecisionTreeRegressor
# ==========================================

import numpy as np

# Obtenemos la importancia de cada variable
importances = tree_model.feature_importances_

# Las asociamos a los nombres de columnas
importance_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

# Mostramos las primeras más importantes
importance_df.head(15)


# Visualización
plt.figure(figsize=(8, 8))
sns.barplot(data=importance_df.head(15),
            x='Importancia', y='Variable', palette='viridis')
plt.title("Importancia de las 15 variables más relevantes (Árbol de decisión)")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.show()


# ==========================================
# Guardar modelo y scaler en archivos
# ==========================================

import joblib

# Guardamos el modelo lineal
joblib.dump(lin_model, "modelo_temperatura.pkl")

# Guardamos el scaler
joblib.dump(scaler, "scaler_temperatura.pkl")

print("Modelo y scaler guardados correctamente.")


