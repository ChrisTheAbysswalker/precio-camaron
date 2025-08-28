import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configurar la semilla aleatoria para reproducibilidad (opcional)
np.random.seed(42)

# Cargar el archivo CSV
file_path = 'precio_camaron.csv'  # Cambia esto si tienes el archivo en otra ruta
df = pd.read_csv(file_path)

# Mostrar las primeras filas para verificar que los datos estén bien cargados
print(df.head())

# Asegurarse de que las columnas se llamen correctamente según tu archivo
df.rename(columns={'month': 'Month', 'price': 'Price', 'change': 'Change'}, inplace=True)

# Verificar el tipo de dato en las columnas
print(df.info())

# Convertir la columna 'Month' a datetime para extraer el mes
df['Month'] = pd.to_datetime(df['Month'], format='%m/%d/%Y').dt.month

# Verificar la transformación de la columna 'Month'
print(df.head())
print(df.info())

# Separar variables independientes (X) y dependiente (y)
X = df[['Month', 'Change']]  # Mes y Cambio serán las variables independientes
y = df['Price']  # Precio es la variable dependiente

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear características polinómicas de grado 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Crear y entrenar el modelo de regresión lineal con las características polinómicas
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Realizar predicciones
y_pred = model.predict(X_test_poly)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse:.2f}")

# Graficar la relación entre el "Month" y el "Price"
plt.figure(figsize=(10, 6))

# Ordenar los valores de prueba para una mejor visualización
sort_idx = np.argsort(X_test['Month'])
X_test_sorted = X_test['Month'].values[sort_idx]
y_test_sorted = y_test.values[sort_idx]
y_pred_sorted = y_pred[sort_idx]

# Graficar los puntos de datos reales
plt.scatter(X_test_sorted, y_test_sorted, color='blue', label='Valores reales')

# Graficar la línea de regresión polinómica
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Línea de regresión polinómica')

plt.xlabel('Mes')
plt.ylabel('Precio del Camarón')
plt.legend()
plt.title('Regresión Polinómica: Relación entre Mes y Precio del Camarón')
plt.show()

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones', alpha=0.6)
plt.xlabel('Índice')
plt.ylabel('Precio del Camarón')
plt.legend()
plt.title('Comparación entre valores reales y predicciones')
plt.show()
