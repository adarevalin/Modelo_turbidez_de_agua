import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statistics as st
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data_7.csv")
img_dir = df.iloc[:,0]
eti = df.iloc[:,1]
y = eti.tolist() # etiquetas 

promedio = []

count = 0; aux = 0
#NOTA: error medio cuadratico con la imagen original, promedio img original, moda con la imagen en gris
for i in img_dir:
    img = cv2.imread(i) ## lectura de las imagenes
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_norm = cv2.normalize(img_gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_array = np.array(image_norm)
    
    # obtener la integral proyectiva
    prom_col = [] # Crear una nueva lista para cada imagen
    suma = 0; aux = 0; aux_1 = 0
    for i in range(len(img_array[0])):
        aux_1 = np.mean(img_array[:,i]) # promedio por columnas 
        prom_col.append(aux_1)

    for j in range(len(img_array)): # columnas 392
        aux = np.mean(img_array[j,:]) # promedio por columnas 
        prom_col.append(aux)

    promedio.append(prom_col)

    count = count + 1
    print(count)

X = np.array(promedio)
print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('paso la split')
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
print('paso la entre')
# Realizar predicciones en los datos de entrenamiento
y_pred = regressor.predict(X_train)

# Calcular los residuos (diferencia entre las etiquetas reales y las predicciones)
residuos = y_train - y_pred

print('paso la resta')
# Graficar los residuos frente a las etiquetas reales
plt.scatter(y_train, residuos, color='black')
plt.axhline(y=0, color='blue', linestyle='--')
plt.title('Gráfica de Residuos: Bosques aleatorios')
plt.xlabel('Etiquetas reales')
plt.ylabel('Residuos')
plt.show()



# Graficar las etiquetas reales contra las etiquetas predichas
plt.scatter(y_train, y_pred, color='black')
plt.title('Gráfica de Dispersión: Bosques aleatorios')
plt.xlabel('Etiqueta real')
plt.ylabel('Etiqueta predicha')
plt.show()



'''
###################### Metricas de validacion #################################################
print('pasamos el entrenamiento')
resultado = regressor.score(X_test, y_test)
print("Precision de los datos de testeo con Score - porcentaje: ",resultado) # con esto vamos a ver cual es la precisión de nuestro modelo en porcentaje.

#Calculando el MSE del entrenamiento y del test.
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_quad_train = regressor.predict(X_train) # prediccion con los datos escalados de entrenamiento
y_quad_test = regressor.predict(X_test)

#validacion de modelo con los datos de entrenamiento
print('quadratic_T: %.3f' % np.sqrt((mean_squared_error(y_train,y_quad_train))))
print('r2 Score_T: %.3f' % (r2_score(y_train,y_quad_train)))
#validacion de modelo con los datos de testeo
print('quadratic: %.3f' % np.sqrt((mean_squared_error(y_test,y_quad_test))))
print('r2 Score: %.3f' % (r2_score(y_test,y_quad_test)))


# Realizar validación cruzada con 5 folds y calcular el R2 y MSE en cada fold
cv_scores = cross_val_score(regressor, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
mse_scores = -cv_scores  # Convertir los puntajes de negativo a positivo
r2_scores = cross_val_score(regressor, X_train, y_train, cv=10, scoring='r2')

# Imprimir los puntajes promedio de validación cruzada
print("Puntajes MSE en Validación Cruzada:", mse_scores)
print("Puntajes R2 en Validación Cruzada:", r2_scores)

# Imprimir los puntajes promedio de validación cruzada
print("Promedio MSE en Validación Cruzada:", np.mean(mse_scores))
print("Promedio R2 en Validación Cruzada:", np.mean(r2_scores))




from sklearn.base import BaseEstimator
from datetime import datetime, timezone
import os
import joblib

def _save_versioned_estimator(estimator: BaseEstimator, output_dir: str):
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M")
    model_dir = os.path.join(output_dir, version)
    os.makedirs(model_dir, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir, "model.joblib"))
    except Exception as e:
        print("Ha ocurrido una excepción al guardar el estimador:", e)

_save_versioned_estimator(regressor, "MiModelo") 
'''