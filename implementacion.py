import joblib
import statistics as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
#estcalamiento de los datos

model = joblib.load("model.joblib")

df = pd.read_csv("data_7.csv")
dff = pd.DataFrame()
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

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_prueba = model.predict(X)

print(y_prueba)
print('r2 Score_T: %.3f' % (r2_score(y_prueba,y)))
print('quadratic_T: %.3f' % np.sqrt((mean_squared_error(y_prueba,y))))


y_pred = model.predict(X)

# Calcular los residuos (diferencia entre las etiquetas reales y las predicciones)
residuos = y - y_pred

print('paso la resta')
# Graficar los residuos frente a las etiquetas reales
plt.scatter(y, residuos, color='black')
plt.axhline(y=0, color='blue', linestyle='--')
plt.title('Gráfica de Residuos: Bosques aleatorios')
plt.xlabel('Etiquetas reales')
plt.ylabel('Residuos')
plt.show()

# Graficar las etiquetas reales contra las etiquetas predichas
plt.scatter(y, y_pred, color='black')
plt.title('Gráfica de Dispersión: Bosques aleatorios')
plt.xlabel('Etiqueta real')
plt.ylabel('Etiqueta predicha')
plt.show()
