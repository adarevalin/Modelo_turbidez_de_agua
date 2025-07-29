# 💧 Sistema de Predicción de Turbidez del Agua con KNN

#![Integrales proyectivas](ruta/a/la/imagen.png)

---

## 📌 Descripción

Este proyecto presenta un modelo basado en K-Nearest Neighbors (KNN) para estimar la **turbidez del agua** a partir de imágenes procesadas. Se enfoca en la predicción eficiente, económica y replicable en contextos reales, comparando su rendimiento con instrumentos de medición tradicionales.

> 📈 **Precisión del modelo: 94%**  
> ⏱️ **Tiempo de respuesta: 0.81 segundos**  
> 💰 **Costo estimado del sistema: $292,000 COP**  
> 🧪 **Comparado con turbidímetros reales (≈ $15,000,000 COP)**

---

## 🧠 Metodología

### 🔹 Datos e Imágenes
- Imágenes en escala de grises
- Características extraídas: **9,900**
- Reducción de características a: **413**
- División de datos:
  - 80% entrenamiento
  - 20% validación

### 🔹 Algoritmo
- Modelo: **KNN (K-Nearest Neighbors)**
- Métricas utilizadas:
  - MSE (Error Cuadrático Medio)
  - R² (Coeficiente de Determinación)

---

## 📊 Resultados

### 🧪 Entrenamiento
| Métrica | Resultado |
|--------|-----------|
| MSE    | 7.805     |
| R²     | 0.997     |

### 🧪 Validación
| Métrica | Resultado |
|--------|-----------|
| MSE    | 4.709     |
| R²     | 0.999     |

> 🔎 **RMSE Entrenamiento:** ~8  
> 🔎 **RMSE Validación:** ~5  
> 💡 El modelo generaliza bien sobre nuevos datos.

---

## 📚 Comparación con Estudios

📖 *Smartphone-based turbidity reader* – Koydemir et al. (2019)  
El sistema propuesto utiliza solo imágenes, evitando complejos métodos de caracterización. A diferencia del artículo, se mejora la replicabilidad y se mantiene una alta precisión con mayor simplicidad.

---

## 🛠️ Mejoras y Futuro

- 📱 **Desarrollo de App Móvil**: Para adquirir y procesar imágenes en campo.
- 💾 **Ampliación del Dataset**: Con más tipos de agua (ríos, lagunas, pantanos, etc.).
- 🌈 **Uso del Canal RGB**: Para enriquecer el análisis espectral y mejorar el reconocimiento.

---

## 🧪 Consideraciones Ambientales

La turbidez está influenciada por factores como:
- Tipo de suelo
- Vegetación
- Sedimentos geográficos

Por ello, es crucial **expandir el entrenamiento** del modelo para mejorar su adaptabilidad en nuevos entornos.

---

## 📎 Referencias Clave

- Torres Castellanos, A. R. (2021) – Universidad de Pamplona  
- Kitchener, B. G., et al. (2017) – Physical Geography  
- Ceylan Koydemir, H. (2019) – *Scientific Reports*  
- González-Salcedo, L., & Garcia-Nuñez, J. (2020) – *Informador Técnico*  
- Feizi, H., et al. (2021) – *Int. J. Env. Sci. and Tech*

[Ver todas las referencias completas...](#referencias)

---

## 🧩 Anexos

- Rango de predicción: **0 a 500 NTU**
- Modelo validado con condiciones reales
- Metodología basada en integrales proyectivas
- Implementación eficaz con bajo costo

---

## 📷 Imagen del Modelo

> A continuación, se muestra la imagen representativa con el uso de integrales proyectivas:

![Integrales Proyectivas](ruta/a/la/imagen.png)

---

## 🤝 Créditos

Desarrollado por: **Andrés David Arévalo Rosero**  
Universidad de Pamplona – Ingeniería Electrónica

---

## 🛠 Cómo usar el proyecto

```bash
# Clonar repositorio
git clone https://github.com/usuario/repositorio.git

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el modelo
python main.py
