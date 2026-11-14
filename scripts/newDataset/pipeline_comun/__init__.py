"""Código común de los dos notebooks de entrenamiento del dataset nuevo.

`PMC/notebooks/LosSimpsons_GPU_4060M.ipynb` (Perceptrón Multicapa) y `CNN/notebooks/LosSimpsons_GPU_4060M.ipynb`
(red convolucional) siguen el mismo proceso sobre los mismos datos y solo cambian el modelo. Todo lo que comparten
vive aquí, una sola vez, y los dos notebooks lo importan. Cada módulo es una etapa del pipeline:

| Módulo              | Etapa del pipeline                                                   |
|---------------------|----------------------------------------------------------------------|
| `pipeline`          | El diagrama y la tabla de las etapas                                 |
| `datos`             | Carga de datos, partición, escenas y análisis exploratorio           |
| `preprocesamiento`  | Redimensionado, normalización, etiquetas one-hot y pesos por clase   |
| `fundamentos`       | Gráficos de las funciones de activación, softmax y pérdida           |
| `entrenamiento`     | Bucle de entrenamiento, regularización L2, parada temprana, curvas   |
| `experimentacion`   | Comparación de configuraciones, búsqueda de hiperparámetros, impacto |
| `evaluacion`        | Métricas (con ROC-AUC), matriz de confusión, aciertos y errores      |
| `auditoria`         | ROC, calibración, umbral de revisión, sesgos por subgrupo y SHAP     |
| `rendimiento`       | Entorno de ejecución, tiempos y uso de la GPU                        |
| `prediccion`        | Clasificar una imagen cualquiera con el modelo final                 |

Los notebooks agregan `scripts/newDataset` a `sys.path` y luego importan, por ejemplo,
`from pipeline_comun.entrenamiento import entrenar`.
"""
