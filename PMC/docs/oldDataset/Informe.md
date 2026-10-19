# Informe técnico — Clasificación de personajes de Los Simpson con MLP

## 1. Identificación del proyecto

| Elemento | Descripción verificable |
|---|---|
| Tipo de proyecto | Clasificación supervisada de imágenes RGB mediante un Perceptrón Multicapa (MLP). |
| Unidad de predicción | Una imagen recibe una sola etiqueta entre 25 personajes. |
| Implementaciones | `PMC/notebooks/oldDataset/LosSimpsonsPMC.ipynb` (CPU/Windows, TensorFlow) y `PMC/notebooks/oldDataset/LosSimpsons_GPU_4060M.ipynb` (GPU/Windows nativo, Keras 3 sobre PyTorch). |
| Evidencia de ejecución | Ambos notebooks conservan sus salidas; los resúmenes de rendimiento están en `PMC/resultados/oldDataset/CPU/` y `PMC/resultados/oldDataset/GPU/`. |
| Integrantes | Vincent Farenden Cerón; Rodrigo Ignacio Martínez Becker; Iván Manuel Pavón Barría; Diego Ignacio Peña y Lillo Luhrs. |

Este informe describe el estado que realmente está implementado. Las métricas, configuraciones y conclusiones se separan por entorno cuando las corridas no produjeron el mismo modelo final.

## 2. Problema, contexto y alcance

### 2.1 Problema abordado

El problema es de **clasificación multiclase de etiqueta única**: dada una imagen RGB que contiene un personaje de Los Simpson, el modelo debe asignarle una de las 25 clases disponibles. No es detección de objetos: no localiza personajes, no dibuja cajas delimitadoras y no admite múltiples etiquetas por imagen.

### 2.2 Problema de negocio

Un archivo audiovisual crece más rápido de lo que un equipo humano puede catalogarlo. Miles de imágenes sin etiquetar son material prácticamente inservible: no se pueden buscar, no se pueden reutilizar y no permiten medir nada. Hoy esa catalogación se hace a mano, imagen por imagen, y su costo crece de forma lineal con el tamaño del archivo.

> **Problema de negocio.** El archivo de imágenes crece más rápido de lo que el equipo puede catalogarlo manualmente, lo que impide buscar material por personaje, encarece cada proyecto de reutilización de contenido y deja sin explotar el catálogo acumulado.

El proyecto aborda ese problema sobre un archivo de unas 20.000 imágenes: asignar automáticamente el personaje que aparece en cada una, para que el archivo pase de ser un conjunto de ficheros sueltos a un catálogo consultable.

| Caso de uso | Qué habilita el modelo |
|---|---|
| Búsqueda en el archivo | Recuperar todos los planos de un personaje para un montaje, un tráiler o un recopilatorio temático. |
| Pre-clasificación asistida | El modelo propone y una persona valida: convierte trabajo de *etiquetar* en trabajo de *revisar*, mucho más rápido. |
| Métricas de contenido | Cuantificar la presencia en pantalla de cada personaje, dato relevante para licenciamiento y merchandising. |
| Control de uso de marca | Detectar personajes con derechos dentro de material aportado por terceros. |

Este escenario es el **marco de uso declarado** del proyecto, no un encargo de un cliente real: no hay contrato, operación comercial ni cifras económicas que permitan calcular un retorno monetario. Por eso los KPIs de la sección 3 se expresan en cantidades medibles sobre los datos y no en unidades de dinero.

### 2.3 Objetivo general

Implementar y evaluar un MLP para clasificar imágenes de personajes en 25 clases, usando un flujo reproducible de preprocesamiento, validación separada y análisis de desempeño sensible al desbalance de clases.

### 2.4 Objetivos específicos

1. Indexar las imágenes de `train` y asociar cada ruta con su carpeta de personaje.
2. Convertir cada imagen a RGB, redimensionarla, normalizar sus píxeles, aplanarla y codificar su etiqueta en one-hot.
3. Separar entrenamiento y validación de manera estratificada, preservando el conjunto opuesto para la prueba final.
4. Entrenar un MLP base y comparar cinco cambios controlados de arquitectura o hiperparámetros.
5. Seleccionar el modelo por F1 macro de validación y medirlo una vez sobre el conjunto de prueba.
6. Interpretar accuracy, precision, recall, F1, top-3 accuracy, matriz de confusión, aciertos, errores y limitaciones del MLP para imágenes.

### 2.5 Alcance de la variante

La variante efectivamente ejecutada se determina por las **25 carpetas de clase presentes tanto en `train` como en `test`**. Los notebooks no documentan el criterio histórico con que se construyó ese subconjunto ni permiten afirmar que se eligieron manualmente desde un universo mayor. Por ello, el alcance se declara como el que está respaldado por las rutas y salidas de ejecución: 25 clases, 16.137 imágenes en `train` y 3.937 en `test`.

## 3. Indicadores de desempeño (KPIs técnicos)

Los KPIs traducen el escenario de uso de la sección 2.2 a cantidades medibles sobre los datos. Sus umbrales se fijan **antes** de evaluar el modelo y su cumplimiento se verifica al cerrar la sección de evaluación de cada notebook, con las cifras de la prueba final.

| KPI | Qué mide en el negocio | Métrica técnica | Umbral |
|---|---|---|---|
| Cobertura de asistencia | Con qué frecuencia el catalogador encuentra la respuesta correcta entre las tres sugerencias del sistema. | Top-3 accuracy en test | ≥ 0,70 |
| Equidad entre personajes | Que el catálogo sirva para todos los personajes y no solo para los más frecuentes. | F1 macro en test | ≥ 0,45 |
| Automatización sin revisión | Proporción de imágenes que el sistema resuelve sin intervención humana. | Accuracy en test | ≥ 0,50 |
| Ventaja sobre el criterio trivial | Que el sistema aporte información real frente a etiquetar todo como el personaje más común. | Accuracy / baseline mayoritaria | ≥ 4× |
| Costo de catalogación | Capacidad de procesar el archivo completo en un tiempo razonable. | Imágenes por segundo en inferencia | ≥ 1.000 img/s |

La elección de **top-3 accuracy** y **F1 macro** como KPIs principales responde al escenario: el sistema es un asistente del catalogador, no su reemplazo, y un catálogo que solo reconoce a los personajes mayoritarios no resuelve el problema de búsqueda. Por eso la selección de modelo se hace por F1 macro y no por accuracy.

Los indicadores técnicos que sustentan esos KPIs son los siguientes:

| Indicador | Qué mide | Uso en el proyecto |
|---|---|---|
| Accuracy | Proporción global de predicciones correctas. | Referencia general, pero puede favorecer a las clases frecuentes. |
| Precision, recall y F1 por clase | Calidad de cada personaje individual. | Identifica qué clases son reconocidas y cuáles quedan rezagadas. |
| F1 macro | Promedio de F1 sin ponderar por tamaño de clase. | Criterio de selección por validación; trata las 25 clases por igual. |
| F1 ponderado | Promedio de F1 ponderado por soporte. | Complementa la lectura global del desempeño. |
| Top-3 accuracy | Casos donde la clase real queda entre las tres probabilidades más altas. | Mide si el modelo conserva la respuesta correcta entre sus candidatas. |
| Matriz de confusión normalizada por fila | Proporción de cada clase real que se confunde con otra. | Explica errores sistemáticos y evita que el tamaño de clase oculte el patrón. |

Las referencias mínimas registradas son: un clasificador aleatorio alcanza `1/25 = 0,0400` de accuracy y predecir siempre la clase mayoritaria alcanza `0,1143` en test. Estas referencias sirven para interpretar si el MLP aprende señal por encima de alternativas triviales; no se presentan como metas fijadas a posteriori.

## 4. Fuente y comprensión de los datos

### 4.1 Fuente

La fuente declarada es [Los Simpson Dataset de Kaggle](https://www.kaggle.com/datasets/alfaro96/los-simpson/data), usuario `alfaro96`. El dataset reúne aproximadamente 20.000 imágenes y ocupa 550 MB (577.675.264 bytes); no está incluido en el repositorio por ese peso.

La estructura requerida por los notebooks es:

```text
LosSimpsonsDataset/
├── train/
│   └── <personaje>/*.jpg
└── test/
    └── <personaje>/*.jpg
```

Cada carpeta de personaje funciona como etiqueta. La primera celda de cada notebook comprueba que ambas carpetas existan antes de procesar imágenes.

### 4.2 Exploración de datos (EDA)

La exploración implementada en los notebooks construye un índice de rutas, personaje y etiqueta; muestra el conteo y porcentaje por clase; inspecciona dimensiones originales; visualiza ejemplos; y compara imágenes originales con su versión redimensionada.

| Hallazgo registrado | Evidencia |
|---|---|
| Conjunto de trabajo | 16.137 imágenes en `train`, 25 clases. |
| Conjunto final | 3.937 imágenes en `test`, con las mismas 25 clases. |
| Desbalance | `homer_simpson` tiene 1.796 imágenes de entrenamiento y `selma_bouvier`, 82; razón aproximada 22:1. |
| Dimensiones originales | Variables; el notebook documenta un rango aproximado entre 256×257 y 720×464 px. |
| Implicancia | Accuracy por sí sola puede ocultar el bajo rendimiento en personajes minoritarios. |

### 4.3 Calidad y supuestos de los datos

El flujo valida que haya imágenes y que las clases de evaluación coincidan con las de entrenamiento. El preprocesamiento también comprueba forma, tipo de datos, rango normalizado, aplanado reversible y alineación entre `X` e `y`.

Persisten supuestos que no se verifican automáticamente:

- La carpeta de cada imagen representa el personaje relevante.
- No existen duplicados entre `train` y `test`.
- Cada imagen admite una sola etiqueta, incluso cuando pueda contener más de un personaje.
- El preprocesamiento no corrige sesgos de fondo, encuadre, pose o representación desigual entre clases.

## 5. Preparación y transformación de datos

La preparación se ejecuta en este orden:

1. Se cargan rutas de extensiones `.jpg`, `.jpeg`, `.png` y `.bmp` desde la carpeta de cada clase.
2. Se convierten las imágenes a RGB y se redimensionan a **128×128 px** con interpolación bilineal.
3. Se convierten a `float32` y se normalizan desde `[0, 255]` a `[0, 1]`.
4. Se aplanan: cada imagen queda como un vector de **49.152 valores** (`128 × 128 × 3`).
5. Las etiquetas se transforman de nombre de carpeta a índice entero y luego a one-hot de 25 posiciones.
6. `train` se divide mediante `train_test_split` estratificado, con semilla `42`: 12.909 imágenes de entrenamiento y 3.228 de validación.
7. El conjunto opuesto (`test` cuando se entrena con `train`) queda reservado para la evaluación final.
8. Se calculan `class_weight` solo desde `y_train`; esta decisión evita que las clases abundantes dominen la pérdida.

La resolución 128×128 es la configuración que aparece en el código y en las salidas guardadas de ambos notebooks. No se presenta 64×64 como la resolución usada para las métricas de este informe.

## 6. Metodología CRISP-DM aplicada

| Fase CRISP-DM | Aplicación verificable en este proyecto |
|---|---|
| Comprensión del problema | Se define clasificación multiclase de etiqueta única, sus límites frente a detección y las métricas apropiadas para clases desbalanceadas. |
| Comprensión de datos | Se inspeccionan conteos por clase, dimensiones originales, ejemplos visuales, desbalance y referencias simples. |
| Preparación de datos | RGB, resize 128×128, normalización, flatten, one-hot, split estratificado y pesos por clase. |
| Modelamiento | MLP base con capas densas, Batch Normalization y Dropout; comparación de E1 a E5. |
| Evaluación | Curvas de entrenamiento/validación, métricas globales y por clase, matriz de confusión, errores, saliencia y comparación con baselines. |
| Despliegue / comunicación | Función de predicción para una imagen, modelos `.keras` regenerables, resultados JSON y este informe. No hay despliegue productivo documentado. |

## 7. Diseño e implementación del modelo

### 7.1 Modelo base

| Componente | Configuración implementada |
|---|---|
| Entrada | Vector de 49.152 características proveniente de una imagen RGB 128×128. |
| Capas ocultas base | Dense `512 → 256`. |
| Activación oculta | ReLU. |
| Regularización | Batch Normalization y Dropout `0,3` después de cada capa oculta. |
| Salida | Dense de 25 neuronas con `softmax`. |
| Pérdida | `categorical_crossentropy`. |
| Optimizador | Adam, tasa de aprendizaje `1e-3`. |
| Batch | 128. |
| Máximo de épocas | 60, controladas mediante parada temprana. |
| Tratamiento del desbalance | `class_weight` balanceado, calculado solo con entrenamiento. |

La primera capa densa de 512 neuronas concentra aproximadamente 25,2 millones de pesos debido a la entrada de 49.152 valores. Esto explica el costo computacional y el riesgo de sobreajuste de un MLP aplicado a imágenes aplanadas.

### 7.2 Entrenamiento y validación

El entrenamiento registra `loss`, `val_loss`, `accuracy`, `val_accuracy` y `top_3`. Se aplican tres mecanismos de control:

| Callback | Configuración / función |
|---|---|
| `EarlyStopping` | Monitorea `val_loss`, paciencia 12 y restaura los mejores pesos. |
| `ReduceLROnPlateau` | Reduce a la mitad el learning rate después de 4 épocas sin mejora de `val_loss`, mínimo `1e-5`. |
| `ModelCheckpoint` | Guarda la mejor época según `val_loss`. |

En el notebook de GPU estos tres mecanismos no son *callbacks* de Keras sino parte de una función `entrenar()` escrita sobre PyTorch, con el mismo comportamiento y la misma interfaz de salida. El motivo es de rendimiento y se documenta en §8.1: `model.fit()` devuelve el control al intérprete de Python entre paso y paso y deja la GPU esperando.

La paciencia es 12 y no 8 porque la `val_loss` de esta red es ruidosa —25 clases desbalanceadas sobre 3.228 imágenes de validación— y una época afortunada puede fijar un mínimo que tarda en repetirse, agotando la paciencia. Medido sobre tres barajados distintos, la mejor `val_loss` alcanzada mejora al pasar de 8 a 12 y no mejora más al subir a 16. El cambio es seguro por construcción: como se restauran siempre los pesos de la mejor época, esperar más nunca devuelve un modelo peor, solo consume más épocas.

### 7.3 Experimentación controlada

Cada experimento cambia una variable respecto del modelo base, conservando semilla, partición, pesos por clase y callbacks:

| Experimento | Cambio |
|---|---|
| Base | Dos capas: `512 → 256`. |
| E1 | Una capa de 512 neuronas. |
| E2 | Tres capas: `1024 → 512 → 256`. |
| E3 | Activación `tanh`. |
| E4 | Learning rate `1e-4`. |
| E5 | Batch de 512. |

El criterio de selección es F1 macro de validación. En caso de un empate práctico, el notebook documenta preferir el modelo con menos parámetros.

## 8. Resultados de las corridas guardadas

Las dos ejecuciones usan la misma resolución, la misma división de datos y seleccionaron la misma arquitectura final (E2), de modo que las métricas de calidad son comparables entre sí. No lo son los tiempos: cada una corre sobre un framework distinto (§8.1).

| Métrica | CPU / Windows | GPU / Windows nativo |
|---|---:|---:|
| Configuración seleccionada | E2: `1024 → 512 → 256` | E2: `1024 → 512 → 256` |
| Parámetros del modelo final | 51.002.393 | 51.002.393 |
| Accuracy de test | 0,5286 | 0,5166 |
| Top-3 accuracy de test | 0,7318 | 0,7181 |
| F1 macro de test | 0,5092 | 0,4885 |
| F1 ponderado de test | 0,5331 | 0,5150 |
| Pérdida de test | 1,7217 | 1,8089 |

La ejecución CPU superó la referencia aleatoria por un factor aproximado de 13,2 y la de clase mayoritaria por 4,6. La ejecución GPU las superó aproximadamente por 12,9 y 4,5, respectivamente. Ambas quedan en la misma banda de calidad, lo esperable cuando solo cambia el dispositivo. Esto muestra que el modelo aprende patrones por encima de las referencias simples, pero no garantiza rendimiento homogéneo para todos los personajes.

### 8.1 Rendimiento computacional registrado

Los resultados JSON indican que, para 128×128:

| Medición | CPU | GPU |
|---|---:|---:|
| Duración total de la corrida | 44:40 | 4:48 |
| Segundos por época (medio) | 15,93 | 0,92 |
| Benchmark segundos por época | 16,49 | 0,89 |
| Épocas totales entrenadas | 147 | 226 |
| Inferencia sobre 3.937 imágenes | 1,41 s | 0,06 s |
| Imágenes por segundo en inferencia | 2.799,4 | 70.862,5 |

**Advertencia sobre esta comparación.** Las dos corridas no usan el mismo framework: CPU
sobre TensorFlow 2.21 y GPU sobre Keras 3 con backend PyTorch. La diferencia de tiempos
mezcla, por tanto, hardware y framework, y no debe leerse como una medida limpia de la
ganancia del hardware. La celda de rendimiento del notebook lo advierte de forma explícita
cuando detecta frameworks distintos. Para una comparación de igual a igual habría que
volver a ejecutar el notebook de CPU con el mismo bucle de entrenamiento.

### 8.2 Sobre el aprovechamiento de la GPU

La medición del uso real de la tarjeta dejó tres observaciones que conviene registrar,
porque afectan a cómo deben leerse todos los tiempos anteriores.

**El porcentaje de uso de GPU no mide potencia entregada, sino ocupación.** Con el equipo
desenchufado, el firmware limita la GPU a 50 W de los 80 W disponibles y el reloj baja de
2.340 a 1.755 MHz. La tarjeta marca 100 % de ocupación y aun así rinde la mitad: el tiempo
por época se duplica. Todas las cifras de este informe se tomaron con el equipo conectado a
la corriente.

**El cuello de botella era `model.fit()`, no el hardware.** `fit()` devuelve el control al
intérprete de Python entre paso y paso. Medido sobre la misma red y los mismos datos, con el
mismo tamaño de lote, sustituirlo por un bucle de entrenamiento escrito a mano sobre PyTorch
hace el entrenamiento **2,3 veces más rápido**: 0,81 frente a 1,89 segundos por época.

La utilización de la GPU, sin embargo, apenas cambia con ese reemplazo: 62 % frente a 65 %.
La ganancia no viene de mantener la tarjeta más ocupada, sino de desperdiciar menos tiempo en
cada paso. Es un buen recordatorio de que el porcentaje de uso, por sí solo, no permite
concluir si una implementación aprovecha el hardware.

**El tamaño de lote es la palanca que queda.** A batch 128 el trabajo de cada paso es
demasiado pequeño para saturar la tarjeta: la corrida a 128×128 registra un 59,7 % de
utilización media, y a 32×32 solo un 29,6 %, porque la entrada más pequeña deja aún menos
trabajo por paso. Manteniendo todo lo demás igual:

| Tamaño de lote | Imágenes por segundo | Utilización GPU | Potencia |
|---:|---:|---:|---:|
| 128 | 15.015 | 61,9 % | 53,8 W |
| 256 | 27.671 | 69,7 % | 60,1 W |
| 512 | 48.775 | 88,0 % | 72,7 W |
| 1.024 | 63.048 | 95,3 % | 79,8 W |
| 2.048 | 70.444 | 96,6 % | 79,0 W |

Multiplicar el lote por dieciséis multiplica el rendimiento por 4,7 y lleva la utilización
del 61,9 % al 96,6 %; a partir de batch 1.024 la tarjeta alcanza su límite de potencia de
80 W, señal de que ya no queda sobrecarga que recuperar. El estudio se mantiene en **batch
128** porque es el valor con el que se comparan las corridas de CPU y GPU, no porque sea el
más rápido. La tabla se mide y se guarda en el JSON de resultados para poder citarla sin
alterar el diseño experimental.

Una advertencia metodológica sobre estas cifras de utilización: se promedian sobre una
ventana de al menos cuatro segundos, muestreando `nvidia-smi` cada 50 ms. Una versión
anterior de la medición muestreaba cada 250 ms sobre épocas que duran décimas de segundo, lo
que dejaba dos o tres lecturas por medida y producía valores sin sentido, como un 20 % a
batch 1.024 entre un 85 % a batch 512 y un 96 % a batch 2.048.

## 9. Análisis de resultados y errores

### 9.1 Interpretación de métricas

La diferencia entre accuracy y F1 macro evidencia el efecto del desbalance. Accuracy pondera más a las clases con muchas imágenes; F1 macro penaliza que las clases minoritarias queden sin reconocer. Por ello, F1 macro se usa para seleccionar configuraciones y no solo accuracy.

Los notebooks incluyen:

- Reporte de precision, recall y F1 para cada personaje.
- Gráfico que relaciona cantidad de imágenes de entrenamiento y F1 por clase.
- Matriz de confusión normalizada por fila.
- Galerías de predicciones correctas, incorrectas y muestras aleatorias.
- Tabla de las confusiones más frecuentes.
- Mapas de saliencia calculados con el gradiente de la probabilidad predicha respecto de los píxeles de entrada.

### 9.2 Causas de error discutidas

Las explicaciones documentadas se sustentan en el tipo de datos y la arquitectura:

1. Personajes con piel, ropa o paletas similares pueden confundirse.
2. Las clases mayoritarias disponen de más variedad visual que las minoritarias.
3. Fondos repetidos pueden convertirse en señales espurias.
4. Una imagen puede contener varios personajes, aunque solo conserve una etiqueta.
5. Un MLP no es invariante al desplazamiento ni a la escala del personaje.
6. Al aplanar 128×128×3, se pierde la vecindad espacial entre píxeles.

## 10. Fortalezas, limitaciones y mejoras

### Fortalezas

- Flujo de preprocesamiento visible y con comprobaciones explícitas.
- Separación entre entrenamiento, validación y test; la selección se hace con validación.
- Semilla fija y experimentos de un factor por vez.
- Métricas variadas e interpretación del desbalance.
- Auditoría visual de predicciones, errores y saliencia.
- Medición separada de CPU y GPU mediante archivos JSON.

### Limitaciones

- El MLP aplanado no conserva estructura espacial ni reutiliza patrones locales.
- El número de parámetros es muy alto respecto de las imágenes de entrenamiento.
- La clase minoritaria tiene pocas observaciones y el uso de `class_weight` no crea nueva variedad visual.
- La ruta del dataset está codificada de forma absoluta en ambos notebooks.
- El dataset no está presente en el árbol actual, por lo que no se puede reentrenar desde un clon limpio sin descargarlo y configurar la ruta.
- Las corridas CPU y GPU se ejecutan sobre frameworks distintos (TensorFlow y PyTorch); sus métricas de calidad son comparables, pero sus tiempos mezclan hardware y framework.

### Mejoras propuestas por el proyecto

| Mejora | Problema que aborda |
|---|---|
| CNN con convolución y pooling | Recupera vecindad espacial, reduce parámetros y reutiliza patrones locales. |
| Aumento de datos | Aporta variación de pose, recorte, brillo o contraste, especialmente a clases minoritarias. |
| Transfer learning | Parte desde representaciones visuales ya aprendidas. |
| Sobremuestreo con aumento o focal loss | Complementa los pesos por clase sin duplicar imágenes idénticas. |
| Recorte/detección previa del personaje | Reduce la influencia de fondos y normaliza encuadre. |
| Validación cruzada con varias semillas | Estima la variabilidad de las conclusiones. |

## 11. Consideraciones éticas y de uso responsable

El proyecto utiliza personajes ficticios, por lo que no se documenta tratamiento de datos personales. Aun así, hay riesgos técnicos que deben comunicarse:

- El modelo puede perjudicar sistemáticamente a clases minoritarias, aunque mantenga una accuracy global aceptable.
- La red siempre devuelve una de las 25 clases: no posee una clase “ninguno de los anteriores”.
- La confianza softmax no debe interpretarse como garantía de que una imagen externa pertenece al dominio del dataset.
- Las predicciones pueden depender de fondo, encuadre o estilo de las escenas más que de la identidad del personaje.

Por estos motivos, el resultado se presenta como ejercicio académico y no como sistema de identificación listo para producción.

## 12. Reproducibilidad y estructura de entrega

### 12.1 Archivos principales

```text
requirements.txt
PMC/README.md
PMC/notebooks/oldDataset/LosSimpsonsPMC.ipynb
PMC/notebooks/oldDataset/LosSimpsons_GPU_4060M.ipynb
PMC/notebooks/oldDataset/README_GPU.md
PMC/resultados/oldDataset/CPU/rendimiento_cpu_*.json
PMC/resultados/oldDataset/GPU/rendimiento_gpu_*.json
PMC/models/oldDataset/.gitkeep
PMC/images/
PMC/docs/oldDataset/Informe.md
```

### 12.2 Requisitos de ejecución

1. Crear un entorno Python e instalar `requirements.txt`.
2. Descargar el dataset desde la fuente citada y mantener la estructura `train/<clase>` y `test/<clase>`.
3. Configurar `DATASET_BASE` en la primera celda del notebook con la ubicación real del dataset.
4. Ejecutar las celdas en orden.
5. Para GPU, seguir `PMC/notebooks/oldDataset/README_GPU.md`, que documenta el entorno de Windows nativo con PyTorch y la verificación de que la tarjeta se está usando.

### 12.3 Estado de reproducibilidad

Las salidas y métricas de las corridas están guardadas, pero la reproducción completa exige una acción manual: el dataset externo no se versiona y `DATASET_BASE` usa una ruta absoluta específica del equipo de referencia. Esta condición debe indicarse durante la entrega; no corresponde afirmar que el repositorio se ejecuta desde cero sin configuración adicional.

## 13. Guion mínimo para la defensa técnica

Cada integrante debe poder explicar:

1. Por qué esta tarea es clasificación multiclase de etiqueta única y no detección.
2. Por qué se separan entrenamiento, validación y test, y qué garantiza la semilla 42.
3. Por qué se normaliza, aplana y codifica one-hot.
4. Por qué el desbalance obliga a mirar F1 macro y usar `class_weight`.
5. Qué representan capas, neuronas, pesos, ReLU, softmax, pérdida y Adam.
6. Qué cambió en E1–E5 y por qué la selección se hace por F1 macro de validación.
7. Por qué el porcentaje de uso de GPU no basta para saber si se está aprovechando la tarjeta, y qué lo limitaba en este proyecto.
8. Cómo leer la matriz de confusión, los errores y los mapas de saliencia.
9. Por qué una CNN sería una mejora estructural frente al MLP.
10. Qué límites de reproducibilidad y uso responsable tiene la entrega actual.

## 14. Conclusiones

El proyecto implementa un flujo completo de clasificación de imágenes con MLP y supera claramente los baselines aleatorio y de clase mayoritaria. La evaluación no se limita a accuracy: incluye F1 macro, métricas por clase, matriz de confusión y análisis de errores, lo que hace visible el impacto del desbalance.

La principal conclusión técnica es que el MLP aprende señal, pero tiene límites estructurales para imágenes. La entrada de 49.152 valores produce redes con decenas de millones de parámetros y elimina la estructura espacial que una CNN podría aprovechar. La mejora futura más coherente es cambiar la representación mediante convoluciones, aumento de datos y una validación más robusta, no solamente aumentar la cantidad de neuronas.
