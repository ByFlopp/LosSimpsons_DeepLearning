# Notebooks de la CNN sobre el dataset nuevo (64 personajes)

Esta carpeta contiene la red neuronal convolucional (CNN) para el dataset ampliado por el equipo: **una
carpeta por personaje**, sin partición previa en `train` y `test`. Es el mismo dataset y el mismo proceso de
entrenamiento del Perceptrón Multicapa de [`../../../PMC/notebooks/newDataset/`](../../../PMC/notebooks/newDataset/);
cambia el modelo.

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 64 × 64 |
| [LosSimpsonsCNN.ipynb](LosSimpsonsCNN.ipynb) | CPU: Keras 3 sobre PyTorch con la GPU oculta, en el mismo `.venv-gpu` o en Docker | 64 × 64 |

Los dos siguen el mismo proceso, con el mismo framework y el mismo bucle de entrenamiento; solo cambia el
dispositivo. El de CPU no usa TensorFlow: su primera celda fija `KERAS_BACKEND=torch` y oculta la GPU
(`CUDA_VISIBLE_DEVICES=-1`) antes de importar PyTorch y Keras.

## Qué es igual que en el MLP

Las celdas que no dependen del modelo son copia literal de las del notebook del MLP del mismo dispositivo:

- **Partición propia, con todo el dataset.** Las originales de cada personaje se reparten 70 % / 20 % / 10 %
  (entrenamiento, validación y test), estratificadas, con semilla 42 y sin tope por personaje.
- **Imágenes alteradas y parecidas.** Cada alterada va al mismo conjunto que su original, y las casi
  idénticas a otra del mismo personaje entran con su versión filtrada (`__parecida`) en lugar de la original.
- **Desbalance.** Pesos por clase suavizados: la raíz cuadrada de los balanceados.
- **Escenas con varios personajes.** Etiqueta suave al entrenar (solo con fotogramas y gemelas de
  entrenamiento) y criterio de acierto tolerante en test: nombrar a otro personaje presente cuenta como
  acierto. Los errores con confianza mayor que 0,80 se verifican por imagen. La accuracy estricta se informa
  como referencia; la validación y la selección del modelo usan siempre la etiqueta de la carpeta.
- **Entrenamiento.** El mismo bucle propio sobre PyTorch (`entrenar` / `_pasada`): Adam con tasa `1e-3`,
  batch 128, suavizado de etiquetas 0,1, hasta 60 épocas con parada temprana (paciencia 12) y
  `ReduceLROnPlateau`; los experimentos, 35 épocas y paciencia 9.
- **Evaluación.** Los mismos KPIs con los mismos umbrales, las mismas galerías y el mismo resumen de
  rendimiento en JSON.

## Qué cambia

1. **La entrada.** La CNN recibe la imagen de 64 × 64 × 3 como tensor, sin aplanarla y sin descriptores de
   color y textura. El escalado a [0, 1] lo hace una capa `Rescaling` dentro del modelo.
2. **El modelo base.** Cuatro bloques convolucionales de 32, 64, 128 y 256 filtros de 3 × 3, cada uno con
   `BatchNormalization`, ReLU y `MaxPooling2D` de 2 × 2; pooling global; cabeza densa de 256 neuronas con
   Dropout 0,3; salida `softmax` de 64.
3. **Los experimentos.** R1 (escala de grises), E1 (3 bloques), E2 (2 convoluciones por bloque), E3 (tanh),
   E4 (lr `1e-4`), E5 (batch 512), E6 (dropout 0,5), E7 (aplanado en lugar de pooling global), P1, P2, S1,
   S2, A1 y la combinación C1. Se elige por F1 macro de validación, como en el MLP.
4. **El análisis.** Fundamentos de la convolución con filtros aplicados a mano, mapas de activación de la red
   entrenada, Grad-CAM y una comparación con la corrida del MLP del mismo entorno y la misma resolución.

## Cómo ejecutarlo

Desde la raíz del repositorio:

1. **El dataset y sus imágenes alteradas.** Son los mismos del MLP: deja el dataset en
   `~/Desktop/simpsons_dataset/simpsons_dataset` (o define `SIMPSONS_DATASET`) y genera las alteradas,
   parecidas y gemelas con el script compartido. Los detalles están en la
   [guía del dataset nuevo](../../../PMC/notebooks/newDataset/README.md#paso-1-colocar-el-dataset).

   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
   ```

2. **En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del MLP:

   ```powershell
   .\CNN\scripts\newDataset\ejecutar_notebook_gpu.ps1
   ```

3. **En Windows, en la CPU.** Abre `LosSimpsonsCNN.ipynb` con el kernel de `.venv-gpu` y ejecútalo desde la
   primera celda, que deja Keras sobre PyTorch y oculta la GPU antes de importar nada.

4. **En Docker.** `docker compose up --build` desde la raíz, y en <http://localhost:8888> entra en
   `CNN/notebooks/newDataset/`.

### Tiempos de referencia

Medido en este equipo (Intel Core i7-13620H y NVIDIA GeForce RTX 4060 Laptop, conectado a la corriente) con
la CNN base y el bucle de entrenamiento del notebook:

| Resolución | Segundos por época en GPU | Segundos por época en CPU |
|---|---:|---:|
| 32 × 32 | ≈ 5 | ≈ 32 |
| 64 × 64 | ≈ 10 | ≈ 130 |
| 128 × 128 | ≈ 37 | — |

Una corrida completa entrena el modelo base y catorce configuraciones, unas 500 épocas en total. A 64 × 64
eso es del orden de hora y media en la GPU y de más de quince horas en la CPU; a 32 × 32, la corrida de CPU
baja a unas cuatro horas.

## Resultados de la corrida del 19-10-2026

El notebook de GPU se ejecutó completo con el dataset real (64 personajes, 40.123 originales) a 64 × 64; sus
salidas guardadas son las de esa corrida. El de CPU no se ha ejecutado completo.

| Corrida | Configuración final | Accuracy | Estricta | Top-3 | F1 macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 64 × 64 | E2 - 2 convoluciones por bloque | 0,830 | 0,777 | 0,932 | 0,788 | 1 h 44 min 33 s |

- **Experimentos.** Solo E2 supera a la base en F1 macro de validación (0,717 frente a 0,692), así que C1 no se
  entrena. Las caídas más grandes son las de la escala de grises (R1, −19,7 puntos), la tasa de aprendizaje
  `1e-4` (E4, −17,8) y tanh (E3, −14,5).
- **Frente al MLP** de la misma resolución y dispositivo (R2, solo color), la accuracy sube 28 puntos, la
  estricta 27, la top-3 20 y el F1 macro 27.
- **Costo.** Cargar las imágenes toma 4,7 minutos y entrenar las catorce configuraciones, 96. La GPU trabaja el
  91 % del tiempo a batch 128, así que un batch mayor casi no acelera (de 3.530 a 3.782 imágenes por segundo a
  batch 512); batch 1.024 ya no cabe en la VRAM.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` y `cnn_final.json` | `CNN/models_gpu/newDataset/` (GPU) y `CNN/models/newDataset/` (CPU), ignorados por Git |
| Resumen de rendimiento de cada corrida | `CNN/resultados/newDataset/<CPU\|GPU>/rendimiento_<entorno>_<resolución>.json` |

El JSON tiene las mismas claves que el del MLP, más `modelo` y `arquitectura_final`; así la última sección de
cada notebook puede comparar las dos redes. `cnn_final.json` guarda la resolución y el orden de las clases
del modelo final; la sección «Probar el modelo con una imagen cualquiera» lo usa para clasificar imágenes
nuevas sin volver a entrenar.
