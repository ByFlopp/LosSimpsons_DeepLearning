# Notebooks de la CNN sobre el dataset nuevo (64 personajes)

Esta carpeta contiene la red neuronal convolucional (CNN) para el dataset ampliado por el equipo: **una
carpeta por personaje**, sin partición previa en `train` y `test`. Es el mismo dataset y el mismo proceso de
entrenamiento del Perceptrón Multicapa de [`../../../PMC/notebooks/newDataset/`](../../../PMC/notebooks/newDataset/);
cambia el modelo.

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 (también se corrió a 32 × 32 y 64 × 64) |
| [LosSimpsonsCNN.ipynb](LosSimpsonsCNN.ipynb) | CPU: Keras 3 sobre PyTorch con la GPU oculta, en el mismo `.venv-gpu` o en Docker | 64 × 64 (aún sin ejecutar completo) |

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

1. **La entrada.** La CNN recibe la imagen de alto × ancho × 3 como tensor —128 × 128 × 3 en la corrida
   guardada—, sin aplanarla y sin descriptores de color y textura. El escalado a [0, 1] lo hace una capa
   `Rescaling` dentro del modelo.
2. **El modelo base.** Cuatro bloques convolucionales de 32, 64, 128 y 256 filtros de 3 × 3, cada uno con
   `BatchNormalization`, ReLU y `MaxPooling2D` de 2 × 2; pooling global; cabeza densa de 256 neuronas con
   Dropout 0,3; salida `softmax` de 64.
3. **Los experimentos.** R1 (escala de grises), E1 (3 bloques), E2 (2 convoluciones por bloque), E3 (tanh),
   E4 (lr `1e-4`), E5 (batch 512), E6 (dropout 0,5), E7 (aplanado en lugar de pooling global), P1, P2, S1,
   S2, A1 y la combinación C1. Se elige por F1 macro de validación, como en el MLP. Antes de entrenar cada
   configuración se mide cuánta memoria de la GPU usa: a 128 × 128, E2 y E5 no caben en los 8 GB y quedan
   fuera de la comparación.
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

| Resolución | Segundos por época en GPU (benchmark) | Corrida completa en GPU | Segundos por época en CPU |
|---|---:|---:|---:|
| 32 × 32 | 5,3 | 59 min 03 s | ≈ 32 |
| 64 × 64 | 9,8 | 1 h 44 min 33 s | ≈ 130 |
| 128 × 128 | 37,1 | 5 h 08 min 56 s | — |

Una corrida completa entrena el modelo base y hasta catorce configuraciones: 536 épocas a 32 × 32, 515 a
64 × 64 y 445 a 128 × 128, donde E2 y E5 no caben en la GPU. En la CPU, a 64 × 64, tomaría más de quince
horas; a 32 × 32, unas cuatro.

## Resultados de las corridas del 19-10-2026

El notebook de GPU se ejecutó completo con el dataset real (64 personajes, 40.123 originales) a tres
resoluciones. Sus salidas guardadas son las de la corrida a 128 × 128; las de 64 × 64 quedaron en el commit
`6db8274`, y de la de 32 × 32 solo queda su JSON. El de CPU no se ha ejecutado completo.

| Corrida | Configuración final | Accuracy | Estricta | Top-3 | F1 macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 32 × 32 | E2 - 2 convoluciones por bloque | 0,786 | 0,732 | 0,887 | 0,741 | 59 min 03 s |
| GPU 64 × 64 | E2 - 2 convoluciones por bloque | 0,830 | 0,777 | 0,932 | 0,788 | 1 h 44 min 33 s |
| GPU 128 × 128 | Base - 1 convolución por bloque | **0,847** | **0,797** | **0,937** | **0,810** | 5 h 08 min 56 s |

- **Experimentos.** A 32 × 32 y 64 × 64 gana E2 (a 64 × 64, F1 macro de validación de 0,717 frente a 0,692 de
  la base). A 128 × 128 E2 y E5 no caben en la GPU y ninguna variante supera a la base (0,734; la segunda,
  S2, queda a 0,6 puntos), así que C1 no se entrena. Las caídas más grandes a 128 × 128 son las de la tasa de
  aprendizaje `1e-4` (E4, −24,1 puntos), tanh (E3, −21,0) y la escala de grises (R1, −11,3, frente a −19,7
  a 64 × 64).
- **Frente al MLP** de la misma resolución y dispositivo, la accuracy sube entre 24,5 y 28,0 puntos y el F1
  macro entre 22,7 y 26,6. A 128 × 128, frente a C1 (solo color): accuracy +26,1, estricta +25,8, top-3 +18,7
  y F1 macro +25,1.
- **Costo.** Cargar las imágenes toma de 2,6 a 10,5 minutos y entrenar, de 53,7 a 290,5: entre el 91 y el
  94 % de cada corrida. La GPU trabaja el 36,5 % del tiempo a 32 × 32, el 90,7 % a 64 × 64 y el 98,8 % a
  128 × 128. A 64 × 64 un batch mayor casi no acelera (de 3.530 a 3.782 imágenes por segundo a batch 512), y a
  128 × 128 ya no cabe más que batch 128.

La [presentación de resultados](../../presentacion_resultados_CNN.html) reúne estas cifras con las figuras del
notebook y la comparación con el MLP.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` y `cnn_final.json` | `CNN/models_gpu/newDataset/` (GPU) y `CNN/models/newDataset/` (CPU), ignorados por Git |
| Resumen de rendimiento de cada corrida | `CNN/resultados/newDataset/<CPU\|GPU>/rendimiento_<entorno>_<resolución>.json` |
| Presentación de resultados | `CNN/presentacion_resultados_CNN.html`, generada con `CNN/scripts/newDataset/generar_presentacion.py` |

El JSON tiene las mismas claves que el del MLP, más `modelo` y `arquitectura_final`; así la última sección de
cada notebook puede comparar las dos redes. `cnn_final.json` guarda la resolución y el orden de las clases
del modelo final —hoy, el de la corrida a 128 × 128—; la sección «Probar el modelo con una imagen cualquiera»
lo usa para clasificar imágenes nuevas sin volver a entrenar.
