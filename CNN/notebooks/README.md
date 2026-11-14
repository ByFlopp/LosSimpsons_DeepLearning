# Notebooks de la CNN sobre el dataset nuevo (64 personajes)

Esta carpeta contiene la red neuronal convolucional (CNN) para el dataset ampliado por el equipo: **una
carpeta por personaje**, sin partición previa en `train` y `test`. Es el mismo dataset y el mismo proceso de
entrenamiento del Perceptrón Multicapa de [`../../PMC/notebooks/`](../../PMC/notebooks/);
cambia el modelo.

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU RTX 4060 Laptop: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |

## Qué es igual que en el MLP

Las celdas que no dependen del modelo son copia literal de las del notebook de GPU del MLP:

- **Partición propia, con todo el dataset.** Las originales de cada personaje se reparten 70 % / 20 % / 10 %
  (entrenamiento, validación y test), estratificadas, con semilla 42 y sin tope por personaje.
- **Imágenes alteradas y parecidas.** Cada alterada va al mismo conjunto que su original, y las casi
  idénticas a otra del mismo personaje entran con su versión filtrada (`__parecida`) en lugar de la original.
- **Desbalance.** Pesos por clase suavizados: la raíz cuadrada de los balanceados.
- **La imagen completa y el criterio de la carpeta.** La red aprende y se evalúa con la imagen completa, y
  una predicción acierta solo si nombra al personaje de la carpeta. Las escenas con varios personajes
  (fotogramas del mismo episodio y gemelas) solo alimentan la accuracy por escena, de referencia.
- **Primero el modelo, después las cajas.** Al terminar, la celda «Selección del modelo final» guarda
  `cnn_final.keras` con su `.json`, su `_entrada.py` y su `_particion.csv`. Con esos archivos,
  `scripts/newDataset/detectar_cajas.py` encuentra después a cada personaje con un detector preentrenado
  (Grounding DINO) y la CNN nombra cada caja con su confianza (`Jasper Beardly · 92 %`), sin anotar nada a
  mano. El notebook no depende de ese script.
- **Entrenamiento.** El mismo bucle propio sobre PyTorch (`entrenar` / `_pasada`): Adam con tasa `1e-3`,
  batch 128, suavizado de etiquetas 0,1, hasta 60 épocas con parada temprana (paciencia 12) y
  `ReduceLROnPlateau`; los experimentos, 35 épocas y paciencia 9.
- **Evaluación.** Los mismos KPIs con los mismos umbrales, las mismas galerías y el mismo resumen de
  rendimiento en JSON.

## Qué cambia

1. **La entrada.** La CNN recibe la imagen de alto × ancho × 3 como tensor —128 × 128 × 3—, sin aplanarla y sin descriptores de color y textura. El escalado a [0, 1] lo hace una capa
   `Rescaling` dentro del modelo.
2. **El modelo base.** Cuatro bloques convolucionales de 32, 64, 128 y 256 filtros de 3 × 3, cada uno con
   `BatchNormalization`, ReLU y `MaxPooling2D` de 2 × 2; pooling global; cabeza densa de 256 neuronas con
   Dropout 0,3; salida `softmax` de 64.
3. **Los experimentos.** R1 (escala de grises), E1 (3 bloques), E2 (2 convoluciones por bloque), E3 (tanh),
   E4 (lr `1e-4`), E5 (batch 512), E6 (dropout 0,5), E7 (aplanado en lugar de pooling global), P1 y P2 (pesos
   por clase), S1 (sin suavizado), A1 (sin alteradas), los de regularización G1 (sin dropout), G2 (sin Batch
   Normalization), G3 (L2 de 5·10⁻⁴) y G4 (aumento de datos en línea), y la combinación C1. Se elige por F1
   macro de validación, como en el MLP. Antes de entrenar cada configuración se mide cuánta memoria de la GPU
   usa: a 128 × 128, E2 y E5 no caben en los 8 GB y quedan fuera de la comparación.
4. **El análisis.** Fundamentos de la convolución con filtros aplicados a mano; el efecto de la parada temprana
   y de cada técnica de regularización, con curvas y la brecha entre entrenamiento y validación; precision,
   recall, F1, matriz de confusión y curvas ROC con su ROC-AUC; mapas de activación de la red entrenada en una
   imagen y en cuatro imágenes de test (Homero, Marge, Bart y Lisa); Grad-CAM de casos concretos, de una imagen
   por personaje, el mapa medio de cada personaje y el del test completo; y una comparación con la corrida del
   MLP en la GPU a 128 × 128: desempeño, parámetros, tiempos, capacidad de generalización, interpretabilidad y
   ventajas y desventajas de cada arquitectura.
5. **El contexto.** Una introducción que ubica cada punto de la pauta de la EP3 en el notebook, una sección
   «Pipeline integrado» que describe y justifica las etapas, por qué se dejó el dataset de la EP1 y una sección
   de impacto ético: derechos de autor de las imágenes, sesgo por el desbalance y por el parecido entre
   personajes, ruido en las etiquetas y revisión humana de las predicciones.

## Cómo ejecutarlo

Desde la raíz del repositorio:

1. **El dataset y sus imágenes alteradas.** Son los mismos del MLP: deja el dataset en
   `~/Desktop/simpsons_dataset` (o define `SIMPSONS_DATASET`) y genera las alteradas,
   parecidas y gemelas con el script compartido. Los detalles están en la
   [guía del dataset nuevo](../../PMC/notebooks/README.md#paso-1-colocar-el-dataset).

   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
   ```

2. **En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del MLP:

   ```powershell
   .\CNN\scripts\newDataset\ejecutar_notebook_gpu.ps1
   ```

3. **En Docker.** `docker compose up --build` desde la raíz, y en <http://localhost:8888> entra en
   `CNN/notebooks/`.

   **En Google Colab**, con un entorno de GPU: sube a Google Drive la carpeta del dataset, con sus alteradas,
   parecidas y gemelas, como `MyDrive/simpsons_dataset`. La celda «Definición del conjunto de datos a utilizar»
   detecta Colab, monta Drive y la lee desde ahí. La comparación con el MLP y la recreación de personajes
   necesitan el repositorio completo; sin él, esas celdas lo avisan y siguen.

   **Sin VS Code**, desde `CNN/notebooks/`, el notebook se puede ejecutar completo y guardar en otro archivo:

   ```powershell
   $env:KERAS_BACKEND = "torch"
   ..\..\.venv-gpu\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 `
       --ExecutePreprocessor.kernel_name=python3 --output corrida.ipynb LosSimpsons_GPU_4060M.ipynb
   ```

4. **Después, si se quiere, las cajas.** Con el modelo ya guardado, el script de cajas encuentra a cada
   personaje y lo nombra con la CNN. La primera vez se instala `transformers` en `.venv-gpu`:

   ```powershell
   .\.venv-gpu\Scripts\python.exe -m pip install -r requirements-cajas.txt
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py --modelo cnn ruta\a\imagen.jpg
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py --modelo cnn --dataset --muestra 5
   ```

   Deja `cajas.csv`, las imágenes con sus cajas y un `resumen.json` en
   `CNN/models_gpu/newDataset/cajas_cnn_final/`. Las opciones están en el
   [paso 4 de la guía del dataset nuevo](../../PMC/notebooks/README.md).

5. **Ruido en las etiquetas.** Con el modelo ya guardado, este script recorre el dataset personaje por personaje
   y marca las imágenes cuya carpeta probablemente está mal. No modifica el dataset; tarda alrededor de un minuto
   en la GPU y menos al repetirlo, porque guarda una caché:

   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_ruido_etiquetas.py
   ```

   Deja `ruido.csv`, `resumen_por_personaje.csv` y una hoja de contacto por personaje en
   `CNN/models_gpu/newDataset/ruido_cnn_final/`. Las marcas van de más a menos sospechosas: otro personaje,
   otro personaje de la escena, personaje parecido y sin personaje claro.

### Tiempos de referencia

Medido en este equipo (Intel Core i7-13620H y NVIDIA GeForce RTX 4060 Laptop, conectado a la corriente) con
la CNN base y el bucle de entrenamiento del notebook:

| Resolución | Segundos por época en GPU (benchmark) | Corrida completa en GPU |
|---|---:|---:|
| 128 × 128 | 52,1 | 9 h 11 min 40 s |

Una corrida completa entrena el modelo base y quince configuraciones más (E2 y E5 no caben en la GPU a
128 × 128): 577 épocas.

## Resultados de la corrida del 13-11-2026

El notebook de GPU se ejecutó completo con el dataset real (64 personajes, 56.676 originales) a 128 × 128,
con la imagen completa. Sus salidas guardadas son las de esa corrida.

| Corrida | Configuración final | Accuracy | Top-3 | F1 macro | ROC-AUC macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 128 × 128 | G4 - aumento de datos en línea | **0,747** | **0,891** | **0,731** | **0,986** | 9 h 11 min 40 s |

- **Regularización.** Sin dropout (G1) el F1 macro de validación cae 6,0 puntos y la brecha sube a 0,31; sin
  Batch Normalization (G2), 1,8; con L2 (G3) sube 0,6, y con aumento de datos en línea (G4), 1,9, con la brecha
  casi en cero. C1, que junta G3 y G4, rinde menos que G4 solo.
- **Experimentos.** Gana G4 por F1 macro, accuracy y pérdida de validación. Las caídas más grandes son las de
  la tasa de aprendizaje `1e-4` (E4, −18,3 puntos), tanh (E3, −18,0) y la escala de grises (R1, −8,6).
- **Frente al MLP** en la GPU a 128 × 128, frente a C1 (solo color): accuracy +20,3, top-3 +15,6 y F1 macro
  +18,0.
- **Costo.** Cargar las imágenes toma 11,5 minutos y entrenar, 522,2: el 95 % de la corrida. La GPU trabaja el
  99,3 % del tiempo y ya no cabe más que batch 128.

La [presentación de resultados](../../presentacion_resultados_CNN.html) todavía muestra las corridas de
octubre; se rehace con la corrida a 128 × 128 del 13-11-2026.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` de la corrida y el modelo final con sus archivos (`cnn_final.keras`, `.json`, `_entrada.py` y `_particion.csv`) | `CNN/models_gpu/newDataset/`, ignorados por Git |
| Resumen de rendimiento de la corrida | `CNN/resultados/newDataset/GPU/rendimiento_gpu_128x128.json`; cada corrida reemplaza el de la anterior |
| Cajas de cada personaje, después del notebook | `CNN/models_gpu/newDataset/cajas_cnn_final/`, ignoradas por Git |
| Presentación de resultados | `CNN/presentacion_resultados_CNN.html`, generada con `CNN/scripts/newDataset/generar_presentacion.py` |

El JSON tiene las mismas claves que el del MLP, más `modelo` y `arquitectura_final`; así la última sección de
cada notebook puede comparar las dos redes. `cnn_final.json` guarda la resolución y el orden de las clases
del modelo final; la sección «Probar el modelo con una imagen cualquiera», `predecir_imagen.py` y el script de
cajas lo usan para clasificar imágenes nuevas sin volver a entrenar.
