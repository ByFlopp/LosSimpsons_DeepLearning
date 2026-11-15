# Notebooks de la CNN sobre el dataset nuevo (64 personajes)

Esta carpeta contiene la red neuronal convolucional (CNN) para el dataset ampliado por el equipo: **una
carpeta por personaje**, sin partición previa en `train` y `test`. Es el mismo dataset y el mismo proceso de
entrenamiento del Perceptrón Multicapa de [`../../PMC/notebooks/`](../../PMC/notebooks/);
cambia el modelo.

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU RTX 4060 Laptop: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |

## Qué es igual que en el MLP

Las funciones que no dependen del modelo están en el módulo común
[`scripts/newDataset/pipeline_comun/`](../../scripts/newDataset/pipeline_comun/), que importan los dos notebooks:

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
- **Entrenamiento.** El mismo bucle propio sobre PyTorch (`pipeline_comun.entrenamiento.entrenar`): Adam con
  tasa `1e-3`, batch 128, suavizado de etiquetas 0,1 y `ReduceLROnPlateau`. **Todas las configuraciones** tienen el
  mismo presupuesto: hasta 60 épocas con parada temprana (paciencia 12). El historial de cada una queda en
  `CNN/models_gpu/newDataset/historiales/`, y si la corrida se corta, lo ya entrenado se recupera en lugar de
  reentrenarse.
- **Evaluación y auditoría.** Los mismos KPIs con los mismos umbrales, las mismas galerías, la misma auditoría
  (`auditar_modelo`: ROC-AUC, calibración, umbral de revisión y sesgos por subgrupo) y el mismo resumen de
  rendimiento en JSON.

## Qué cambia

1. **La entrada.** La CNN recibe la imagen de alto × ancho × 3 como tensor —128 × 128 × 3—, sin aplanarla y sin
   descriptores de color y textura. El escalado a [0, 1] lo hace una capa `Rescaling` dentro del modelo.
2. **El modelo base.** Cuatro bloques convolucionales de 32, 64, 128 y 256 filtros de 3 × 3, cada uno con
   `BatchNormalization`, ReLU y `MaxPooling2D` de 2 × 2; pooling global; cabeza densa de 256 neuronas con
   Dropout 0,3; salida `softmax` de 64.
3. **Los experimentos.** R1 (escala de grises), E1 (3 bloques), E2 (2 convoluciones por bloque), E3 (tanh),
   E4 (lr `1e-4`), E5 (batch 512), E6 (dropout 0,5), E7 (aplanado en lugar de pooling global), P1 y P2 (pesos
   por clase), S1 (sin suavizado), A1 (sin alteradas), los de regularización G1 (sin dropout), G2 (sin Batch
   Normalization), G3 (L2 de 5·10⁻⁴) y G4 (aumento de datos en línea), y la combinación C1. Después, un
   **Random Search** en dos etapas (`optimizar_modelo`) sobre la tasa de aprendizaje, el dropout, L2 y el aumento
   en línea. Se elige por F1 macro de validación, como en el MLP. Antes de entrenar cada configuración se mide
   cuánta memoria de la GPU usa: a 128 × 128, E2 y E5 no caben en los 8 GB y quedan fuera de la comparación.
4. **El análisis.** Fundamentos de la convolución con filtros aplicados a mano; el efecto de la parada temprana
   y de cada técnica de regularización, con curvas y la brecha entre entrenamiento y validación; precision,
   recall, F1, matriz de confusión y curvas ROC con su ROC-AUC; la auditoría de calibración, umbral de revisión y
   sesgos por origen, tamaño de la clase, familia Simpson y tipo de imagen; mapas de activación de la red
   entrenada; Grad-CAM de casos concretos, de una imagen por personaje y del test completo; y una comparación con
   el MLP: desempeño, generalización, interpretabilidad, tiempos y arquitectura seleccionada.
5. **El contexto.** Una introducción que ubica cada punto de la pauta de la EP3 en el notebook, una sección
   «Pipeline integrado» con el diagrama y la justificación de las etapas, por qué se dejó el dataset de la EP1 y
   una sección de impacto ético.

## Cómo ejecutarlo

La ejecución es local, en la RTX 4060 Laptop: se acordó con el profesor en lugar de Google Colab, porque la
corrida completa toma unas 16 horas y media. Desde la raíz del repositorio:

1. **El dataset y sus imágenes alteradas.** Son los mismos del MLP: deja el dataset en
   `~/Desktop/simpsons_dataset` (o define `SIMPSONS_DATASET`) y genera las alteradas,
   parecidas y gemelas con el script compartido. Los detalles están en la
   [guía del dataset nuevo](../../PMC/notebooks/README.md).

   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
   ```

2. **En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del MLP:

   ```powershell
   .\CNN\scripts\newDataset\ejecutar_notebook_gpu.ps1
   ```

3. **En Docker.** `docker compose up --build` desde la raíz, y en <http://localhost:8888> entra en
   `CNN/notebooks/`.

   **Sin VS Code**, desde `CNN/notebooks/`, el notebook se puede ejecutar completo y guardar en otro archivo.
   Con `SIMPSONS_PRUEBA=1` corre de punta a punta con 40 imágenes por personaje y 2 épocas, y escribe en carpetas
   `prueba/` aparte, para comprobarlo antes de la corrida completa:

   ```powershell
   $env:KERAS_BACKEND = "torch"
   $env:SIMPSONS_PRUEBA = "1"   # quitar para la corrida completa
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
   `CNN/models_gpu/newDataset/cajas_cnn_final/`.

5. **Ruido en las etiquetas.** Con el modelo ya guardado, este script recorre el dataset personaje por personaje
   y marca las imágenes cuya carpeta probablemente está mal. No modifica el dataset; tarda alrededor de un minuto
   en la GPU y menos al repetirlo, porque guarda una caché:

   ```powershell
   .\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_ruido_etiquetas.py
   ```

   Deja `ruido.csv`, `resumen_por_personaje.csv` y una hoja de contacto por personaje en
   `CNN/models_gpu/newDataset/ruido_cnn_final/`.

### Tiempos de referencia

Medido en este equipo (Intel Core i7-13620H y NVIDIA GeForce RTX 4060 Laptop, conectado a la corriente) con
la CNN base y el bucle de entrenamiento del notebook:

| Resolución | Segundos por época en GPU (benchmark) | Corrida completa en GPU |
|---|---:|---:|
| 128 × 128 | 51,7 | 16 h 28 min 18 s |

Una corrida completa entrena 26 configuraciones —la base, catorce variantes de un factor, C1 y las diez del Random
Search; E2 y E5 no caben en la GPU a 128 × 128—: 981 épocas en total.

## Resultados de la corrida del 14-11-2026

El notebook de GPU se ejecutó completo con el dataset real (64 personajes, 56.268 originales) a 128 × 128,
con la imagen completa. Sus salidas guardadas son las de esa corrida.

| Corrida | Configuración final | Accuracy | Top-3 | F1 macro | ROC-AUC macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 128 × 128 | G4 - aumento de datos en línea | **0,775** | **0,908** | **0,755** | **0,987** | 16 h 28 min 18 s |

- **Regularización.** Sin dropout (G1) el F1 macro de validación cae 5,7 puntos y la brecha sube a 0,31; sin
  Batch Normalization (G2), 2,5; L2 (G3) y dropout 0,5 (E6) no cambian nada, y el aumento de datos en línea (G4)
  sube 3,8, con la brecha en 0,033. C1, que junta las tres mejoras individuales, rinde la mitad que G4 solo.
- **Optimización.** Gana G4. La mejor del Random Search (RS1), también con aumento en línea, queda a 0,1 puntos.
  Las caídas más grandes son las de la tasa de aprendizaje `1e-4` (E4, −10,9 puntos), tanh (E3, −9,2) y la escala
  de grises (R1, −8,1).
- **Auditoría.** ECE de 0,078 (subconfiada); con el umbral de 0,529 elegido en validación resuelve sola el 71,6 %
  del test al 90,4 % de acierto. Accuracy de 0,933 en las imágenes de Kaggle y de 0,670 en los fotogramas de
  episodio.
- **Frente al MLP** en la GPU a 128 × 128 (C1, solo color, corrida del 09-11-2026): accuracy +23,1, top-3 +17,3 y
  F1 macro +20,4.
- **Costo.** Cargar las imágenes toma 11,9 minutos y entrenar, 946,6. La GPU trabaja el 99,4 % del tiempo y no cabe
  más que batch 128.

La [presentación de resultados](../presentacion_resultados_CNN.html) y el [informe](../docs/newDataset/Informe.md)
usan las cifras de esta corrida.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` de la corrida y el modelo final con sus archivos (`cnn_final.keras`, `.json`, `_entrada.py` y `_particion.csv`) | `CNN/models_gpu/newDataset/`, ignorados por Git |
| Historial de cada configuración | `CNN/models_gpu/newDataset/historiales/`, ignorados por Git |
| Resumen de rendimiento de la corrida | `CNN/resultados/newDataset/GPU/rendimiento_gpu_128x128.json`; cada corrida reemplaza el de la anterior |
| Figuras del informe | `CNN/docs/newDataset/figuras/`, guardadas por el notebook al ejecutarse |
| Cajas de cada personaje, después del notebook | `CNN/models_gpu/newDataset/cajas_cnn_final/`, ignoradas por Git |
| Presentación de resultados | `CNN/presentacion_resultados_CNN.html`, generada con `scripts/newDataset/generar_presentacion.py --modelo cnn` |

El JSON tiene las mismas claves que el del MLP, más `modelo` y `arquitectura_final`; así la comparación de cada
notebook puede leer las dos redes desde el disco. `cnn_final.json` guarda la resolución y el orden de las clases
del modelo final; la sección «Probar el modelo con una imagen cualquiera», `predecir_imagen.py` y el script de
cajas lo usan para clasificar imágenes nuevas sin volver a entrenar.
