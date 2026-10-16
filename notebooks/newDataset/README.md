# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El trabajo con el dataset anterior (25 personajes)
se conserva intacto en [`../oldDataset/`](../oldDataset/).

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |
| [LosSimpsonsPMC.ipynb](LosSimpsonsPMC.ipynb) | CPU: TensorFlow, en Docker o en un Python con TensorFlow | 64 × 64 |

Los dos tienen la misma estructura que los del dataset anterior. Se guardan sin salidas: las genera la
primera corrida sobre el dataset real.

## Qué cambia respecto del dataset anterior

1. **Partición propia.** El notebook reparte las imágenes originales de cada personaje en entrenamiento
   (70 %), validación (15 %) y test (15 %), de forma estratificada y con semilla fija. Validación y test
   conservan la distribución real del dataset y solo contienen originales.
2. **Unas 300 imágenes por personaje para entrenar.** Del 70 % de cada personaje se toman como máximo 300
   originales. El desbalance de entrenamiento baja de 35:1 (3.060 imágenes de Homer frente a 87 de Louie)
   a unas 5:1, y `class_weight` compensa lo que queda.
3. **Imágenes alteradas.** Un cuarto de las imágenes de cada personaje tiene una variante con un cambio de
   color (brillo, contraste, saturación o tono) y un filtro de textura (desenfoque, nitidez, ruido o
   realce de bordes). Solo entran al entrenamiento, y solo las de originales elegidas para entrenar.
4. **Descriptores de color y textura.** El modelo base no recibe los píxeles, sino unos 2.460 descriptores
   por imagen: histogramas HSV y Lab, LBP, filtros de Gabor y HOG. La experimentación los compara con los
   píxeles crudos (R1), con el color solo (R2) y con la textura sola (R3), además de las variantes de
   arquitectura E1–E5 de siempre.

## Paso 1: colocar el dataset

```text
simpsons_dataset/
├── abraham_grampa_simpson/
├── agnes_skinner/
├── ...
└── waylon_smithers/
```

Por defecto los notebooks y el script buscan el dataset en `~/Desktop/simpsons_dataset/simpsons_dataset`.
Si está en otra ruta, defínela en la variable de entorno `SIMPSONS_DATASET` antes de abrir Jupyter.

La etiqueta de cada imagen es siempre el nombre de su carpeta. Algunas carpetas contienen fotogramas cuyo
nombre de archivo es el de otro personaje: los añadió el equipo a propósito, porque en ellos el
protagonista es el personaje de la carpeta.

## Paso 2: generar las imágenes alteradas (una sola vez)

Desde la raíz del repositorio:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
```

El script elige al azar un cuarto de las imágenes de cada personaje y guarda junto a cada una su variante,
`<original>__alterada.<ext>`, con el mismo formato. Además deja un registro de qué transformación recibió
cada imagen en `_alteraciones.csv`, en la raíz del dataset.

- **Las originales no se modifican nunca**: el script solo las lee.
- **Es reproducible**: con la misma semilla genera exactamente los mismos archivos, byte a byte.
- **No deja rastro**: cada ejecución borra primero lo que generó la anterior, así que se puede repetir
  sin acumular archivos.
- `--limpiar` borra todo lo generado y deja el dataset como se descargó.
- `--fraccion` y `--semilla` permiten cambiar la proporción de alteradas (0,25 por defecto) y la semilla
  (42).

Si se usa Docker en lugar de `.venv-gpu`, la misma imagen sirve para ejecutar el script montando el
dataset con permiso de escritura:

```powershell
docker run --rm -v "${PWD}:/workspace" -v "C:/ruta/al/simpsons_dataset:/data/simpsons_dataset" `
  -w /workspace simpsons-pmc:gpu python scripts/newDataset/generar_imagenes_alteradas.py
```

## Paso 3: ejecutar un notebook

**En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del dataset anterior
([`../oldDataset/README_GPU.md`](../oldDataset/README_GPU.md)), más una dependencia:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-newDataset.txt
.\scripts\newDataset\ejecutar_notebook_gpu.ps1
```

**En Docker.** Copia `.env.example` como `.env`, pon en `SIMPSONS_DATASET_HOST` la ruta del dataset y
levanta el contenedor:

```powershell
docker compose up --build
```

Después abre <http://localhost:8888> y entra en `notebooks/newDataset/`. Cada notebook abre con su
kernel: el de GPU con PyTorch y el de CPU con TensorFlow.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` y `mlp_final.json` | `models_gpu/newDataset/` (GPU) y `models/newDataset/` (CPU), ignorados por Git |
| Resumen de rendimiento de cada corrida | `resultados/newDataset/<CPU\|GPU>/rendimiento_<entorno>_<resolución>.json` |

`mlp_final.json` guarda la representación, la resolución y el orden de las clases del modelo final. La
sección «Probar el modelo con una imagen cualquiera» lo usa para repetir el mismo pre-procesamiento sin
volver a entrenar.

## Estado y tiempos de referencia

- Los dos notebooks y el script se probaron de principio a fin con un **dataset sintético** que imita la
  estructura del real: 20 personajes, archivos `.png` y `.webp`, nombres con `__` y nombres de otro
  personaje. No se han ejecutado todavía con el dataset real.
- Las celdas de "Lectura" que interpretan resultados están marcadas como **pendientes**. Se escribieron
  para la corrida del dataset anterior y hay que reescribirlas con las cifras de la primera corrida.
- El cálculo de descriptores es la etapa más lenta del pre-procesamiento: en el portátil del equipo, con
  imágenes sintéticas, rinde entre 60 y 160 imágenes por segundo. Con las unas 27.000 imágenes que usa
  el notebook de GPU son varios minutos. El KPI de costo de catalogación incluye ese cálculo, porque es
  parte del costo real de clasificar una imagen nueva.
