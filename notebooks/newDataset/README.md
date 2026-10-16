# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El trabajo con el dataset anterior (25 personajes)
se conserva intacto en [`../oldDataset/`](../oldDataset/).

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |
| [LosSimpsonsPMC.ipynb](LosSimpsonsPMC.ipynb) | CPU: TensorFlow, en Docker o en un Python con TensorFlow | 64 × 64 |

Los dos tienen la misma estructura que los del dataset anterior.

## Qué cambia respecto del dataset anterior

1. **Partición propia, con todo el dataset.** El notebook de GPU reparte todas las imágenes originales de
   cada personaje en entrenamiento (70 %), validación (20 %) y test (10 %), de forma estratificada y con
   semilla fija, sin tope por personaje. El desbalance (35:1, de 3.060 imágenes de Homer a 87 de Louie) se
   compensa con pesos por clase suavizados, la raíz cuadrada de los balanceados. El notebook de CPU
   conserva todavía el esquema anterior: 70/15/15 y un máximo de 300 originales por personaje para entrenar.
2. **Imágenes alteradas.** Un cuarto de las imágenes de cada personaje tiene una variante con un giro de 10
   a 20°, un cambio de color (brillo, contraste, saturación o tono) y un filtro de textura (desenfoque,
   nitidez, ruido o realce de bordes). Solo entran al entrenamiento, y solo las de originales que quedan en
   él.
3. **Imágenes parecidas.** Las imágenes casi idénticas a otra del mismo personaje (fotogramas seguidos de
   una escena, recortes repetidos) entran con una versión filtrada de la misma forma, en cualquier
   conjunto: así el modelo no puede acertar en test solo porque vio casi la misma imagen al entrenar.
4. **Descriptores de color y textura.** El modelo base no recibe los píxeles, sino unos 2.460 descriptores
   por imagen: histogramas HSV y Lab, LBP, filtros de Gabor y HOG. La experimentación los compara con los
   píxeles crudos (R1), con el color solo (R2) y con la textura sola (R3), además de las variantes de
   arquitectura E1–E5 de siempre. El notebook de GPU suma dropout (E6), pesos por clase (P1, P2),
   suavizado de etiquetas (S1), etiquetas de escena (S2), entrenamiento sin alteradas (A1) y una
   combinación de los cambios que mejoran al modelo base (C1).
5. **Escenas con varios personajes (GPU).** Los fotogramas de episodio indican qué otros personajes salen
   en la misma escena: se usan como etiqueta secundaria al entrenar y para medir una accuracy tolerante,
   que da por buena la predicción de un personaje que también está en pantalla.

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

## Paso 2: generar las imágenes alteradas y parecidas

Desde la raíz del repositorio:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
```

El script hace dos cosas, y cada imagen que genera lleva un giro de 10 a 20° (con las esquinas vacías
recortadas), un cambio de color y un filtro de textura:

- **Alteradas.** Elige al azar un cuarto de las imágenes de cada personaje y guarda junto a cada una su
  variante, `<original>__alterada.<ext>`, con el mismo formato.
- **Parecidas.** Compara entre sí las imágenes de cada personaje (miniaturas a color de 16 × 16,
  similitud coseno ≥ 0,90) y agrupa las casi idénticas. En cada grupo una queda intacta y cada una de las
  demás recibe su versión filtrada, `<original>__parecida.<ext>`, que se vuelve a sortear hasta que deja de
  parecerse a las del grupo. Los notebooks usan esa versión en lugar de la original.

Deja un registro de cada transformación en `_alteraciones.csv` y `_parecidas.csv`, en la raíz del
dataset. Con el dataset del 15-10-2026 (39.000 originales) generó 9.756 alteradas y 7.894 parecidas, de
5.573 grupos, en unos 3 minutos, y bajó los pares casi idénticos de 11.223 a 237.

- **Las originales no se modifican nunca**: el script solo las lee.
- **Es reproducible**: con la misma semilla genera exactamente los mismos archivos, byte a byte.
- **No deja rastro**: cada ejecución borra primero lo que generó la anterior, así que se puede repetir
  sin acumular archivos.
- `--limpiar` borra todo lo generado y deja el dataset como se descargó.
- `--fraccion`, `--umbral` y `--semilla` cambian la proporción de alteradas (0,25 por defecto), la
  similitud a partir de la cual dos imágenes son parecidas (0,90) y la semilla (42).

Hay que volver a ejecutarlo cada vez que se agregan o quitan imágenes del dataset: qué imágenes son
parecidas depende de las demás de su carpeta.

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

- El notebook de GPU se ejecutó de principio a fin con el dataset real (64 personajes, 39.000 originales)
  el 15-10-2026: unos 25 minutos, de los que cerca de 7 son la carga de las imágenes y el cálculo de los
  descriptores en la CPU. El de CPU solo se ha probado con un **dataset sintético** que imita la
  estructura del real: 20 personajes, archivos `.png` y `.webp`, nombres con `__` y nombres de otro
  personaje.
- Las celdas de "Lectura" marcadas como **pendientes** se escribieron para la corrida del dataset anterior
  y hay que reescribirlas con las cifras de la corrida nueva.
- El cálculo de descriptores es la etapa más lenta del pre-procesamiento. El KPI de costo de catalogación
  incluye ese cálculo, porque es parte del costo real de clasificar una imagen nueva.
