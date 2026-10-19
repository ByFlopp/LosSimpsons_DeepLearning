# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El trabajo con el dataset anterior (25 personajes)
se conserva intacto en [`../oldDataset/`](../oldDataset/).

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |
| [LosSimpsonsPMC.ipynb](LosSimpsonsPMC.ipynb) | CPU: Keras 3 sobre PyTorch con la GPU oculta, en el mismo `.venv-gpu` o en Docker | 128 × 128 |

Los dos tienen la misma estructura que los del dataset anterior y siguen el mismo proceso: la misma
partición, los mismos descriptores, las mismas etiquetas de escena, los mismos experimentos, el mismo
criterio de acierto y el mismo framework, Keras 3 sobre PyTorch con un bucle de entrenamiento propio. Solo
cambia el dispositivo. El de CPU no usa TensorFlow: su primera celda fija `KERAS_BACKEND=torch` y oculta la
GPU (`CUDA_VISIBLE_DEVICES=-1`) antes de importar PyTorch y Keras.

## Qué cambia respecto del dataset anterior

1. **Partición propia, con todo el dataset.** Los notebooks reparten todas las imágenes originales de
   cada personaje en entrenamiento (70 %), validación (20 %) y test (10 %), de forma estratificada y con
   semilla fija, sin tope por personaje. El desbalance (33:1, de 3.118 imágenes de Homer a 94 de Louie) se
   compensa con pesos por clase suavizados, la raíz cuadrada de los balanceados.
2. **Imágenes alteradas.** Un cuarto de las imágenes de cada personaje tiene una variante con un giro de 10
   a 20°, un cambio de color (brillo, contraste, saturación o tono) y un filtro de textura (desenfoque,
   nitidez, ruido o realce de bordes). Cada variante entra al mismo conjunto que su original
   (entrenamiento, validación o test), así los tres conjuntos tienen las mismas condiciones.
3. **Imágenes parecidas.** Las imágenes casi idénticas a otra del mismo personaje (fotogramas seguidos de
   una escena, recortes repetidos) entran con una versión filtrada de la misma forma, en cualquier
   conjunto: así el modelo no puede acertar en test solo porque vio casi la misma imagen al entrenar.
4. **Descriptores de color y textura.** El modelo base no recibe los píxeles, sino unos 2.460 descriptores
   por imagen: histogramas HSV y Lab, LBP, filtros de Gabor y HOG. La experimentación los compara con los
   píxeles crudos (R1), con el color solo (R2) y con la textura sola (R3), además de las variantes de
   arquitectura E1–E5 de siempre. Los notebooks suman dropout (E6), pesos por clase (P1, P2),
   suavizado de etiquetas (S1), etiquetas de escena (S2), entrenamiento sin alteradas (A1) y una
   combinación de los cambios que mejoran al modelo base (C1).
5. **Escenas con varios personajes.** Los fotogramas de episodio y las imágenes casi idénticas
   guardadas en la carpeta de otro personaje (gemelas) indican quién más sale en cada imagen. Se usan como
   etiqueta secundaria al entrenar y, en el test, una predicción que nombra a cualquier personaje presente
   cuenta como acierto en todas las métricas y en los KPIs. La accuracy estricta, que solo acepta el
   personaje de la carpeta, se informa como referencia. La validación sigue usando la etiqueta de la
   carpeta. Además, cada error de test con confianza mayor que 0,80 se verifica por imagen: si la carpeta
   del personaje predicho tiene una imagen de la misma toma (similitud de miniaturas ≥ 0,80), ese
   personaje sale en la imagen y la predicción cuenta como acierto por escena; si no, es un error real.

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

El script hace tres cosas. Cada imagen que genera en las dos primeras lleva un giro de 10 a 20° (con las
esquinas vacías recortadas), un cambio de color y un filtro de textura:

- **Alteradas.** Elige al azar un cuarto de las imágenes de cada personaje y guarda junto a cada una su
  variante, `<original>__alterada.<ext>`, con el mismo formato.
- **Parecidas.** Compara entre sí las imágenes de cada personaje (miniaturas a color de 16 × 16,
  similitud coseno ≥ 0,90) y agrupa las casi idénticas. En cada grupo una queda intacta y cada una de las
  demás recibe su versión filtrada, `<original>__parecida.<ext>`, que se vuelve a sortear hasta que deja de
  parecerse a las del grupo. Los notebooks usan esa versión en lugar de la original.
- **Gemelas entre personajes.** Compara con las mismas miniaturas las imágenes de carpetas *distintas*:
  una imagen casi idéntica guardada en la carpeta de otro personaje es la misma escena, así que los dos
  personajes aparecen en ella, y los notebooks cuentan como acierto que el modelo nombre a cualquiera de
  los dos. Además copia cada imagen, con su nombre original, a la subcarpeta del otro personaje en
  `gemelas_por_revisar/`, **fuera del dataset** (junto a la carpeta `simpsons_dataset/simpsons_dataset`).
  Desde ahí el equipo decide a mano si la suma también a la carpeta de ese personaje. No propone una
  imagen que la carpeta del otro personaje ya tiene.

Deja un registro de cada transformación en `_alteraciones.csv`, `_parecidas.csv` y `_gemelas.csv`, en la
raíz del dataset. Con el dataset del 15-10-2026 (39.000 originales) generó 9.756 alteradas y 7.894
parecidas, de 5.573 grupos, en unos 3 minutos, y bajó los pares casi idénticos de 11.223 a 237. El
16-10-2026 (38.999 originales) encontró además 756 pares de gemelas en 1.174 imágenes y dejó 1.233 copias
para revisar. En las corridas del 16-10-2026 el dataset ya tenía 40.123 originales, con 10.043 alteradas,
9.041 parecidas y 3.326 pares de gemelas.

- **Las originales no se modifican nunca**: el script solo las lee. Tampoco mueve nada a otra carpeta de
  personaje: eso lo hace el equipo desde `gemelas_por_revisar/`.
- **Es reproducible**: con la misma semilla genera exactamente los mismos archivos, byte a byte.
- **No deja rastro**: cada ejecución borra primero lo que generó la anterior, incluidas las copias de
  `gemelas_por_revisar/`, así que se puede repetir sin acumular archivos. Por eso esa carpeta no es lugar
  para guardar nada propio: solo se vacía si lleva la marca `_generado_por_script.txt` del script.
- `--limpiar` borra todo lo generado y deja el dataset como se descargó.
- `--fraccion`, `--umbral`, `--umbral-gemelas` y `--semilla` cambian la proporción de alteradas (0,25 por
  defecto), la similitud a partir de la cual dos imágenes del mismo personaje son parecidas (0,90), la de
  dos imágenes de personajes distintos para ser gemelas (0,90) y la semilla (42). `--revision` cambia la
  carpeta de revisión, que nunca puede quedar dentro del dataset.

Hay que volver a ejecutarlo cada vez que se agregan o quitan imágenes del dataset: qué imágenes son
parecidas o gemelas depende de las demás.

Si se usa Docker en lugar de `.venv-gpu`, la misma imagen sirve para ejecutar el script montando el
dataset con permiso de escritura, y la carpeta de revisión para que las copias queden en el equipo:

```powershell
docker run --rm -v "${PWD}:/workspace" -v "C:/ruta/al/simpsons_dataset:/data/simpsons_dataset" `
  -v "C:/ruta/a/gemelas_por_revisar:/data/gemelas_por_revisar" `
  -w /workspace simpsons-pmc:gpu python scripts/newDataset/generar_imagenes_alteradas.py
```

## Paso 3: ejecutar un notebook

**En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del dataset anterior
([`../oldDataset/README_GPU.md`](../oldDataset/README_GPU.md)), más una dependencia:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-newDataset.txt
.\PMC\scripts\newDataset\ejecutar_notebook_gpu.ps1
```

**En Windows, en la CPU.** El mismo entorno `.venv-gpu`, sin TensorFlow: abre `LosSimpsonsPMC.ipynb` con
ese kernel y ejecútalo desde la primera celda, que deja Keras sobre PyTorch y oculta la GPU antes de
importar nada. Si el kernel ya había importado `torch` o `keras` con otra configuración, la celda pide
reiniciarlo.

**En Docker.** Copia `.env.example` como `.env`, pon en `SIMPSONS_DATASET_HOST` la ruta del dataset y
levanta el contenedor:

```powershell
docker compose up --build
```

Después abre <http://localhost:8888> y entra en `PMC/notebooks/newDataset/`. Los dos notebooks usan Keras
sobre PyTorch; el de CPU oculta la GPU desde su primera celda.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` y `mlp_final.json` | `PMC/models_gpu/newDataset/` (GPU) y `PMC/models/newDataset/` (CPU), ignorados por Git |
| Resumen de rendimiento de cada corrida | `PMC/resultados/newDataset/<CPU\|GPU>/rendimiento_<entorno>_<resolución>.json` |

`mlp_final.json` guarda la representación, la resolución y el orden de las clases del modelo final. La
sección «Probar el modelo con una imagen cualquiera» lo usa para repetir el mismo pre-procesamiento sin
volver a entrenar.

## Resultados de las corridas del 16-10-2026

Los dos notebooks se ejecutaron completos con el dataset real (64 personajes, 40.123 originales), cada uno
a 32 × 32, 64 × 64 y 128 × 128. La accuracy usa el criterio de acierto del proyecto (el personaje de la
carpeta o uno presente en la escena); la estricta solo acepta el de la carpeta.

| Corrida | Configuración final | Accuracy | Estricta | Top-3 | F1 macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 32 × 32 | R2 - solo color | 0,541 | 0,501 | 0,724 | 0,515 | 19 min 36 s |
| GPU 64 × 64 | R2 - solo color | 0,550 | 0,507 | 0,731 | 0,522 | 23 min 19 s |
| GPU 128 × 128 | C1 - combinación | 0,586 | 0,539 | 0,751 | 0,559 | 28 min 14 s |
| CPU 32 × 32 | R2 - solo color | 0,533 | 0,492 | 0,718 | 0,510 | 30 min 04 s |
| CPU 64 × 64 | C1 - combinación | 0,584 | 0,537 | 0,751 | 0,567 | 37 min 44 s |
| CPU 128 × 128 | R2 - solo color | 0,561 | 0,516 | 0,734 | 0,538 | 52 min 44 s |

- **El dispositivo cambia el tiempo, no el modelo.** Con la misma configuración y resolución (R2 a
  32 × 32), GPU y CPU quedan a menos de un punto de accuracy. La mayor fuente de variación es qué
  configuración gana la validación: las dos corridas que eligieron C1 son las mejores en test.
- **El color solo gana en todas las corridas.** R2 (370 histogramas de color) supera a la combinación de
  color y textura del modelo base; la textura sola y los píxeles crudos quedan muy atrás.
- **Más resolución ayuda poco.** De 32 × 32 a 128 × 128, la accuracy de R2 sube unos 2 a 3 puntos, pero
  la carga de imágenes y el cálculo de descriptores pasan de unos 2,6 minutos a entre 9 y 11. 64 × 64 es el
  mejor equilibrio.
- **La GPU acelera el entrenamiento entre 1,7 y 2,7 veces, pero la corrida completa solo entre 1,5 y 1,9.**
  La red es pequeña para la tarjeta (a batch 128 la GPU trabaja entre un 22 y un 34 % del tiempo) y el
  pre-procesamiento corre en la CPU en las dos.
- **Los KPIs se cumplen en las seis corridas.** Con el criterio estricto, a 32 × 32 la accuracy queda en
  el umbral de 0,50 o por debajo.
- **Frente a la corrida del 15-10-2026** (0,643 de accuracy estricta), la caída a 0,539 con la misma
  configuración mide sobre todo la fuga que se eliminó: aquel test no filtraba las imágenes casi idénticas
  a otras de entrenamiento ni tenía alteradas.

El detalle, con los tiempos de cada etapa, está al final de cada notebook, en «Comparación de las corridas
del 16-10-2026».

## Estado y tiempos de referencia

- Los dos notebooks se ejecutaron de principio a fin con el dataset real el 16-10-2026, a tres
  resoluciones. Con la GPU, una corrida tarda de 20 a 28 minutos; con la CPU, de 30 a 53. La carga de las
  imágenes y el cálculo de los descriptores, que se hacen en la CPU en los dos notebooks, toman de 2,6 a 11
  minutos según la resolución.
- Las salidas guardadas en cada notebook son las de su última corrida: 64 × 64 en el de GPU y 32 × 32 en
  el de CPU, aunque los dos tienen hoy `ALTO_OBJETIVO = 128`. Las lecturas de resultados citan cifras de
  varias corridas e indican de cuál sale cada una.
- El cálculo de descriptores es la etapa más lenta del pre-procesamiento. El KPI de costo de catalogación
  incluye ese cálculo, porque es parte del costo real de clasificar una imagen nueva.
