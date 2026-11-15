# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El notebook del dataset anterior (25 personajes, EP1)
quedó en el historial de Git (`PMC/notebooks/oldDataset/`, commit `4b8b80e`), y su informe en
[`../docs/oldDataset/Informe.md`](../docs/oldDataset/Informe.md). El informe de la EP3 está en
[`../docs/newDataset/Informe.md`](../docs/newDataset/Informe.md).

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU RTX 4060 Laptop: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |

Tiene la misma estructura que el notebook de GPU del dataset anterior: Keras 3 sobre PyTorch con un bucle
de entrenamiento propio y los datos enteros en la VRAM.

**Primero el modelo, después las cajas** (punto 6 y paso 4). El notebook entrena y evalúa con la imagen
completa, sin cajas, y al final guarda el modelo con todo lo necesario para usarlo fuera de él. Las cajas
alrededor de cada personaje las pone después `scripts/newDataset/detectar_cajas.py` con ese modelo; el
notebook no depende de ese script.

## Qué cambia respecto del dataset anterior

1. **Partición propia, con todo el dataset.** El notebook reparte todas las imágenes originales de
   cada personaje en entrenamiento (70 %), validación (20 %) y test (10 %), de forma estratificada y con
   semilla fija, sin tope por personaje. El desbalance (unos 20:1, de 3.118 imágenes de Homer a 151 de Maggie) se
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
   arquitectura E1–E5 de siempre. El notebook suma dropout (E6), pesos por clase (P1, P2),
   suavizado de etiquetas (S1), entrenamiento sin alteradas (A1) y una combinación de los cambios que mejoran al modelo base (C1).
   La EP3 suma las variantes de regularización G1 (sin dropout), G2 (sin Batch Normalization) y G3 (L2), un Grid
   Search de 36 configuraciones, la ROC-AUC, la auditoría (calibración, umbral de revisión y sesgos por subgrupo),
   SHAP y la comparación con la CNN, todo con el mismo presupuesto: 60 épocas y paciencia 12.
5. **Escenas con varios personajes.** Los fotogramas de episodio y las imágenes casi idénticas
   guardadas en la carpeta de otro personaje (gemelas) indican quién más sale en cada imagen. No cambian
   el acierto: solo alimentan la accuracy por escena, una referencia para comparar con la corrida del
   16-10-2026, en la que nombrar a cualquier personaje presente contaba como acierto.
6. **Primero el modelo, después las cajas.** El MLP aprende y se evalúa con la imagen completa, y en el
   test una predicción acierta solo si nombra al personaje de la carpeta. Al terminar, la celda «Selección
   del modelo final» guarda `mlp_final.keras` con su `.json`, su `_entrada.py` (las funciones de
   descriptores con que se entrenó) y su `_particion.csv`. Con esos archivos,
   `scripts/newDataset/detectar_cajas.py` encuentra después a cada personaje con un detector preentrenado
   (Grounding DINO) y el MLP nombra cada caja con su confianza (`Jasper Beardly · 92 %`), sin anotar nada
   a mano.

## Paso 1: colocar el dataset

```text
simpsons_dataset/
├── abraham_grampa_simpson/
├── agnes_skinner/
├── ...
└── waylon_smithers/
```

Por defecto los notebooks y el script buscan el dataset en `~/Desktop/simpsons_dataset`.
Si está en otra ruta, defínela en la variable de entorno `SIMPSONS_DATASET` antes de abrir Jupyter.

La etiqueta de cada imagen es siempre el nombre de su carpeta, y cada original se llama
`<personaje>_<número>.<ext>` (`homer_simpson_2294.jpg`). Antes los fotogramas de Frinkiac se llamaban por el
episodio y el instante (`apu_nahasapeemapetilon__S21E21__t0659367.jpg`), a veces con el nombre de otro personaje:
ese nombre anterior queda en `_renombres.csv`, en la raíz del dataset, y de ahí lo leen los notebooks para saber
quién más aparece en la escena.

Si se amplió el dataset con Frinkiac ([scripts/newDataset/frinkiac/](../../scripts/newDataset/frinkiac/README.md)), copia primero las aprobadas de `final/` a cada carpeta de personaje. Son originales y deben estar en su carpeta antes del paso 2, que genera las sintéticas a partir de ellas.

## Paso 2: renombrar las originales y generar las imágenes alteradas y parecidas

Desde la raíz del repositorio:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\newDataset\renombrar_imagenes.py             # vista previa
.\.venv-gpu\Scripts\python.exe scripts\newDataset\renombrar_imagenes.py --aplicar
.\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
```

`renombrar_imagenes.py` sin `--aplicar` solo muestra cómo quedaría cada carpeta. Con `--aplicar` borra las
alteradas y parecidas anteriores, deja cada original como `<personaje>_<número>` (las que ya lo cumplen
conservan su número) y anota cada cambio en `_renombres.csv`. `generar_imagenes_alteradas.py` se detiene si
alguna original no cumple ese formato, así que las variantes siempre llevan el nombre de su original.

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
  `gemelas_por_revisar/`, **fuera del dataset** (en el Escritorio, junto a la carpeta `simpsons_dataset`).
  Desde ahí el equipo decide a mano si la suma también a la carpeta de ese personaje. No propone una
  imagen que la carpeta del otro personaje ya tiene.

Deja un registro de cada transformación en `_alteraciones.csv`, `_parecidas.csv` y `_gemelas.csv`, en la
raíz del dataset. Con el dataset del 15-10-2026 (39.000 originales) generó 9.756 alteradas y 7.894
parecidas, de 5.573 grupos, en unos 3 minutos, y bajó los pares casi idénticos de 11.223 a 237. El
16-10-2026 (38.999 originales) encontró además 756 pares de gemelas en 1.174 imágenes y dejó 1.233 copias
para revisar. En las corridas del 16-10-2026 el dataset ya tenía 40.123 originales, con 10.043 alteradas,
9.041 parecidas y 3.326 pares de gemelas. Con el dataset del 15-11-2026 (56.268 originales) hay 14.077 alteradas,
9.040 parecidas y 6.101 pares de gemelas.

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

## Paso 3: ejecutar el notebook

La ejecución es local, en la RTX 4060 Laptop: se acordó con el profesor en lugar de Google Colab.

**En Windows, con la GPU.** El entorno `.venv-gpu` se crea con `requirements-gpu-windows.txt` (sus pasos de
instalación quedaron en `PMC/notebooks/oldDataset/README_GPU.md`, en el historial de Git, commit `4b8b80e`), más una
dependencia:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-newDataset.txt
.\PMC\scripts\newDataset\ejecutar_notebook_gpu.ps1
```

**Sin VS Code**, desde `PMC/notebooks/`, el notebook se puede ejecutar completo y guardar en otro archivo. Con
`SIMPSONS_PRUEBA=1` corre de punta a punta en un par de minutos, con 40 imágenes por personaje, 2 épocas y una grilla
de 2 configuraciones, y escribe en carpetas `prueba/` aparte, para comprobarlo antes de la corrida completa:

```powershell
$env:KERAS_BACKEND = "torch"
$env:SIMPSONS_PRUEBA = "1"   # quitar para la corrida completa
..\..\.venv-gpu\Scripts\python.exe -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 `
    --ExecutePreprocessor.kernel_name=python3 --output corrida.ipynb LosSimpsons_GPU_4060M.ipynb
```

**En Docker.** Copia `.env.example` como `.env`, pon en `SIMPSONS_DATASET_HOST` la ruta del dataset y
levanta el contenedor:

```powershell
docker compose up --build
```

Después abre <http://localhost:8888> y entra en `PMC/notebooks/`.

## Paso 4 (después de entrenar, opcional): cada personaje en su caja

Va después del notebook y no lo cambia: el notebook no usa este script ni lee nada de lo que deja. Usa el
modelo final que guarda la celda «Selección del modelo final» en `PMC/models_gpu/newDataset/`:

| Archivo | Contenido |
|---|---|
| `mlp_final.keras` | La red entrenada |
| `mlp_final.json` | Clases, resolución y representación de la entrada |
| `mlp_final_entrada.py` | Las funciones de descriptores con que se entrenó, copiadas del notebook. El notebook comprueba que dan exactamente la entrada de test |
| `mlp_final_particion.csv` | El conjunto (train, val o test) de cada imagen original |

Desde la raíz del repositorio, la primera vez se instala `transformers` en el mismo `.venv-gpu`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-cajas.txt
.\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py --modelo mlp ruta\a\imagen.jpg ruta\a\carpeta
.\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py --modelo mlp --dataset --muestra 5
```

Cada caja sale de dos etapas, sin anotar nada a mano:

1. **Dónde.** Grounding DINO (`IDEA-Research/grounding-dino-base`, ~0,9 GB que se bajan una vez) encuentra
   a cada personaje animado del fotograma. No sabe quién es cada uno: solo dónde está. Si no encuentra a
   nadie, busca otra vez con un umbral más bajo.
2. **Quién.** Cada caja se recorta ampliada un 10 % por lado y llevada a un cuadrado, se redimensiona a
   128 × 128, se prepara con `mlp_final_entrada.py` y el MLP la nombra con su personaje más probable y su
   confianza: `Jasper Beardly · 92 %`. También nombra la imagen completa.

Deja en `PMC/models_gpu/newDataset/cajas_mlp_final/` (o en `--salida`):

- `cajas.csv`: una fila por caja, con sus coordenadas, el puntaje del detector, el personaje, la confianza y
  las otras dos candidatas. La caja 0 de cada imagen es la imagen completa.
- `imagenes/`: cada imagen con sus cajas; en verde las que nombran al personaje de la carpeta, en azul las
  demás y con un marco naranjo las imágenes en que el detector no encontró a nadie.
- `hojas/<personaje>.jpg`: con `--dataset`, una hoja de contacto por personaje.
- `resumen.json`: modelo, detector, umbrales y conteos, y en cuántas imágenes alguna caja o la imagen
  completa nombran al personaje de la carpeta.

Más opciones:

- Con `--dataset` toma originales del dataset y, por defecto, solo las de test (`--conjunto`), que el modelo
  no vio al entrenar; `--muestra N` limita a N por personaje y `--personajes` a algunos personajes.
- El MLP aprendió con fotogramas completos, así que nombra un recorte con menos seguridad que la imagen entera, y con descriptores de color le cuesta separar dentro de una caja a personajes de paleta parecida. Con `--modelo cnn` el mismo script usa la CNN.
- Se puede detener con Ctrl+C: `cajas.csv` queda con las imágenes ya procesadas.
- Pruebas del script, sin GPU: `.venv-frinkiac\Scripts\python.exe -m pytest scripts\newDataset\tests -q`.
## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` de la corrida y el modelo final con sus archivos (`mlp_final.keras`, `.json`, `_entrada.py` y `_particion.csv`) | `PMC/models_gpu/newDataset/`, ignorados por Git |
| Historial de cada configuración | `PMC/models_gpu/newDataset/historiales/`, ignorados por Git |
| Resumen de la corrida | `PMC/resultados/newDataset/GPU/rendimiento_gpu_128x128.json`; cada corrida reemplaza el de la anterior |
| Figuras del informe | `PMC/docs/newDataset/figuras/`, guardadas por el notebook al ejecutarse |
| Recreación de cada personaje (anexo del notebook) | `PMC/resultados/newDataset/recreaciones/` |
| Presentación de resultados | `PMC/presentacion_resultados_PMC.html`, generada con `scripts/newDataset/generar_presentacion.py --modelo mlp` |
| Cajas de cada personaje, después del notebook (paso 4) | `PMC/models_gpu/newDataset/cajas_mlp_final/`, ignoradas por Git |

`mlp_final.json` guarda la representación, la resolución y el orden de las clases del modelo final, y
`mlp_final_entrada.py` repite su pre-procesamiento. La sección «Probar el modelo con una imagen cualquiera»
y el script de cajas los usan para clasificar sin volver a entrenar.

## Resultados de la corrida del 15-11-2026

El notebook de GPU se ejecutó completo con el dataset real (64 personajes, 56.268 originales) a 128 × 128, con la
imagen completa. Sus salidas guardadas son las de esa corrida.

| Corrida | Configuración final | Accuracy | Top-3 | F1 macro | ROC-AUC macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 128 × 128 | GS01 - solo color, 3 capas, dropout 0,2 | **0,556** | **0,732** | **0,557** | **0,940** | 2 h 2 min 44 s |

- **Representación.** El color solo (R2) supera a color y textura por 6,9 puntos de F1 macro de validación; la
  textura sola (−17,4) y los píxeles crudos (−24,6) quedan muy atrás.
- **Regularización.** Sin dropout (G1) el F1 macro cae 7,5 puntos; sin suavizado (S1), 3,5; sin alteradas (A1), 2,6;
  sin Batch Normalization (G2), 1,1. L2 (G3) resta 0,7 y el dropout 0,5 (E6) no cambia nada.
- **Optimización.** Gana GS01, del Grid Search: la entrada y las tres capas de C1 con dropout 0,2 en lugar de 0,5,
  5,8 puntos sobre C1 y 10,5 sobre la base. El dropout es lo único que mueve la grilla; 35 de sus 36 configuraciones
  llegaron al tope de 60 épocas.
- **Auditoría.** ECE de 0,044 (levemente sobreconfiado); con el umbral de 0,958 elegido en validación resuelve solo
  el 14,3 % del test al 89,7 % de acierto. Accuracy de 0,627 en Kaggle y de 0,508 en los fotogramas de episodio, y
  11,3 puntos menos en la familia Simpson que en el resto.
- **Frente a la CNN** en la GPU a 128 × 128 (G4, corrida del 14-11-2026): accuracy −22,0, top-3 −17,7 y F1 macro
  −19,8; la CNN cataloga 3.567 imágenes por segundo de extremo a extremo y el MLP 1.360.

La [presentación de resultados](../presentacion_resultados_PMC.html) y el [informe](../docs/newDataset/Informe.md)
usan las cifras de esta corrida.

## Estado y tiempos de referencia

- La corrida del 15-11-2026 tardó 2 h 2 min en la GPU: 11,4 minutos en cargar las imágenes y 2,7 en calcular los
  descriptores, ambos en la CPU, y 1 h 46 min en entrenar las 54 configuraciones (76 minutos de ellos, el Grid
  Search). Las 84 celdas de código terminaron sin errores.
- Después de la corrida solo se volvió a ejecutar la celda de los KPIs, con la medición del costo corregida: la
  primera versión medía una sola vez en frío y daba 988 imágenes por segundo; la mediana de tres repeticiones,
  después de un calentamiento, da 1.360.
- La red es pequeña para la tarjeta: a batch 128 la GPU trabaja el 27,8 % del tiempo. El cálculo de descriptores
  es la etapa más lenta de catalogar una imagen nueva, y el KPI de costo lo incluye.
- El anexo «Corrida del 15-11-2026 frente a la del 09-11-2026», al final del notebook, compara las dos corridas.
