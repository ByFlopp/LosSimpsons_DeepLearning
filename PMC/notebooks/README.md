# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El trabajo con el dataset anterior (25 personajes)
se conserva intacto en [`../oldDataset/`](../oldDataset/).

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
   arquitectura E1–E5 de siempre. El notebook suma dropout (E6), pesos por clase (P1, P2),
   suavizado de etiquetas (S1), entrenamiento sin alteradas (A1) y una combinación de los cambios que mejoran al modelo base (C1).
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

La etiqueta de cada imagen es siempre el nombre de su carpeta. Algunas carpetas contienen fotogramas cuyo
nombre de archivo es el de otro personaje: los añadió el equipo a propósito, porque en ellos el
protagonista es el personaje de la carpeta.

Si se amplió el dataset con Frinkiac ([scripts/newDataset/frinkiac/](../../scripts/newDataset/frinkiac/README.md)), copia primero las aprobadas de `final/` a cada carpeta de personaje. Son originales y deben estar en su carpeta antes del paso 2, que genera las sintéticas a partir de ellas.

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
  `gemelas_por_revisar/`, **fuera del dataset** (en el Escritorio, junto a la carpeta `simpsons_dataset`).
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

## Paso 3: ejecutar el notebook

**En Windows, con la GPU.** El entorno `.venv-gpu` es el mismo del dataset anterior
([`../oldDataset/README_GPU.md`](../oldDataset/README_GPU.md)), más una dependencia:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-newDataset.txt
.\PMC\scripts\newDataset\ejecutar_notebook_gpu.ps1
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
| Resumen de la corrida | `PMC/resultados/newDataset/GPU/rendimiento_gpu_128x128.json`; cada corrida reemplaza el de la anterior |
| Cajas de cada personaje, después del notebook (paso 4) | `PMC/models_gpu/newDataset/cajas_mlp_final/`, ignoradas por Git |

`mlp_final.json` guarda la representación, la resolución y el orden de las clases del modelo final, y
`mlp_final_entrada.py` repite su pre-procesamiento. La sección «Probar el modelo con una imagen cualquiera»
y el script de cajas los usan para clasificar sin volver a entrenar.

## Resultados de la corrida del 16-10-2026, con la imagen completa

El notebook se ejecutó completo con el dataset real (64 personajes, 40.123 originales) en la GPU a
128 × 128, con la imagen completa, como ahora. La accuracy usa el criterio de acierto de entonces (el personaje
de la carpeta o uno presente en la escena); la estricta solo acepta el de la carpeta y es la comparable con
el criterio actual.

| Corrida | Configuración final | Accuracy | Estricta | Top-3 | F1 macro | Duración |
|---|---|---:|---:|---:|---:|---:|
| GPU 128 × 128 | C1 - combinación | 0,586 | 0,539 | 0,751 | 0,559 | 28 min 14 s |

- **El color solo gana.** El modelo final usa solo los 370 histogramas de color, con las tres capas de E2.
- **La red es pequeña para la tarjeta.** A batch 128 la GPU trabaja el 33,7 % del tiempo, y el
  pre-procesamiento, que corre en la CPU, toma 11 minutos: el 39 % de la corrida.
- **Los KPIs de calidad se cumplen**, también con el criterio estricto (0,539 de accuracy).
- **Frente a la corrida del 15-10-2026** (0,643 de accuracy estricta), la caída a 0,539 con la misma
  configuración mide sobre todo la fuga que se eliminó: aquel test no filtraba las imágenes casi idénticas
  a otras de entrenamiento ni tenía alteradas.

El detalle, con los tiempos de cada etapa, está al final del notebook, en «Corrida con la imagen completa
del 16-10-2026».

## Estado y tiempos de referencia

- La corrida del 16-10-2026, con la imagen completa, tardó 28 minutos en la GPU. La carga de las imágenes y
  el cálculo de los descriptores, que se hacen en la CPU, tomaron 11 minutos.
- El notebook no tiene salidas guardadas: las de su última corrida eran de 64 × 64 y se borraron. Vuelven a
  aparecer al ejecutarlo. Las lecturas de resultados todavía describen la corrida del 16-10-2026 y se
  revisan con las salidas nuevas.
- El cálculo de descriptores es la etapa más lenta del pre-procesamiento. El KPI de costo de catalogación
  incluye ese cálculo, porque es parte del costo real de clasificar una imagen nueva.
