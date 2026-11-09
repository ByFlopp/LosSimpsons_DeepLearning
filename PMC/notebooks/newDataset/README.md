# Notebooks del dataset nuevo (64 personajes)

Esta carpeta contiene la versión del MLP para el dataset ampliado por el equipo: **una carpeta por
personaje**, sin partición previa en `train` y `test`. El trabajo con el dataset anterior (25 personajes)
se conserva intacto en [`../oldDataset/`](../oldDataset/).

| Notebook | Entorno | Resolución de trabajo |
|---|---|---|
| [LosSimpsons_GPU_4060M.ipynb](LosSimpsons_GPU_4060M.ipynb) | GPU RTX 4060 Laptop: Keras 3 sobre PyTorch, en Windows nativo o en Docker | 128 × 128 |

Tiene la misma estructura que el notebook de GPU del dataset anterior: Keras 3 sobre PyTorch con un bucle
de entrenamiento propio y los datos enteros en la VRAM.

**Desde el 07-11-2026 el notebook trabaja con la caja de cada personaje** (punto 6 y paso 3): calcula
los descriptores sobre el recorte del personaje de la carpeta y una predicción acierta solo si nombra a ese
personaje.

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
   guardadas en la carpeta de otro personaje (gemelas) indican quién más sale en cada imagen. Ya no
   cambian el acierto: solo alimentan la accuracy por escena, una referencia para comparar con la corrida
   con la imagen completa, en la que nombrar a cualquier personaje presente contaba como acierto.
6. **Cada personaje en su caja.** `scripts/newDataset/detectar_cajas.py` encuentra a cada
   personaje con un detector preentrenado (Grounding DINO) y una CNN le pone nombre a cada caja, sin anotar
   nada a mano. El MLP calcula sus descriptores sobre el recorte del personaje de la carpeta, y en el test
   una predicción acierta solo si nombra a ese personaje. Una sección nueva muestra, en los fotogramas con
   varios personajes, cada caja con el nombre y la confianza que le da el modelo
   (`Jasper Beardly · 92 %`).

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

Si se amplió el dataset con Frinkiac ([scripts/newDataset/frinkiac/](../../../scripts/newDataset/frinkiac/README.md)), copia primero las aprobadas de `final/` a cada carpeta de personaje. Son originales y deben estar en su carpeta antes del paso 2, que genera las sintéticas a partir de ellas.

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

## Paso 3: detectar a cada personaje (cajas)

Lo necesitan los notebooks de GPU de la CNN y del MLP. Desde la raíz del repositorio, la primera vez se
instala `transformers` en el mismo `.venv-gpu`:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-cajas.txt
.\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py --muestra 5   # piloto: 5 por personaje
.\.venv-gpu\Scripts\python.exe scripts\newDataset\detectar_cajas.py               # todo el dataset
```

Cada caja sale de dos etapas, sin anotar nada a mano:

1. **Dónde.** Grounding DINO (`IDEA-Research/grounding-dino-base`, ~0,9 GB que se bajan una vez) encuentra
   a cada personaje animado del fotograma. No sabe quién es cada uno: solo dónde está.
2. **Quién.** La CNN final del proyecto pone nombre a cada caja con su confianza. Como esa CNN se entrenó
   con fotogramas completos, el script primero afina una copia con las cajas sin dudas (la única de la
   imagen, o una que la CNN reconoce claramente) y con ella vuelve a nombrarlas todas. La copia queda en
   `CNN/models_gpu/newDataset/cajas/maestra_recortes.keras` y solo sirve para anotar.

De cada imagen entra al notebook la caja del personaje de la carpeta: la que la CNN reconoce mejor como él.
Si no hay cajas o nadie lo reconoce, entra la imagen completa, como antes. Las alteradas y parecidas no
pasan por el detector: usan la caja de su original, girada igual que la imagen.

Deja en la raíz del dataset `_cajas.csv` (una fila por imagen que usan los notebooks), `_cajas_candidatas.csv`
(todas las cajas, con el personaje y la confianza de cada una) y `_cajas_detecciones.jsonl` (una caché).
Junto al dataset, nunca dentro, deja `cajas_por_revisar/<personaje>.jpg`: una hoja por personaje con sus
cajas rotuladas, para mirar sin anotar nada. Al terminar imprime la cobertura de los 64 personajes.

- En la RTX 4060 procesa unas 4,5 originales por segundo: del orden de 2 a 2,5 horas para todo el dataset,
  más unos minutos de afinado. Se puede detener con Ctrl+C y retomar: cada ejecución solo procesa lo nuevo.
- Hay que repetirlo **después del paso 2** cada vez que cambia el dataset; las originales ya procesadas no
  vuelven a pasar por el detector.
- `--reafinar` vuelve a afinar la copia de la CNN, `--rehacer` vuelve a detectar y `--personajes` limita
  la detección a algunos personajes.
- Pruebas de la geometría de las cajas, sin GPU: `.venv-frinkiac\Scripts\python.exe -m pytest scripts\newDataset\tests -q`.

## Paso 4: ejecutar el notebook

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

Después abre <http://localhost:8888> y entra en `PMC/notebooks/newDataset/`.

## Dónde quedan los resultados

| Qué | Dónde |
|---|---|
| Modelos `.keras` y `mlp_final.json` con cajas | `PMC/models_gpu/newDataset/cajas/` (GPU), ignorados por Git |
| Modelos `.keras` y `mlp_final.json` con la imagen completa | `PMC/models_gpu/newDataset/` (corrida anterior), ignorados por Git |
| Resumen de la corrida con cajas | `PMC/resultados/newDataset/cajas/GPU/rendimiento_gpu_128x128.json` |
| Resumen de la corrida con la imagen completa | `PMC/resultados/newDataset/GPU/rendimiento_gpu_128x128.json` |

`mlp_final.json` guarda la representación, la resolución y el orden de las clases del modelo final. La
sección «Probar el modelo con una imagen cualquiera» lo usa para repetir el mismo pre-procesamiento sin
volver a entrenar.

## Resultados de la corrida del 16-10-2026, con la imagen completa

El notebook se ejecutó completo con el dataset real (64 personajes, 40.123 originales) en la GPU a
128 × 128, todavía con la imagen completa. La accuracy usa el criterio de acierto de entonces (el personaje
de la carpeta o uno presente en la escena); la estricta solo acepta el de la carpeta y es la comparable con
la corrida con cajas.

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
  aparecer al ejecutarlo a 128 × 128 con cajas. Las lecturas de resultados todavía describen la corrida con
  la imagen completa y se revisan con las salidas nuevas.
- El cálculo de descriptores es la etapa más lenta del pre-procesamiento. El KPI de costo de catalogación
  incluye ese cálculo, porque es parte del costo real de clasificar una imagen nueva.
