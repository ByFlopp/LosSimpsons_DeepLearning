# Cómo ejecutar el notebook en GPU

`LosSimpsons_GPU_4060M.ipynb` corre en **Windows nativo**, sin WSL, sobre una RTX 4060
Laptop. Esta guía explica cómo montar el entorno, cómo comprobar que la GPU se está usando
de verdad y qué decisiones de rendimiento hay detrás.

El entorno anterior (WSL2 + TensorFlow), con el que se hicieron las corridas guardadas en
`resultados/oldDataset/GPU/`, está documentado en el [apéndice](#apéndice-entorno-anterior-wsl2--tensorflow).

---

## Por qué PyTorch y no TensorFlow

**TensorFlow no soporta GPU en Windows desde la versión 2.11.** Las opciones eran meterse
en WSL2 o cambiar de backend. Keras 3 puede apoyarse en PyTorch, que sí tiene CUDA nativo
en Windows, así que el notebook usa **Keras 3 sobre backend PyTorch**: el modelo se sigue
definiendo con la API de Keras (`Dense`, `BatchNormalization`, `Dropout`) y solo cambia el
motor que hay debajo.

Hay una segunda diferencia, más importante para el rendimiento: **el entrenamiento no usa
`model.fit()`**, sino un bucle escrito a mano sobre PyTorch. La razón está medida más
abajo, en [Por qué un bucle propio](#por-qué-un-bucle-propio-en-lugar-de-fit).

---

## Puesta en marcha paso a paso

**Todos los comandos se ejecutan desde la raíz del repositorio**, en PowerShell.

<details>
<summary><b>Desde cero, todo seguido</b> (pasos 2 a 6, para quien ya sabe lo que hace)</summary>

```powershell
py -3.13 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-windows.txt
.\.venv-gpu\Scripts\python.exe -m ipykernel install --user --name simpsons-gpu-win --display-name "Python 3.13 (Simpsons GPU Windows)"

$k = "$env:APPDATA\jupyter\kernels\simpsons-gpu-win\kernel.json"
$j = Get-Content $k -Raw | ConvertFrom-Json
$j | Add-Member -NotePropertyName env -NotePropertyValue @{ KERAS_BACKEND = "torch" } -Force
[System.IO.File]::WriteAllText($k, ($j | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
```

</details>

### Paso 0 — Enchufa el portátil

Con batería, el firmware recorta la potencia de la GPU y **el tiempo por época se duplica**.
Conecta la corriente y pon Windows en modo de máximo rendimiento: botón derecho en el icono
de batería → *Configuración de energía y suspensión* → **Modo de energía: Máximo
rendimiento**. El porqué, con cifras, está en [Enchufa el portátil](#enchufa-el-portátil).

### Paso 1 — Comprobar el driver NVIDIA

Basta con un driver reciente (580 o superior). **No hace falta instalar el CUDA Toolkit**: el
wheel de PyTorch trae sus propias bibliotecas CUDA.

```powershell
nvidia-smi
```

Debe listar la tarjeta. Si el comando no existe, instala o actualiza el driver de NVIDIA.

### Paso 2 — Crear el entorno virtual

Python 3.13 (también sirve 3.12). Comprueba primero qué tienes instalado:

```powershell
py -0p
py -3.13 -m venv .venv-gpu
```

### Paso 3 — Instalar PyTorch con CUDA

Se instala desde el índice de PyTorch, **no desde PyPI**: el paquete `torch` de PyPI es solo
para CPU.

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

Son unos 3 GB de descarga: tarda varios minutos.

### Paso 4 — Instalar el resto de dependencias

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu-windows.txt
```

Instala Keras 3, NumPy, pandas, matplotlib, seaborn, scikit-learn, Pillow y Jupyter.
**TensorFlow no se instala**: Keras 3 no lo necesita con el backend de PyTorch.

### Paso 5 — Registrar el kernel de Jupyter

```powershell
.\.venv-gpu\Scripts\python.exe -m ipykernel install --user --name simpsons-gpu-win --display-name "Python 3.13 (Simpsons GPU Windows)"
```

### Paso 6 — Fijar `KERAS_BACKEND` en el kernel

`KERAS_BACKEND` tiene que valer `torch` **antes** de que se importe `keras`, y la forma
fiable de garantizarlo es dejarlo en el bloque `env` del kernel. Este fragmento parchea el
`kernel.json` sin tener que editarlo a mano:

```powershell
$k = "$env:APPDATA\jupyter\kernels\simpsons-gpu-win\kernel.json"
$j = Get-Content $k -Raw | ConvertFrom-Json
$j | Add-Member -NotePropertyName env -NotePropertyValue @{ KERAS_BACKEND = "torch" } -Force
[System.IO.File]::WriteAllText($k, ($j | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
```

Comprobación:

```powershell
Get-Content "$env:APPDATA\jupyter\kernels\simpsons-gpu-win\kernel.json" | Select-String KERAS_BACKEND
```

> **Por qué `WriteAllText` y no `Set-Content -Encoding utf8`:** en PowerShell 5.1 esa opción
> escribe el archivo **con BOM**, y Jupyter no puede leer un `kernel.json` con BOM: el kernel
> deja de arrancar con `Unexpected UTF-8 BOM`. `WriteAllText` con `UTF8Encoding($false)` lo
> escribe sin BOM. Si ya te pasó, vuelve a ejecutar el fragmento corregido.

> **Ojo:** volver a ejecutar `ipykernel install` **sobrescribe el archivo y borra el bloque
> `env`**. Si el notebook empieza a quejarse de que el backend no es `torch`, es casi siempre
> esto: vuelve a lanzar el fragmento de arriba y reinicia el kernel.

### Paso 7 — Verificar el entorno

```powershell
.\.venv-gpu\Scripts\python.exe -c "import os; os.environ.setdefault('KERAS_BACKEND','torch'); import torch, keras; print('backend:', keras.backend.backend()); print('cuda   :', torch.cuda.is_available()); print('gpu    :', torch.cuda.get_device_name(0))"
```

Salida esperada:

```
backend: torch
cuda   : True
gpu    : NVIDIA GeForce RTX 4060 Laptop GPU
```

### Paso 8 — Colocar el dataset

El notebook espera esta estructura:

```
C:/Users/Laptop/Desktop/Trabajos/LosSimpsonsDataset/
├── train/<personaje>/*.jpg
└── test/<personaje>/*.jpg
```

No está versionado (≈550 MB). Descárgalo de
[Kaggle](https://www.kaggle.com/datasets/alfaro96/los-simpson/data). Si lo dejas en otra
ubicación, cambia `DATASET_BASE` en la celda de configuración del notebook.

### Paso 9 — Abrir el notebook

```powershell
.\scripts\oldDataset\ejecutar_notebook_gpu.ps1
```

El script valida el entorno, avisa si el equipo está con batería y abre Jupyter. Como
alternativa, abre el notebook en VS Code y selecciona el kernel *Python 3.13 (Simpsons GPU
Windows)*.

### Paso 10 — Confirmar antes de entrenar

Ejecuta la primera celda (sección 0). Debe imprimir el backend, la tarjeta y el límite de
potencia:

```
Keras  : 3.15.1   (backend: torch)
PyTorch: 2.14.0+cu130   (compilado para CUDA 13.0)
  - NVIDIA GeForce RTX 4060 Laptop GPU
    compute capability : 8.9
Operacion de prueba ejecutada en: cuda:0
Limite de potencia de la GPU: 80 W de 115 W posibles
```

Si el límite de potencia sale bajo (50 W o menos), vuelve al paso 0. La celda falla con un
error explícito si el backend no es `torch` o si PyTorch no ve la tarjeta.

Con eso, *Run All* entrena las seis configuraciones y escribe el resumen en
`resultados/oldDataset/GPU/`. La corrida completa a 128 × 128 tarda unos 4 minutos y medio.

---

## Enchufa el portátil

Es el ajuste con más impacto de toda esta guía y no cuesta nada.

Con batería, el firmware recorta la potencia que puede consumir la GPU. Medido en este
equipo, con la tarjeta al 100 % de ocupación:

| | Con batería | Enchufado |
| --- | ---: | ---: |
| Límite de potencia | 50 W | 80 W |
| Reloj SM | 1.755 MHz | 2.340 MHz |
| GEMM TF32 | 9,8 TFLOPS | 13,6 TFLOPS |
| Entrenamiento (WSL+TF, línea base) | 5,63 s/época | 2,86 s/época |

**El tiempo por época se dobla con el equipo desenchufado.** Antes de medir nada, conecta
la corriente y pon el modo de energía de Windows en máximo rendimiento. La primera celda
del notebook lee el límite de potencia y avisa si lo encuentra bajo.

Un detalle que confunde: el monitor puede marcar **100 % de uso de GPU** y aun así estar
rindiendo la mitad. El porcentaje mide *ocupación*, no potencia entregada.

---

## Verificar que realmente está usando la GPU

La celda de la sección 0 lo comprueba sola: falla con un error explícito si `torch` no ve
la tarjeta o si el backend de Keras no es `torch`. Debe imprimir algo así:

```
Keras  : 3.15.1   (backend: torch)
PyTorch: 2.14.0+cu130   (compilado para CUDA 13.0)
  - NVIDIA GeForce RTX 4060 Laptop GPU
    compute capability : 8.9
Operacion de prueba ejecutada en: cuda:0
```

Durante el entrenamiento, en otra terminal:

```powershell
nvidia-smi dmon -s um
```

---

## Decisiones de rendimiento

### Los datos viven en la VRAM, en uint8

El conjunto completo se sube a la memoria de la tarjeta **sin convertirlo a `float32`**. A
128×128 son 634 MB de entrenamiento y 159 MB de validación; en `float32` serían 2,5 GB y
0,6 GB. El escalado a [0, 1] lo hace la capa `Rescaling` del propio modelo, sobre cada lote
y a coste despreciable.

Así ningún paso espera a que le copien datos desde la RAM del equipo. La corrida completa
se mantiene por debajo de 3 GB de VRAM, holgada en los 8 GB de la tarjeta.

### Por qué un bucle propio en lugar de `fit()`

`fit()` devuelve el control al intérprete de Python entre paso y paso, y la GPU se queda
esperando.

El notebook solo ejecuta el bucle propio, así que la comparación contra `fit()` viene de un
**micro-benchmark controlado**: misma red de 25,3 M parámetros, mismo tamaño de entrada,
datos sintéticos, las cuatro variantes medidas seguidas en la misma sesión y con el equipo
enchufado. Se descarta la primera época de cada una.

| Configuración | s/época | img/s | utilización GPU | potencia |
| --- | ---: | ---: | ---: | ---: |
| `model.fit()`, batch 128 | 1,89 | 6.815 | 65 % | 50 W |
| Bucle propio, batch 128 | 0,81 | 15.882 | 62 % | 53 W |
| `model.fit()`, batch 512 | 0,65 | 19.842 | 67 % | 54 W |
| **Bucle propio, batch 512** | **0,24** | **54.023** | **97 %** | **74 W** |

**Cuidado con leer la columna de utilización como si fuera la ganancia.** A batch 128 el
bucle propio marca 62 %, prácticamente lo mismo que `fit()`, y aun así es 2,3 × más rápido:
no mantiene la tarjeta más ocupada, sino que desperdicia menos tiempo en cada paso. La
utilización solo se dispara al 97 % cuando el lote sube a 512, porque entonces cada paso
trae trabajo suficiente para que la GPU no se quede sin nada que hacer entre lanzamientos.

Las corridas reales del notebook confirman ese comportamiento. Estas cifras sí salen de
`resultados/oldDataset/GPU/rendimiento_gpu_<resolución>.json`, con el bucle propio a batch 128:

| Resolución | s/época (benchmark) | utilización GPU | potencia |
| --- | ---: | ---: | ---: |
| 32 × 32 | 0,57 | 29,6 % | 10,8 W |
| 64 × 64 | 0,59 | 30,9 % | 18,5 W |
| 128 × 128 | 0,89 | 59,7 % | 52,6 W |

Cuanto más pequeña es la entrada, menos trabajo hay por paso y más se nota la sobrecarga: a
32 × 32 la tarjeta pasa cerca del 70 % del tiempo esperando, y el consumo se queda en 10,8 W
de los 80 W disponibles.

Estas medias se toman sobre una ventana de al menos cuatro segundos, muestreando cada 50 ms
(unas 45 lecturas por medición). No es un detalle menor: con el muestreo anterior, de 250 ms
sobre las propias épocas cronometradas, quedaban dos o tres lecturas por medida y las medias
resultantes eran ruido.

Funciona porque un `keras.Sequential` sobre el backend de PyTorch **es** un
`torch.nn.Module`: se puede entrenar con un bucle a mano sin renunciar a la API de Keras
para construirlo, guardarlo o usarlo después.

La función `entrenar()` del notebook reproduce los tres *callbacks* del original:
`EarlyStopping` con `restore_best_weights`, `ReduceLROnPlateau` y el guardado de la mejor
época. Devuelve un objeto con la misma interfaz `.history` que `fit()`, así que el resto
del notebook —curvas, tablas, comparaciones— no cambia.

### Precisión mixta: medida y descartada

`mixed_float16` **empeora** el rendimiento en esta red: 2,50 s/época frente a 1,89 s en
`float32`. A batch pequeño el cuello de botella es el lanzamiento de kernels, no el
cálculo, y las conversiones a `float16` solo añaden trabajo. Lo que sí se activa es
**TF32** (`torch.backends.cuda.matmul.allow_tf32`), que usa los *tensor cores* de Ada en
las multiplicaciones `float32` sin cambiar nada del modelo.

### `torch.compile` no está disponible en Windows

`jit_compile=True` falla con `AssertionError: internal error`. El compilador Inductor de
PyTorch necesita Triton, que no tiene versión para Windows (`triton not found`). Es la
única ventaja real que conserva el entorno WSL: allí sí funciona XLA, y acelera
TensorFlow de 2,86 a 0,87 s/época.

### El tamaño de lote es la palanca que queda

La celda de rendimiento del notebook mide el barrido completo en cada corrida y lo guarda en
`escalado_batch_imagenes_por_segundo`. Imágenes por segundo, con el bucle propio:

| batch | 32 × 32 | 64 × 64 | 128 × 128 |
| ---: | ---: | ---: | ---: |
| 128 | 22.256 | 20.857 | 15.015 |
| 256 | 44.912 | 41.778 | 27.671 |
| 512 | 87.243 | 80.900 | 48.775 |
| 1.024 | 172.412 | 146.455 | 63.048 |
| 2.048 | **354.971** | **237.912** | **70.444** |

De batch 128 a 2.048 el rendimiento se multiplica por 15,9 en 32 × 32 y por 4,7 en
128 × 128. La diferencia entre columnas es esperable: con la entrada más grande cada paso ya
trae bastante trabajo, así que queda menos sobrecarga que recuperar.

La utilización acompaña. A 128 × 128, medida en la misma corrida:

| batch | s/época | img/s | utilización GPU | potencia |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0,860 | 15.015 | 61,9 % | 53,8 W |
| 256 | 0,467 | 27.671 | 69,7 % | 60,1 W |
| 512 | 0,265 | 48.775 | 88,0 % | 72,7 W |
| 1.024 | 0,205 | 63.048 | 95,3 % | 79,8 W |
| 2.048 | 0,183 | 70.444 | 96,6 % | 79,0 W |

A partir de batch 1.024 la tarjeta llega al límite de potencia de 80 W: ahí ya no queda
sobrecarga que recuperar, y es el hardware el que manda.

El notebook entrena a **batch 128** porque es el valor con el que se comparan las corridas
de CPU y GPU en el informe, no porque sea el más rápido. El barrido se mide y se guarda para
poder citarlo sin cambiar la configuración del estudio.

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
| --- | --- | --- |
| `Keras esta usando el backend "tensorflow"` | El bloque `env` del kernel se perdió | Reañadir `KERAS_BACKEND` a `kernel.json` y reiniciar el kernel |
| `Unexpected UTF-8 BOM` al abrir el kernel | Se escribió `kernel.json` con `Set-Content -Encoding utf8`, que añade BOM | Reescribirlo con `WriteAllText` y `UTF8Encoding($false)` (paso 6) |
| `PyTorch no detecta ninguna GPU` | Driver antiguo, o se instaló el `torch` de PyPI (solo CPU) | Reinstalar desde el índice `cu130` |
| Todo va el doble de lento | El portátil está con batería | Conectar a la corriente, modo máximo rendimiento |
| `torch.cuda.OutOfMemoryError` | Batch demasiado grande en el barrido | El notebook lo captura y sigue; reducir `BATCHES_A_MEDIR` |
| `AssertionError: internal error` al compilar | `torch.compile` en Windows | No usar `jit_compile`; no hay Triton para Windows |
| Aviso `Skipping variable loading for optimizer` | El optimizador es de PyTorch, no de Keras | Inocuo; la celda de predicción ya carga con `compile=False` |

---

## Configuración de referencia

| Componente | Versión |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, CC 8.9 |
| Driver | 616.92 |
| Sistema | Windows 11 |
| Python | 3.13.15 |
| PyTorch | 2.14.0+cu130 |
| Keras | 3.15.1 |
| Entorno | `.venv-gpu/` en la raíz del repositorio |
| Modelos | `models_gpu/oldDataset/` |

---

## Apéndice: entorno anterior (WSL2 + TensorFlow)

Las corridas guardadas en `resultados/oldDataset/GPU/` se hicieron con este montaje, anterior a la
migración a Windows nativo. Se conserva porque documenta esos resultados y porque sigue
siendo la única forma de usar TensorFlow con GPU en este equipo.

### Montaje

TensorFlow instalado con pip no encuentra las bibliotecas CUDA que él mismo trae como
dependencias, porque no están en el `LD_LIBRARY_PATH`. Había dos formas de resolverlo:

**Kernel dedicado.** En `~/.local/share/jupyter/kernels/simpsons-gpu/kernel.json`, un
bloque `env` con `LD_LIBRARY_PATH` apuntando a los directorios `lib` de cada paquete
`nvidia-*` del entorno, más `/usr/lib/wsl/lib`, y
`XLA_FLAGS=--xla_gpu_cuda_data_dir=.../cuda_nvcc`. Ejecutar `ipykernel install` borraba
ese bloque, así que había que volver a ponerlo.

**Script de arranque.** `scripts/ejecutar_notebook_gpu.sh` (eliminado en la migración)
construía el `LD_LIBRARY_PATH` al vuelo:

```bash
NVIDIA_DIR=$(python -c "import nvidia,os;print(os.path.dirname(nvidia.__file__))")
CUDA_LIBS=$(find "$NVIDIA_DIR" -path '*/lib/*.so*' -type f -printf '%h\n' | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CUDA_LIBS}"
```

### RAM de WSL

Por defecto WSL toma la mitad de la RAM de Windows. El notebook convertía todas las
imágenes a `float32`, con picos de 7,14 GB a 128×128 y 28,55 GB a 256×256, y el OOM killer
mataba el kernel de Jupyter sin mensaje. La solución era `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
```

seguido de `wsl --shutdown`.

Ese problema **ya no existe**: los datos se mantienen en `uint8` y nunca se materializa la
copia `float32` completa.

### Por qué allí los datos se anclaban a la RAM del host

En el notebook antiguo, un bloque `with tf.device("/CPU:0")` fijaba los tensores en la RAM
del equipo. Sin él, Keras subía las matrices `float32` completas a la VRAM y la tarjeta se
quedaba sin memoria en el experimento E5 con `Dst tensor is not initialized`. El coste era
copiar cada lote por el bus en cada paso, que es justamente lo que limitaba la utilización.

La solución actual —`uint8` en la VRAM más `Rescaling` dentro del modelo— elimina el
dilema: los datos caben enteros en la tarjeta y no hay copias por paso.

### Rendimiento de aquel entorno

Medido de nuevo con el equipo enchufado, para que sea comparable:

| Configuración (batch 128) | s/época | utilización GPU |
| --- | ---: | ---: |
| Como estaba el notebook | 2,86 | 68 % |
| + datos uint8 en VRAM | 2,65 | 67 % |
| + XLA (`jit_compile=True`) | 0,87 | 70 % |
| + XLA + `mixed_float16` | 0,91 | 71 % |
