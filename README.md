# Clasificador de personajes de Los Simpson

Este repositorio reúne los modelos que el equipo entrena para identificar qué personaje de Los Simpson aparece en una imagen RGB. Cada tipo de modelo tiene su propia carpeta, con sus notebooks, modelos, resultados y documentación. El dataset, su preparación y el entorno de ejecución se comparten.

| Modelo | Carpeta | Estado | Resultado en test, dataset nuevo |
|---|---|---|---|
| Perceptrón multicapa (MLP) | [PMC/](PMC/) | Terminado, con el dataset anterior (25 personajes) y el nuevo (64) | Accuracy de 0,53 a 0,59 en seis corridas |
| Red neuronal convolucional (CNN) | [CNN/](CNN/) | Corrida de GPU a 64 × 64 completa; la de CPU, pendiente | Accuracy de 0,830 (F1 macro 0,788) |

<p align="center">
  <img src="images/logoLosSimpsons.png" alt="Logo de Los Simpson" width="480">
</p>

## Lo que comparten los modelos

**Dataset.** 64 personajes, una carpeta por personaje, sin partición previa. No se versiona: por defecto se busca en `~/Desktop/simpsons_dataset/simpsons_dataset`, o en la ruta de la variable de entorno `SIMPSONS_DATASET`. La etiqueta de cada imagen es siempre el nombre de su carpeta.

**Preparación del dataset.** [scripts/newDataset/generar_imagenes_alteradas.py](scripts/newDataset/generar_imagenes_alteradas.py) genera dentro del dataset las variantes alteradas y las versiones filtradas de las imágenes casi idénticas, y registra las imágenes repetidas entre carpetas de personajes distintos. Nunca modifica las originales. Se ejecuta desde la raíz del repositorio cada vez que cambia el dataset:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\newDataset\generar_imagenes_alteradas.py
```

Los pasos completos, con sus opciones y la versión en Docker, están en la [guía del dataset nuevo](PMC/notebooks/newDataset/README.md).

**Entorno.** Keras 3 sobre PyTorch, en Windows nativo (`.venv-gpu`) o en Docker:

| Archivo | Para qué |
|---|---|
| `requirements-gpu-windows.txt` | Entorno `.venv-gpu`: Keras 3 sobre PyTorch con CUDA. Los pasos de instalación están en [PMC/notebooks/oldDataset/README_GPU.md](PMC/notebooks/oldDataset/README_GPU.md). |
| `requirements-newDataset.txt` | scikit-image, para los descriptores de color y textura del MLP del dataset nuevo. |
| `requirements.txt` | Dependencias del notebook de CPU del dataset anterior, que usa TensorFlow. |
| `Dockerfile` y `docker-compose.yml` | Jupyter Lab con la GPU y el dataset montados: copia `.env.example` como `.env`, pon la ruta del dataset y ejecuta `docker compose up --build`. |

## Estructura del repositorio

```text
.
├── PMC/                            # Perceptrón multicapa: todo su trabajo, con su propio README
│   ├── notebooks/<oldDataset|newDataset>/
│   ├── models/ y models_gpu/       # Modelos .keras regenerables, ignorados por Git
│   ├── resultados/<oldDataset|newDataset>/<CPU|GPU>/
│   ├── scripts/                    # Lanzadores de los notebooks de GPU y diagrama de la red
│   ├── images/                     # Imágenes de los notebooks y del README del MLP
│   ├── docs/                       # Informe, pauta de la EP1 y material de la unidad
│   ├── presentacion_resultados.html
│   └── README.md
├── CNN/                            # Red convolucional: mismo dataset y mismo proceso, con su propio README
│   ├── notebooks/newDataset/       # Notebooks de GPU y CPU (sin versión para el dataset anterior)
│   ├── models/ y models_gpu/       # Modelos .keras regenerables, ignorados por Git
│   ├── resultados/newDataset/<CPU|GPU>/
│   ├── scripts/                    # Lanzador del notebook de GPU
│   ├── images/                     # Imágenes de los notebooks
│   └── README.md
├── scripts/newDataset/
│   └── generar_imagenes_alteradas.py  # Prepara el dataset nuevo: alteradas, parecidas y gemelas
├── images/logoLosSimpsons.png
├── Dockerfile
├── docker-compose.yml
├── .env.example                    # Ruta del dataset para Docker; se copia como .env
├── requirements.txt
├── requirements-gpu-windows.txt
├── requirements-newDataset.txt
└── README.md                       # Este documento
```

Cada modelo sigue la misma organización: una carpeta propia con `notebooks/`, `models/`, `resultados/` y `scripts/`, y dentro de cada una una subcarpeta por dataset.

## Integrantes

- Vincent Farenden Cerón
- Rodrigo Ignacio Martínez Becker
- Iván Manuel Pavón Barría
- Diego Ignacio Peña y Lillo Luhrs

Trabajo para la asignatura **Técnicas Avanzadas para Machine Learning_001D**.
