# Entorno de los notebooks de PMC/notebooks/newDataset y CNN/notebooks/newDataset (CPU y GPU)
# en un solo contenedor Linux.
#
# - GPU: Keras 3 sobre PyTorch con CUDA 13.0. El wheel de torch trae sus propias
#   bibliotecas CUDA; el contenedor solo necesita el driver NVIDIA del host.
# - CPU: el mismo Keras 3 sobre PyTorch, sin TensorFlow. El notebook oculta la GPU
#   (CUDA_VISIBLE_DEVICES=-1) desde su primera celda, y su kernel la oculta también.
# - Descriptores de color y textura del MLP: scikit-image (requirements-newDataset.txt).
#
# Cada notebook abre con su kernel, registrado con el mismo nombre que guarda en
# sus metadatos y con KERAS_BACKEND fijado en el kernel.json. Los notebooks de
# PMC/notebooks/oldDataset apuntan a una ruta absoluta de Windows y se ejecutan fuera
# del contenedor.
#
# Uso (desde la raiz del repositorio, con Docker Desktop sobre WSL2):
#   docker compose up --build
# y abrir http://localhost:8888

FROM python:3.13.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SIMPSONS_DATASET=/data/simpsons_dataset

# torch va en su propia capa: son ~3 GB y no cambian al tocar los requirements.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt requirements-gpu-windows.txt requirements-newDataset.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/requirements-gpu-windows.txt \
                -r /tmp/requirements/requirements.txt \
                -r /tmp/requirements/requirements-newDataset.txt \
 && rm -rf /tmp/requirements

RUN python -m ipykernel install --name simpsons-gpu-win \
        --display-name "Python 3.13 (Simpsons GPU Docker)" --env KERAS_BACKEND torch \
 && python -m ipykernel install --name python313-global \
        --display-name "Python 3.13 (Simpsons CPU Docker)" --env KERAS_BACKEND torch \
        --env KERAS_TORCH_DEVICE cpu --env CUDA_VISIBLE_DEVICES -1

RUN useradd --create-home --uid 1000 simpsons
USER simpsons
WORKDIR /workspace

EXPOSE 8888

# Sin token: compose publica el puerto solo en 127.0.0.1.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.root_dir=/workspace", "--IdentityProvider.token="]
