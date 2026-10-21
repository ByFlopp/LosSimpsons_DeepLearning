"""Clasifica imágenes con la CNN entrenada: qué personaje de Los Simpson aparece y con qué probabilidad.

Uso, desde la raíz del repositorio:
    .venv-gpu/Scripts/python.exe CNN/scripts/newDataset/predecir_imagen.py ruta/a/imagen.jpg
    .venv-gpu/Scripts/python.exe CNN/scripts/newDataset/predecir_imagen.py imagen1.jpg imagen2.png --top 3
    .venv-gpu/Scripts/python.exe CNN/scripts/newDataset/predecir_imagen.py imagen.jpg --json

Usa el modelo final que guarda el notebook (cnn_final.keras) y su cnn_final.json, que trae la resolución y el
orden de las clases. Por defecto lo busca en models_gpu/newDataset/ y, si no está, en models/newDataset/.

El pre-procesamiento es el mismo del notebook: la imagen se convierte a RGB, se redimensiona con interpolación
bilineal al tamaño del modelo y entra en uint8; el escalado a [0, 1] lo hace la capa Rescaling del modelo.

Si la imagen está en una carpeta con el nombre de un personaje (como las del dataset), también dice si el
modelo acertó. La red siempre responde con alguno de sus 64 personajes, aunque la imagen no tenga a ninguno:
con una confianza baja, la predicción conviene revisarla a mano.
"""

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
MODELOS = (RAIZ / "CNN/models_gpu/newDataset/cnn_final.keras", RAIZ / "CNN/models/newDataset/cnn_final.keras")
# Bajo esta confianza la predicción se marca para revisión humana
UMBRAL_REVISION = 0.50


def leer_argumentos():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("imagenes", nargs="+", type=Path, help="una o más rutas de imágenes (jpg, png, webp, ...)")
    parser.add_argument("--top", type=int, default=5, help="cuántas candidatas mostrar (por defecto %(default)s)")
    parser.add_argument("--modelo", type=Path, help="archivo .keras; a su lado debe estar el .json con las clases")
    parser.add_argument("--json", action="store_true", help="escribe el resultado en JSON en lugar de texto")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU")
    return parser.parse_args()


def cargar_modelo(ruta_modelo, usar_cpu):
    if ruta_modelo is None:
        ruta_modelo = next((ruta for ruta in MODELOS if ruta.exists()), None)
        if ruta_modelo is None:
            raise SystemExit(
                "No hay modelo entrenado en " + " ni en ".join(str(ruta.relative_to(RAIZ)) for ruta in MODELOS)
                + ": ejecuta antes el notebook de la CNN."
            )
    ruta_metadatos = ruta_modelo.with_suffix(".json")
    if not ruta_modelo.is_file() or not ruta_metadatos.is_file():
        raise SystemExit(f"Faltan {ruta_modelo} o {ruta_metadatos}")

    # El backend se fija antes de importar Keras, y la GPU se oculta antes de importar PyTorch
    os.environ.setdefault("KERAS_BACKEND", "torch")
    if usar_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import keras
    import torch

    # compile=False: solo se predice. El entrenamiento usa un optimizador de PyTorch, así que el de
    # Keras guardado en el archivo está vacío
    modelo = keras.models.load_model(ruta_modelo, compile=False)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = modelo.to(dispositivo)
    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    return modelo, metadatos, dispositivo


def predecir(modelo, metadatos, dispositivo, ruta_imagen, top):
    import numpy as np
    import torch
    from PIL import Image

    clases = metadatos["clases"]
    with Image.open(ruta_imagen) as archivo:
        # np.array y no np.asarray: la copia es escribible, y PyTorch avisa si recibe la memoria de solo
        # lectura de PIL
        reducida = np.array(
            archivo.convert("RGB").resize((metadatos["ancho"], metadatos["alto"]), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    # La imagen entra con su forma (1, alto, ancho, 3), sin aplanar
    entrada = torch.as_tensor(reducida[np.newaxis]).to(dispositivo)
    probabilidades = modelo.predict(entrada, verbose=0)[0]
    mejores = np.argsort(-probabilidades)[:top]

    prediccion = clases[int(mejores[0])]
    carpeta = ruta_imagen.parent.name if ruta_imagen.parent.name in clases else None
    return {
        "imagen": str(ruta_imagen),
        "prediccion": prediccion,
        "confianza": round(float(probabilidades[mejores[0]]), 4),
        "revisar": bool(probabilidades[mejores[0]] < UMBRAL_REVISION),
        "carpeta": carpeta,
        "acierto": None if carpeta is None else prediccion == carpeta,
        "candidatas": [
            {"personaje": clases[int(i)], "probabilidad": round(float(probabilidades[i]), 4)} for i in mejores
        ],
    }


def mostrar(resultado):
    print(resultado["imagen"])
    for posicion, candidata in enumerate(resultado["candidatas"], start=1):
        print(f"  {posicion}. {candidata['personaje'].replace('_', ' '):<30} {candidata['probabilidad']:7.1%}")
    linea = f"  -> {resultado['prediccion'].replace('_', ' ')} ({resultado['confianza']:.1%} de confianza)"
    if resultado["carpeta"] is not None:
        linea += f"   |   carpeta: {resultado['carpeta'].replace('_', ' ')}, {'acierto' if resultado['acierto'] else 'error'}"
    print(linea)
    if resultado["revisar"]:
        print(f"  Confianza menor que {UMBRAL_REVISION:.0%}: conviene revisar la imagen a mano.")


def main():
    argumentos = leer_argumentos()
    faltan = [str(ruta) for ruta in argumentos.imagenes if not ruta.is_file()]
    if faltan:
        raise SystemExit("No se encontró: " + ", ".join(faltan))
    if argumentos.top < 1:
        raise SystemExit("--top debe ser al menos 1")

    modelo, metadatos, dispositivo = cargar_modelo(argumentos.modelo, argumentos.cpu)
    top = min(argumentos.top, len(metadatos["clases"]))
    resultados = [predecir(modelo, metadatos, dispositivo, ruta, top) for ruta in argumentos.imagenes]

    if argumentos.json:
        print(json.dumps(resultados, indent=2, ensure_ascii=False))
    else:
        print(f"Modelo: {metadatos['configuracion']} ({metadatos['alto']}x{metadatos['ancho']}, {dispositivo})\n")
        for resultado in resultados:
            mostrar(resultado)
            print()


if __name__ == "__main__":
    sys.exit(main())
