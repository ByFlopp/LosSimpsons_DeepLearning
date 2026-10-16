"""Genera variantes alteradas (color + textura) de un cuarto de las imágenes de cada personaje.

Uso:
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py --dataset RUTA --fraccion 0.25 --semilla 42
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py --limpiar

El dataset es una carpeta por personaje. Para cada una se elige al azar una fracción de sus
imágenes originales y, junto a cada elegida, se guarda una copia con un cambio de color y un
filtro de textura, en el mismo formato:

    homer_simpson/pic_0012.jpg  ->  homer_simpson/pic_0012__alterada.jpg

Las originales solo se leen; la etiqueta de cada imagen la da su carpeta, nunca su nombre.

Reproducible y sin rastro: cada ejecución borra primero todo lo que generó la anterior (los
archivos cuyo nombre termina en "__alterada" y el registro _alteraciones.csv) y lo vuelve a
generar con la misma semilla, de modo que el resultado es idéntico byte a byte. --limpiar solo
borra, y deja el dataset exactamente como estaba antes de la primera ejecución.

La ruta del dataset sale de --dataset, de la variable SIMPSONS_DATASET o, si no hay ninguna,
de ~/Desktop/simpsons_dataset/simpsons_dataset.
"""

import argparse
import csv
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARCA = "__alterada"
SUFIJO_TEMPORAL = ".tmp"
REGISTRO = "_alteraciones.csv"
CAMPOS_REGISTRO = ("personaje", "original", "alterada", "color", "parametro_color",
                   "textura", "parametro_textura")
FRACCION = 0.25
SEMILLA = 42
CALIDAD = 92
RUTA_POR_DEFECTO = Path.home() / "Desktop" / "simpsons_dataset" / "simpsons_dataset"

# Formato de PIL para cada extensión. Se pasa explícito porque el archivo se escribe
# primero con un nombre temporal, cuya extensión PIL no reconocería.
FORMATOS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
OPCIONES_GUARDADO = {"JPEG": {"quality": CALIDAD}, "WEBP": {"quality": CALIDAD}}


# --- Cambios de color --------------------------------------------------------------------
# Rangos moderados: la variante debe seguir siendo el mismo personaje (piel amarilla, pelo
# azul de Marge...), solo con otra iluminación o saturación.

def cambiar_brillo(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    factor = float(rng.uniform(0.7, 1.3))
    return ImageEnhance.Brightness(imagen).enhance(factor), factor


def cambiar_contraste(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    factor = float(rng.uniform(0.7, 1.3))
    return ImageEnhance.Contrast(imagen).enhance(factor), factor


def cambiar_saturacion(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    factor = float(rng.uniform(0.6, 1.4))
    return ImageEnhance.Color(imagen).enhance(factor), factor


def cambiar_tono(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    """Gira el tono (H de HSV) como mucho un 4 % de la rueda de color, unos 14 grados."""
    desplazamiento = float(rng.uniform(-0.04, 0.04))
    hsv = np.asarray(imagen.convert("HSV"), dtype=np.int16).copy()
    hsv[..., 0] = (hsv[..., 0] + round(desplazamiento * 256)) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), desplazamiento


# --- Filtros de textura ------------------------------------------------------------------

def desenfocar(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    radio = float(rng.uniform(0.5, 1.5))
    return imagen.filter(ImageFilter.GaussianBlur(radio)), radio


def enfocar(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    porcentaje = int(rng.integers(100, 201))
    filtro = ImageFilter.UnsharpMask(radius=2, percent=porcentaje, threshold=3)
    return imagen.filter(filtro), porcentaje


def anadir_ruido(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    sigma = float(rng.uniform(5, 15))
    pixeles = np.asarray(imagen, dtype=np.float32)
    ruidosa = pixeles + rng.normal(0.0, sigma, pixeles.shape)
    return Image.fromarray(np.clip(ruidosa, 0, 255).round().astype(np.uint8), "RGB"), sigma


def realzar_bordes(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    return imagen.filter(ImageFilter.EDGE_ENHANCE), 1.0


CAMBIOS_COLOR = {
    "brillo": cambiar_brillo,
    "contraste": cambiar_contraste,
    "saturacion": cambiar_saturacion,
    "tono": cambiar_tono,
}
FILTROS_TEXTURA = {
    "desenfoque": desenfocar,
    "nitidez": enfocar,
    "ruido": anadir_ruido,
    "bordes": realzar_bordes,
}


def es_generado(ruta: Path) -> bool:
    """True para los archivos que crea este script, incluidos los temporales a medio escribir."""
    nombre = ruta.name.removesuffix(SUFIJO_TEMPORAL)
    return Path(nombre).stem.endswith(MARCA)


def carpetas_personaje(dataset: Path) -> list[Path]:
    return sorted(p for p in dataset.iterdir() if p.is_dir() and not p.name.startswith("."))


def originales_de(carpeta: Path) -> list[Path]:
    return sorted(
        p for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS and not es_generado(p)
    )


def semilla_de(*partes: str) -> list[int]:
    """Semilla estable entre ejecuciones y sistemas (hash() de Python cambia en cada proceso)."""
    return [zlib.crc32(parte.encode("utf-8")) for parte in partes]


def limpiar(dataset: Path) -> int:
    """Borra todo lo generado por ejecuciones anteriores. Nunca toca una original."""
    borrados = 0
    for carpeta in carpetas_personaje(dataset):
        for ruta in carpeta.iterdir():
            if ruta.is_file() and es_generado(ruta):
                ruta.unlink()
                borrados += 1
    registro = dataset / REGISTRO
    for ruta in (registro, registro.with_name(registro.name + SUFIJO_TEMPORAL)):
        if ruta.exists():
            ruta.unlink()
    return borrados


def alterar(original: Path, semilla: int) -> dict:
    """Escribe la variante alterada de una imagen y devuelve su fila del registro."""
    # Generador propio de cada imagen: el resultado no depende del orden de proceso
    # (se procesa en paralelo) ni de las demás imágenes de la carpeta.
    rng = np.random.default_rng([semilla, *semilla_de(original.parent.name, original.name)])
    nombre_color = list(CAMBIOS_COLOR)[rng.integers(len(CAMBIOS_COLOR))]
    nombre_textura = list(FILTROS_TEXTURA)[rng.integers(len(FILTROS_TEXTURA))]

    with Image.open(original) as archivo:
        imagen = archivo.convert("RGB")
    imagen, parametro_color = CAMBIOS_COLOR[nombre_color](imagen, rng)
    imagen, parametro_textura = FILTROS_TEXTURA[nombre_textura](imagen, rng)

    destino = original.with_name(f"{original.stem}{MARCA}{original.suffix}")
    assert destino != original and es_generado(destino)
    formato = FORMATOS[original.suffix.lower()]
    temporal = destino.with_name(destino.name + SUFIJO_TEMPORAL)
    imagen.save(temporal, format=formato, **OPCIONES_GUARDADO.get(formato, {}))
    os.replace(temporal, destino)

    return {
        "personaje": original.parent.name,
        "original": original.name,
        "alterada": destino.name,
        "color": nombre_color,
        "parametro_color": round(parametro_color, 4),
        "textura": nombre_textura,
        "parametro_textura": round(parametro_textura, 4),
    }


def validar(dataset: Path) -> list[Path]:
    if not dataset.is_dir():
        sys.exit(f"No existe el dataset: {dataset}")
    if (dataset / "train").is_dir() or (dataset / "test").is_dir():
        sys.exit(
            f"{dataset} tiene la estructura antigua (train/ y test/). Este script trabaja "
            "con una carpeta por personaje directamente bajo la ruta del dataset."
        )
    carpetas = [c for c in carpetas_personaje(dataset) if originales_de(c)]
    if not carpetas:
        sys.exit(f"No hay carpetas de personaje con imágenes en {dataset}")
    return carpetas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=None,
                        help="carpeta con una subcarpeta por personaje")
    parser.add_argument("--fraccion", type=float, default=FRACCION,
                        help="fracción de imágenes de cada personaje que reciben una alteración")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--limpiar", action="store_true",
                        help="solo borra lo generado y deja el dataset como estaba")
    args = parser.parse_args()

    dataset = args.dataset or Path(os.environ.get("SIMPSONS_DATASET", RUTA_POR_DEFECTO))
    dataset = dataset.expanduser().resolve()
    if not 0 < args.fraccion <= 1:
        sys.exit("--fraccion debe estar en (0, 1]")
    if args.semilla < 0:
        sys.exit("--semilla debe ser un entero no negativo")

    carpetas = validar(dataset)
    print(f"Dataset: {dataset}  ({len(carpetas)} personajes)")

    borrados = limpiar(dataset)
    print(f"Limpieza: {borrados} archivos generados en ejecuciones anteriores eliminados")
    if args.limpiar:
        return

    inicio = time.perf_counter()
    elegidas = []
    resumen = []
    for carpeta in carpetas:
        originales = originales_de(carpeta)
        cantidad = int(args.fraccion * len(originales) + 0.5)  # redondeo al entero más cercano
        rng = np.random.default_rng([args.semilla, *semilla_de(carpeta.name)])
        indices = np.sort(rng.choice(len(originales), size=cantidad, replace=False))
        elegidas.extend(originales[i] for i in indices)
        resumen.append((carpeta.name, len(originales), cantidad))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ejecutor:
        filas = list(ejecutor.map(lambda ruta: alterar(ruta, args.semilla), elegidas))

    # El registro se escribe al final y de forma atómica: si aparece, la ejecución terminó.
    registro = dataset / REGISTRO
    temporal = registro.with_name(registro.name + SUFIJO_TEMPORAL)
    with open(temporal, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_REGISTRO, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    os.replace(temporal, registro)

    ancho = max(len(nombre) for nombre, _, _ in resumen)
    print(f"\n{'personaje':<{ancho}}  originales  alteradas")
    for nombre, originales, alteradas in resumen:
        print(f"{nombre:<{ancho}}  {originales:>10}  {alteradas:>9}")
    total_originales = sum(o for _, o, _ in resumen)
    print(f"{'TOTAL':<{ancho}}  {total_originales:>10}  {len(filas):>9}")
    print(f"\n{len(filas)} imágenes alteradas en {time.perf_counter() - inicio:.1f} s."
          f" Registro: {registro}")


if __name__ == "__main__":
    main()
