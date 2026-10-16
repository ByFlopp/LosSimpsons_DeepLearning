"""Genera variantes alteradas de las imágenes de cada personaje, diversifica las casi idénticas y
detecta las imágenes que comparten personajes de distintas carpetas.

Uso:
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py --dataset RUTA --fraccion 0.25 --umbral 0.90 --umbral-gemelas 0.90 --semilla 42
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py --limpiar

El dataset es una carpeta por personaje. El script hace tres cosas con las imágenes originales; cada
imagen que genera en los pasos 1 y 2 lleva un giro de 10 a 20 grados, un cambio de color y un filtro de
textura:

1. Variantes alteradas. Para cada personaje elige al azar una fracción de sus originales y guarda,
   junto a cada elegida, una variante en el mismo formato. El notebook de GPU pone cada variante en el
   mismo conjunto que su original (entrenamiento, validación o test); el de CPU solo la usa para entrenar:

       homer_simpson/pic_0012.jpg  ->  homer_simpson/pic_0012__alterada.jpg

2. Imágenes parecidas. Compara entre sí las imágenes de cada personaje con miniaturas a color de
   16 x 16 y agrupa las casi idénticas (similitud coseno >= --umbral): fotogramas seguidos de una
   misma escena o recortes repetidos. En cada grupo una imagen queda intacta y cada una de las demás
   recibe su propia variante, que se vuelve a sortear hasta que deja de parecerse a las del grupo:

       homer_simpson/pic_0013.jpg  ->  homer_simpson/pic_0013__parecida.jpg

   El notebook usa la variante __parecida EN LUGAR de su original, en cualquier conjunto: así el
   dataset deja de tener dos imágenes casi idénticas del mismo personaje.

3. Gemelas entre personajes. Compara las originales de carpetas DISTINTAS con las mismas miniaturas
   (similitud >= --umbral-gemelas): una imagen casi idéntica guardada en la carpeta de otro personaje
   es la misma escena, así que los dos personajes aparecen en ella. El notebook cuenta como acierto
   que el modelo nombre a cualquiera de los dos. Además, copia cada imagen a una carpeta de revisión
   FUERA del dataset, en la subcarpeta del otro personaje y con su nombre original, para que el
   equipo decida a mano si la agrega también a esa carpeta:

       homer_simpson/homer_simpson__S18E11__t1225141.jpg  (gemela de una imagen de rich_texan)
           ->  gemelas_por_revisar/rich_texan/homer_simpson__S18E11__t1225141.jpg

   Una imagen no se copia si la carpeta del otro personaje ya la tiene (similitud >= 0,999).

Las originales solo se leen; la etiqueta de cada imagen la da su carpeta, nunca su nombre.

Reproducible y sin rastro: cada ejecución borra primero todo lo que generó la anterior (los archivos
cuyo nombre termina en "__alterada" o "__parecida", los registros _alteraciones.csv, _parecidas.csv y
_gemelas.csv, y las copias de la carpeta de revisión) y lo vuelve a generar con la misma semilla, de
modo que el resultado es idéntico byte a byte. --limpiar solo borra, y deja el dataset exactamente como
estaba antes de la primera ejecución. La carpeta de revisión solo se vacía si lleva la marca que deja
el script (_generado_por_script.txt).

La ruta del dataset sale de --dataset, de la variable SIMPSONS_DATASET o, si no hay ninguna,
de ~/Desktop/simpsons_dataset/simpsons_dataset. La carpeta de revisión sale de --revision o, si no
se indica, es gemelas_por_revisar junto a la carpeta del dataset.
"""

import argparse
import csv
import math
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARCA_ALTERADA = "__alterada"
MARCA_PARECIDA = "__parecida"
MARCAS = (MARCA_ALTERADA, MARCA_PARECIDA)
SUFIJO_TEMPORAL = ".tmp"
REGISTRO = "_alteraciones.csv"
REGISTRO_PARECIDAS = "_parecidas.csv"
REGISTRO_GEMELAS = "_gemelas.csv"
CAMPOS_TRANSFORMACION = ("rotacion", "color", "parametro_color", "textura", "parametro_textura")
CAMPOS_REGISTRO = ("personaje", "original", "alterada", *CAMPOS_TRANSFORMACION)
CAMPOS_PARECIDAS = ("personaje", "grupo", "representante", "original", "parecida",
                    "similitud_original", "similitud_final", "intentos", *CAMPOS_TRANSFORMACION)
# copia_original: dónde quedó la copia de "original" en la carpeta de revisión (vacío si la carpeta
# de otro_personaje ya la tenía); copia_otro, lo mismo para "otro_original"
CAMPOS_GEMELAS = ("personaje", "original", "otro_personaje", "otro_original", "similitud",
                  "copia_original", "copia_otro")
FRACCION = 0.25
UMBRAL = 0.90
UMBRAL_GEMELAS = 0.90
SEMILLA = 42
CALIDAD = 92
RUTA_POR_DEFECTO = Path.home() / "Desktop" / "simpsons_dataset" / "simpsons_dataset"
# Carpeta de revisión de las gemelas, junto a la del dataset (nunca dentro: cada subcarpeta del
# dataset se toma como un personaje)
NOMBRE_REVISION = "gemelas_por_revisar"
MARCA_REVISION = "_generado_por_script.txt"
TEXTO_MARCA_REVISION = (
    "Esta carpeta la genera scripts/newDataset/generar_imagenes_alteradas.py.\n"
    "Cada subcarpeta es un personaje y contiene imagenes de OTRAS carpetas del dataset en las que\n"
    "tambien aparece ese personaje (una imagen casi identica esta guardada en su carpeta).\n"
    "Si una imagen debe sumarse a ese personaje, copiala a su carpeta del dataset.\n"
    "Cada ejecucion del script vacia esta carpeta y la vuelve a llenar: no guardes aqui nada propio.\n"
)
# Similitud a partir de la cual la carpeta del otro personaje ya tiene la imagen y no se propone
SIMILITUD_IDENTICA = 0.999
# Filas de la matriz de similitud que se calculan a la vez al buscar gemelas entre personajes
BLOQUE_GEMELAS = 1024

# Giro de cada variante: entre ROTACION_MINIMA y ROTACION_MAXIMA grados, en un sentido al azar
ROTACION_MINIMA = 10.0
ROTACION_MAXIMA = 20.0
# Las imágenes se comparan como las ve el notebook (128 x 128), reducidas a 16 x 16 a color
LADO_TRABAJO = 128
LADO_MINIATURA = 16
# Sorteos que se prueban, como máximo, para que una imagen parecida deje de parecerse a su grupo
INTENTOS_PARECIDA = 5

# Formato de PIL para cada extensión. Se pasa explícito porque el archivo se escribe
# primero con un nombre temporal, cuya extensión PIL no reconocería.
FORMATOS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".webp": "WEBP"}
OPCIONES_GUARDADO = {"JPEG": {"quality": CALIDAD}, "WEBP": {"quality": CALIDAD}}


# --- Giro --------------------------------------------------------------------------------

def girar(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, float]:
    """Gira la imagen entre ROTACION_MINIMA y ROTACION_MAXIMA grados y recorta el mayor rectángulo
    centrado que no tiene esquinas vacías, devuelto al tamaño original: la variante no gana bordes
    negros que el modelo pudiera aprender, a cambio de acercar un poco el encuadre."""
    angulo = float(rng.uniform(ROTACION_MINIMA, ROTACION_MAXIMA)) * float(rng.choice((-1.0, 1.0)))
    ancho, alto = imagen.size
    radianes = math.radians(abs(angulo))
    coseno, seno = math.cos(radianes), math.sin(radianes)
    # Un rectángulo centrado (s * ancho, s * alto) cabe en la imagen girada si, girado de vuelta,
    # su caja envolvente cabe en la original
    escala = min(ancho / (ancho * coseno + alto * seno), alto / (ancho * seno + alto * coseno))
    margen_x = (ancho - ancho * escala) / 2
    margen_y = (alto - alto * escala) / 2
    # Un píxel más de margen: el borde del recorte mezclaría el relleno negro al interpolar
    caja = (math.ceil(margen_x) + 1, math.ceil(margen_y) + 1,
            math.floor(ancho - margen_x) - 1, math.floor(alto - margen_y) - 1)
    girada = imagen.rotate(angulo, resample=Image.Resampling.BICUBIC)
    if caja[2] <= caja[0] or caja[3] <= caja[1]:  # imagen diminuta: no hay recorte posible
        return girada, angulo
    return girada.crop(caja).resize((ancho, alto), Image.Resampling.BICUBIC), angulo


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


def transformar(imagen: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict]:
    """Giro, cambio de color y filtro de textura; devuelve la variante y lo que se le aplicó."""
    nombre_color = list(CAMBIOS_COLOR)[rng.integers(len(CAMBIOS_COLOR))]
    nombre_textura = list(FILTROS_TEXTURA)[rng.integers(len(FILTROS_TEXTURA))]
    imagen, angulo = girar(imagen, rng)
    imagen, parametro_color = CAMBIOS_COLOR[nombre_color](imagen, rng)
    imagen, parametro_textura = FILTROS_TEXTURA[nombre_textura](imagen, rng)
    return imagen, {
        "rotacion": round(angulo, 2),
        "color": nombre_color,
        "parametro_color": round(parametro_color, 4),
        "textura": nombre_textura,
        "parametro_textura": round(parametro_textura, 4),
    }


# --- Archivos ----------------------------------------------------------------------------

def es_generado(ruta: Path) -> bool:
    """True para los archivos que crea este script, incluidos los temporales a medio escribir."""
    nombre = ruta.name.removesuffix(SUFIJO_TEMPORAL)
    return Path(nombre).stem.endswith(MARCAS)


def con_marca(original: Path, marca: str) -> Path:
    destino = original.with_name(f"{original.stem}{marca}{original.suffix}")
    assert destino != original and es_generado(destino)
    return destino


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


def abrir_rgb(ruta: Path) -> Image.Image:
    with Image.open(ruta) as archivo:
        return archivo.convert("RGB")


def guardar(imagen: Image.Image, destino: Path, formato: str) -> None:
    """Escribe con un nombre temporal y lo renombra: un archivo a medio escribir nunca parece válido."""
    temporal = destino.with_name(destino.name + SUFIJO_TEMPORAL)
    imagen.save(temporal, format=formato, **OPCIONES_GUARDADO.get(formato, {}))
    os.replace(temporal, destino)


def escribir_registro(ruta: Path, campos: tuple[str, ...], filas: list[dict]) -> None:
    """El registro se escribe al final y de forma atómica: si aparece, la ejecución terminó."""
    temporal = ruta.with_name(ruta.name + SUFIJO_TEMPORAL)
    with open(temporal, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    os.replace(temporal, ruta)


def limpiar(dataset: Path) -> int:
    """Borra todo lo generado por ejecuciones anteriores. Nunca toca una original."""
    borrados = 0
    for carpeta in carpetas_personaje(dataset):
        for ruta in carpeta.iterdir():
            if ruta.is_file() and es_generado(ruta):
                ruta.unlink()
                borrados += 1
    for nombre in (REGISTRO, REGISTRO_PARECIDAS, REGISTRO_GEMELAS):
        registro = dataset / nombre
        for ruta in (registro, registro.with_name(registro.name + SUFIJO_TEMPORAL)):
            if ruta.exists():
                ruta.unlink()
    return borrados


def limpiar_revision(revision: Path) -> int:
    """Vacía la carpeta de revisión de una ejecución anterior: borra las imágenes copiadas, la marca
    y las subcarpetas que quedan vacías. Solo toca una carpeta que lleve la marca del script."""
    if not revision.exists():
        return 0
    if not (revision / MARCA_REVISION).is_file():
        if any(revision.iterdir()):
            sys.exit(f"{revision} existe y no la creó este script: indica otra carpeta con --revision")
        return 0
    borrados = 0
    # Primero los archivos y luego las carpetas, de la más profunda a la raíz
    for ruta in sorted(revision.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if ruta.is_file() and (ruta.suffix.lower() in EXTENSIONES_VALIDAS or ruta.name == MARCA_REVISION):
            ruta.unlink()
            borrados += 1
        elif ruta.is_dir() and not any(ruta.iterdir()):
            ruta.rmdir()
    if not any(revision.iterdir()):
        revision.rmdir()
    return borrados


# --- 1. Variantes alteradas --------------------------------------------------------------

def alterar(original: Path, semilla: int) -> dict:
    """Escribe la variante alterada de una imagen y devuelve su fila del registro."""
    # Generador propio de cada imagen: el resultado no depende del orden de proceso
    # (se procesa en paralelo) ni de las demás imágenes de la carpeta.
    rng = np.random.default_rng([semilla, *semilla_de(original.parent.name, original.name)])
    variante, aplicado = transformar(abrir_rgb(original), rng)
    destino = con_marca(original, MARCA_ALTERADA)
    guardar(variante, destino, FORMATOS[original.suffix.lower()])
    return {"personaje": original.parent.name, "original": original.name, "alterada": destino.name, **aplicado}


# --- 2. Imágenes parecidas ---------------------------------------------------------------

def miniatura(imagen: Image.Image) -> np.ndarray:
    """Vector con el que se comparan las imágenes: la imagen a 128 x 128, como la ve el notebook,
    reducida a 16 x 16 a color, centrada y de norma 1. El producto de dos es su similitud coseno,
    que no cambia con el brillo global pero sí con el encuadre."""
    reducida = imagen.resize((LADO_TRABAJO, LADO_TRABAJO), Image.Resampling.BILINEAR)
    vector = np.asarray(
        reducida.resize((LADO_MINIATURA, LADO_MINIATURA), Image.Resampling.BOX), dtype=np.float32
    ).ravel()
    vector = vector - vector.mean()
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def pares_parecidos(miniaturas: np.ndarray, umbral: float) -> tuple[np.ndarray, np.ndarray]:
    """Índices (i, j), con i < j, de los pares de imágenes con similitud >= umbral."""
    similitud = miniaturas @ miniaturas.T
    filas, columnas = np.nonzero(np.triu(similitud >= umbral, k=1))
    return filas, columnas


def agrupar(miniaturas: np.ndarray, umbral: float) -> tuple[list[list[int]], int]:
    """Grupos de imágenes parecidas (componentes conexas de los pares por encima del umbral),
    cada uno ordenado, y el número de pares. Un grupo encadena fotogramas seguidos de una escena
    aunque el primero y el último ya no se parezcan tanto entre sí."""
    filas, columnas = pares_parecidos(miniaturas, umbral)
    padre = list(range(len(miniaturas)))

    def raiz(nodo: int) -> int:
        while padre[nodo] != nodo:
            padre[nodo] = padre[padre[nodo]]
            nodo = padre[nodo]
        return nodo

    for a, b in zip(filas.tolist(), columnas.tolist()):
        raiz_a, raiz_b = raiz(a), raiz(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)

    grupos: dict[int, list[int]] = {}
    for nodo in range(len(miniaturas)):
        grupos.setdefault(raiz(nodo), []).append(nodo)
    return [grupo for grupo in grupos.values() if len(grupo) > 1], len(filas)


def diversificar(rutas: list[Path], miniaturas: np.ndarray, grupo: list[int], id_grupo: int,
                 semilla: int, umbral: float) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Deja intacta la primera imagen del grupo y reemplaza cada una de las demás por una variante
    __parecida que ya no se parece a ninguna versión final del grupo. Devuelve las filas del
    registro y la miniatura final de cada imagen reemplazada."""
    representante = grupo[0]
    finales = [miniaturas[representante]]
    filas, nuevas = [], {}
    for indice in grupo[1:]:
        original = rutas[indice]
        similitud_original = max(float(miniaturas[indice] @ miniaturas[otro]) for otro in grupo if otro != indice)
        imagen = abrir_rgb(original)
        for intento in range(1, INTENTOS_PARECIDA + 1):
            # Semilla distinta de la de la variante alterada de la misma imagen
            rng = np.random.default_rng(
                [semilla, *semilla_de(MARCA_PARECIDA, original.parent.name, original.name), intento]
            )
            variante, aplicado = transformar(imagen, rng)
            vector = miniatura(variante)
            similitud_final = max(float(vector @ otra) for otra in finales)
            if similitud_final < umbral:
                break
        destino = con_marca(original, MARCA_PARECIDA)
        guardar(variante, destino, FORMATOS[original.suffix.lower()])
        finales.append(vector)
        nuevas[indice] = vector
        filas.append({
            "personaje": original.parent.name,
            "grupo": id_grupo,
            "representante": rutas[representante].name,
            "original": original.name,
            "parecida": destino.name,
            "similitud_original": round(similitud_original, 4),
            "similitud_final": round(similitud_final, 4),
            "intentos": intento,
            **aplicado,
        })
    return filas, nuevas


# --- 3. Gemelas entre personajes ---------------------------------------------------------

def gemelas_entre_personajes(miniaturas: dict[str, np.ndarray], umbral: float
                             ) -> tuple[list[tuple[str, int, str, int, float]], set[tuple[str, int, str]]]:
    """Pares de originales de personajes distintos con similitud >= umbral, como
    (personaje_a, i, personaje_b, j, similitud), con índices dentro de cada carpeta y cada par una
    sola vez. Devuelve también las (personaje, i, otro_personaje) cuya imagen ya está en la carpeta de
    otro_personaje (similitud >= SIMILITUD_IDENTICA): esas no se proponen para revisión."""
    nombres = list(miniaturas)
    todas = np.concatenate([miniaturas[nombre] for nombre in nombres])
    personaje = np.concatenate([np.full(len(miniaturas[nombre]), k) for k, nombre in enumerate(nombres)])
    primera = np.concatenate(([0], np.cumsum([len(miniaturas[nombre]) for nombre in nombres])[:-1]))

    def local(indice: int) -> tuple[str, int]:
        k = int(personaje[indice])
        return nombres[k], indice - int(primera[k])

    pares, ya_en_carpeta = [], set()
    # Por bloques de filas: la matriz completa (decenas de miles al cuadrado) no cabe en memoria
    for desde in range(0, len(todas), BLOQUE_GEMELAS):
        similitud = todas[desde:desde + BLOQUE_GEMELAS] @ todas.T
        distinto = personaje[desde:desde + len(similitud), None] != personaje[None, :]
        filas, columnas = np.nonzero((similitud >= umbral) & distinto)
        for fila, columna in zip(filas.tolist(), columnas.tolist()):
            indice = desde + fila
            valor = float(similitud[fila, columna])
            if valor >= SIMILITUD_IDENTICA:
                ya_en_carpeta.add((*local(indice), nombres[int(personaje[columna])]))
            if indice < columna:
                pares.append((*local(indice), *local(columna), valor))
    return pares, ya_en_carpeta


def copiar_gemelas(pares: list[tuple[str, int, str, int, float]], ya_en_carpeta: set[tuple[str, int, str]],
                   originales: dict[str, list[Path]], revision: Path) -> tuple[list[dict], int]:
    """Copia cada imagen de un par a la subcarpeta del OTRO personaje en la carpeta de revisión,
    con su nombre original, y devuelve las filas del registro y el número de copias."""
    destinos: dict[Path, Path] = {}  # copia -> original copiado

    def copia(nombre: str, indice: int, otro: str) -> str:
        if (nombre, indice, otro) in ya_en_carpeta:
            return ""
        original = originales[nombre][indice]
        destino = revision / otro / original.name
        # El mismo nombre hacia la misma carpeta es la misma imagen guardada en dos carpetas
        # (los nombres llevan personaje, episodio e instante o un número): se copia una sola vez
        destinos.setdefault(destino, original)
        return destino.relative_to(revision).as_posix()

    filas = []
    for nombre_a, i, nombre_b, j, similitud in pares:
        filas.append({
            "personaje": nombre_a,
            "original": originales[nombre_a][i].name,
            "otro_personaje": nombre_b,
            "otro_original": originales[nombre_b][j].name,
            "similitud": round(similitud, 4),
            "copia_original": copia(nombre_a, i, nombre_b),
            "copia_otro": copia(nombre_b, j, nombre_a),
        })

    if destinos:
        revision.mkdir(parents=True, exist_ok=True)
        (revision / MARCA_REVISION).write_text(TEXTO_MARCA_REVISION, encoding="utf-8")
        for destino in {destino.parent for destino in destinos}:
            destino.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ejecutor:
            list(ejecutor.map(lambda par: shutil.copy2(par[1], par[0]), destinos.items()))
    return filas, len(destinos)


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
                        help="fracción de imágenes de cada personaje que reciben una variante alterada")
    parser.add_argument("--umbral", type=float, default=UMBRAL,
                        help="similitud coseno de las miniaturas a partir de la cual dos imágenes son parecidas")
    parser.add_argument("--umbral-gemelas", type=float, default=UMBRAL_GEMELAS,
                        help="similitud a partir de la cual dos imágenes de personajes distintos son gemelas")
    parser.add_argument("--revision", type=Path, default=None,
                        help="carpeta, fuera del dataset, donde se copian las gemelas para revisarlas "
                             f"(por defecto, {NOMBRE_REVISION} junto a la carpeta del dataset)")
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--limpiar", action="store_true",
                        help="solo borra lo generado y deja el dataset como estaba")
    args = parser.parse_args()

    dataset = args.dataset or Path(os.environ.get("SIMPSONS_DATASET", RUTA_POR_DEFECTO))
    dataset = dataset.expanduser().resolve()
    revision = (args.revision or dataset.parent / NOMBRE_REVISION).expanduser().resolve()
    if not 0 < args.fraccion <= 1:
        sys.exit("--fraccion debe estar en (0, 1]")
    if not 0 < args.umbral <= 1:
        sys.exit("--umbral debe estar en (0, 1]")
    if not 0 < args.umbral_gemelas <= 1:
        sys.exit("--umbral-gemelas debe estar en (0, 1]")
    if args.semilla < 0:
        sys.exit("--semilla debe ser un entero no negativo")
    if revision == dataset or revision.is_relative_to(dataset):
        sys.exit("--revision debe quedar fuera del dataset: cada subcarpeta del dataset es un personaje")

    carpetas = validar(dataset)
    print(f"Dataset: {dataset}  ({len(carpetas)} personajes)")

    borrados = limpiar(dataset)
    borrados_revision = limpiar_revision(revision)
    print(f"Limpieza: {borrados} archivos generados en ejecuciones anteriores eliminados"
          f" y {borrados_revision} de la carpeta de revisión")
    if args.limpiar:
        return

    inicio = time.perf_counter()
    originales = {carpeta.name: originales_de(carpeta) for carpeta in carpetas}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ejecutor:
        # Imágenes parecidas: miniaturas de todas las originales, grupos y variantes de reemplazo
        todas = [ruta for rutas in originales.values() for ruta in rutas]
        vectores = list(ejecutor.map(lambda ruta: miniatura(abrir_rgb(ruta)), todas))
        miniaturas, desde = {}, 0
        for nombre, rutas in originales.items():
            miniaturas[nombre] = np.stack(vectores[desde:desde + len(rutas)])
            desde += len(rutas)
        print(f"Comparadas {len(todas)} imágenes en {time.perf_counter() - inicio:.1f} s")

        trabajos, pares_antes, grupos_por_personaje = [], 0, {}
        for nombre in originales:
            grupos, pares = agrupar(miniaturas[nombre], args.umbral)
            pares_antes += pares
            grupos_por_personaje[nombre] = len(grupos)
            trabajos.extend((nombre, grupo) for grupo in grupos)
        def diversificar_grupo(numerado):
            id_grupo, (nombre, grupo) = numerado
            return diversificar(originales[nombre], miniaturas[nombre], grupo, id_grupo,
                                args.semilla, args.umbral)

        resultados = list(ejecutor.map(diversificar_grupo, enumerate(trabajos, start=1)))
        filas_parecidas = [fila for filas, _ in resultados for fila in filas]

        # Pares que quedan por encima del umbral con las versiones finales de cada personaje
        finales = {nombre: matriz.copy() for nombre, matriz in miniaturas.items()}
        for (nombre, _), (_, nuevas) in zip(trabajos, resultados):
            for indice, vector in nuevas.items():
                finales[nombre][indice] = vector
        pares_despues = sum(len(pares_parecidos(matriz, args.umbral)[0]) for matriz in finales.values())

        # Variantes alteradas: una fracción al azar de las originales de cada personaje
        elegidas = []
        resumen = []
        for nombre, rutas in originales.items():
            cantidad = int(args.fraccion * len(rutas) + 0.5)  # redondeo al entero más cercano
            rng = np.random.default_rng([args.semilla, *semilla_de(nombre)])
            indices = np.sort(rng.choice(len(rutas), size=cantidad, replace=False))
            elegidas.extend(rutas[i] for i in indices)
            parecidas = sum(1 for fila in filas_parecidas if fila["personaje"] == nombre)
            resumen.append((nombre, len(rutas), cantidad, grupos_por_personaje[nombre], parecidas))
        filas_alteradas = list(ejecutor.map(lambda ruta: alterar(ruta, args.semilla), elegidas))

    # Gemelas entre personajes: se comparan las miniaturas de las originales, no las parecidas
    inicio_gemelas = time.perf_counter()
    pares_gemelas, ya_en_carpeta = gemelas_entre_personajes(miniaturas, args.umbral_gemelas)
    filas_gemelas, copias = copiar_gemelas(pares_gemelas, ya_en_carpeta, originales, revision)
    segundos_gemelas = time.perf_counter() - inicio_gemelas

    escribir_registro(dataset / REGISTRO, CAMPOS_REGISTRO, filas_alteradas)
    escribir_registro(dataset / REGISTRO_PARECIDAS, CAMPOS_PARECIDAS, filas_parecidas)
    escribir_registro(dataset / REGISTRO_GEMELAS, CAMPOS_GEMELAS, filas_gemelas)

    ancho = max(len(nombre) for nombre, *_ in resumen)
    print(f"\n{'personaje':<{ancho}}  originales  alteradas  grupos_parecidas  parecidas")
    for nombre, n_originales, n_alteradas, n_grupos, n_parecidas in resumen:
        print(f"{nombre:<{ancho}}  {n_originales:>10}  {n_alteradas:>9}  {n_grupos:>16}  {n_parecidas:>9}")
    totales = [sum(fila[i] for fila in resumen) for i in range(1, 5)]
    print(f"{'TOTAL':<{ancho}}  {totales[0]:>10}  {totales[1]:>9}  {totales[2]:>16}  {totales[3]:>9}")

    sin_resolver = sum(1 for fila in filas_parecidas if fila["similitud_final"] >= args.umbral)
    print(f"\nPares casi idénticos dentro de cada personaje (similitud >= {args.umbral:.2f}): "
          f"{pares_antes} antes -> {pares_despues} después")
    if sin_resolver:
        print(f"  {sin_resolver} parecidas siguen sobre el umbral tras {INTENTOS_PARECIDA} sorteos")
    imagenes_con_gemela = len({(fila[k], fila[o]) for fila in filas_gemelas
                               for k, o in (("personaje", "original"), ("otro_personaje", "otro_original"))})
    print(f"Gemelas entre personajes (similitud >= {args.umbral_gemelas:.2f}): {len(filas_gemelas)} pares en "
          f"{imagenes_con_gemela} imágenes ({segundos_gemelas:.1f} s)")
    if copias:
        print(f"  {copias} copias para revisar en {revision}")
    print(f"{len(filas_alteradas)} alteradas y {len(filas_parecidas)} parecidas en "
          f"{time.perf_counter() - inicio:.1f} s. Registros: {dataset / REGISTRO}, {dataset / REGISTRO_PARECIDAS}, "
          f"{dataset / REGISTRO_GEMELAS}")


if __name__ == "__main__":
    main()
