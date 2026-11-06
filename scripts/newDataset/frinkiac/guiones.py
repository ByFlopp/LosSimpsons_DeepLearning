"""Obtiene los guiones de "The Simpsons by the Data" (simpsons_script_lines.csv y simpsons_episodes.csv).

Orden de búsqueda, del más completo al menos:
    1. Las rutas indicadas con --guiones / --episodios.
    2. Los CSV en carpetas habituales (junto al dataset, Descargas, Escritorio, la raíz del repo) o dentro
       de un .zip descargado de Kaggle en esas carpetas.
    3. La API de Kaggle, si hay credenciales (KAGGLE_USERNAME y KAGGLE_KEY, o ~/.kaggle/kaggle.json).
    4. La copia de TidyTuesday en GitHub. No pide credenciales, pero solo trae los episodios desde 2010
       (temporada 21 en adelante): se usa como último recurso y se avisa.

Los ids de las líneas son los mismos en todas las copias, así que cambiar después a una más completa
solo agrega líneas: lo ya recolectado se conserva.
"""

import csv
import io
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from comun import escribir_atomico, sha256_hex

log = logging.getLogger("frinkiac.guiones")

LINEAS, EPISODIOS = "simpsons_script_lines.csv", "simpsons_episodes.csv"
KAGGLE = "prashant111/the-simpsons-dataset"
URL_KAGGLE = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE}"
URL_TIDYTUESDAY = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-02-04/"
COLUMNAS_LINEAS = {"id", "episode_id", "timestamp_in_ms", "speaking_line", "raw_character_text", "spoken_words"}
COLUMNAS_EPISODIOS = {"id", "season", "number_in_season"}


@dataclass
class Guiones:
    lineas: Path
    episodios: Path
    fuente: str
    completo: bool  # False si es la copia parcial de TidyTuesday


def _columnas(ruta: Path) -> set[str]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(ruta, newline="", encoding="utf-8", errors="replace") as archivo:
        return set(next(csv.reader(archivo), []))


def validos(lineas: Path, episodios: Path) -> bool:
    try:
        return COLUMNAS_LINEAS <= _columnas(lineas) and COLUMNAS_EPISODIOS <= _columnas(episodios)
    except OSError:
        return False


def carpetas_habituales(dataset: Path, raiz_repo: Path) -> list[Path]:
    casa = Path.home()
    candidatas = [dataset.parent, dataset.parent.parent, raiz_repo, casa / "Downloads", casa / "Descargas",
                  casa / "Desktop", casa / "Escritorio"]
    if os.environ.get("FRINKIAC_GUIONES"):
        candidatas.insert(0, Path(os.environ["FRINKIAC_GUIONES"]))
    vistas, unicas = set(), []
    for carpeta in candidatas:
        clave = str(carpeta.resolve()) if carpeta.exists() else None
        if clave and clave not in vistas:
            vistas.add(clave)
            unicas.append(carpeta)
    return unicas


def _buscar_csv(carpetas: list[Path]) -> tuple[Path, Path] | None:
    for carpeta in carpetas:
        for lugar in (carpeta, *sorted(p for p in carpeta.iterdir() if p.is_dir() and not p.name.startswith("."))):
            lineas, episodios = lugar / LINEAS, lugar / EPISODIOS
            if lineas.is_file() and episodios.is_file() and validos(lineas, episodios):
                return lineas, episodios
    return None


def _buscar_zip(carpetas: list[Path]) -> Path | None:
    for carpeta in carpetas:
        for comprimido in sorted(carpeta.glob("*.zip")):
            nombre = comprimido.name.lower()
            if "simpson" not in nombre and not nombre.startswith("archive"):
                continue
            try:
                with zipfile.ZipFile(comprimido) as zip_:
                    nombres = {Path(n).name for n in zip_.namelist()}
                if {LINEAS, EPISODIOS} <= nombres:
                    return comprimido
            except (zipfile.BadZipFile, OSError):
                continue
    return None


def _extraer(zip_: zipfile.ZipFile, destino: Path) -> tuple[Path, Path]:
    rutas = {}
    for nombre in zip_.namelist():
        base = Path(nombre).name
        if base in (LINEAS, EPISODIOS):
            escribir_atomico(destino / base, zip_.read(nombre))
            rutas[base] = destino / base
    return rutas[LINEAS], rutas[EPISODIOS]


def _credenciales_kaggle() -> tuple[str, str] | None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    archivo = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    try:
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        return datos["username"], datos["key"]
    except (OSError, ValueError, KeyError):
        return None


def descargar_kaggle(destino: Path, sesion=requests) -> tuple[Path, Path] | None:
    credenciales = _credenciales_kaggle()
    if not credenciales:
        return None
    try:
        respuesta = sesion.get(URL_KAGGLE, auth=credenciales, timeout=(10, 300))
        respuesta.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(respuesta.content)) as zip_:
            return _extraer(zip_, destino)
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as error:
        log.warning("no se pudo descargar de Kaggle", extra={"datos": {"error": str(error)}})
        return None


def descargar_tidytuesday(destino: Path, sesion=requests) -> tuple[Path, Path]:
    rutas = []
    for nombre in (LINEAS, EPISODIOS):
        respuesta = sesion.get(URL_TIDYTUESDAY + nombre, timeout=(10, 300))
        respuesta.raise_for_status()
        escribir_atomico(destino / nombre, respuesta.content)
        rutas.append(destino / nombre)
    return rutas[0], rutas[1]


def obtener(destino: Path, carpetas: list[Path], lineas: str | None = None, episodios: str | None = None,
            sesion=requests) -> Guiones:
    """Deja en `destino` una copia de los dos CSV y dice de dónde salieron."""
    destino.mkdir(parents=True, exist_ok=True)
    registro = destino / "fuente.json"
    if lineas or episodios:
        if not (lineas and episodios):
            raise SystemExit("Indica los dos archivos: --guiones y --episodios")
        if not validos(Path(lineas), Path(episodios)):
            raise SystemExit(f"{lineas} o {episodios} no tienen las columnas del guion de Kaggle")
        return _copiar(Path(lineas), Path(episodios), destino, registro, f"archivos indicados ({lineas})", True)
    encontrados = _buscar_csv(carpetas)
    if encontrados:
        return _copiar(*encontrados, destino, registro, f"local ({encontrados[0].parent})", True)
    comprimido = _buscar_zip(carpetas)
    if comprimido:
        with zipfile.ZipFile(comprimido) as zip_:
            rutas = _extraer(zip_, destino)
        return _registrar(*rutas, registro, f"zip ({comprimido})", True)
    previo = _previo(destino, registro)
    if previo and previo.completo:
        return previo
    kaggle = descargar_kaggle(destino, sesion)
    if kaggle:
        return _registrar(*kaggle, registro, f"Kaggle ({KAGGLE})", True)
    if previo:
        return previo
    rutas = descargar_tidytuesday(destino, sesion)
    return _registrar(*rutas, registro, "TidyTuesday (solo episodios desde 2010)", False)


def _previo(destino: Path, registro: Path) -> Guiones | None:
    lineas, episodios = destino / LINEAS, destino / EPISODIOS
    if not (registro.is_file() and validos(lineas, episodios)):
        return None
    datos = json.loads(registro.read_text(encoding="utf-8"))
    return Guiones(lineas, episodios, datos["fuente"], datos["completo"])


def _copiar(lineas: Path, episodios: Path, destino: Path, registro: Path, fuente: str, completo: bool) -> Guiones:
    copias = []
    for origen in (lineas, episodios):
        copia = destino / origen.name if origen.name in (LINEAS, EPISODIOS) else destino / (
            LINEAS if origen == lineas else EPISODIOS)
        if origen.resolve() != copia.resolve():
            datos = origen.read_bytes()
            if not copia.is_file() or sha256_hex(copia.read_bytes()) != sha256_hex(datos):
                escribir_atomico(copia, datos)
        copias.append(copia)
    return _registrar(copias[0], copias[1], registro, fuente, completo)


def _registrar(lineas: Path, episodios: Path, registro: Path, fuente: str, completo: bool) -> Guiones:
    escribir_atomico(registro, json.dumps({"fuente": fuente, "completo": completo}, ensure_ascii=False).encode())
    return Guiones(lineas, episodios, fuente, completo)
