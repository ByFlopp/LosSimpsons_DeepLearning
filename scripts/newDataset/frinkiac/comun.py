"""Piezas compartidas por los módulos de la ampliación con Frinkiac.

Reutiliza del script de preparación del dataset las mismas reglas que ya usa el proyecto: qué es una
imagen original (las variantes sintéticas __alterada y __parecida nunca cuentan), cómo se reduce una
imagen a su miniatura de comparación y el umbral a partir del cual dos imágenes son casi idénticas.
"""

import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

# scripts/newDataset, para importar generar_imagenes_alteradas sin instalarlo como paquete
CARPETA_SCRIPTS = Path(__file__).resolve().parent.parent
if str(CARPETA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CARPETA_SCRIPTS))

from generar_imagenes_alteradas import (  # noqa: E402
    UMBRAL,
    abrir_rgb,
    carpetas_personaje,
    es_generado,
    miniatura,
    originales_de,
)

__all__ = [
    "UMBRAL", "abrir_rgb", "carpetas_personaje", "es_generado", "miniatura", "originales_de",
    "OBJETIVO", "MARGEN", "REPO", "MENCIONES", "nombre_fotograma", "clave_de_nombre", "sha256_hex",
    "escribir_atomico", "meta_recoleccion", "configurar_logs", "ahora",
]

OBJETIVO = 1000
# Fracción extra que se recolecta sobre lo que falta para absorber los rechazos de la revisión manual
MARGEN = 0.15
REPO = "ByFlopp/LosSimpsons_DeepLearning"
MENCIONES = ("DiegoPyLL",)
# Mismo formato de nombre que ya usan las imágenes de Frinkiac del dataset:
#     apu_nahasapeemapetilon__S21E21__t0659367.jpg
PATRON_FOTOGRAMA = re.compile(r"__(S\d{2,}E\d{2,})__t(\d+)")
SUFIJO_TEMPORAL = ".tmp"


def nombre_fotograma(personaje: str, episodio: str, ts: int) -> str:
    return f"{personaje}__{episodio}__t{ts:07d}.jpg"


def clave_de_nombre(nombre: str) -> tuple[str, int] | None:
    """(episodio, timestamp) de un archivo con nombre de Frinkiac, o None si no lo tiene."""
    encontrado = PATRON_FOTOGRAMA.search(nombre)
    if not encontrado:
        return None
    return encontrado.group(1), int(encontrado.group(2))


def sha256_hex(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def escribir_atomico(ruta: Path, datos: bytes) -> None:
    """Escribe con nombre temporal, fuerza el volcado a disco y renombra: un archivo a medio escribir
    nunca parece válido (mismo patrón que guardar() del script de preparación)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Nombre único: dos escritores del mismo archivo nunca comparten temporal
    temporal = ruta.with_name(f"{ruta.name}.{os.getpid()}-{threading.get_ident()}{SUFIJO_TEMPORAL}")
    with open(temporal, "wb") as archivo:
        archivo.write(datos)
        archivo.flush()
        os.fsync(archivo.fileno())
    # En Windows el reemplazo falla mientras otro proceso (un lector, el antivirus, el indexador) tiene
    # abierto el destino: se reintenta unos segundos antes de rendirse
    for intento in range(60):
        try:
            os.replace(temporal, ruta)
            return
        except PermissionError:
            if os.name != "nt" or intento == 59:
                temporal.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * min(intento + 1, 10))


def meta_recoleccion(objetivo: int, margen: float) -> int:
    """Total (originales + nuevas vigentes) al que se recolecta antes de pasar a revisión."""
    return math.ceil(objetivo * (1 + margen))


def ahora() -> float:
    return time.time()


class FormatoJson(logging.Formatter):
    """Una línea JSON por evento: journald y CloudWatch la indexan sin configuración extra."""

    def format(self, registro: logging.LogRecord) -> str:
        evento = {
            "t": round(registro.created, 3),
            "nivel": registro.levelname,
            "modulo": registro.name,
            "msg": registro.getMessage(),
        }
        extra = getattr(registro, "datos", None)
        if extra:
            evento.update(extra)
        if registro.exc_info:
            evento["error"] = self.formatException(registro.exc_info)
        return json.dumps(evento, ensure_ascii=False)


class FormatoTexto(logging.Formatter):
    """Una línea legible por evento, para la consola del PC: 14:03:22 subida personaje=louie ..."""

    def format(self, registro: logging.LogRecord) -> str:
        extra = getattr(registro, "datos", None) or {}
        detalles = " ".join(f"{k}={v}" for k, v in extra.items() if k != "lineas")
        nivel = "" if registro.levelno == logging.INFO else f"[{registro.levelname}] "
        linea = f"{time.strftime('%H:%M:%S', time.localtime(registro.created))} {nivel}{registro.getMessage()}"
        if detalles:
            linea += f"  {detalles}"
        if registro.exc_info:
            linea += "\n" + self.formatException(registro.exc_info)
        return linea


def configurar_logs(nivel: str = "INFO", formato: str = "json") -> None:
    manejador = logging.StreamHandler(sys.stdout)
    manejador.setFormatter(FormatoTexto() if formato == "texto" else FormatoJson())
    raiz = logging.getLogger()
    raiz.handlers[:] = [manejador]
    raiz.setLevel(nivel)
    # botocore y urllib3 son muy verbosos en INFO
    for ruidoso in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
