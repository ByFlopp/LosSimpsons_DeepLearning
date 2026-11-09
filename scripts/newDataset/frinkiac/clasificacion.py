"""Clasificación automática de por_revisar/ con la CNN: reparte las imágenes y traslada la decisión.

`pnpm frinkiac clasificar` (clasificar_con_cnn.py, en el entorno .venv-gpu) pasa por la CNN cada imagen
pendiente de por_revisar/<personaje>/ y la deja en uno de tres grupos:

    frinkiac_ampliacion/
    ├── por_revisar/<personaje>/       las imágenes de siempre (no cambian al clasificar)
    └── clasificacion/
        ├── seguras/<personaje>/       la CNN reconoce al personaje con confianza alta: se aprueban
        ├── dudosas/<personaje>/       AQUÍ SE REVISA: borra las malas y, al terminar, _PENDIENTE.txt
        ├── ruido/<personaje>/         la CNN no lo ve en ninguna vista: se rechazan
        └── _registro.csv              grupo y probabilidades de cada imagen clasificada

Las tres carpetas tienen enlaces duros a las imágenes de por_revisar/ (sin copias ni espacio extra).
Decide la carpeta en la que queda cada imagen: seguras/ y dudosas/ se aprueban; ruido/ y lo borrado se
rechazan. Para rescatar una imagen de ruido/, muévela a seguras/.

Cuando dudosas/<personaje>/ ya no tiene _PENDIENTE.txt (o no hubo dudosas), el siguiente `pnpm frinkiac`
traslada la decisión a por_revisar/<personaje>/: borra ahí las rechazadas y el _PENDIENTE.txt. Después la
aplica igual que una revisión a mano: aprueba, rechaza y repone.

Este módulo solo usa la biblioteca estándar: lo importan el flujo (.venv-frinkiac) y el clasificador
(.venv-gpu).
"""

import csv
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CARPETA = "clasificacion"
SEGURAS, DUDOSAS, RUIDO = "seguras", "dudosas", "ruido"
GRUPOS = (SEGURAS, DUDOSAS, RUIDO)
APRUEBAN = (SEGURAS, DUDOSAS)
REGISTRO = "_registro.csv"
COLUMNAS = ("personaje", "archivo", "grupo", "p_completa", "p_recorte", "prediccion", "p_prediccion", "fecha")
# Los mismos nombres que usa ampliar_con_frinkiac.py en por_revisar/<personaje>/
MARCA_PENDIENTE = "_PENDIENTE.txt"
REGISTRO_REVISION = "_descargadas.json"

TEXTO_DUDOSAS = """Dudosas de {personaje}

La CNN no está segura de que {personaje} aparezca en estas imágenes.

1. Borra las imágenes en las que {personaje} NO aparece.
2. Cuando termines esta carpeta, borra también este archivo ({marca}).
3. Ejecuta 'pnpm frinkiac'. Se aprueban las que queden aquí y en seguras/{personaje}/. Se rechazan
   las borradas y las de ruido/{personaje}/, y el flujo las repone.

Si 'pnpm frinkiac clasificar' agrega más dudosas, este archivo vuelve a aparecer.
"""

LEEME = """Clasificación automática con la CNN ('pnpm frinkiac clasificar')

seguras/<personaje>/   la CNN reconoce al personaje con confianza alta. Se aprueban.
dudosas/<personaje>/   AQUÍ SE REVISA: borra las imágenes en las que el personaje no aparece y, al
                       terminar, borra su _PENDIENTE.txt.
ruido/<personaje>/     la CNN no ve al personaje. Se rechazan; si alguna sirve, muévela a seguras/.
_registro.csv          grupo y probabilidades de cada imagen clasificada.

Son enlaces a las imágenes de por_revisar/: borrar aquí no borra nada de staging/. Cuando una carpeta de
dudosas queda sin _PENDIENTE.txt, 'pnpm frinkiac' aplica la decisión y repone las rechazadas.
"""


@dataclass(frozen=True)
class Umbrales:
    """Probabilidad que da la CNN al personaje de la carpeta."""

    segura: float = 0.80  # en el fotograma completo
    segura_recorte: float = 0.90  # o en alguno de los recortes cuadrados
    ruido: float = 0.05  # por debajo en todas las vistas

    def validar(self) -> None:
        if not all(0 < self.ruido < segura <= 1 for segura in (self.segura, self.segura_recorte)):
            raise ValueError(f"Umbrales incoherentes: hace falta 0 < ruido < segura <= 1 ({self})")


@dataclass(frozen=True)
class Resultado:
    archivo: str
    p_completa: float
    p_recorte: float
    prediccion: str
    p_prediccion: float
    grupo: str


@dataclass(frozen=True)
class Sincronizacion:
    personaje: str
    estado: str  # trasladada, dudosas_pendientes, sin_clasificar
    aprobadas: int = 0
    rechazadas: int = 0
    faltan: int = 0  # dudosas por revisar o imágenes sin clasificar


def grupo_de(p_completa: float, p_recorte: float, umbrales: Umbrales) -> str:
    if p_completa >= umbrales.segura or p_recorte >= umbrales.segura_recorte:
        return SEGURAS
    if max(p_completa, p_recorte) < umbrales.ruido:
        return RUIDO
    return DUDOSAS


def raiz_de(por_revisar: Path) -> Path:
    """clasificacion/ queda junto a por_revisar/."""
    return Path(por_revisar).parent / CARPETA


def leer_registro(raiz: Path) -> dict[tuple[str, str], dict]:
    ruta = raiz / REGISTRO
    if not ruta.exists():
        return {}
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return {(fila["personaje"], fila["archivo"]): fila for fila in csv.DictReader(archivo)}


def _descargadas(carpeta: Path) -> set[str]:
    ruta = carpeta / REGISTRO_REVISION
    if not ruta.exists():
        return set()
    return set(json.loads(ruta.read_text(encoding="utf-8"))["archivos"])


def _en_revision(carpeta: Path) -> set[str]:
    """Imágenes de por_revisar/<personaje>/ que esperan una decisión. Una carpeta sin _PENDIENTE.txt ya
    se revisó a mano y no se toca."""
    if not (carpeta / MARCA_PENDIENTE).exists():
        return set()
    return {r.name for r in carpeta.glob("*.jpg")} & _descargadas(carpeta)


def pendientes(por_revisar: Path, personaje: str, registro: dict) -> list[Path]:
    """Imágenes en revisión de un personaje que todavía no se clasifican."""
    carpeta = Path(por_revisar) / personaje
    return [carpeta / nombre for nombre in sorted(_en_revision(carpeta)) if (personaje, nombre) not in registro]


def _enlazar(origen: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return
    try:
        os.link(origen, destino)
    except OSError:  # otro volumen o un sistema de archivos sin enlaces duros
        shutil.copy2(origen, destino)


def repartir(por_revisar: Path, personaje: str, resultados: list[Resultado]) -> Counter:
    """Enlaza cada imagen en la carpeta de su grupo y la anota en _registro.csv. Si hay dudosas nuevas,
    deja el _PENDIENTE.txt de dudosas/<personaje>/."""
    raiz = raiz_de(por_revisar)
    if not (raiz / "LEEME.txt").exists():
        raiz.mkdir(parents=True, exist_ok=True)
        (raiz / "LEEME.txt").write_text(LEEME, encoding="utf-8")
    for r in resultados:
        for grupo in GRUPOS:  # una imagen está en un solo grupo, aunque se reclasifique tras un corte
            if grupo != r.grupo:
                (raiz / grupo / personaje / r.archivo).unlink(missing_ok=True)
        _enlazar(Path(por_revisar) / personaje / r.archivo, raiz / r.grupo / personaje / r.archivo)
    cuenta = Counter(r.grupo for r in resultados)
    if cuenta[DUDOSAS]:
        (raiz / DUDOSAS / personaje / MARCA_PENDIENTE).write_text(
            TEXTO_DUDOSAS.format(personaje=personaje, marca=MARCA_PENDIENTE), encoding="utf-8")
    # El registro se escribe al final: si se corta antes, la próxima vez se vuelven a clasificar
    ruta = raiz / REGISTRO
    nuevo = not ruta.exists()
    fecha = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow(COLUMNAS)
        for r in resultados:
            escritor.writerow([personaje, r.archivo, r.grupo, f"{r.p_completa:.4f}", f"{r.p_recorte:.4f}",
                               r.prediccion, f"{r.p_prediccion:.4f}", fecha])
        archivo.flush()
        os.fsync(archivo.fileno())
    return cuenta


def _jpgs(carpeta: Path) -> set[str]:
    return {r.name for r in carpeta.glob("*.jpg")} if carpeta.is_dir() else set()


def _vaciar(raiz: Path, personaje: str) -> None:
    for grupo in GRUPOS:
        carpeta = raiz / grupo / personaje
        if not carpeta.is_dir():
            continue
        for ruta in carpeta.glob("*.jpg"):
            ruta.unlink()
        (carpeta / MARCA_PENDIENTE).unlink(missing_ok=True)
        if not any(carpeta.iterdir()):
            carpeta.rmdir()


def sincronizar(por_revisar: Path) -> list[Sincronizacion]:
    """Traslada a por_revisar/ la decisión de cada personaje que ya está listo. Borra en
    por_revisar/<personaje>/ las rechazadas y el _PENDIENTE.txt, y vacía sus carpetas de clasificación.
    Un personaje está listo cuando todas sus imágenes en revisión están clasificadas y sus dudosas,
    revisadas."""
    from almacen import bloqueo_archivo  # importa botocore: solo existe en el entorno del flujo

    por_revisar = Path(por_revisar)
    raiz = raiz_de(por_revisar)
    if not raiz.is_dir():
        return []
    registro = leer_registro(raiz)
    salida = []
    for personaje in sorted({p for p, _ in registro}):
        carpeta = por_revisar / personaje
        # Mismo cerrojo que usa ampliar_con_frinkiac.py al agregar imágenes a la carpeta
        with bloqueo_archivo(por_revisar / ".bloqueos" / personaje):
            en_revision = _en_revision(carpeta)
            clasificadas = {nombre for nombre in en_revision if (personaje, nombre) in registro}
            if not clasificadas:
                continue
            sin_clasificar = en_revision - clasificadas
            if sin_clasificar:
                salida.append(Sincronizacion(personaje, "sin_clasificar", faltan=len(sin_clasificar)))
                continue
            if (raiz / DUDOSAS / personaje / MARCA_PENDIENTE).exists():
                faltan = len(_jpgs(raiz / DUDOSAS / personaje) & en_revision)
                salida.append(Sincronizacion(personaje, "dudosas_pendientes", faltan=faltan))
                continue
            aprobadas = set().union(*(_jpgs(raiz / grupo / personaje) for grupo in APRUEBAN)) & en_revision
            rechazadas = en_revision - aprobadas
            for nombre in rechazadas:
                (carpeta / nombre).unlink(missing_ok=True)
            (carpeta / MARCA_PENDIENTE).unlink(missing_ok=True)
            _vaciar(raiz, personaje)
            salida.append(Sincronizacion(personaje, "trasladada", len(aprobadas), len(rechazadas)))
    return salida


def resumen(por_revisar: Path) -> dict[str, Counter]:
    """Por personaje con revisión abierta: cuántas hay en cada grupo, cuántas sin clasificar y si sus
    dudosas siguen pendientes (clave 'dudosas_pendientes': 0 o 1)."""
    por_revisar = Path(por_revisar)
    raiz = raiz_de(por_revisar)
    registro = leer_registro(raiz)
    salida = {}
    if not por_revisar.is_dir():
        return salida
    for carpeta in sorted(p for p in por_revisar.iterdir() if p.is_dir() and not p.name.startswith(".")):
        en_revision = _en_revision(carpeta)
        if not en_revision:
            continue
        cuenta = Counter({grupo: len(_jpgs(raiz / grupo / carpeta.name) & en_revision) for grupo in GRUPOS})
        cuenta["sin_clasificar"] = sum(1 for nombre in en_revision if (carpeta.name, nombre) not in registro)
        cuenta["dudosas_pendientes"] = int((raiz / DUDOSAS / carpeta.name / MARCA_PENDIENTE).exists())
        salida[carpeta.name] = cuenta
    return salida
