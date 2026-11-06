"""Inventario de las imágenes ORIGINALES del dataset local, para que la EC2 no necesite el dataset.

Solo cuentan las originales: las variantes __alterada y __parecida (y sus .tmp) las genera
generar_imagenes_alteradas.py DESPUÉS de completar las 1000, así que se ignoran aquí, en el conteo y en
la comparación de unicidad. Se usa originales_de() del mismo script para que la regla sea idéntica.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from comun import abrir_rgb, carpetas_personaje, clave_de_nombre, escribir_atomico, miniatura, originales_de


@dataclass
class Inventario:
    objetivo: int
    conteos: dict[str, int]
    claves: dict[str, set[tuple[str, int]]] = field(default_factory=dict)
    miniaturas: dict[str, np.ndarray] = field(default_factory=dict)

    def por_debajo(self) -> list[str]:
        """Personajes con menos originales que el objetivo, del más escaso al más cercano."""
        return sorted((p for p, n in self.conteos.items() if n < self.objetivo), key=lambda p: (self.conteos[p], p))


def _miniatura_de(ruta: Path) -> np.ndarray:
    return miniatura(abrir_rgb(ruta))


def originales_propias(carpeta: Path, excluir: set[str]) -> list[Path]:
    """Originales de la carpeta sin las que vienen de esta misma ampliación (ya exportadas): esas ya
    están contadas en staging/ y contarlas dos veces adelantaría la meta."""
    return [r for r in originales_de(carpeta) if r.name not in excluir]


def conteos_actuales(dataset: Path, excluir: dict[str, set[str]] | None = None) -> dict[str, int]:
    """Conteo rápido (sin abrir imágenes) para saber si el dataset cambió desde el inventario."""
    excluir = excluir or {}
    conteos = {}
    for carpeta in carpetas_personaje(dataset):
        n = len(originales_propias(carpeta, excluir.get(carpeta.name, set())))
        if n:
            conteos[carpeta.name] = n
    return conteos


def construir(dataset: Path, objetivo: int, hilos: int = 8, progreso=print,
              excluir: dict[str, set[str]] | None = None) -> Inventario:
    conteos, claves, miniaturas = {}, {}, {}
    excluir = excluir or {}
    for carpeta in carpetas_personaje(dataset):
        originales = originales_propias(carpeta, excluir.get(carpeta.name, set()))
        if not originales:
            continue
        personaje = carpeta.name
        conteos[personaje] = len(originales)
        claves[personaje] = {c for c in (clave_de_nombre(r.name) for r in originales) if c}
        # Las miniaturas solo hacen falta para los personajes que se van a ampliar
        if len(originales) < objetivo:
            with ThreadPoolExecutor(hilos) as grupo:
                miniaturas[personaje] = np.stack(list(grupo.map(_miniatura_de, originales)))
            progreso(f"{personaje}: {len(originales)} originales")
    return Inventario(objetivo, conteos, claves, miniaturas)


def guardar(inventario: Inventario, ruta: Path) -> None:
    personajes = sorted(inventario.conteos)
    indice = {p: i for i, p in enumerate(personajes)}
    clave_p, clave_e, clave_t = [], [], []
    for p in personajes:
        for episodio, ts in sorted(inventario.claves.get(p, ())):
            clave_p.append(indice[p])
            clave_e.append(episodio)
            clave_t.append(ts)
    mini_p, mini = [], []
    for p in personajes:
        bloque = inventario.miniaturas.get(p)
        if bloque is not None and len(bloque):
            mini_p.extend([indice[p]] * len(bloque))
            mini.append(bloque.astype(np.float16))
    temporal = ruta.with_name(ruta.name + ".partial.npz")
    np.savez_compressed(
        temporal,
        objetivo=np.int64(inventario.objetivo),
        personajes=np.array(personajes, dtype=str),
        conteos=np.array([inventario.conteos[p] for p in personajes], dtype=np.int64),
        clave_personaje=np.array(clave_p, dtype=np.int32),
        clave_episodio=np.array(clave_e, dtype="U16"),
        clave_ts=np.array(clave_t, dtype=np.int64),
        mini_personaje=np.array(mini_p, dtype=np.int32),
        mini=np.concatenate(mini) if mini else np.zeros((0, 0), np.float16),
    )
    escribir_atomico(ruta, temporal.read_bytes())
    temporal.unlink()


def cargar(ruta: Path) -> Inventario:
    with np.load(ruta, allow_pickle=False) as datos:
        personajes = [str(p) for p in datos["personajes"]]
        conteos = {p: int(n) for p, n in zip(personajes, datos["conteos"])}
        claves: dict[str, set[tuple[str, int]]] = {p: set() for p in personajes}
        for i, episodio, ts in zip(datos["clave_personaje"], datos["clave_episodio"], datos["clave_ts"]):
            claves[personajes[int(i)]].add((str(episodio), int(ts)))
        miniaturas = {}
        mini_p, mini = datos["mini_personaje"], datos["mini"].astype(np.float32)
        for i, p in enumerate(personajes):
            seleccion = mini_p == i
            if seleccion.any():
                miniaturas[p] = mini[seleccion]
        return Inventario(int(datos["objetivo"]), conteos, claves, miniaturas)
