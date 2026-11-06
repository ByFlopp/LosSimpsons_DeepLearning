"""Estado local de la recolección en SQLite.

Cada transición se escribe en su propia transacción con journal WAL y synchronous=FULL: si la
instancia se apaga de golpe, la base queda en el último estado confirmado. El estado local nunca es la
única copia: S3 es la fuente de verdad (ver Recolector.reconciliar) y la base se respalda en S3.

Ciclo de un candidato (un fotograma de un personaje):

    pendiente --descarga--> descargada --S3 confirma--> subida
        |                      |
        +--> rechazada_duplicada, rechazada_invalida, no_existe, fallida

Una fila "descargada" tiene el archivo en la caché local y todavía no está confirmada en S3; al
reanudar se vuelve a subir (la clave es determinista y la escritura es condicional, así que repetirla
nunca crea un duplicado).
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PENDIENTE = "pendiente"
DESCARGADA = "descargada"
SUBIDA = "subida"
DUPLICADA = "rechazada_duplicada"
INVALIDA = "rechazada_invalida"
NO_EXISTE = "no_existe"
FALLIDA = "fallida"
# Tras estos fallos temporales seguidos en un mismo fotograma se deja de intentar
MAX_INTENTOS = 10

ESQUEMA = """
CREATE TABLE IF NOT EXISTS candidatos (
    personaje   TEXT    NOT NULL,
    episodio    TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    linea_id    INTEGER,
    texto       TEXT,
    estado      TEXT    NOT NULL,
    sha256      TEXT,
    miniatura   BLOB,
    intentos    INTEGER NOT NULL DEFAULT 0,
    detalle     TEXT,
    actualizado REAL    NOT NULL,
    PRIMARY KEY (personaje, episodio, ts)
);
CREATE INDEX IF NOT EXISTS candidatos_estado ON candidatos (personaje, estado);
CREATE TABLE IF NOT EXISTS lineas (
    personaje  TEXT    NOT NULL,
    linea_id   INTEGER NOT NULL,
    pasada     INTEGER NOT NULL,
    resultado  TEXT    NOT NULL,
    actualizado REAL   NOT NULL,
    PRIMARY KEY (personaje, linea_id, pasada)
);
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT NOT NULL
);
"""


class Estado:
    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: sin transacciones implícitas; cada una se abre con BEGIN IMMEDIATE
        self.con = sqlite3.connect(self.ruta, isolation_level=None, timeout=30)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.executescript(ESQUEMA)

    @staticmethod
    def integra(ruta: Path) -> bool:
        try:
            con = sqlite3.connect(ruta)
            try:
                return con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            finally:
                con.close()
        except sqlite3.DatabaseError:
            return False

    @contextmanager
    def transaccion(self):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            yield self.con
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")

    def cerrar(self) -> None:
        self.con.close()

    def respaldar(self, destino: Path) -> None:
        """Copia consistente de la base aunque haya escrituras en curso (API backup de sqlite)."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        copia = sqlite3.connect(destino)
        try:
            self.con.backup(copia)
        finally:
            copia.close()

    # --- Meta ------------------------------------------------------------------------------

    def leer_meta(self, clave: str, defecto: str | None = None) -> str | None:
        fila = self.con.execute("SELECT valor FROM meta WHERE clave = ?", (clave,)).fetchone()
        return fila[0] if fila else defecto

    def escribir_meta(self, clave: str, valor: str) -> None:
        with self.transaccion() as con:
            con.execute("INSERT INTO meta (clave, valor) VALUES (?, ?) "
                        "ON CONFLICT (clave) DO UPDATE SET valor = excluded.valor", (clave, valor))

    # --- Líneas del guion ------------------------------------------------------------------

    def linea_procesada(self, personaje: str, linea_id: int, pasada: int) -> bool:
        return self.con.execute(
            "SELECT 1 FROM lineas WHERE personaje = ? AND linea_id = ? AND pasada = ?",
            (personaje, linea_id, pasada)).fetchone() is not None

    def lineas_procesadas(self, personaje: str) -> set[tuple[int, int]]:
        return {(l, p) for l, p in self.con.execute(
            "SELECT linea_id, pasada FROM lineas WHERE personaje = ?", (personaje,))}

    def registrar_linea(self, personaje: str, linea_id: int, pasada: int, resultado: str,
                        fotogramas: list[tuple[str, int]], texto: str) -> int:
        """Marca la línea como procesada y agrega sus fotogramas como pendientes, en una sola
        transacción: o quedan las dos cosas o ninguna. Devuelve cuántos fotogramas eran nuevos."""
        momento = time.time()
        with self.transaccion() as con:
            nuevos = 0
            for episodio, ts in fotogramas:
                cursor = con.execute(
                    "INSERT OR IGNORE INTO candidatos (personaje, episodio, ts, linea_id, texto, estado, "
                    "actualizado) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (personaje, episodio, ts, linea_id, texto, PENDIENTE, momento))
                nuevos += cursor.rowcount
            con.execute(
                "INSERT INTO lineas (personaje, linea_id, pasada, resultado, actualizado) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (personaje, linea_id, pasada) DO UPDATE SET resultado = excluded.resultado, "
                "actualizado = excluded.actualizado",
                (personaje, linea_id, pasada, resultado, momento))
        return nuevos

    # --- Candidatos ------------------------------------------------------------------------

    def candidatos(self, personaje: str, estados: tuple[str, ...]) -> list[sqlite3.Row]:
        marcas = ",".join("?" * len(estados))
        self.con.row_factory = sqlite3.Row
        try:
            return self.con.execute(
                f"SELECT * FROM candidatos WHERE personaje = ? AND estado IN ({marcas}) ORDER BY episodio, ts",
                (personaje, *estados)).fetchall()
        finally:
            self.con.row_factory = None

    def fila(self, personaje: str, episodio: str, ts: int) -> sqlite3.Row | None:
        self.con.row_factory = sqlite3.Row
        try:
            return self.con.execute("SELECT * FROM candidatos WHERE personaje = ? AND episodio = ? AND ts = ?",
                                    (personaje, episodio, ts)).fetchone()
        finally:
            self.con.row_factory = None

    def estado_de(self, personaje: str, episodio: str, ts: int) -> str | None:
        fila = self.con.execute(
            "SELECT estado FROM candidatos WHERE personaje = ? AND episodio = ? AND ts = ?",
            (personaje, episodio, ts)).fetchone()
        return fila[0] if fila else None

    def marcar(self, personaje: str, episodio: str, ts: int, estado: str, detalle: str | None = None) -> None:
        with self.transaccion() as con:
            con.execute("UPDATE candidatos SET estado = ?, detalle = ?, actualizado = ? "
                        "WHERE personaje = ? AND episodio = ? AND ts = ?",
                        (estado, detalle, time.time(), personaje, episodio, ts))

    def marcar_descargada(self, personaje: str, episodio: str, ts: int, sha256: str,
                          miniatura: np.ndarray) -> None:
        with self.transaccion() as con:
            con.execute("UPDATE candidatos SET estado = ?, sha256 = ?, miniatura = ?, detalle = NULL, "
                        "actualizado = ? WHERE personaje = ? AND episodio = ? AND ts = ?",
                        (DESCARGADA, sha256, np.asarray(miniatura, np.float32).tobytes(), time.time(),
                         personaje, episodio, ts))

    def registrar_fallo(self, personaje: str, episodio: str, ts: int, detalle: str) -> bool:
        """Suma un fallo temporal. Devuelve True si el fotograma se dio por perdido."""
        with self.transaccion() as con:
            con.execute("UPDATE candidatos SET intentos = intentos + 1, detalle = ?, actualizado = ? "
                        "WHERE personaje = ? AND episodio = ? AND ts = ?",
                        (detalle, time.time(), personaje, episodio, ts))
            intentos = con.execute("SELECT intentos FROM candidatos WHERE personaje = ? AND episodio = ? "
                                   "AND ts = ?", (personaje, episodio, ts)).fetchone()[0]
            if intentos >= MAX_INTENTOS:
                con.execute("UPDATE candidatos SET estado = ? WHERE personaje = ? AND episodio = ? AND ts = ?",
                            (FALLIDA, personaje, episodio, ts))
                return True
        return False

    def asegurar_subida(self, personaje: str, episodio: str, ts: int, sha256: str, miniatura: np.ndarray,
                        texto: str | None = None) -> None:
        """Registra como subido un objeto que ya está en S3 (reconstrucción desde S3)."""
        with self.transaccion() as con:
            con.execute(
                "INSERT INTO candidatos (personaje, episodio, ts, texto, estado, sha256, miniatura, actualizado) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (personaje, episodio, ts) DO UPDATE SET "
                "estado = excluded.estado, sha256 = excluded.sha256, miniatura = excluded.miniatura, "
                "actualizado = excluded.actualizado",
                (personaje, episodio, ts, texto, SUBIDA, sha256, np.asarray(miniatura, np.float32).tobytes(),
                 time.time()))

    def claves_subidas(self, personaje: str) -> set[tuple[str, int]]:
        return {(e, t) for e, t in self.con.execute(
            "SELECT episodio, ts FROM candidatos WHERE personaje = ? AND estado = ?", (personaje, SUBIDA))}

    def shas(self, personaje: str) -> set[str]:
        return {s for (s,) in self.con.execute(
            "SELECT sha256 FROM candidatos WHERE personaje = ? AND estado IN (?, ?) AND sha256 IS NOT NULL",
            (personaje, DESCARGADA, SUBIDA))}

    def miniaturas_aceptadas(self, personaje: str) -> np.ndarray:
        filas = self.con.execute(
            "SELECT miniatura FROM candidatos WHERE personaje = ? AND estado IN (?, ?) AND miniatura IS NOT NULL",
            (personaje, DESCARGADA, SUBIDA)).fetchall()
        if not filas:
            return np.zeros((0, 0), np.float32)
        return np.stack([np.frombuffer(blob, np.float32) for (blob,) in filas])

    def contar(self, personaje: str) -> dict[str, int]:
        return dict(self.con.execute(
            "SELECT estado, COUNT(*) FROM candidatos WHERE personaje = ? GROUP BY estado", (personaje,)))
