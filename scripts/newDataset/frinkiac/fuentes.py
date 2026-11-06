"""Fuentes de la ampliación: el cliente HTTP de Frinkiac y los guiones de Kaggle.

Frinkiac (https://frinkiac.com) indexa los subtítulos de cada episodio y sirve el fotograma exacto de
cada instante. Los guiones de Kaggle ("The Simpsons by the Data", simpsons_script_lines.csv) dicen qué
personaje dice cada línea. Juntando las dos cosas se obtienen los fotogramas de los momentos en que
habla un personaje; la revisión manual descarta después los planos en los que no aparece.

Endpoints usados (verificados):
    /api/search?q=TEXTO                      -> [{Episode, Timestamp, ...}]
    /api/caption?e=EP&t=TS                   -> {Subtitles: [{StartTimestamp, EndTimestamp, Content}]}
    /api/frames/EP/TS/ANTES/DESPUES          -> [{Episode, Timestamp}] cada ~200 ms
    /img/EP/TS.jpg                           -> el fotograma (inmutable)
"""

import csv
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from rapidfuzz import fuzz
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

log = logging.getLogger("frinkiac.fuentes")

URL_FRINKIAC = "https://frinkiac.com"
USER_AGENT = "LosSimpsons-DeepLearning-dataset/1.0 (+https://github.com/ByFlopp/LosSimpsons_DeepLearning)"


class ErrorPermanente(Exception):
    """La petición no va a funcionar aunque se repita (4xx distinto de 429)."""


class NoEncontrado(ErrorPermanente):
    """404, o ningún subtítulo coincide con la línea del guion."""


class ErrorTemporal(Exception):
    """Se agotaron los reintentos por red, 429 o 5xx. El ítem queda pendiente para otra vuelta."""


class _Reintentable(Exception):
    def __init__(self, mensaje: str, espera: float | None = None):
        super().__init__(mensaje)
        self.espera = espera


def _segundos_retry_after(valor: str | None) -> float | None:
    if not valor:
        return None
    try:
        return max(0.0, float(valor))
    except ValueError:
        return None


class ClienteFrinkiac:
    """Cliente con límite de peticiones por segundo, reintentos con backoff exponencial y jitter,
    y respeto de Retry-After. Nunca reintenta un 404: ese fotograma no existe."""

    def __init__(self, sesion: requests.Session | None = None, rps: float = 2.0, intentos: int = 8,
                 espera_maxima: float = 120.0, timeout: tuple[float, float] = (10.0, 30.0),
                 base: str = URL_FRINKIAC, dormir=time.sleep):
        self.sesion = sesion or requests.Session()
        self.sesion.headers.setdefault("User-Agent", USER_AGENT)
        self.intervalo = 1.0 / rps if rps > 0 else 0.0
        self.intentos = intentos
        self.espera_maxima = espera_maxima
        self.timeout = timeout
        self.base = base.rstrip("/")
        self._dormir = dormir
        self._siguiente = 0.0
        self._cerrojo = threading.Lock()
        self._backoff = wait_exponential_jitter(initial=1, max=espera_maxima)

    # --- Transporte ------------------------------------------------------------------------

    def _esperar_turno(self) -> None:
        with self._cerrojo:
            ahora = time.monotonic()
            if self._siguiente > ahora:
                self._dormir(self._siguiente - ahora)
                ahora = time.monotonic()
            self._siguiente = ahora + self.intervalo

    def _espera(self, estado) -> float:
        error = estado.outcome.exception()
        if isinstance(error, _Reintentable) and error.espera is not None:
            return min(error.espera, self.espera_maxima)
        return self._backoff(estado)

    def _una_peticion(self, ruta: str, params: dict | None) -> requests.Response:
        self._esperar_turno()
        try:
            respuesta = self.sesion.get(f"{self.base}{ruta}", params=params, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as error:
            raise _Reintentable(f"red: {error}") from error
        if respuesta.status_code == 404:
            raise NoEncontrado(f"404 {ruta}")
        if respuesta.status_code == 429 or respuesta.status_code >= 500:
            raise _Reintentable(f"HTTP {respuesta.status_code} {ruta}",
                                _segundos_retry_after(respuesta.headers.get("Retry-After")))
        if respuesta.status_code >= 400:
            raise ErrorPermanente(f"HTTP {respuesta.status_code} {ruta}")
        return respuesta

    def _pedir(self, ruta: str, params: dict | None = None, json: bool = True):
        reintentos = Retrying(
            stop=stop_after_attempt(self.intentos), wait=self._espera, sleep=self._dormir,
            retry=retry_if_exception_type(_Reintentable), reraise=True,
            before_sleep=lambda estado: log.warning(
                "reintento", extra={"datos": {"ruta": ruta, "intento": estado.attempt_number,
                                              "error": str(estado.outcome.exception())}}),
        )
        try:
            for intento in reintentos:
                with intento:
                    respuesta = self._una_peticion(ruta, params)
                    if not json:
                        return respuesta
                    try:
                        return respuesta.json()
                    except ValueError as error:
                        # Una página de Cloudflare en vez de JSON: es temporal
                        raise _Reintentable(f"respuesta no JSON en {ruta}") from error
        except _Reintentable as error:
            raise ErrorTemporal(str(error)) from error

    # --- API -------------------------------------------------------------------------------

    def buscar(self, texto: str) -> list[dict]:
        return self._pedir("/api/search", {"q": texto}) or []

    def subtitulos(self, episodio: str, ts: int) -> list[dict]:
        return (self._pedir("/api/caption", {"e": episodio, "t": ts}) or {}).get("Subtitles") or []

    def fotogramas(self, episodio: str, ts: int, antes: int, despues: int) -> list[dict]:
        return self._pedir(f"/api/frames/{episodio}/{ts}/{antes}/{despues}") or []

    def imagen(self, episodio: str, ts: int) -> bytes:
        respuesta = self._pedir(f"/img/{episodio}/{ts}.jpg", json=False)
        tipo = respuesta.headers.get("Content-Type", "")
        if not tipo.startswith("image/") or not respuesta.content:
            raise ErrorTemporal(f"respuesta sin imagen ({tipo!r}) para {episodio}/{ts}")
        return respuesta.content


# --- Guiones de Kaggle ---------------------------------------------------------------------

@dataclass(frozen=True)
class Linea:
    id: int
    personaje_guion: str
    episodio: str
    ts: int
    texto: str


def _abrir_csv(ruta: Path):
    # Algunas filas del CSV de Kaggle traen textos muy largos
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    return open(ruta, newline="", encoding="utf-8", errors="replace")


def leer_episodios(ruta: Path) -> dict[int, str]:
    """id del episodio en Kaggle -> clave de Frinkiac (S05E10)."""
    episodios = {}
    with _abrir_csv(ruta) as archivo:
        for fila in csv.DictReader(archivo):
            try:
                episodios[int(fila["id"])] = f"S{int(fila['season']):02d}E{int(fila['number_in_season']):02d}"
            except (KeyError, TypeError, ValueError):
                continue
    return episodios


def _filas_de_lineas(ruta: Path):
    with _abrir_csv(ruta) as archivo:
        for fila in csv.DictReader(archivo):
            if None in fila or not fila.get("raw_character_text"):
                continue  # fila mal formada del CSV original
            yield fila


def contar_personajes(ruta: Path) -> dict[str, int]:
    """Cuántas líneas habladas tiene cada nombre de personaje del guion."""
    conteo: dict[str, int] = {}
    for fila in _filas_de_lineas(ruta):
        if str(fila.get("speaking_line", "")).lower() == "true":
            nombre = fila["raw_character_text"].strip()
            conteo[nombre] = conteo.get(nombre, 0) + 1
    return conteo


def leer_lineas(ruta: Path, episodios: dict[int, str], alias: dict[str, list[str]]) -> dict[str, list[Linea]]:
    """Líneas habladas de cada personaje del dataset, según sus alias en el guion. Cada lista queda
    ordenada por (episodio, ts, id): el orden de proceso es el mismo en cada ejecución."""
    personaje_de = {}
    for personaje, nombres in alias.items():
        for nombre in nombres:
            personaje_de.setdefault(nombre.strip().lower(), []).append(personaje)
    lineas: dict[str, list[Linea]] = {personaje: [] for personaje in alias}
    for fila in _filas_de_lineas(ruta):
        if str(fila.get("speaking_line", "")).lower() != "true":
            continue
        destinos = personaje_de.get(fila["raw_character_text"].strip().lower())
        if not destinos:
            continue
        try:
            linea = Linea(
                id=int(fila["id"]),
                personaje_guion=fila["raw_character_text"].strip(),
                episodio=episodios[int(fila["episode_id"])],
                ts=int(float(fila["timestamp_in_ms"])),
                texto=(fila.get("spoken_words") or fila.get("normalized_text") or "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if len(normalizar(linea.texto).split()) < 2:
            continue  # "Ha!" o "Hmm" coinciden con cualquier subtítulo
        for personaje in destinos:
            lineas[personaje].append(linea)
    for lista in lineas.values():
        lista.sort(key=lambda l: (l.episodio, l.ts, l.id))
    return lineas


# --- De una línea del guion a fotogramas --------------------------------------------------

def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", texto.lower())).strip()


def _consulta(texto: str, palabras: int = 12) -> str:
    return " ".join(normalizar(texto).split()[:palabras])


def _similitud(a: str, b: str) -> float:
    a, b = normalizar(a), normalizar(b)
    if not a or not b:
        return 0.0
    return max(fuzz.partial_ratio(a, b), fuzz.token_set_ratio(a, b))


def ubicar_subtitulo(cliente: ClienteFrinkiac, linea: Linea, tolerancia_ms: int = 15000,
                     similitud_minima: float = 80.0) -> dict:
    """Subtítulo de Frinkiac que corresponde a la línea del guion. Primero busca el texto y se queda
    con el resultado del mismo episodio más cercano al instante del guion; si la búsqueda no lo
    encuentra, mira los subtítulos alrededor del instante del guion. Lanza NoEncontrado si ningún
    subtítulo se parece lo suficiente al texto."""
    referencia = linea.ts
    resultados = [r for r in cliente.buscar(_consulta(linea.texto)) if r.get("Episode") == linea.episodio]
    if resultados:
        cercanos = [r for r in resultados if abs(int(r["Timestamp"]) - linea.ts) <= tolerancia_ms]
        if cercanos:
            referencia = int(min(cercanos, key=lambda r: abs(int(r["Timestamp"]) - linea.ts))["Timestamp"])
        elif len(resultados) == 1:
            referencia = int(resultados[0]["Timestamp"])
    subtitulos = cliente.subtitulos(linea.episodio, referencia)
    puntuados = []
    for subtitulo in subtitulos:
        try:
            inicio, fin = int(subtitulo["StartTimestamp"]), int(subtitulo["EndTimestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        puntaje = _similitud(subtitulo.get("Content", ""), linea.texto)
        if puntaje >= similitud_minima:
            contiene = inicio <= referencia <= fin
            puntuados.append((contiene, puntaje, -abs((inicio + fin) // 2 - referencia), subtitulo))
    if not puntuados:
        raise NoEncontrado(f"sin subtítulo para la línea {linea.id} ({linea.episodio})")
    return max(puntuados, key=lambda p: p[:3])[3]


def elegir_espaciados(instantes: list[int], cantidad: int, separacion_ms: int) -> list[int]:
    """Hasta `cantidad` instantes separados al menos `separacion_ms`, repartidos por todo el tramo."""
    elegidos: list[int] = []
    for instante in sorted(set(instantes)):
        if not elegidos or instante - elegidos[-1] >= separacion_ms:
            elegidos.append(instante)
    if len(elegidos) <= cantidad:
        return elegidos
    if cantidad == 1:
        return [elegidos[len(elegidos) // 2]]
    paso = (len(elegidos) - 1) / (cantidad - 1)
    return [elegidos[round(i * paso)] for i in range(cantidad)]


# Cada pasada amplía la ventana alrededor del subtítulo: la primera usa solo el tramo en que se oye la
# línea; las siguientes, los segundos de antes y después (el personaje suele seguir en pantalla).
EXTRA_POR_PASADA_MS = (0, 3000, 6000)


def fotogramas_de_linea(cliente: ClienteFrinkiac, linea: Linea, pasada: int, cantidad: int = 3,
                        separacion_ms: int = 1000) -> list[int]:
    """Instantes de Frinkiac de la línea para esta pasada. En las pasadas > 0 solo se devuelven los
    instantes fuera de la ventana de la pasada anterior, para no repetir trabajo."""
    subtitulo = ubicar_subtitulo(cliente, linea)
    inicio, fin = int(subtitulo["StartTimestamp"]), int(subtitulo["EndTimestamp"])
    extra = EXTRA_POR_PASADA_MS[pasada]
    previo = EXTRA_POR_PASADA_MS[pasada - 1] if pasada > 0 else None
    desde, hasta = inicio - extra, fin + extra
    centro = (desde + hasta) // 2
    mitad = (hasta - desde) // 2 + 1
    instantes = []
    for fotograma in cliente.fotogramas(linea.episodio, centro, mitad, mitad):
        if fotograma.get("Episode") != linea.episodio:
            continue
        instante = int(fotograma["Timestamp"])
        if not desde <= instante <= hasta:
            continue
        if previo is not None and inicio - previo <= instante <= fin + previo:
            continue
        instantes.append(instante)
    return elegir_espaciados(instantes, cantidad, separacion_ms)
