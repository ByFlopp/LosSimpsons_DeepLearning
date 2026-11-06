"""Bucle de recolección: de las líneas del guion a imágenes únicas en S3.

Garantías:
- Sin pérdida: una imagen solo se da por subida cuando S3 confirma la escritura y su sha256; lo que
  estaba a medio camino al morir el proceso se retoma al arrancar. S3 es la fuente de verdad: al
  arrancar, reconciliar() agrega a la base todo lo que ya esté en staging/.
- Sin repetición: la clave de cada imagen sale del episodio y del instante; S3 la escribe con
  If-None-Match: * y la base tiene esa misma clave primaria. Además se descarta todo fotograma casi
  idéntico (similitud >= UMBRAL) a una original o a una nueva del mismo personaje.
- Solo originales: las sintéticas no se generan ni se cuentan aquí (ver inventario.py).
"""

import io
import logging
import shutil
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image

import estado as E
from almacen import Almacen, ErrorVerificacion
from comun import UMBRAL, clave_de_nombre, escribir_atomico, meta_recoleccion, miniatura, nombre_fotograma, sha256_hex
from fuentes import EXTRA_POR_PASADA_MS, ClienteFrinkiac, ErrorPermanente, ErrorTemporal, Linea, NoEncontrado, \
    fotogramas_de_linea
from inventario import Inventario

log = logging.getLogger("frinkiac.recolector")

RECOLECTANDO = "recolectando"
POR_REVISAR = "por_revisar"
COMPLETO = "completo"
AGOTADO = "agotado"
SIN_LINEAS = "sin_lineas"
LADO_MINIMO = 64


@dataclass
class Config:
    objetivo: int = 1000
    margen: float = 0.15
    fotogramas_por_linea: int = 3
    separacion_ms: int = 1000
    umbral: float = UMBRAL
    respaldo_cada_s: float = 300
    fallos_para_pausa: int = 20
    pausa_s: float = 600
    espacio_minimo_mb: int = 500

    @property
    def meta(self) -> int:
        return meta_recoleccion(self.objetivo, self.margen)


@dataclass
class Evaluacion:
    personaje: str
    originales: int
    subidas: int
    rechazadas: int
    aprobadas: int
    pendientes_revision: int
    agotado: bool
    estado: str
    nombres_aprobados: list[str] = field(default_factory=list, repr=False)
    nombres_vigentes: list[str] = field(default_factory=list, repr=False)

    @property
    def vigentes(self) -> int:
        return self.subidas - self.rechazadas

    @property
    def total_aprobado(self) -> int:
        return self.originales + self.aprobadas

    def como_dict(self) -> dict:
        return {"originales": self.originales, "subidas": self.subidas, "rechazadas": self.rechazadas,
                "aprobadas": self.aprobadas, "pendientes_revision": self.pendientes_revision,
                "agotado": self.agotado, "estado": self.estado}


class Indice:
    """Miniaturas de un personaje (originales + nuevas aceptadas) para medir similitud coseno."""

    def __init__(self, base: np.ndarray):
        self._datos = np.asarray(base, np.float32).reshape(len(base), -1) if len(base) else None
        self._n = len(base) if len(base) else 0

    def maxima(self, vector: np.ndarray) -> float:
        if not self._n:
            return -1.0
        return float((self._datos[:self._n] @ vector).max())

    def agregar(self, vector: np.ndarray) -> None:
        vector = np.asarray(vector, np.float32).ravel()
        if self._datos is None:
            self._datos = np.empty((64, vector.size), np.float32)
        elif self._n == len(self._datos):
            self._datos = np.concatenate([self._datos, np.empty_like(self._datos)])
        self._datos[self._n] = vector
        self._n += 1


def _metadato(texto: str, largo: int = 500) -> str:
    """Los metadatos de S3 solo admiten ASCII."""
    return urllib.parse.quote(texto or "", safe=" ")[:largo]


class Recolector:
    def __init__(self, almacen: Almacen, cliente: ClienteFrinkiac, estado: E.Estado, inventario: Inventario,
                 lineas: dict[str, list[Linea]], datos: Path, personajes: list[str], config: Config | None = None,
                 parada: Event | None = None, al_terminar_personaje=None):
        # al_terminar_personaje(ev): se llama cuando un personaje cierra su vuelta, para que el equipo
        # pueda ir revisándolo sin esperar a los demás
        self.al_terminar_personaje = al_terminar_personaje
        self.almacen = almacen
        self.cliente = cliente
        self.db = estado
        self.inventario = inventario
        self.lineas = lineas
        self.cache = Path(datos) / "cache"
        self.datos = Path(datos)
        self.personajes = personajes
        self.config = config or Config()
        self.parada = parada or Event()
        self._fallos_seguidos = 0
        self._ultimo_respaldo = time.monotonic()
        self._indices: dict[str, Indice] = {}
        self._revisiones: dict[str, tuple[set[str], set[str]]] = {}
        self.progreso = 0  # líneas procesadas + imágenes subidas en esta vuelta

    # --- Claves -----------------------------------------------------------------------------

    def _clave_staging(self, personaje: str, nombre: str) -> str:
        return self.almacen.clave("staging", personaje, nombre)

    def _imagenes_en(self, prefijo: str) -> dict[str, str]:
        """nombre -> clave de las imágenes bajo un prefijo (sin los _manifiesto/_RESUMEN)."""
        imagenes = {}
        for clave in self.almacen.listar(prefijo):
            nombre = clave.rsplit("/", 1)[-1]
            if nombre.lower().endswith(".jpg") and not nombre.startswith("_"):
                imagenes[nombre] = clave
        return imagenes

    # --- Arranque ----------------------------------------------------------------------------

    def reconciliar(self) -> dict[str, int]:
        """Hace que la base refleje S3. Lo que está en staging/ cuenta como subido aunque la base no lo
        sepa (se perdió el disco o el proceso murió tras subir); lo que la base cree subido y no está
        en S3 vuelve a pendiente para subirse de nuevo."""
        recuperadas = {}
        for personaje in self.personajes:
            en_s3 = self._imagenes_en(self.almacen.clave("staging", personaje) + "/")
            subidas = self.db.claves_subidas(personaje)
            agregadas = 0
            for nombre, clave in sorted(en_s3.items()):
                clave_frinkiac = clave_de_nombre(nombre)
                if clave_frinkiac is None or clave_frinkiac in subidas:
                    continue
                datos = self.almacen.leer_bytes(clave)
                cabecera = self.almacen.cabecera(clave) or {}
                sha = sha256_hex(datos)
                if cabecera.get("Metadata", {}).get("sha256") not in (None, sha):
                    log.error("objeto corrupto en staging", extra={"datos": {"clave": clave}})
                    continue
                imagen = Image.open(io.BytesIO(datos)).convert("RGB")
                texto = urllib.parse.unquote(cabecera.get("Metadata", {}).get("texto", ""))
                self.db.asegurar_subida(personaje, *clave_frinkiac, sha, miniatura(imagen), texto)
                agregadas += 1
            for episodio, ts in subidas:
                if nombre_fotograma(personaje, episodio, ts) not in en_s3:
                    log.warning("falta en S3 lo que la base daba por subido; se repite",
                                extra={"datos": {"personaje": personaje, "episodio": episodio, "ts": ts}})
                    self.db.marcar(personaje, episodio, ts, E.PENDIENTE, "faltaba en S3")
            if agregadas:
                recuperadas[personaje] = agregadas
                log.info("recuperadas desde S3", extra={"datos": {"personaje": personaje, "imagenes": agregadas}})
        self._indices.clear()
        return recuperadas

    # --- Revisión y evaluación ------------------------------------------------------------------

    def _leer_revision(self, personaje: str) -> tuple[set[str], set[str]]:
        contenido, _ = self.almacen.leer_json(self.almacen.clave("revision", f"{personaje}.json"))
        contenido = contenido or {}
        revision = set(contenido.get("aprobadas", [])), set(contenido.get("rechazadas", []))
        self._revisiones[personaje] = revision
        return revision

    def evaluar(self, personaje: str, releer: bool = True) -> Evaluacion:
        aprobadas_rev, rechazadas_rev = (self._leer_revision(personaje) if releer or personaje not in
                                         self._revisiones else self._revisiones[personaje])
        subidas = {nombre_fotograma(personaje, e, t) for e, t in self.db.claves_subidas(personaje)}
        rechazadas = subidas & rechazadas_rev
        aprobadas = (subidas & aprobadas_rev) - rechazadas
        vigentes = subidas - rechazadas
        originales = self.inventario.conteos.get(personaje, 0)
        lineas = self.lineas.get(personaje, [])
        procesadas = self.db.lineas_procesadas(personaje)
        agotado = all((l.id, p) in procesadas for l in lineas for p in range(len(EXTRA_POR_PASADA_MS)))
        pendientes = len(vigentes - aprobadas)
        if originales + len(aprobadas) >= self.config.objetivo:
            estado = COMPLETO
        elif originales + len(vigentes) >= self.config.meta or (agotado and pendientes):
            estado = POR_REVISAR
        elif not lineas:
            estado = SIN_LINEAS
        elif agotado:
            estado = AGOTADO
        else:
            estado = RECOLECTANDO
        return Evaluacion(personaje, originales, len(subidas), len(rechazadas), len(aprobadas), pendientes,
                          agotado, estado, sorted(aprobadas), sorted(vigentes))

    def _meta_alcanzada(self, personaje: str) -> bool:
        ev = self.evaluar(personaje, releer=False)
        return ev.originales + ev.vigentes >= self.config.meta or ev.estado == COMPLETO

    # --- Red ---------------------------------------------------------------------------------

    def _exito_red(self) -> None:
        self._fallos_seguidos = 0

    def _fallo_red(self, error: Exception) -> None:
        self._fallos_seguidos += 1
        log.warning("fallo temporal", extra={"datos": {"error": str(error), "seguidos": self._fallos_seguidos}})
        if self._fallos_seguidos >= self.config.fallos_para_pausa:
            log.error("demasiados fallos seguidos; pausa", extra={"datos": {"segundos": self.config.pausa_s}})
            self.respaldar(forzar=True)
            self.parada.wait(self.config.pausa_s)
            self._fallos_seguidos = 0

    def _hay_espacio(self) -> bool:
        self.cache.mkdir(parents=True, exist_ok=True)
        libre_mb = shutil.disk_usage(self.cache).free / 2**20
        if libre_mb < self.config.espacio_minimo_mb:
            log.error("poco espacio en disco; espero", extra={"datos": {"libre_mb": round(libre_mb)}})
            self.parada.wait(60)
            return False
        return True

    # --- Respaldo ------------------------------------------------------------------------------

    def respaldar(self, forzar: bool = False) -> None:
        if not forzar and time.monotonic() - self._ultimo_respaldo < self.config.respaldo_cada_s:
            return
        copia = self.datos / "respaldo.sqlite"
        try:
            self.db.respaldar(copia)
            self.almacen.subir_archivo(copia, self.almacen.clave("estado", "ultimo.sqlite"))
            self._ultimo_respaldo = time.monotonic()
        except Exception:  # noqa: BLE001 - el respaldo es redundante: S3 ya tiene las imágenes
            log.exception("no se pudo respaldar la base")
        finally:
            copia.unlink(missing_ok=True)

    # --- Candidatos -----------------------------------------------------------------------------

    def _indice(self, personaje: str) -> Indice:
        if personaje not in self._indices:
            base = self.inventario.miniaturas.get(personaje, np.zeros((0, 0), np.float32))
            nuevas = self.db.miniaturas_aceptadas(personaje)
            partes = [m for m in (base, nuevas) if len(m)]
            self._indices[personaje] = Indice(np.concatenate(partes) if partes else np.zeros((0, 0)))
        return self._indices[personaje]

    def _descargar(self, personaje: str, episodio: str, ts: int) -> bool:
        """pendiente -> descargada (o un rechazo). True si quedó lista para subir."""
        if not self._hay_espacio():
            return False
        try:
            datos = self.cliente.imagen(episodio, ts)
        except NoEncontrado:
            self.db.marcar(personaje, episodio, ts, E.NO_EXISTE)
            return False
        except ErrorPermanente as error:
            self.db.marcar(personaje, episodio, ts, E.INVALIDA, str(error))
            return False
        except ErrorTemporal as error:
            self.db.registrar_fallo(personaje, episodio, ts, str(error))
            self._fallo_red(error)
            return False
        self._exito_red()
        try:
            with Image.open(io.BytesIO(datos)) as prueba:
                prueba.verify()
            imagen = Image.open(io.BytesIO(datos)).convert("RGB")
        except Exception as error:  # noqa: BLE001 - cualquier fallo de decodificación invalida el archivo
            self.db.marcar(personaje, episodio, ts, E.INVALIDA, f"no decodifica: {error}")
            return False
        if min(imagen.size) < LADO_MINIMO:
            self.db.marcar(personaje, episodio, ts, E.INVALIDA, f"tamaño {imagen.size}")
            return False
        sha = sha256_hex(datos)
        if sha in self.db.shas(personaje):
            self.db.marcar(personaje, episodio, ts, E.DUPLICADA, "sha256 repetido")
            return False
        vector = miniatura(imagen)
        indice = self._indice(personaje)
        similitud = indice.maxima(vector)
        if similitud >= self.config.umbral:
            self.db.marcar(personaje, episodio, ts, E.DUPLICADA, f"similitud {similitud:.3f}")
            return False
        escribir_atomico(self.cache / personaje / nombre_fotograma(personaje, episodio, ts), datos)
        self.db.marcar_descargada(personaje, episodio, ts, sha, vector)
        indice.agregar(vector)
        return True

    def _subir(self, fila) -> None:
        """descargada -> subida. Solo marca subida cuando S3 confirmó contenido y sha256."""
        personaje, episodio, ts, sha = fila["personaje"], fila["episodio"], fila["ts"], fila["sha256"]
        nombre = nombre_fotograma(personaje, episodio, ts)
        local = self.cache / personaje / nombre
        datos = local.read_bytes() if local.exists() else None
        if datos is None or sha256_hex(datos) != sha:
            # La caché se perdió (disco nuevo): Frinkiac sirve imágenes inmutables, se vuelve a bajar
            try:
                datos = self.cliente.imagen(episodio, ts)
            except ErrorTemporal as error:
                self._fallo_red(error)
                return
            except ErrorPermanente as error:
                self.db.marcar(personaje, episodio, ts, E.NO_EXISTE, str(error))
                return
            if sha256_hex(datos) != sha:
                self.db.marcar(personaje, episodio, ts, E.PENDIENTE, "cambió el contenido; se reevalúa")
                self._indices.pop(personaje, None)
                return
        metadatos = {"personaje": personaje, "episodio": episodio, "ts": str(ts),
                     "linea": str(fila["linea_id"] or ""), "texto": _metadato(fila["texto"])}
        try:
            resultado = self.almacen.subir_fotograma(self._clave_staging(personaje, nombre), datos, sha, metadatos)
        except ErrorVerificacion as error:
            # Ya hay en S3 otro contenido con esa clave: no se sobrescribe ni se reintenta en bucle
            log.error("verificación fallida", extra={"datos": {"archivo": nombre, "error": str(error)}})
            self.db.marcar(personaje, episodio, ts, E.FALLIDA, str(error))
            return
        self.db.marcar(personaje, episodio, ts, E.SUBIDA, resultado)
        local.unlink(missing_ok=True)
        self.progreso += 1
        log.info("subida", extra={"datos": {"personaje": personaje, "archivo": nombre, "resultado": resultado}})

    def _vaciar_pendientes(self, personaje: str) -> None:
        for fila in self.db.candidatos(personaje, (E.DESCARGADA,)):
            if self.parada.is_set():
                return
            self._subir(fila)
        for fila in self.db.candidatos(personaje, (E.PENDIENTE,)):
            if self.parada.is_set() or self._meta_alcanzada(personaje):
                return
            if self._descargar(personaje, fila["episodio"], fila["ts"]):
                self._subir(self.db.fila(personaje, fila["episodio"], fila["ts"]))
            self.respaldar()

    # --- Personaje --------------------------------------------------------------------------------

    def procesar_personaje(self, personaje: str) -> None:
        self._leer_revision(personaje)
        self._vaciar_pendientes(personaje)
        existentes = self.inventario.claves.get(personaje, set())
        lineas = self.lineas.get(personaje, [])
        procesadas = self.db.lineas_procesadas(personaje)
        for pasada in range(len(EXTRA_POR_PASADA_MS)):
            for linea in lineas:
                if self.parada.is_set() or self._meta_alcanzada(personaje):
                    return
                if (linea.id, pasada) in procesadas:
                    continue
                try:
                    instantes = fotogramas_de_linea(self.cliente, linea, pasada, self.config.fotogramas_por_linea,
                                                    self.config.separacion_ms)
                    resultado = "con_fotogramas" if instantes else "sin_fotogramas"
                except NoEncontrado:
                    instantes, resultado = [], "sin_coincidencia"
                except ErrorPermanente as error:
                    instantes, resultado = [], f"error: {error}"[:200]
                except ErrorTemporal as error:
                    self._fallo_red(error)  # la línea queda sin marcar y se reintenta en otra vuelta
                    continue
                self._exito_red()
                nuevos = [(linea.episodio, t) for t in instantes if (linea.episodio, t) not in existentes]
                self.db.registrar_linea(personaje, linea.id, pasada, resultado, nuevos, linea.texto)
                procesadas.add((linea.id, pasada))
                self.progreso += 1
                self._vaciar_pendientes(personaje)
                self.respaldar()

    # --- Cierre por personaje ------------------------------------------------------------------------

    def _escribir_manifiesto(self, ev: Evaluacion) -> None:
        vigentes = []
        for fila in self.db.candidatos(ev.personaje, (E.SUBIDA,)):
            nombre = nombre_fotograma(ev.personaje, fila["episodio"], fila["ts"])
            if nombre in ev.nombres_vigentes:
                vigentes.append({"archivo": nombre, "sha256": fila["sha256"]})
        self.almacen.escribir_json(self.almacen.clave("staging", ev.personaje, "_manifiesto.json"), {
            "personaje": ev.personaje, "originales": ev.originales, "vigentes": vigentes,
            "generado": time.time(), **ev.como_dict()})

    def finalizar(self, ev: Evaluacion) -> int:
        """Copia a final/ las aprobadas: exactamente las que faltan para el objetivo si el personaje
        está completo, o todas si se agotaron sus líneas. Quita de final/ lo que ya no corresponde
        (siempre queda en staging/). Devuelve cuántas quedaron en final/."""
        necesarias = max(0, self.config.objetivo - ev.originales)
        elegidas = ev.nombres_aprobados[:necesarias] if ev.estado == COMPLETO else ev.nombres_aprobados
        prefijo_final = self.almacen.clave("final", ev.personaje)
        for nombre in elegidas:
            self.almacen.copiar_si_falta(self._clave_staging(ev.personaje, nombre), f"{prefijo_final}/{nombre}")
        for nombre, clave in self._imagenes_en(prefijo_final + "/").items():
            if nombre not in elegidas and self.almacen.cabecera(self._clave_staging(ev.personaje, nombre)):
                self.almacen.borrar(clave)
        self.almacen.escribir_json(f"{prefijo_final}/_RESUMEN.json", {
            "personaje": ev.personaje, "completo": ev.estado == COMPLETO, "originales": ev.originales,
            "nuevas": len(elegidas), "total": ev.originales + len(elegidas), "objetivo": self.config.objetivo,
            "archivos": elegidas, "generado": time.time()})
        return len(elegidas)

    # --- Vuelta completa --------------------------------------------------------------------------------

    def vuelta(self) -> list[Evaluacion]:
        """Una pasada por todos los personajes. Devuelve la evaluación final de cada uno."""
        self.progreso = 0
        resultados = []
        for personaje in self.personajes:
            if self.parada.is_set():
                break
            ev = self.evaluar(personaje)
            if ev.estado == RECOLECTANDO:
                log.info("recolectando", extra={"datos": {"personaje": personaje, "faltan":
                                                          self.config.meta - ev.originales - ev.vigentes}})
                self.procesar_personaje(personaje)
                ev = self.evaluar(personaje)
            if ev.estado in (POR_REVISAR, AGOTADO, COMPLETO) and ev.subidas:
                self._escribir_manifiesto(ev)
            if ev.estado == COMPLETO or (ev.estado == AGOTADO and not ev.pendientes_revision and ev.aprobadas):
                self.finalizar(ev)
            resultados.append(ev)
            if self.al_terminar_personaje and not self.parada.is_set():
                try:
                    self.al_terminar_personaje(ev)
                except Exception:  # noqa: BLE001 - preparar la revisión nunca debe frenar la recolección
                    log.exception("no se pudo preparar la revisión", extra={"datos": {"personaje": personaje}})
        self.respaldar(forzar=True)
        if not self.parada.is_set():
            self.almacen.escribir_json(self.almacen.clave("control", "progreso.json"), {
                "actualizado": time.time(), "objetivo": self.config.objetivo, "meta": self.config.meta,
                "personajes": {ev.personaje: ev.como_dict() for ev in resultados}})
        return resultados
