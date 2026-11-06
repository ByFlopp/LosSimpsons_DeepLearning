"""Acceso a S3: subida condicional y verificada, JSON con control de concurrencia y candado con lease.

Distribución del bucket:

    inputs/inventario.npz                  originales actuales (conteo, claves y miniaturas)
    inputs/simpsons_script_lines.csv       guiones de Kaggle
    inputs/simpsons_episodes.csv
    control/lock.json                      candado de la única instancia que recolecta
    control/run.json                       identificador de la corrida (marca del issue)
    control/progreso.json                  estado de cada personaje al terminar una vuelta
    estado/ultimo.sqlite                   respaldo de la base local (el bucket tiene versionado)
    staging/<personaje>/<archivo>.jpg      imágenes nuevas, una por fotograma
    staging/<personaje>/_manifiesto.json   lista y sha256 de las vigentes al cerrar la recolección
    revision/<personaje>.json              decisiones de la revisión manual
    final/<personaje>/<archivo>.jpg        aprobadas que completan las 1000
    final/<personaje>/_RESUMEN.json

Nunca se sobrescribe una imagen: todas se escriben con If-None-Match: * y su clave sale del episodio
y del instante, así que repetir una subida es inofensivo. Los JSON de control se escriben con If-Match
sobre su ETag para que dos escritores no se pisen.

AlmacenLocal ofrece la misma interfaz y las mismas garantías sobre una carpeta del PC, para correr sin
AWS: la misma distribución de carpetas, imágenes creadas de forma atómica y sin sobrescribir, y JSON con
compare-and-swap bajo un cerrojo de archivo.
"""

import base64
import json
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from botocore.config import Config
from botocore.exceptions import ClientError

from comun import escribir_atomico, sha256_hex

log = logging.getLogger("frinkiac.almacen")

CONFIG_BOTO = Config(retries={"max_attempts": 10, "mode": "adaptive"}, connect_timeout=10, read_timeout=60)


class Conflicto(Exception):
    """La escritura condicional no se aplicó: el objeto ya existe o cambió desde que se leyó."""


class ErrorVerificacion(Exception):
    """El objeto en S3 no coincide con lo que se subió."""


class CandadoOcupado(Exception):
    pass


def _codigo(error: ClientError) -> str:
    return str(error.response.get("Error", {}).get("Code", ""))


def _estado_http(error: ClientError) -> int:
    return int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def _es_conflicto(error: ClientError) -> bool:
    return _codigo(error) in ("PreconditionFailed", "ConditionalRequestConflict") or _estado_http(error) in (409, 412)


def _no_existe(error: ClientError) -> bool:
    return _codigo(error) in ("NoSuchKey", "404", "NotFound") or _estado_http(error) == 404


class _Base:
    """Lo común a los dos almacenes. Cada uno implementa cabecera, listar, leer_bytes,
    subir_fotograma, subir_archivo, descargar_archivo, copiar_si_falta, borrar, leer_json,
    escribir_json y ubicacion."""

    prefijo = ""

    def clave(self, *partes: str) -> str:
        return "/".join(p for p in (self.prefijo, *partes) if p)

    def actualizar_json(self, clave: str, cambio, intentos: int = 8) -> dict:
        """Lee, aplica cambio(dict | None) -> dict y escribe de forma condicional; repite si otro
        escribió entre medio."""
        for intento in range(intentos):
            actual, etag = self.leer_json(clave)
            nuevo = cambio(actual)
            try:
                self.escribir_json(clave, nuevo, si_coincide=etag, si_no_existe=etag is None)
                return nuevo
            except Conflicto:
                time.sleep(min(2 ** intento, 30) * 0.1)
        raise Conflicto(f"{clave}: demasiados escritores a la vez")


class Almacen(_Base):
    def __init__(self, cliente_s3, bucket: str, prefijo: str = ""):
        self.s3 = cliente_s3
        self.bucket = bucket
        self.prefijo = prefijo.strip("/")

    def ubicacion(self, clave: str = "") -> str:
        return f"s3://{self.bucket}/{clave}"

    def borrar(self, clave: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=clave)

    def relativa(self, clave: str) -> str:
        return clave[len(self.prefijo) + 1:] if self.prefijo else clave

    # --- Objetos ---------------------------------------------------------------------------

    def cabecera(self, clave: str) -> dict | None:
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=clave)
        except ClientError as error:
            if _no_existe(error):
                return None
            raise

    def listar(self, prefijo: str):
        """Claves bajo el prefijo, paginando."""
        paginador = self.s3.get_paginator("list_objects_v2")
        for pagina in paginador.paginate(Bucket=self.bucket, Prefix=prefijo):
            for objeto in pagina.get("Contents", []):
                yield objeto["Key"]

    def leer_bytes(self, clave: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=clave)["Body"].read()
        except ClientError as error:
            if _no_existe(error):
                return None
            raise

    def subir_fotograma(self, clave: str, datos: bytes, sha256: str, metadatos: dict[str, str]) -> str:
        """Sube la imagen sin sobrescribir nunca y verifica lo que quedó guardado.

        Devuelve "creada" o "ya_existia" (una subida anterior se completó pero el proceso murió antes
        de anotarla). S3 rechaza el cuerpo si su SHA-256 no coincide con ChecksumSHA256."""
        metadatos = {**metadatos, "sha256": sha256}
        try:
            self.s3.put_object(
                Bucket=self.bucket, Key=clave, Body=datos, ContentType="image/jpeg",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(sha256)).decode(),
                Metadata=metadatos, IfNoneMatch="*")
            resultado = "creada"
        except ClientError as error:
            if not _es_conflicto(error):
                raise
            resultado = "ya_existia"
        cabecera = self.cabecera(clave)
        if cabecera is None:
            raise ErrorVerificacion(f"{clave} no aparece después de subirla")
        if cabecera.get("Metadata", {}).get("sha256") != sha256 or cabecera.get("ContentLength") != len(datos):
            raise ErrorVerificacion(f"{clave} en S3 no coincide con el fotograma descargado")
        return resultado

    def subir_archivo(self, ruta: Path, clave: str) -> None:
        self.s3.upload_file(str(ruta), self.bucket, clave)

    def descargar_archivo(self, clave: str, ruta: Path) -> bool:
        """Descarga a un temporal y renombra. False si el objeto no existe."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_name(ruta.name + ".descarga")
        try:
            self.s3.download_file(self.bucket, clave, str(temporal))
        except ClientError as error:
            if _no_existe(error):
                return False
            raise
        os.replace(temporal, ruta)
        return True

    def copiar_si_falta(self, origen: str, destino: str) -> bool:
        """Copia servidor a servidor si el destino no tiene ya el mismo contenido."""
        actual = self.cabecera(destino)
        fuente = self.cabecera(origen)
        if fuente is None:
            raise ErrorVerificacion(f"falta el origen {origen}")
        if actual is not None and actual.get("Metadata", {}).get("sha256") == fuente.get("Metadata", {}).get("sha256"):
            return False
        self.s3.copy_object(Bucket=self.bucket, Key=destino, CopySource={"Bucket": self.bucket, "Key": origen},
                            MetadataDirective="COPY")
        return True

    # --- JSON con concurrencia optimista -------------------------------------------------------

    def leer_json(self, clave: str) -> tuple[dict | None, str | None]:
        try:
            respuesta = self.s3.get_object(Bucket=self.bucket, Key=clave)
        except ClientError as error:
            if _no_existe(error):
                return None, None
            raise
        return json.loads(respuesta["Body"].read()), respuesta["ETag"]

    def escribir_json(self, clave: str, contenido: dict, si_coincide: str | None = None,
                      si_no_existe: bool = False) -> str:
        """Escribe el JSON y devuelve su ETag. Con si_coincide / si_no_existe la escritura es
        condicional y lanza Conflicto si no se cumple."""
        argumentos = {"Bucket": self.bucket, "Key": clave, "ContentType": "application/json",
                      "Body": json.dumps(contenido, ensure_ascii=False, indent=2, sort_keys=True).encode()}
        if si_no_existe:
            argumentos["IfNoneMatch"] = "*"
        elif si_coincide:
            argumentos["IfMatch"] = si_coincide
        try:
            return self.s3.put_object(**argumentos)["ETag"]
        except ClientError as error:
            if _es_conflicto(error) or (si_coincide and _no_existe(error)):
                raise Conflicto(clave) from error
            raise


# --- Almacén local -------------------------------------------------------------------------------

@contextmanager
def bloqueo_archivo(ruta: Path):
    """Cerrojo exclusivo y bloqueante entre procesos (y entre hilos: cada uso abre su propio handle)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    archivo = open(ruta, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            archivo.seek(0)
            while True:
                try:
                    msvcrt.locking(archivo.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.02)
            try:
                yield
            finally:
                archivo.seek(0)
                msvcrt.locking(archivo.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(archivo.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(archivo.fileno(), fcntl.LOCK_UN)
    finally:
        archivo.close()


class AlmacenLocal(_Base):
    """La misma interfaz que Almacen sobre una carpeta, para correr sin AWS.

    - Las imágenes se crean sin sobrescribir nunca: se escriben a un temporal y se publican con un
      enlace duro (POSIX) o un rename (Windows), que fallan de forma atómica si el destino ya existe.
    - El sha256 de cabecera() se calcula del contenido real del archivo: no depende de metadatos que un
      corte de luz pudiera dejar a medias.
    - Los JSON se escriben con compare-and-swap (ETag = sha256 del contenido) bajo un cerrojo de archivo,
      así que el servicio y los comandos de revisión pueden correr a la vez.
    - Los metadatos van aparte, en .meta/, para que staging/ y final/ solo tengan imágenes y JSON.
    """

    def __init__(self, raiz: Path):
        self.raiz = Path(raiz).resolve()
        self.raiz.mkdir(parents=True, exist_ok=True)
        self.prefijo = ""
        self._meta = self.raiz / ".meta"
        self._hilos = threading.Lock()

    def ubicacion(self, clave: str = "") -> str:
        return str(self._ruta(clave)) if clave else str(self.raiz) + os.sep

    def _ruta(self, clave: str) -> Path:
        ruta = (self.raiz / clave).resolve()
        if ruta != self.raiz and self.raiz not in ruta.parents:
            raise ValueError(f"clave fuera del almacén: {clave}")
        return ruta

    def _ruta_meta(self, clave: str) -> Path:
        return self._meta / f"{clave}.json"

    @contextmanager
    def _bloqueo(self):
        with self._hilos, bloqueo_archivo(self._meta / ".bloqueo"):
            yield

    # --- Objetos ---------------------------------------------------------------------------

    def cabecera(self, clave: str) -> dict | None:
        ruta = self._ruta(clave)
        if not ruta.is_file():
            return None
        datos = ruta.read_bytes()
        meta_ruta = self._ruta_meta(clave)
        try:
            metadatos = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.is_file() else {}
        except ValueError:
            metadatos = {}
        metadatos["sha256"] = sha256_hex(datos)
        return {"Metadata": metadatos, "ContentLength": len(datos), "ETag": metadatos["sha256"]}

    def listar(self, prefijo: str):
        """Claves bajo el prefijo. Recorre solo la carpeta del prefijo y omite las carpetas internas
        que empiezan con punto (.meta con los metadatos, .datos con la base del servicio)."""
        base = self.raiz / prefijo.rsplit("/", 1)[0] if "/" in prefijo else self.raiz
        if not base.is_dir():
            return
        claves = []
        for ruta in base.rglob("*"):
            if not ruta.is_file() or ruta.name.endswith((".tmp", ".descarga")):
                continue
            relativa = ruta.relative_to(self.raiz).as_posix()
            if relativa.startswith(".") or not relativa.startswith(prefijo):
                continue
            claves.append(relativa)
        yield from sorted(claves)

    def leer_bytes(self, clave: str) -> bytes | None:
        ruta = self._ruta(clave)
        return ruta.read_bytes() if ruta.is_file() else None

    def subir_fotograma(self, clave: str, datos: bytes, sha256: str, metadatos: dict[str, str]) -> str:
        if sha256_hex(datos) != sha256:
            raise ErrorVerificacion(f"{clave}: el contenido no coincide con su sha256")
        ruta = self._ruta(clave)
        resultado = "creada" if crear_sin_sobrescribir(ruta, datos) else "ya_existia"
        cabecera = self.cabecera(clave)
        if cabecera is None or cabecera["Metadata"]["sha256"] != sha256:
            raise ErrorVerificacion(f"{clave} en el almacén no coincide con el fotograma descargado")
        if resultado == "creada" or not self._ruta_meta(clave).is_file():
            escribir_atomico(self._ruta_meta(clave), json.dumps({**metadatos, "sha256": sha256}).encode())
        return resultado

    def subir_archivo(self, ruta: Path, clave: str) -> None:
        escribir_atomico(self._ruta(clave), Path(ruta).read_bytes())

    def descargar_archivo(self, clave: str, ruta: Path) -> bool:
        datos = self.leer_bytes(clave)
        if datos is None:
            return False
        escribir_atomico(Path(ruta), datos)
        return True

    def copiar_si_falta(self, origen: str, destino: str) -> bool:
        fuente = self.cabecera(origen)
        if fuente is None:
            raise ErrorVerificacion(f"falta el origen {origen}")
        actual = self.cabecera(destino)
        if actual is not None and actual["Metadata"]["sha256"] == fuente["Metadata"]["sha256"]:
            return False
        escribir_atomico(self._ruta(destino), self._ruta(origen).read_bytes())
        if self._ruta_meta(origen).is_file():
            escribir_atomico(self._ruta_meta(destino), self._ruta_meta(origen).read_bytes())
        return True

    def borrar(self, clave: str) -> None:
        self._ruta(clave).unlink(missing_ok=True)
        self._ruta_meta(clave).unlink(missing_ok=True)

    # --- JSON con compare-and-swap ---------------------------------------------------------------

    def leer_json(self, clave: str) -> tuple[dict | None, str | None]:
        # Bajo el mismo cerrojo que las escrituras: en Windows un archivo abierto para leer no se puede
        # reemplazar, y así nunca se lee un JSON a medio cambiar
        with self._bloqueo():
            datos = self.leer_bytes(clave)
        if datos is None:
            return None, None
        return json.loads(datos), sha256_hex(datos)

    def escribir_json(self, clave: str, contenido: dict, si_coincide: str | None = None,
                      si_no_existe: bool = False) -> str:
        cuerpo = json.dumps(contenido, ensure_ascii=False, indent=2, sort_keys=True).encode()
        with self._bloqueo():
            actual = self.leer_bytes(clave)
            if si_no_existe and actual is not None:
                raise Conflicto(clave)
            if si_coincide and (actual is None or sha256_hex(actual) != si_coincide):
                raise Conflicto(clave)
            escribir_atomico(self._ruta(clave), cuerpo)
        return sha256_hex(cuerpo)


def crear_sin_sobrescribir(ruta: Path, datos: bytes) -> bool:
    """Crea el archivo con su contenido completo o no hace nada si ya existe. Nunca deja un archivo a
    medio escribir con el nombre final ni reemplaza uno existente. True si lo creó."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f"{ruta.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(temporal, "wb") as archivo:
        archivo.write(datos)
        archivo.flush()
        os.fsync(archivo.fileno())
    try:
        if os.name == "nt":
            os.rename(temporal, ruta)  # en Windows falla si el destino existe
        else:
            try:
                os.link(temporal, ruta)  # falla si el destino existe
            except FileExistsError:
                raise
            except OSError:
                # Sistema de archivos sin enlaces duros: se decide bajo un cerrojo
                with bloqueo_archivo(ruta.parent / ".crear.bloqueo"):
                    if ruta.exists():
                        return False
                    os.replace(temporal, ruta)
        return True
    except FileExistsError:
        return False
    finally:
        temporal.unlink(missing_ok=True)


def identidad_host() -> str:
    """Identidad estable entre reinicios del proceso en la misma máquina: el instance-id de EC2 si se
    puede leer de IMDSv2, si no el nombre del host."""
    try:
        import requests

        token = requests.put("http://169.254.169.254/latest/api/token", timeout=1,
                             headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}).text
        instancia = requests.get("http://169.254.169.254/latest/meta-data/instance-id", timeout=1,
                                 headers={"X-aws-ec2-metadata-token": token})
        if instancia.ok and instancia.text.startswith("i-"):
            return instancia.text
    except Exception:  # noqa: BLE001 - fuera de EC2 no hay IMDS
        pass
    return socket.gethostname()


class CerrojoLocal:
    """Un solo proceso por máquina: cerrojo exclusivo del sistema operativo sobre un archivo. Si el
    proceso muere, el sistema lo libera solo."""

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self._archivo = None

    def adquirir(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        archivo = open(self.ruta, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                archivo.seek(0)
                msvcrt.locking(archivo.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(archivo.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            archivo.close()
            raise CandadoOcupado(f"otro proceso de esta máquina usa {self.ruta}") from error
        self._archivo = archivo

    def liberar(self) -> None:
        if self._archivo is not None:
            self._archivo.close()
            self._archivo = None


class Candado:
    """Candado distribuido en S3 con lease: garantiza que una sola instancia recolecta a la vez.

    Se crea con If-None-Match: * y se renueva con If-Match sobre su ETag. Si el dueño anterior dejó
    vencer el lease (se cayó), otro lo toma con If-Match; si dos lo intentan a la vez, solo uno gana.
    La misma máquina puede retomar su propio candado sin esperar el vencimiento, porque el
    CerrojoLocal ya garantiza que el proceso anterior de esa máquina terminó."""

    def __init__(self, almacen: Almacen, clave: str, host: str, lease_s: float = 600, renovar_s: float = 120,
                 al_perder=None, reloj=time.time):
        self.almacen = almacen
        self.clave = clave
        self.host = host
        self.dueno = f"{host}/{os.getpid()}/{uuid.uuid4().hex[:8]}"
        self.lease_s = lease_s
        self.renovar_s = renovar_s
        self.al_perder = al_perder or (lambda: None)
        self.reloj = reloj
        self.etag: str | None = None
        self.expira = 0.0
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    def _cuerpo(self, expira: float) -> dict:
        return {"dueno": self.dueno, "host": self.host, "expira": expira, "renovado": self.reloj()}

    def adquirir(self, renovar_en_segundo_plano: bool = True) -> None:
        expira = self.reloj() + self.lease_s
        try:
            self.etag = self.almacen.escribir_json(self.clave, self._cuerpo(expira), si_no_existe=True)
        except Conflicto:
            actual, etag = self.almacen.leer_json(self.clave)
            if actual and actual.get("expira", 0) > self.reloj() and actual.get("host") != self.host:
                raise CandadoOcupado(
                    f"lo tiene {actual.get('dueno')} hasta {time.ctime(actual['expira'])}") from None
            try:
                self.etag = self.almacen.escribir_json(self.clave, self._cuerpo(expira), si_coincide=etag)
            except Conflicto:
                raise CandadoOcupado("otra instancia lo tomó al mismo tiempo") from None
        self.expira = expira
        log.info("candado adquirido", extra={"datos": {"dueno": self.dueno}})
        if renovar_en_segundo_plano:
            self._hilo = threading.Thread(target=self._bucle, name="candado", daemon=True)
            self._hilo.start()

    def renovar(self) -> bool:
        expira = self.reloj() + self.lease_s
        try:
            self.etag = self.almacen.escribir_json(self.clave, self._cuerpo(expira), si_coincide=self.etag)
            self.expira = expira
            return True
        except Conflicto:
            log.error("candado perdido: otra instancia lo tomó")
            return False
        except Exception:  # noqa: BLE001 - un fallo de red no es una pérdida mientras dure el lease
            log.exception("no se pudo renovar el candado")
            return self.reloj() < self.expira - self.renovar_s

    def _bucle(self) -> None:
        while not self._detener.wait(self.renovar_s):
            if not self.renovar():
                self.al_perder()
                return

    def vigente(self) -> bool:
        return self.etag is not None and self.reloj() < self.expira

    def liberar(self) -> None:
        self._detener.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
        if self.etag is not None:
            try:
                self.almacen.escribir_json(self.clave, self._cuerpo(0), si_coincide=self.etag)
            except Exception:  # noqa: BLE001 - si no se puede liberar, vence solo
                log.warning("no se pudo liberar el candado; vencerá solo")
            self.etag = None
