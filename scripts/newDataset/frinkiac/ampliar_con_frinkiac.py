"""Amplía con fotogramas de Frinkiac a los personajes con menos de 1000 imágenes ORIGINALES.

Las imágenes sintéticas (__alterada, __parecida) no cuentan ni se generan aquí: se crean después, con
generar_imagenes_alteradas.py, cuando el dataset ya tiene sus 1000 originales por personaje.

La forma normal de usarlo es `pnpm frinkiac` desde la raíz del repo, que ejecuta el subcomando `flujo`:
todo el proceso de una vez, reanudable. Por defecto todo queda en una carpeta junto al dataset (nunca
dentro), por ejemplo:

    Desktop/
    ├── simpsons_dataset/            el dataset: no se toca hasta ejecutar 'exportar'
    └── frinkiac_ampliacion/
        ├── por_revisar/<personaje>/ aquí se revisa a mano
        ├── final/<personaje>/       aprobadas que completan las 1000, listas para exportar
        └── staging/ revision/ control/ inputs/ estado/ .datos/

Revisión: en por_revisar/<personaje>/ se borran las imágenes en las que el personaje no aparece y, al
terminar esa carpeta, también su _PENDIENTE.txt. El siguiente `pnpm frinkiac` aplica las carpetas
terminadas, repone las rechazadas y deja listas en por_revisar/ las nuevas. Para revisar menos,
`pnpm frinkiac clasificar` (clasificar_con_cnn.py) separa antes por_revisar/ con la CNN en seguras,
dudosas y ruido; solo se revisan las dudosas y el paso 4 traslada la decisión (clasificacion.py).

Subcomandos:
    flujo             todo lo de abajo en orden (es lo que corre `pnpm frinkiac`)
    inventario        cuenta las originales del dataset y guarda en el almacén el inventario y los guiones
    mapear            revisa qué nombres del guion corresponden a cada carpeta (personajes_frinkiac.json)
    recolectar        baja fotogramas únicos a staging/ y al terminar abre el issue (servicio de la EC2)
    estado            avance por personaje
    revisar           prepara en por_revisar/<personaje>/ las nuevas sin revisar
    aplicar-revision  anota las borradas como rechazadas y el resto como aprobadas
    exportar          copia las aprobadas de final/ al dataset, sin sobrescribir nada

Almacén: --local [CARPETA] (por defecto) o --bucket B para S3 (EC2).

Códigos de salida: 0 terminó; 2 falta configurar algo (no se reintenta); 3 detenido con Ctrl+C o por
aviso de spot; 4 perdió el candado; 5 otro proceso está recolectando; 1 error inesperado. El lanzador
de pnpm y systemd reinician ante 1 y 4, y cada arranque retoma desde donde quedó.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import boto3

import aviso_github
import clasificacion
import guiones as G
import inventario as inv
from almacen import (CONFIG_BOTO, Almacen, AlmacenLocal, Candado, CandadoOcupado, CerrojoLocal, Conflicto,
                     bloqueo_archivo, crear_sin_sobrescribir, identidad_host)
from comun import MARGEN, MENCIONES, OBJETIVO, REPO, configurar_logs, escribir_atomico, sha256_hex
from estado import Estado
from fuentes import ClienteFrinkiac, contar_personajes, leer_episodios, leer_lineas
from generar_imagenes_alteradas import RUTA_POR_DEFECTO
from recolector import AGOTADO, COMPLETO, POR_REVISAR, RECOLECTANDO, SIN_LINEAS, Config, Evaluacion, Recolector

log = logging.getLogger("frinkiac")

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[2]
ALIAS_POR_DEFECTO = CARPETA / "personajes_frinkiac.json"
DATASET_POR_DEFECTO = Path(os.environ.get("SIMPSONS_DATASET", str(RUTA_POR_DEFECTO)))
# Junto al dataset, nunca dentro: cada subcarpeta del dataset se toma como un personaje
LOCAL_POR_DEFECTO = DATASET_POR_DEFECTO.parent / "frinkiac_ampliacion"
ENTRADAS = {"inventario": "inventario.npz", "guiones": "simpsons_script_lines.csv",
            "episodios": "simpsons_episodes.csv"}
SALIDA_OK, SALIDA_CONFIG, SALIDA_PARADA, SALIDA_CANDADO_PERDIDO, SALIDA_OCUPADO = 0, 2, 3, 4, 5
SCRIPT = "python scripts/newDataset/frinkiac/ampliar_con_frinkiac.py"
REGISTRO_REVISION = "_descargadas.json"
MARCA_PENDIENTE = "_PENDIENTE.txt"
# Rechazar todo un lote grande casi siempre es un accidente (se vació la carpeta): se pide --forzar
LOTE_SOSPECHOSO = 10


# --- Comunes ------------------------------------------------------------------------------------

def es_local(args) -> bool:
    return bool(getattr(args, "local", None))


def crear_almacen(args):
    if es_local(args):
        raiz = Path(args.local).resolve()
        dataset = Path(getattr(args, "dataset", None) or DATASET_POR_DEFECTO).resolve()
        if raiz == dataset or dataset in raiz.parents:
            sys.exit(f"El almacén local no puede estar dentro del dataset ({dataset}): cada subcarpeta del "
                     "dataset se toma como un personaje. Usa otra carpeta con --local.")
        return AlmacenLocal(raiz)
    if getattr(args, "bucket", None):
        s3 = boto3.client("s3", region_name=args.region, config=CONFIG_BOTO)
        return Almacen(s3, args.bucket, args.prefijo)
    sys.exit("Indica dónde guardar: --local [carpeta] para este PC, o --bucket <bucket> (o FRINKIAC_BUCKET) para S3")


def opcion_almacen(args) -> str:
    """La opción que hay que repetir en los demás comandos para usar el mismo almacén."""
    if es_local(args):
        return f'--local "{Path(args.local).resolve()}"'
    return f"--bucket {args.bucket}" + (f" --prefijo {args.prefijo}" if args.prefijo else "")


def base_local(args) -> Path:
    return Path(args.local) if es_local(args) else LOCAL_POR_DEFECTO


def carpeta_revision(args) -> Path:
    """Donde el equipo revisa a mano: por_revisar/ dentro de la carpeta de la ampliación."""
    return Path(args.destino) if getattr(args, "destino", None) else base_local(args) / "por_revisar"


def cargar_alias(ruta: Path) -> dict[str, list[str]]:
    contenido = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {p: list(n) for p, n in contenido.items() if not p.startswith("_")}


def asegurar_entrada(almacen, nombre: str, carpeta: Path) -> Path:
    """Copia local de una entrada del almacén. Se vuelve a bajar si cambió en el almacén (su sha256
    queda anotado en control/entradas.json al guardarla)."""
    ruta = carpeta / ENTRADAS[nombre]
    clave = almacen.clave("inputs", ENTRADAS[nombre])
    esperado = (almacen.leer_json(almacen.clave("control", "entradas.json"))[0] or {}).get(nombre)
    if ruta.exists() and esperado and sha256_hex(ruta.read_bytes()) != esperado:
        ruta.unlink()
    if not ruta.exists() and not almacen.descargar_archivo(clave, ruta):
        sys.exit(f"Falta {almacen.ubicacion(clave)}: ejecuta antes 'inventario' (o 'pnpm frinkiac')")
    return ruta


def guardar_entrada(almacen, nombre: str, ruta: Path) -> bool:
    """Guarda una entrada en el almacén si cambió. True si la subió."""
    sha = sha256_hex(Path(ruta).read_bytes())
    clave_registro = almacen.clave("control", "entradas.json")
    clave = almacen.clave("inputs", ENTRADAS[nombre])
    if (almacen.leer_json(clave_registro)[0] or {}).get(nombre) == sha and almacen.cabecera(clave):
        return False
    almacen.subir_archivo(Path(ruta), clave)
    almacen.actualizar_json(clave_registro, lambda actual: {**(actual or {}), nombre: sha})
    return True


def cargar_inventario(almacen, carpeta: Path) -> inv.Inventario:
    return inv.cargar(asegurar_entrada(almacen, "inventario", carpeta))


def imagenes_propias(almacen) -> dict[str, set[str]]:
    """Nombres de las imágenes que ya trajo esta ampliación, por personaje (las de staging/)."""
    propias: dict[str, set[str]] = {}
    for clave in almacen.listar(almacen.clave("staging") + "/"):
        partes = clave.split("/")
        if clave.endswith(".jpg") and len(partes) >= 2:
            propias.setdefault(partes[-2], set()).add(partes[-1])
    return propias


# --- inventario -----------------------------------------------------------------------------------

def cmd_inventario(args) -> int:
    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"No existe el dataset: {dataset}")
    almacen = crear_almacen(args) if (args.bucket or es_local(args)) else None
    excluir = imagenes_propias(almacen) if almacen is not None else {}
    inventario = inv.construir(dataset, args.objetivo, excluir=excluir)
    salida = Path(args.salida) if args.salida else base_local(args) / ".datos" / "inventario.npz"
    salida.parent.mkdir(parents=True, exist_ok=True)
    inv.guardar(inventario, salida)
    debajo = inventario.por_debajo()
    print(f"{len(inventario.conteos)} personajes; {len(debajo)} con menos de {args.objetivo} originales "
          f"(faltan {sum(args.objetivo - inventario.conteos[p] for p in debajo)}). Inventario: {salida}")
    if almacen is not None:
        for nombre, ruta in {"inventario": salida, "guiones": args.guiones, "episodios": args.episodios}.items():
            if ruta is None:
                print(f"  (sin --{nombre}: no se guarda {ENTRADAS[nombre]})")
                continue
            guardar_entrada(almacen, nombre, Path(ruta))
            print(f"  guardado {almacen.ubicacion(almacen.clave('inputs', ENTRADAS[nombre]))}")
    return 0


# --- mapear ------------------------------------------------------------------------------------------

def cmd_mapear(args) -> int:
    from rapidfuzz import fuzz, process

    conteo = contar_personajes(Path(args.guiones))
    por_minuscula = {n.lower(): n for n in conteo}
    alias = cargar_alias(args.alias) if Path(args.alias).exists() else {}
    personajes = sorted(alias)
    if args.inventario:
        personajes = inv.cargar(Path(args.inventario)).por_debajo()
    sugeridos = {}
    for personaje in personajes:
        encontrados = [(n, conteo[por_minuscula[n.lower()]]) for n in alias.get(personaje, [])
                       if n.lower() in por_minuscula]
        faltan = [n for n in alias.get(personaje, []) if n.lower() not in por_minuscula]
        legible = personaje.replace("_", " ")
        sugerencias = [(n, round(s)) for n, s, _ in process.extract(legible, list(conteo), scorer=fuzz.WRatio, limit=4)
                       if s >= 80 and n not in dict(encontrados)]
        lineas = sum(n for _, n in encontrados)
        print(f"{personaje:32s} {lineas:6d} líneas  usa={encontrados}")
        if faltan:
            print(f"{'':32s} no están en el guion: {faltan}")
        if sugerencias:
            print(f"{'':32s} sugerencias: {sugerencias}")
        if not encontrados and sugerencias:
            sugeridos[personaje] = [sugerencias[0][0]]
    if args.escribir and sugeridos:
        nuevo = {**alias, **sugeridos}
        escribir_atomico(Path(args.alias), (json.dumps(nuevo, ensure_ascii=False, indent=2, sort_keys=True)
                                             + "\n").encode("utf-8"))
        print(f"Agregadas {len(sugeridos)} sugerencias a {args.alias}: revísalas antes de recolectar.")
    return 0


# --- recolectar -----------------------------------------------------------------------------------------

class VigiaSpot(threading.Thread):
    """Detiene el proceso con tiempo (AWS avisa 2 minutos antes) si la instancia spot va a ser
    reclamada. Fuera de EC2 o en instancias on-demand no hace nada."""

    def __init__(self, parada: threading.Event, cada_s: float = 5):
        super().__init__(name="spot", daemon=True)
        self.parada = parada
        self.cada_s = cada_s

    def run(self) -> None:
        import requests

        base = "http://169.254.169.254/latest"
        fallos = 0
        while not self.parada.wait(self.cada_s):
            try:
                token = requests.put(f"{base}/api/token", timeout=2,
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}).text
                aviso = requests.get(f"{base}/meta-data/spot/instance-action", timeout=2,
                                     headers={"X-aws-ec2-metadata-token": token})
                fallos = 0
            except Exception:  # noqa: BLE001 - el vigía es opcional: nunca debe tumbar la recolección
                fallos += 1
                if fallos >= 3:
                    return  # tres fallos seguidos: no hay IMDS (no es EC2)
                continue
            if aviso.status_code == 200:
                log.warning("aviso de interrupción spot; deteniendo", extra={"datos": {"aviso": aviso.text}})
                self.parada.set()
                return


def obtener_run_id(almacen) -> str:
    clave = almacen.clave("control", "run.json")
    actual, _ = almacen.leer_json(clave)
    if actual:
        return actual["run_id"]
    try:
        almacen.escribir_json(clave, {"run_id": uuid.uuid4().hex, "creado": time.time()}, si_no_existe=True)
    except Conflicto:
        pass  # otro lo creó primero: vale el suyo
    return almacen.leer_json(clave)[0]["run_id"]


def abrir_estado(almacen, datos: Path) -> Estado:
    """Abre la base local. Si falta o está corrupta, parte del último respaldo del almacén; si tampoco
    hay, de una vacía. En todos los casos reconciliar() la completa después con lo que hay en staging/."""
    ruta = datos / "estado.sqlite"
    if ruta.exists() and not Estado.integra(ruta):
        apartada = ruta.with_name(f"estado.corrupta-{int(time.time())}.sqlite")
        os.replace(ruta, apartada)
        log.error("base local corrupta; apartada", extra={"datos": {"ruta": str(apartada)}})
    if not ruta.exists():
        respaldo = datos / "descargado.sqlite"
        if almacen.descargar_archivo(almacen.clave("estado", "ultimo.sqlite"), respaldo) and Estado.integra(respaldo):
            os.replace(respaldo, ruta)
            log.info("base restaurada desde el respaldo del almacén")
        else:
            respaldo.unlink(missing_ok=True)
    return Estado(ruta)


ETIQUETAS = {COMPLETO: "completo", POR_REVISAR: "por revisar", AGOTADO: "agotado", SIN_LINEAS: "sin líneas",
             RECOLECTANDO: "recolectando"}


def redactar_aviso(resultados: list[Evaluacion], fase: str, args, objetivo: int, almacen) -> tuple[str, str]:
    menciones = " ".join(f"@{m.lstrip('@')}" for m in args.mencionar)
    staging, final = almacen.ubicacion(almacen.clave("staging")), almacen.ubicacion(almacen.clave("final"))
    donde = "en el PC donde corre la recolección" if es_local(args) else "en S3"
    cuenta = {e: sum(r.estado == e for r in resultados) for e in ETIQUETAS}
    if fase == "revision":
        titulo = "Dataset ampliado con Frinkiac: imágenes listas para revisar"
        apertura = (f"{menciones} terminó la recolección de fotogramas de Frinkiac. Las imágenes nuevas "
                    f"están {donde}, en `{staging}`, y hay que revisarlas a mano antes de usarlas.")
    else:
        titulo = "Dataset ampliado con Frinkiac: completo"
        apertura = (f"{menciones} todos los personajes quedaron revisados. Las aprobadas que completan las "
                    f"{objetivo} originales por personaje están {donde}, en `{final}`.")
    filas = ["| Personaje | Originales | Nuevas | Rechazadas | Aprobadas | Total aprobado | Estado |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in sorted(resultados, key=lambda r: r.personaje):
        filas.append(f"| `{r.personaje}` | {r.originales} | {r.vigentes} | {r.rechazadas} | {r.aprobadas} | "
                     f"{min(r.total_aprobado, objetivo) if r.estado == COMPLETO else r.total_aprobado} | "
                     f"{ETIQUETAS[r.estado]} |")
    if es_local(args):
        pasos = [
            "### Pasos",
            f"- [ ] Revisar cada carpeta de `{carpeta_revision(args)}`: borrar las imágenes en las que el personaje no "
            f"aparece y, al terminar esa carpeta, borrar también su `{MARCA_PENDIENTE}`.",
            "- [ ] Volver a ejecutar `pnpm frinkiac`: aplica las carpetas revisadas, repone las rechazadas y copia a "
            "`final/` los personajes completos. Este issue recibe un comentario con el resultado.",
            "- [ ] Con todo completo: `pnpm frinkiac exportar` copia las aprobadas al dataset y después hay que volver "
            "a ejecutar `scripts/newDataset/generar_imagenes_alteradas.py`. Las sintéticas se generan recién ahí y "
            "nunca cuentan para las originales.",
        ]
    else:
        opcion = opcion_almacen(args)
        pasos = [
            "### Pasos",
            f"- [ ] En un PC: `pnpm frinkiac revisar {opcion} --personaje <p>` (o `{SCRIPT} revisar ...`), borrar de "
            f"`por_revisar/<p>/` las imágenes en las que el personaje no aparece, borrar `{MARCA_PENDIENTE}` y "
            f"aplicar con `pnpm frinkiac aplicar-revision {opcion} --personaje <p>`.",
            "- [ ] Volver a iniciar el servicio en la EC2 (`sudo systemctl start frinkiac`): repone las rechazadas "
            "y copia a `final/` los personajes completos. Este issue recibe un comentario con el resultado.",
            f"- [ ] Con todo completo: `pnpm frinkiac exportar {opcion}` y después volver a ejecutar "
            "`scripts/newDataset/generar_imagenes_alteradas.py`. Las sintéticas se generan recién ahí y nunca "
            "cuentan para las originales.",
        ]
    notas = []
    if cuenta[AGOTADO] or cuenta[SIN_LINEAS]:
        notas.append(f"> {cuenta[AGOTADO]} personaje(s) agotaron sus líneas del guion sin llegar a {objetivo} y "
                     f"{cuenta[SIN_LINEAS]} no tienen líneas mapeadas (revisar `personajes_frinkiac.json`).")
    contenido = "\n\n".join([
        apertura,
        f"**Resumen:** {cuenta[COMPLETO]} completos, {cuenta[POR_REVISAR]} por revisar, "
        f"{cuenta[AGOTADO]} agotados, {cuenta[SIN_LINEAS]} sin líneas.",
        "\n".join(filas), *notas, "\n".join(pasos),
        "<sub>Generado por scripts/newDataset/frinkiac/ampliar_con_frinkiac.py</sub>",
    ])
    return titulo, contenido


def obtener_token(args) -> str:
    """GITHUB_TOKEN, el parámetro de SSM (EC2) o, en un PC, la sesión del GitHub CLI (gh auth login)."""
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    if args.ssm_token:
        ssm = boto3.client("ssm", region_name=args.region, config=CONFIG_BOTO)
        return aviso_github.token_desde_ssm(ssm, args.ssm_token)
    if shutil.which("gh"):
        resultado = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=30)
        if resultado.returncode == 0 and resultado.stdout.strip():
            return resultado.stdout.strip()
    raise RuntimeError("Falta el token de GitHub: define GITHUB_TOKEN, usa --ssm-token o inicia sesión con "
                       "'gh auth login'")


def directorio_datos(args) -> Path:
    if getattr(args, "datos", None):
        return Path(args.datos)
    if es_local(args):
        return Path(args.local) / ".datos"  # junto a lo recolectado: se encuentra desde cualquier carpeta
    if os.environ.get("FRINKIAC_DATOS"):
        return Path(os.environ["FRINKIAC_DATOS"])
    # En un PC con S3, cada bucket tiene su propia base local
    return Path("/var/lib/frinkiac") if os.name != "nt" else LOCAL_POR_DEFECTO / f".datos-s3-{args.bucket}"


def cmd_recolectar(args) -> int:
    datos = directorio_datos(args)
    datos.mkdir(parents=True, exist_ok=True)
    cerrojo = CerrojoLocal(datos / "proceso.lock")
    try:
        cerrojo.adquirir()
    except CandadoOcupado as error:
        log.error(str(error))
        return SALIDA_OCUPADO
    parada = threading.Event()

    def detener(senal, _marco):
        log.warning("señal recibida; terminando el ítem en curso", extra={"datos": {"senal": senal}})
        parada.set()

    anteriores = {senal: signal.signal(senal, detener) for senal in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _recolectar(args, datos, cerrojo, parada)
    finally:
        for senal, manejador in anteriores.items():
            signal.signal(senal, manejador)


def _recolectar(args, datos: Path, cerrojo: CerrojoLocal, parada: threading.Event) -> int:
    codigo = {"valor": SALIDA_PARADA}

    def candado_perdido():
        codigo["valor"] = SALIDA_CANDADO_PERDIDO
        parada.set()

    almacen = crear_almacen(args)
    host = socket.gethostname() if es_local(args) else identidad_host()
    candado = Candado(almacen, almacen.clave("control", "lock.json"), host, al_perder=candado_perdido)
    try:
        candado.adquirir()
    except CandadoOcupado as error:
        log.error("otro proceso está recolectando en este almacén", extra={"datos": {"detalle": str(error)}})
        cerrojo.liberar()
        return SALIDA_OCUPADO
    recolector = None
    estado = None
    try:
        if not es_local(args):
            VigiaSpot(parada).start()
        run_id = obtener_run_id(almacen)
        estado = abrir_estado(almacen, datos)
        entradas = datos / "inputs"
        inventario = cargar_inventario(almacen, entradas)
        objetivo = args.objetivo or inventario.objetivo
        alias = cargar_alias(args.alias)
        personajes = [p for p in inventario.conteos if inventario.conteos[p] < objetivo]
        personajes.sort(key=lambda p: (inventario.conteos[p], p))
        if args.personajes:
            pedidos = set(args.personajes)
            personajes = [p for p in personajes if p in pedidos]
        sin_alias = [p for p in personajes if not alias.get(p)]
        if sin_alias:
            log.warning("personajes sin nombres en personajes_frinkiac.json", extra={"datos": {"personajes": sin_alias}})
        episodios = leer_episodios(asegurar_entrada(almacen, "episodios", entradas))
        lineas = leer_lineas(asegurar_entrada(almacen, "guiones", entradas), episodios,
                             {p: alias.get(p, []) for p in personajes})
        log.info("inicio", extra={"datos": {"run_id": run_id, "almacen": almacen.ubicacion(), "datos": str(datos),
                                            "personajes": len(personajes), "objetivo": objetivo,
                                            "lineas": {p: len(l) for p, l in lineas.items()}}})
        config = Config(objetivo=objetivo, margen=args.margen)
        # Las esperas de los reintentos se cortan apenas llega Ctrl+C o SIGTERM
        cliente = ClienteFrinkiac(rps=args.rps, dormir=parada.wait)
        recolector = Recolector(almacen, cliente, estado, inventario, lineas, datos, personajes, config, parada,
                                al_terminar_personaje=getattr(args, "al_terminar_personaje", None))
        recolector.reconciliar()
        resultados: list[Evaluacion] = []
        while not parada.is_set():
            resultados = recolector.vuelta()
            if parada.is_set() or not any(r.estado == RECOLECTANDO for r in resultados):
                break
            if recolector.progreso == 0:
                log.warning("vuelta sin avances (fallos temporales); espero antes de reintentar")
                parada.wait(config.pausa_s)
        if parada.is_set():
            return codigo["valor"]
        fase = "revision" if any(r.estado == POR_REVISAR for r in resultados) else "final"
        titulo, contenido = redactar_aviso(resultados, fase, args, objetivo, almacen)
        if args.sin_issue:
            print(f"\n# {titulo}\n\n{contenido}\n")
            return SALIDA_OK
        try:
            url = aviso_github.publicar_con_reintentos(aviso_github.GitHub(obtener_token(args), args.repo),
                                                       run_id, fase, titulo, contenido)
        except Exception:
            # La corrida no se da por terminada sin aviso: queda anotado y el próximo arranque reintenta
            almacen.escribir_json(almacen.clave("control", "aviso_pendiente.json"),
                                  {"fase": fase, "titulo": titulo, "contenido": contenido, "intento": time.time()})
            log.exception("no se pudo publicar el issue; se reintentará al reiniciar")
            return 1
        almacen.escribir_json(almacen.clave("control", "aviso.json"), {"fase": fase, "url": url, "publicado": time.time()})
        estado.escribir_meta(f"aviso:{fase}", url)
        log.info("issue publicado", extra={"datos": {"fase": fase, "url": url}})
        return SALIDA_OK
    finally:
        if recolector is not None:
            recolector.respaldar(forzar=True)
        if estado is not None:
            estado.cerrar()  # deja la base consolidada y sin archivos abiertos
        candado.liberar()
        cerrojo.liberar()


# --- estado ---------------------------------------------------------------------------------------------

def cmd_estado(args) -> int:
    almacen = crear_almacen(args)
    with tempfile.TemporaryDirectory() as temporal:
        inventario = cargar_inventario(almacen, Path(temporal))
    progreso = (almacen.leer_json(almacen.clave("control", "progreso.json"))[0] or {}).get("personajes", {})
    objetivo = args.objetivo or inventario.objetivo
    print(f"Almacén: {almacen.ubicacion()}")
    print(f"{'personaje':32s} {'orig':>5s} {'staging':>8s} {'rechaz':>7s} {'aprob':>6s} {'final':>6s}  estado")
    for personaje in inventario.por_debajo():
        staging = sum(1 for c in almacen.listar(almacen.clave("staging", personaje) + "/") if c.endswith(".jpg"))
        final = sum(1 for c in almacen.listar(almacen.clave("final", personaje) + "/") if c.endswith(".jpg"))
        revision = almacen.leer_json(almacen.clave("revision", f"{personaje}.json"))[0] or {}
        print(f"{personaje:32s} {inventario.conteos[personaje]:5d} {staging:8d} {len(revision.get('rechazadas', [])):7d} "
              f"{len(revision.get('aprobadas', [])):6d} {final:6d}  {progreso.get(personaje, {}).get('estado', '-')}")
    aviso = almacen.leer_json(almacen.clave("control", "aviso.json"))[0]
    if aviso:
        print(f"\nIssue: {aviso['url']} (fase {aviso['fase']})")
    print(f"Objetivo: {objetivo} originales por personaje")
    return 0


# --- Revisión manual -------------------------------------------------------------------------------------------
#
# Cada personaje tiene su carpeta en por_revisar/. Al prepararla se copian las nuevas sin revisar, se
# anota la lista en _descargadas.json y se deja _PENDIENTE.txt. El revisor borra las imágenes malas y,
# al terminar, borra _PENDIENTE.txt. Al aplicar: lo que sigue en la carpeta queda aprobado y lo borrado,
# rechazado; las copias aprobadas se quitan de la carpeta (siguen en staging/). Mientras _PENDIENTE.txt
# exista, nuevas imágenes del mismo personaje se agregan a la carpeta en vez de crear otra.

TEXTO_PENDIENTE = """Revisión de {personaje}

1. Borra las imágenes en las que {personaje} NO aparece.
2. Cuando termines esta carpeta, borra también este archivo ({marca}).
3. Vuelve a ejecutar 'pnpm frinkiac': lo que quede aquí se aprueba, lo borrado se rechaza y se repone.

Mientras este archivo exista la carpeta se considera en revisión y pueden llegar más imágenes.
"""


def preparar_revision(almacen, base: Path, personaje: str, todas: bool = False) -> int:
    """Copia a base/<personaje>/ las nuevas sin revisar. Devuelve cuántas agregó."""
    carpeta = base / personaje
    with bloqueo_archivo(base / ".bloqueos" / personaje):
        registro_ruta = carpeta / REGISTRO_REVISION
        registro = json.loads(registro_ruta.read_text(encoding="utf-8")) if registro_ruta.exists() else None
        if registro is not None and not (carpeta / MARCA_PENDIENTE).exists():
            return 0  # revisada y a la espera de aplicarse: no se toca
        revision = almacen.leer_json(almacen.clave("revision", f"{personaje}.json"))[0] or {}
        decididas = set() if todas else set(revision.get("aprobadas", [])) | set(revision.get("rechazadas", []))
        ya = set(registro["archivos"]) if registro else set()
        nuevas = []
        for clave in almacen.listar(almacen.clave("staging", personaje) + "/"):
            nombre = clave.rsplit("/", 1)[-1]
            if not nombre.endswith(".jpg") or nombre in decididas or nombre in ya:
                continue
            if not (carpeta / nombre).exists():
                almacen.descargar_archivo(clave, carpeta / nombre)
            nuevas.append(nombre)
        if not nuevas:
            return 0
        escribir_atomico(registro_ruta, json.dumps({"personaje": personaje, "archivos": sorted(ya | set(nuevas)),
                                                    "actualizado": time.time()}, indent=2).encode("utf-8"))
        if not (carpeta / MARCA_PENDIENTE).exists():
            escribir_atomico(carpeta / MARCA_PENDIENTE, TEXTO_PENDIENTE.format(
                personaje=personaje, marca=MARCA_PENDIENTE).encode("utf-8"))
        return len(nuevas)


def aplicar_revision(almacen, base: Path, personaje: str, forzar: bool = False,
                     exigir_terminada: bool = True) -> tuple[str, int, int]:
    """Aplica la revisión de base/<personaje>/. Devuelve (resultado, aprobadas, rechazadas) con
    resultado en: aplicada, sin_revision, en_revision (aún tiene _PENDIENTE.txt), sospechosa."""
    carpeta = base / personaje
    with bloqueo_archivo(base / ".bloqueos" / personaje):
        registro_ruta = carpeta / REGISTRO_REVISION
        if not registro_ruta.exists():
            return "sin_revision", 0, 0
        if exigir_terminada and (carpeta / MARCA_PENDIENTE).exists():
            return "en_revision", 0, 0
        descargadas = set(json.loads(registro_ruta.read_text(encoding="utf-8"))["archivos"])
        presentes = {r.name for r in carpeta.iterdir() if r.is_file()}
        aprobadas, rechazadas = descargadas & presentes, descargadas - presentes
        if not forzar and not aprobadas and len(rechazadas) >= LOTE_SOSPECHOSO:
            return "sospechosa", 0, len(rechazadas)

        def combinar(actual):
            actual = actual or {"personaje": personaje, "aprobadas": [], "rechazadas": []}
            return {**actual, "aprobadas": sorted((set(actual["aprobadas"]) - rechazadas) | aprobadas),
                    "rechazadas": sorted((set(actual["rechazadas"]) - aprobadas) | rechazadas),
                    "actualizado": time.time()}

        almacen.actualizar_json(almacen.clave("revision", f"{personaje}.json"), combinar)
        # Registro aplicado al historial; las copias aprobadas salen de la carpeta (siguen en staging/)
        historial = base / ".historial"
        historial.mkdir(parents=True, exist_ok=True)
        os.replace(registro_ruta, historial / f"{personaje}-{int(time.time() * 1000)}.json")
        for nombre in aprobadas:
            (carpeta / nombre).unlink(missing_ok=True)
        (carpeta / MARCA_PENDIENTE).unlink(missing_ok=True)
        if carpeta.is_dir() and not any(carpeta.iterdir()):
            carpeta.rmdir()
        return "aplicada", len(aprobadas), len(rechazadas)


def cmd_revisar(args) -> int:
    almacen = crear_almacen(args)
    base = carpeta_revision(args)
    nuevas = preparar_revision(almacen, base, args.personaje, args.todas)
    carpeta = base / args.personaje
    if (carpeta / REGISTRO_REVISION).exists() and not (carpeta / MARCA_PENDIENTE).exists():
        print(f"{carpeta} ya está revisada: aplica la revisión antes de preparar otra.")
    elif nuevas:
        print(f"{nuevas} imágenes nuevas en {carpeta}. Borra las que NO muestran a {args.personaje} y, al "
              f"terminar, borra {MARCA_PENDIENTE}.")
    else:
        print(f"No hay imágenes nuevas sin revisar de {args.personaje}.")
    return 0


def cmd_aplicar_revision(args) -> int:
    almacen = crear_almacen(args)
    resultado, aprobadas, rechazadas = aplicar_revision(almacen, carpeta_revision(args), args.personaje,
                                                        args.forzar, exigir_terminada=False)
    if resultado == "sin_revision":
        sys.exit(f"No hay revisión preparada para {args.personaje}: ejecuta antes 'revisar'")
    if resultado == "sospechosa":
        sys.exit(f"Se rechazarían las {rechazadas} imágenes de {args.personaje}. Si es correcto, repite con --forzar.")
    print(f"{args.personaje}: {aprobadas} aprobadas y {rechazadas} rechazadas.")
    return 0


# --- exportar ---------------------------------------------------------------------------------------------------

def cmd_exportar(args) -> int:
    """Copia las aprobadas de final/<personaje>/ a la carpeta del personaje en el dataset. Nunca
    sobrescribe: un archivo que ya está en el dataset se deja como está."""
    almacen = crear_almacen(args)
    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"No existe el dataset: {dataset}")
    pedidos = set(args.personajes or [])
    copiadas = existentes = 0
    personajes = sorted({c.split("/")[-2] for c in almacen.listar(almacen.clave("final") + "/") if c.endswith(".jpg")})
    for personaje in personajes:
        if pedidos and personaje not in pedidos:
            continue
        if not (dataset / personaje).is_dir():
            print(f"  {personaje}: no tiene carpeta en el dataset; se omite")
            continue
        nuevas = 0
        for clave in almacen.listar(almacen.clave("final", personaje) + "/"):
            if not clave.endswith(".jpg"):
                continue
            destino = dataset / personaje / clave.rsplit("/", 1)[-1]
            if destino.exists():
                existentes += 1
                continue
            if crear_sin_sobrescribir(destino, almacen.leer_bytes(clave)):
                nuevas += 1
        copiadas += nuevas
        print(f"  {personaje}: {nuevas} copiadas")
    print(f"{copiadas} imágenes copiadas a {dataset} ({existentes} ya estaban). Ahora vuelve a ejecutar "
          "scripts/newDataset/generar_imagenes_alteradas.py para regenerar las sintéticas.")
    return 0


# --- limpiar ----------------------------------------------------------------------------------------------------

# Lo único que el script crea en la carpeta de la ampliación; lo demás que haya ahí no se toca
GENERADO = ("staging", "final", "revision", "control", "inputs", "estado", "por_revisar", clasificacion.CARPETA,
            ".datos", ".meta", "LEEME.txt")


def cmd_limpiar(args) -> int:
    """Borra todo lo que generó la ampliación local para empezar de cero: imágenes recolectadas,
    revisiones, base interna y caché. Nunca toca el dataset ni archivos que no creó el script."""
    if not es_local(args):
        if args.bucket:
            sys.exit("limpiar solo borra la carpeta local. Para S3: aws s3 rm --recursive "
                     f"s3://{args.bucket}/{args.prefijo} (el bucket tiene versionado)")
        args.local = str(LOCAL_POR_DEFECTO)
    crear_almacen(args)  # valida que la carpeta no esté dentro del dataset
    raiz = Path(args.local).resolve()
    if not raiz.is_dir():
        print(f"No hay nada que limpiar: {raiz} no existe.")
        return 0
    cerrojo = CerrojoLocal(raiz / ".datos" / "proceso.lock")
    if (raiz / ".datos").is_dir():
        try:
            cerrojo.adquirir()
        except CandadoOcupado:
            sys.exit("Hay una recolección en curso en esta carpeta: detenla con Ctrl+C y vuelve a intentarlo.")
    try:
        objetivos = [raiz / nombre for nombre in GENERADO if (raiz / nombre).exists()]
        if not objetivos:
            print(f"No hay nada que limpiar en {raiz}.")
            return 0
        imagenes = sum(1 for o in objetivos if o.is_dir() for _ in o.rglob("*.jpg"))
        tamano = sum(f.stat().st_size for o in objetivos for f in ([o] if o.is_file() else o.rglob("*")) if f.is_file())
        print(f"Se borrará de {raiz}:")
        for objetivo in objetivos:
            print(f"  {objetivo.name}")
        print(f"({imagenes} imágenes, {tamano / 2**20:.0f} MB). El dataset no se toca.")
        if not args.si:
            respuesta = input("Escribe 'si' para confirmar: ").strip().lower()
            if respuesta not in ("si", "sí"):
                print("Cancelado: no se borró nada.")
                return 0
    finally:
        cerrojo.liberar()
    for objetivo in objetivos:
        if objetivo.is_dir():
            shutil.rmtree(objetivo)
        else:
            objetivo.unlink()
    restantes = list(raiz.iterdir())
    if not restantes:
        raiz.rmdir()
    print("Listo: la próxima ejecución de 'pnpm frinkiac' parte de cero"
          + (f" (se conservaron {len(restantes)} archivos que no creó el script)." if restantes else "."))
    return 0


# --- flujo (pnpm frinkiac) -----------------------------------------------------------------------------------------

LEEME = """Ampliación del dataset con Frinkiac (scripts/newDataset/frinkiac)

por_revisar/<personaje>/   AQUÍ SE REVISA: borra las imágenes en las que el personaje no aparece y,
                           al terminar la carpeta, borra su _PENDIENTE.txt. Luego vuelve a ejecutar
                           'pnpm frinkiac'.
clasificacion/             o, para revisar menos, 'pnpm frinkiac clasificar': la CNN separa por_revisar
                           en seguras/, dudosas/ y ruido/, y solo se revisan las dudosas (ver su LEEME).
final/<personaje>/         aprobadas que completan las 1000 originales. 'pnpm frinkiac exportar' las
                           copia al dataset (sin sobrescribir nada).
staging/<personaje>/       todas las imágenes nuevas, también las rechazadas. NO la borres a mano: es
                           la copia de referencia; si falta, el script vuelve a descargarlo todo.
revision/                  decisiones de la revisión.
control/, estado/, inputs/ y las carpetas ocultas: uso interno del script; no las modifiques.

Para empezar de cero: 'pnpm frinkiac limpiar' (borra solo lo que creó el script).

Nada de esto toca el dataset hasta que ejecutes 'pnpm frinkiac exportar'.
"""


def _paso(numero: int, texto: str) -> None:
    print(f"\n=== {numero}/6 {texto}", flush=True)


def cmd_flujo(args) -> int:
    """Todo el proceso en orden. Cada paso es idempotente: volver a ejecutarlo continúa donde quedó."""
    if not es_local(args) and not args.bucket:
        args.local = str(LOCAL_POR_DEFECTO)
    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"No existe el dataset: {dataset}. Indica la ruta con --dataset o SIMPSONS_DATASET.")
    almacen = crear_almacen(args)
    datos = directorio_datos(args)
    revision_base = carpeta_revision(args)
    if es_local(args) and not (Path(args.local) / "LEEME.txt").exists():
        escribir_atomico(Path(args.local) / "LEEME.txt", LEEME.encode("utf-8"))
    print(f"Ampliación en {almacen.ubicacion()}  (dataset: {dataset})")

    _paso(1, "Guiones del programa")
    g = G.obtener(datos / "guiones", G.carpetas_habituales(dataset, RAIZ_REPO), args.guiones, args.episodios)
    print(f"Fuente: {g.fuente}")
    if not g.completo:
        print("AVISO: esta copia solo trae los episodios desde 2010, así que habrá menos imágenes por personaje.\n"
              f"       Para el guion completo baja https://www.kaggle.com/datasets/{G.KAGGLE} y deja el .zip en\n"
              "       Descargas o junto al dataset; el próximo 'pnpm frinkiac' lo usa y conserva lo ya recolectado.")
    for nombre, ruta in (("guiones", g.lineas), ("episodios", g.episodios)):
        guardar_entrada(almacen, nombre, ruta)

    _paso(2, "Inventario de originales")
    excluir = imagenes_propias(almacen)
    objetivo = args.objetivo or OBJETIVO
    previo = None
    if almacen.cabecera(almacen.clave("inputs", ENTRADAS["inventario"])):
        previo = cargar_inventario(almacen, datos / "inputs")
    if previo is None or previo.objetivo != objetivo or previo.conteos != inv.conteos_actuales(dataset, excluir):
        print("El dataset cambió o no hay inventario: se cuentan las originales (tarda unos segundos)...")
        inventario = inv.construir(dataset, objetivo, excluir=excluir, progreso=lambda *_: None)
        salida = datos / "inventario.npz"
        inv.guardar(inventario, salida)
        guardar_entrada(almacen, "inventario", salida)
    else:
        inventario = previo
    debajo = inventario.por_debajo()
    if args.personajes:
        debajo = [p for p in debajo if p in set(args.personajes)]
    print(f"{len(debajo)} personajes bajo {objetivo} originales; faltan "
          f"{sum(objetivo - inventario.conteos[p] for p in debajo)} imágenes (solo originales: las sintéticas no cuentan).")

    _paso(3, "Nombres de los personajes en el guion")
    alias = cargar_alias(args.alias)
    lineas = leer_lineas(g.lineas, leer_episodios(g.episodios), {p: alias.get(p, []) for p in debajo})
    sin_lineas = [p for p in debajo if not lineas.get(p)]
    print(f"{len(debajo) - len(sin_lineas)} con líneas habladas en el guion.")
    if sin_lineas:
        print(f"Sin líneas (revisa {ALIAS_POR_DEFECTO.name} o ejecuta 'pnpm frinkiac mapear --guiones "
              f"\"{g.lineas}\"'): {', '.join(sin_lineas)}")

    _paso(4, "Revisiones terminadas")
    for s in clasificacion.sincronizar(revision_base):
        if s.estado == "trasladada":
            print(f"  {s.personaje}: clasificación de la CNN lista ({s.aprobadas} seguras o dudosas que quedan, "
                  f"{s.rechazadas} de ruido o borradas)")
        elif s.estado == "dudosas_pendientes":
            print(f"  {s.personaje}: faltan revisar {s.faltan} dudosas en {clasificacion.CARPETA}/"
                  f"{clasificacion.DUDOSAS}/{s.personaje}/ (al terminar, borra su {MARCA_PENDIENTE})")
        else:
            print(f"  {s.personaje}: {s.faltan} imágenes nuevas sin clasificar; ejecuta 'pnpm frinkiac clasificar'")
    aplicadas = 0
    if revision_base.is_dir():
        for carpeta in sorted(p for p in revision_base.iterdir() if p.is_dir() and not p.name.startswith(".")):
            resultado, aprobadas, rechazadas = aplicar_revision(almacen, revision_base, carpeta.name, args.forzar)
            if resultado == "aplicada":
                aplicadas += 1
                print(f"  {carpeta.name}: {aprobadas} aprobadas, {rechazadas} rechazadas")
            elif resultado == "en_revision":
                print(f"  {carpeta.name}: en revisión (todavía tiene {MARCA_PENDIENTE})")
            elif resultado == "sospechosa":
                print(f"  {carpeta.name}: se borraron las {rechazadas} imágenes; si es correcto usa "
                      f"'pnpm frinkiac --forzar'. Por ahora no se aplica.")
    if not aplicadas:
        print("Ninguna carpeta terminada de revisar.")

    _paso(5, "Recolección (Ctrl+C la detiene de forma segura; se retoma con 'pnpm frinkiac')")
    if not args.sin_issue:
        try:
            obtener_token(args)
        except Exception as error:  # noqa: BLE001
            print(f"{error}\nO ejecuta 'pnpm frinkiac --sin-issue' para recolectar sin publicar el issue.")
            return SALIDA_CONFIG

    def al_terminar(ev: Evaluacion) -> None:
        if ev.pendientes_revision:
            nuevas = preparar_revision(almacen, revision_base, ev.personaje)
            if nuevas:
                log.info("lista para revisar", extra={"datos": {"personaje": ev.personaje, "imagenes": nuevas,
                                                                "carpeta": str(revision_base / ev.personaje)}})

    args.al_terminar_personaje = al_terminar
    args.objetivo = objetivo
    codigo = cmd_recolectar(args)
    if codigo != SALIDA_OK:
        return codigo

    _paso(6, "Siguiente paso")
    progreso = (almacen.leer_json(almacen.clave("control", "progreso.json"))[0] or {}).get("personajes", {})
    for personaje, ev in sorted(progreso.items()):
        if ev.get("pendientes_revision"):
            preparar_revision(almacen, revision_base, personaje)
    carpetas =sorted(p.parent.name for p in revision_base.glob(f"*/{MARCA_PENDIENTE}")) if revision_base.is_dir() else []
    completos = sorted(p for p, ev in progreso.items() if ev.get("estado") == COMPLETO)
    if carpetas:
        print(f"Revisa {len(carpetas)} carpeta(s) en {revision_base}:")
        for personaje in carpetas:
            cuantas = len(list((revision_base / personaje).glob("*.jpg")))
            print(f"  {personaje}: {cuantas} imágenes")
        print(f"Borra las imágenes malas y el {MARCA_PENDIENTE} de cada carpeta terminada; luego 'pnpm frinkiac'.")
        sin_clasificar = sum(c["sin_clasificar"] for c in clasificacion.resumen(revision_base).values())
        if sin_clasificar:
            print(f"Para revisar menos: 'pnpm frinkiac clasificar' separa con la CNN las {sin_clasificar} sin "
                  f"clasificar en seguras, dudosas y ruido ({clasificacion.raiz_de(revision_base)}); solo se "
                  "revisan las dudosas.")
    if completos:
        print(f"{len(completos)} personaje(s) completos en {almacen.ubicacion(almacen.clave('final'))}. "
              "Para copiarlos al dataset: 'pnpm frinkiac exportar'.")
    if not carpetas and len(completos) == len(progreso):
        print("Todo completo. 'pnpm frinkiac exportar' y después generar_imagenes_alteradas.py.")
    return SALIDA_OK


# --- CLI ------------------------------------------------------------------------------------------------------

def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=os.environ.get("FRINKIAC_LOG", "INFO"))
    parser.add_argument("--log-formato", choices=("json", "texto"), default=os.environ.get("FRINKIAC_LOG_FORMATO", "json"))
    sub = parser.add_subparsers(dest="comando", required=True)

    def con_almacen(p):
        grupo = p.add_mutually_exclusive_group()
        grupo.add_argument("--bucket", default=os.environ.get("FRINKIAC_BUCKET"), help="bucket de S3")
        grupo.add_argument("--local", nargs="?", const=str(LOCAL_POR_DEFECTO), default=os.environ.get("FRINKIAC_LOCAL"),
                           metavar="CARPETA", help=f"carpeta de este PC en vez de S3 (por defecto {LOCAL_POR_DEFECTO})")
        p.add_argument("--prefijo", default=os.environ.get("FRINKIAC_PREFIJO", ""))
        p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        return p

    def con_dataset(p):
        p.add_argument("--dataset", default=str(DATASET_POR_DEFECTO))
        return p

    def lista(texto: str) -> list[str]:
        return [x.strip() for x in texto.split(",") if x.strip()]

    def opciones_recolectar(p):
        p.add_argument("--datos", default=None,
                       help="base y caché locales (por defecto <almacén local>/.datos, FRINKIAC_DATOS o /var/lib/frinkiac)")
        p.add_argument("--alias", default=str(ALIAS_POR_DEFECTO))
        p.add_argument("--personajes", type=lista, help="solo estos personajes, separados por comas")
        p.add_argument("--objetivo", type=int, default=None, help=f"por defecto {OBJETIVO}")
        p.add_argument("--margen", type=float, default=float(os.environ.get("FRINKIAC_MARGEN", MARGEN)))
        p.add_argument("--rps", type=float, default=float(os.environ.get("FRINKIAC_RPS", 2.0)),
                       help="peticiones por segundo a Frinkiac")
        p.add_argument("--repo", default=os.environ.get("FRINKIAC_REPO", REPO))
        p.add_argument("--mencionar", nargs="+", default=os.environ.get("FRINKIAC_MENCIONAR", " ".join(MENCIONES)).split())
        p.add_argument("--ssm-token", default=os.environ.get("FRINKIAC_SSM_TOKEN"),
                       help="parámetro SecureString de SSM con el token de GitHub")
        p.add_argument("--sin-issue", action="store_true", help="imprime el aviso en vez de publicarlo")
        return p

    p = opciones_recolectar(con_dataset(con_almacen(sub.add_parser("flujo", help="todo el proceso (pnpm frinkiac)"))))
    p.add_argument("--guiones", help="simpsons_script_lines.csv (si no, se busca o se descarga)")
    p.add_argument("--episodios", help="simpsons_episodes.csv")
    p.add_argument("--destino", default=None, help="carpeta de revisión (por defecto <almacén>/por_revisar)")
    p.add_argument("--forzar", action="store_true", help="aplica también carpetas en las que se borró todo")
    p.set_defaults(funcion=cmd_flujo)

    p = con_dataset(con_almacen(sub.add_parser("inventario", help="cuenta las originales del dataset local")))
    p.add_argument("--objetivo", type=int, default=OBJETIVO)
    p.add_argument("--salida", default=None)
    p.add_argument("--guiones", help="simpsons_script_lines.csv de Kaggle")
    p.add_argument("--episodios", help="simpsons_episodes.csv de Kaggle")
    p.set_defaults(funcion=cmd_inventario)

    p = sub.add_parser("mapear", help="revisa los nombres del guion de cada personaje")
    p.add_argument("--guiones", required=True)
    p.add_argument("--alias", default=str(ALIAS_POR_DEFECTO))
    p.add_argument("--inventario", help="inventario.npz local: revisa solo los personajes bajo el objetivo")
    p.add_argument("--escribir", action="store_true", help="agrega al JSON la mejor sugerencia de los que no tienen")
    p.set_defaults(funcion=cmd_mapear)

    p = opciones_recolectar(con_almacen(sub.add_parser("recolectar", help="baja fotogramas únicos y avisa (EC2)")))
    p.set_defaults(funcion=cmd_recolectar)

    p = con_almacen(sub.add_parser("estado", help="avance por personaje"))
    p.add_argument("--objetivo", type=int, default=None)
    p.set_defaults(funcion=cmd_estado)

    for nombre, funcion, ayuda in (("revisar", cmd_revisar, "prepara las nuevas sin revisar"),
                                   ("aplicar-revision", cmd_aplicar_revision, "aplica una carpeta revisada")):
        p = con_almacen(sub.add_parser(nombre, help=ayuda))
        p.add_argument("--personaje", required=True)
        p.add_argument("--destino", default=None, help="carpeta de revisión (por defecto <almacén>/por_revisar)")
        if nombre == "revisar":
            p.add_argument("--todas", action="store_true", help="también las ya revisadas")
        else:
            p.add_argument("--forzar", action="store_true", help="permite rechazar todas las de la carpeta")
        p.set_defaults(funcion=funcion)

    p = con_dataset(con_almacen(sub.add_parser("exportar", help="copia final/ al dataset sin sobrescribir")))
    p.add_argument("--personajes", type=lista)
    p.set_defaults(funcion=cmd_exportar)

    p = con_dataset(con_almacen(sub.add_parser("limpiar", help="borra lo generado para empezar de cero")))
    p.add_argument("--si", action="store_true", help="no pide confirmación")
    p.set_defaults(funcion=cmd_limpiar)
    return parser


def main(argv=None) -> int:
    # La consola de Windows (cp1252) no tiene caracteres como "♪" de algunos subtítulos: se reemplazan
    # en vez de cortar la ejecución
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = construir_parser().parse_args(argv)
    configurar_logs(args.log, args.log_formato)
    try:
        return args.funcion(args)
    except SystemExit as salida:
        if isinstance(salida.code, str):
            # Falta configurar algo: se explica y se sale con 2 para que el lanzador no reintente en bucle
            print(salida.code, file=sys.stderr)
            return SALIDA_CONFIG
        raise


if __name__ == "__main__":
    sys.exit(main())
