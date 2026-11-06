"""Issue de aviso en GitHub, idempotente.

Cada corrida tiene un identificador (control/run.json en S3) que se escribe como comentario HTML
oculto en el cuerpo del issue. Antes de crear se busca ese marcador: si el issue ya existe, se comenta
en él. Cada comentario lleva a su vez la huella de su contenido, así que repetir el mismo aviso (por
ejemplo, tras un reinicio entre la creación y la confirmación) no agrega nada.

Los POST no se reintentan a ciegas: si una creación falla sin respuesta, el reintento vuelve a empezar
por la búsqueda, que encuentra el issue si sí se había creado.
"""

import hashlib
import logging

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

log = logging.getLogger("frinkiac.github")

API = "https://api.github.com"
ETIQUETA = "dataset-frinkiac"


class ErrorGitHub(Exception):
    pass


class _Temporal(Exception):
    pass


class GitHub:
    def __init__(self, token: str, repo: str, sesion: requests.Session | None = None, api: str = API):
        self.repo = repo
        self.api = api.rstrip("/")
        self.sesion = sesion or requests.Session()
        self.sesion.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "LosSimpsons-DeepLearning-dataset",
        })

    def _pedir(self, metodo: str, ruta: str, **argumentos):
        try:
            respuesta = self.sesion.request(metodo, f"{self.api}{ruta}", timeout=30, **argumentos)
        except (requests.ConnectionError, requests.Timeout) as error:
            raise _Temporal(str(error)) from error
        limite_agotado = respuesta.status_code == 403 and respuesta.headers.get("X-RateLimit-Remaining") == "0"
        if respuesta.status_code in (429, 502, 503, 504) or respuesta.status_code >= 500 or limite_agotado:
            raise _Temporal(f"HTTP {respuesta.status_code} {ruta}")
        if respuesta.status_code >= 400:
            raise ErrorGitHub(f"HTTP {respuesta.status_code} {ruta}: {respuesta.text[:300]}")
        return respuesta

    def _paginas(self, ruta: str, params: dict):
        pagina = 1
        while True:
            lote = self._pedir("GET", ruta, params={**params, "per_page": 100, "page": pagina}).json()
            yield from lote
            if len(lote) < 100:
                return
            pagina += 1

    def buscar_issue(self, marcador: str, etiqueta: str = ETIQUETA) -> dict | None:
        for issue in self._paginas(f"/repos/{self.repo}/issues", {"state": "all", "labels": etiqueta}):
            if marcador in (issue.get("body") or ""):
                return issue
        return None

    def comentarios(self, numero: int) -> list[dict]:
        return list(self._paginas(f"/repos/{self.repo}/issues/{numero}/comments", {}))

    def asegurar_etiqueta(self, etiqueta: str = ETIQUETA) -> None:
        try:
            self._pedir("POST", f"/repos/{self.repo}/labels",
                        json={"name": etiqueta, "color": "fbca04",
                              "description": "Ampliación del dataset con Frinkiac"})
        except ErrorGitHub as error:
            if "HTTP 422" not in str(error):  # 422: ya existe
                raise

    def crear_issue(self, titulo: str, cuerpo: str, etiqueta: str = ETIQUETA) -> dict:
        return self._pedir("POST", f"/repos/{self.repo}/issues",
                           json={"title": titulo, "body": cuerpo, "labels": [etiqueta]}).json()

    def comentar(self, numero: int, cuerpo: str) -> dict:
        return self._pedir("POST", f"/repos/{self.repo}/issues/{numero}/comments", json={"body": cuerpo}).json()


def huella(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16]


def publicar(gh: GitHub, run_id: str, fase: str, titulo: str, contenido: str) -> str:
    """Crea el issue de la corrida o comenta en él. Devuelve la URL del issue. Idempotente."""
    marcador = f"<!-- frinkiac-run:{run_id} -->"
    marca_fase = f"<!-- frinkiac-fase:{fase}:{huella(contenido)} -->"
    gh.asegurar_etiqueta()
    issue = gh.buscar_issue(marcador)
    if issue is None:
        creado = gh.crear_issue(titulo, f"{marcador}\n{marca_fase}\n\n{contenido}")
        log.info("issue creado", extra={"datos": {"url": creado["html_url"]}})
        return creado["html_url"]
    if marca_fase in (issue.get("body") or "") or any(
            marca_fase in (c.get("body") or "") for c in gh.comentarios(issue["number"])):
        log.info("el aviso ya estaba publicado", extra={"datos": {"url": issue["html_url"]}})
        return issue["html_url"]
    gh.comentar(issue["number"], f"{marca_fase}\n\n{contenido}")
    log.info("comentario agregado", extra={"datos": {"url": issue["html_url"]}})
    return issue["html_url"]


def publicar_con_reintentos(gh: GitHub, run_id: str, fase: str, titulo: str, contenido: str,
                            intentos: int = 8, dormir=None) -> str:
    """Reintenta el ciclo completo (buscar y luego crear o comentar) ante fallos temporales."""
    opciones = {"sleep": dormir} if dormir else {}
    for intento in Retrying(stop=stop_after_attempt(intentos), wait=wait_exponential_jitter(initial=2, max=300),
                            retry=retry_if_exception_type(_Temporal), reraise=True, **opciones):
        with intento:
            return publicar(gh, run_id, fase, titulo, contenido)
    raise AssertionError("inalcanzable")


def token_desde_ssm(cliente_ssm, nombre: str) -> str:
    return cliente_ssm.get_parameter(Name=nombre, WithDecryption=True)["Parameter"]["Value"]
