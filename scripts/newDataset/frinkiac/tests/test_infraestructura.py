"""Candado, subida condicional, cliente HTTP de Frinkiac, issue de GitHub y la CLI de revisión."""

import json

import pytest
import responses

import aviso_github
from almacen import Candado, CandadoOcupado, ErrorVerificacion
from ampliar_con_frinkiac import main
from comun import sha256_hex
from conftest import imagen_ruido
from fuentes import ClienteFrinkiac, ErrorTemporal, NoEncontrado


# --- S3 ----------------------------------------------------------------------------------------------

def test_subida_nunca_sobrescribe(almacen):
    datos = imagen_ruido(1)
    assert almacen.subir_fotograma("staging/p/a.jpg", datos, sha256_hex(datos), {}) == "creada"
    assert almacen.subir_fotograma("staging/p/a.jpg", datos, sha256_hex(datos), {}) == "ya_existia"
    otros = imagen_ruido(2)
    with pytest.raises(ErrorVerificacion):
        almacen.subir_fotograma("staging/p/a.jpg", otros, sha256_hex(otros), {})
    assert almacen.leer_bytes("staging/p/a.jpg") == datos


class Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_candado_excluye_y_recupera_tras_vencer(almacen):
    reloj = Reloj()
    a = Candado(almacen, "control/lock.json", "host-a", lease_s=600, reloj=reloj)
    b = Candado(almacen, "control/lock.json", "host-b", lease_s=600, reloj=reloj)
    a.adquirir(renovar_en_segundo_plano=False)
    with pytest.raises(CandadoOcupado):
        b.adquirir(renovar_en_segundo_plano=False)
    assert a.renovar()
    reloj.t += 601  # a murió sin renovar
    b.adquirir(renovar_en_segundo_plano=False)
    assert not a.renovar()  # a se entera de que lo perdió
    b.liberar()
    c = Candado(almacen, "control/lock.json", "host-c", reloj=reloj)
    c.adquirir(renovar_en_segundo_plano=False)  # liberado: se toma al instante


def test_misma_maquina_retoma_su_candado(almacen):
    reloj = Reloj()
    Candado(almacen, "control/lock.json", "host-a", reloj=reloj).adquirir(renovar_en_segundo_plano=False)
    Candado(almacen, "control/lock.json", "host-a", reloj=reloj).adquirir(renovar_en_segundo_plano=False)


# --- Frinkiac ------------------------------------------------------------------------------------------

def cliente(intentos=4):
    return ClienteFrinkiac(rps=0, intentos=intentos, dormir=lambda _s: None, base="https://frinkiac.test")


@responses.activate
def test_frinkiac_reintenta_429_y_respeta_404():
    responses.get("https://frinkiac.test/api/search", status=429, headers={"Retry-After": "0"})
    responses.get("https://frinkiac.test/api/search", status=503)
    responses.get("https://frinkiac.test/api/search", json=[{"Episode": "S01E01", "Timestamp": 1}])
    assert cliente().buscar("hola") == [{"Episode": "S01E01", "Timestamp": 1}]
    responses.get("https://frinkiac.test/img/S01E01/5.jpg", status=404)
    with pytest.raises(NoEncontrado):
        cliente().imagen("S01E01", 5)


@responses.activate
def test_frinkiac_agota_reintentos_como_error_temporal():
    responses.get("https://frinkiac.test/api/caption", status=502)
    with pytest.raises(ErrorTemporal):
        cliente(intentos=3).subtitulos("S01E01", 5)
    assert len(responses.calls) == 3


# --- GitHub ----------------------------------------------------------------------------------------------

API = "https://api.github.com/repos/o/r"


@responses.activate
def test_issue_se_crea_una_vez_y_luego_se_comenta():
    gh = aviso_github.GitHub("t", "o/r")
    responses.post(f"{API}/labels", status=422, json={})
    responses.get(f"{API}/issues", json=[])
    responses.post(f"{API}/issues", json={"number": 7, "html_url": "https://github.com/o/r/issues/7"})
    url = aviso_github.publicar(gh, "run1", "revision", "Título", "@DiegoPyLL contenido")
    assert url.endswith("/7")
    cuerpo = json.loads(responses.calls[-1].request.body)["body"]
    assert "<!-- frinkiac-run:run1 -->" in cuerpo and "@DiegoPyLL" in cuerpo

    responses.reset()
    responses.post(f"{API}/labels", status=422, json={})
    responses.get(f"{API}/issues", json=[{"number": 7, "html_url": "https://github.com/o/r/issues/7", "body": cuerpo}])
    responses.get(f"{API}/issues/7/comments", json=[])
    # mismo aviso repetido (reinicio): no se publica nada
    aviso_github.publicar(gh, "run1", "revision", "Título", "@DiegoPyLL contenido")
    assert not [c for c in responses.calls if c.request.method == "POST" and "/issues" in c.request.url]
    # aviso nuevo: comentario en el mismo issue, nunca un segundo issue
    responses.post(f"{API}/issues/7/comments", json={"id": 1})
    aviso_github.publicar(gh, "run1", "final", "Título", "@DiegoPyLL completo")
    publicados = [c.request.url for c in responses.calls if c.request.method == "POST" and "/issues" in c.request.url]
    assert publicados == [f"{API}/issues/7/comments"]


# --- CLI de recolección de punta a punta ----------------------------------------------------------------------

@pytest.fixture
def bucket_preparado(almacen, opcion_almacen, tmp_path, monkeypatch):
    """Ejecuta 'inventario' sobre un dataset falso y reemplaza Frinkiac por el doble."""
    import ampliar_con_frinkiac
    from conftest import FrinkiacFalso, escribir_guiones, subtitulos_para

    lineas = [(i, "Louie", 10_000 * i, f"linea{i} dice algo") for i in range(1, 6)]
    carpeta = tmp_path / "dataset" / "louie"
    carpeta.mkdir(parents=True)
    for i in range(3):
        (carpeta / f"louie_{i:04d}.jpg").write_bytes(imagen_ruido(500 + i))
        (carpeta / f"louie_{i:04d}__alterada.jpg").write_bytes(imagen_ruido(600 + i))
    guiones, episodios = escribir_guiones(tmp_path, lineas)
    assert main(["inventario", *opcion_almacen, "--dataset", str(tmp_path / "dataset"), "--objetivo", "8",
                 "--salida", str(tmp_path / "inventario.npz"), "--guiones", str(guiones),
                 "--episodios", str(episodios)]) == 0
    monkeypatch.setattr(ampliar_con_frinkiac, "ClienteFrinkiac", lambda **_: FrinkiacFalso(subtitulos_para(lineas)))
    # Nunca tocar GitHub de verdad: ni el gh del PC ni el repo real
    monkeypatch.setattr(ampliar_con_frinkiac.shutil, "which", lambda _nombre: None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    alias = tmp_path / "alias.json"
    alias.write_text(json.dumps({"louie": ["Louie"]}))
    # En modo local la base va por defecto a <almacén>/.datos; en S3 se indica
    datos = [] if opcion_almacen[0] == "--local" else ["--datos", str(tmp_path / "datos")]
    return ["recolectar", *opcion_almacen, *datos, "--alias", str(alias), "--margen", "0", "--repo", "prueba/no-existe"]


def test_recolectar_termina_y_avisa(bucket_preparado, almacen, capsys):
    assert main([*bucket_preparado, "--sin-issue"]) == 0
    salida = capsys.readouterr().out
    assert "@DiegoPyLL" in salida and "| `louie` | 3 | 5 |" in salida
    assert len([k for k in almacen.listar("staging/louie/") if k.endswith(".jpg")]) == 5  # 3 + 5 = 8
    assert almacen.leer_json("control/lock.json")[0]["expira"] == 0  # candado liberado
    assert almacen.leer_json("control/progreso.json")[0]["personajes"]["louie"]["estado"] == "por_revisar"
    # una segunda ejecución retoma el estado y no agrega nada
    assert main([*bucket_preparado, "--sin-issue"]) == 0
    assert len([k for k in almacen.listar("staging/louie/") if k.endswith(".jpg")]) == 5


def test_recolectar_sin_github_no_termina(bucket_preparado, almacen):
    assert main(bucket_preparado) == 1  # sin token no hay aviso: se reintentará al reiniciar
    pendiente, _ = almacen.leer_json("control/aviso_pendiente.json")
    assert pendiente["fase"] == "revision" and "@DiegoPyLL" in pendiente["contenido"]


# --- CLI de revisión ---------------------------------------------------------------------------------------

def test_revisar_y_aplicar_revision(almacen, opcion_almacen, tmp_path):
    for i in range(4):
        datos = imagen_ruido(i)
        almacen.subir_fotograma(f"staging/louie/louie__S01E01__t{i:07d}.jpg", datos, sha256_hex(datos), {})
    destino = tmp_path / "por_revisar"
    base = opcion_almacen
    assert main(["revisar", *base, "--personaje", "louie", "--destino", str(destino)]) == 0
    carpeta = destino / "louie"
    imagenes = sorted(carpeta.glob("*.jpg"))
    assert len(imagenes) == 4 and (carpeta / "_PENDIENTE.txt").exists()
    imagenes[0].unlink()  # el revisor descarta una
    # preparar otra vez no vuelve a traer la descartada ni duplica nada
    assert main(["revisar", *base, "--personaje", "louie", "--destino", str(destino)]) == 0
    assert len(list(carpeta.glob("*.jpg"))) == 3
    assert main(["aplicar-revision", *base, "--personaje", "louie", "--destino", str(destino)]) == 0
    revision, _ = almacen.leer_json("revision/louie.json")
    assert revision["rechazadas"] == [imagenes[0].name]
    assert len(revision["aprobadas"]) == 3
    assert not carpeta.exists()  # aplicada: las copias aprobadas salen de la carpeta (siguen en staging/)
    # una nueva revisión solo trae lo que falta decidir: nada
    assert main(["revisar", *base, "--personaje", "louie", "--destino", str(destino)]) == 0
    assert not carpeta.exists()


# --- Modo local ------------------------------------------------------------------------------------------------

def test_exportar_copia_final_sin_sobrescribir(almacen, opcion_almacen, tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "louie").mkdir(parents=True)
    propia = imagen_ruido(99)
    (dataset / "louie" / "louie__S01E01__t0000001.jpg").write_bytes(propia)  # ya estaba en el dataset
    for i in range(3):
        datos = imagen_ruido(i)
        almacen.subir_fotograma(f"final/louie/louie__S01E01__t{i:07d}.jpg", datos, sha256_hex(datos), {})
    almacen.escribir_json("final/louie/_RESUMEN.json", {"completo": True})
    assert main(["exportar", *opcion_almacen, "--dataset", str(dataset)]) == 0
    copiadas = sorted(p.name for p in (dataset / "louie").iterdir())
    assert copiadas == [f"louie__S01E01__t{i:07d}.jpg" for i in range(3)]  # sin el _RESUMEN.json
    assert (dataset / "louie" / "louie__S01E01__t0000001.jpg").read_bytes() == propia  # no se sobrescribió
    assert main(["exportar", *opcion_almacen, "--dataset", str(dataset)]) == 0  # repetir es inofensivo


def test_almacen_local_no_puede_estar_dentro_del_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "louie").mkdir(parents=True)
    # 2 = falta configurar algo: el lanzador de pnpm no lo reintenta
    assert main(["exportar", "--local", str(dataset / "frinkiac"), "--dataset", str(dataset)]) == 2
    assert not (dataset / "frinkiac").exists()


def test_local_crea_sin_sobrescribir_aunque_compitan_hilos(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from almacen import crear_sin_sobrescribir

    destino = tmp_path / "x" / "img.jpg"
    contenidos = [imagen_ruido(i) for i in range(16)]
    with ThreadPoolExecutor(16) as grupo:
        creadas = list(grupo.map(lambda datos: crear_sin_sobrescribir(destino, datos), contenidos))
    assert creadas.count(True) == 1
    assert destino.read_bytes() == contenidos[creadas.index(True)]
    assert [p.name for p in destino.parent.iterdir()] == ["img.jpg"]  # no quedan temporales


def test_local_compare_and_swap_no_pierde_escrituras(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from almacen import AlmacenLocal

    almacen = AlmacenLocal(tmp_path / "almacen")

    def sumar(_):
        almacen.actualizar_json("revision/louie.json",
                                lambda actual: {"n": (actual or {"n": 0})["n"] + 1}, intentos=200)

    with ThreadPoolExecutor(8) as grupo:
        list(grupo.map(sumar, range(40)))
    assert almacen.leer_json("revision/louie.json")[0]["n"] == 40
