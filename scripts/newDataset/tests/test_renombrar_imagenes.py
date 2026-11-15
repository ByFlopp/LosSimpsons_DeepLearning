"""Pruebas de renombrar_imagenes.py y de la comprobación de nombres de generar_imagenes_alteradas.py.

No necesitan GPU ni el dataset: usan un dataset de juguete.
    .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generar_imagenes_alteradas as G  # noqa: E402
import renombrar_imagenes as N  # noqa: E402

CAJAS = [
    {"personaje": "ana", "archivo": "ana_0001.jpg", "cajas": []},
    {"personaje": "ana", "archivo": "beto__S01E01__t0010000.jpg", "cajas": [{"nombre": "Beto · 90 %"}]},
    {"personaje": "beto", "archivo": "beto__S01E01__t0012000.jpg", "cajas": []},
]


def imagen(ruta: Path, semilla: int) -> None:
    datos = np.random.default_rng(semilla).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    Image.fromarray(datos).save(ruta)


def listado(dataset: Path) -> dict[str, list[str]]:
    return {c.name: sorted(p.name for p in c.iterdir()) for c in sorted(dataset.iterdir()) if c.is_dir()}


def filas_renombres(dataset: Path) -> list[tuple[str, str, str]]:
    return [(f["personaje"], f["antiguo"], f["nuevo"]) for f in N.leer_renombres(dataset)]


@pytest.fixture
def dataset(tmp_path):
    """ana mezcla Kaggle (con un número repetido en .jpg y .png), un fotograma de beto, un videoframe, variantes
    y una alterada huérfana; beto solo tiene un fotograma. Hay registros, cajas y una carpeta de revisión."""
    raiz = tmp_path / "simpsons_dataset"
    ana, beto = raiz / "ana", raiz / "beto"
    ana.mkdir(parents=True)
    beto.mkdir()
    for semilla, nombre in enumerate(["ana_0001.jpg", "ana_0001__alterada.jpg", "ana_0002.jpg", "ana_0002.png",
                                      "beto__S01E01__t0010000.jpg", "beto__S01E01__t0010000__parecida.jpg",
                                      "videoframe_5.png", "huerfana__alterada.jpg"]):
        imagen(ana / nombre, semilla)
    imagen(beto / "beto__S01E01__t0012000.jpg", 20)
    (ana / "_PENDIENTE.txt").write_text("no es una imagen", encoding="utf-8")
    for registro in (G.REGISTRO, G.REGISTRO_PARECIDAS, G.REGISTRO_GEMELAS):
        (raiz / registro).write_text("personaje,original\nana,ana_0001.jpg\n", encoding="utf-8")
    (raiz / N.REGISTRO_CAJAS).write_bytes(
        "".join(json.dumps(c, ensure_ascii=False) + "\r\n" for c in CAJAS).encode("utf-8"))
    revision = tmp_path / G.NOMBRE_REVISION
    (revision / "beto").mkdir(parents=True)
    (revision / G.MARCA_REVISION).write_text("generada", encoding="utf-8")
    imagen(revision / "beto" / "ana_0001.jpg", 30)
    return raiz


def ejecutar(monkeypatch, *argumentos):
    monkeypatch.setattr(sys, "argv", ["renombrar_imagenes.py", *argumentos])
    N.main()


# --- Regla de nombres ----------------------------------------------------------------------------------------

def test_numeradas_separa_las_que_cumplen_y_las_repetidas(dataset):
    con_numero, sin_formato = G.numeradas(dataset / "ana")
    assert {r.name: n for r, n in con_numero.items()} == {"ana_0001.jpg": 1, "ana_0002.jpg": 2}
    assert [r.name for r in sin_formato] == ["ana_0002.png", "beto__S01E01__t0010000.jpg", "videoframe_5.png"]


def test_planificar_sigue_desde_el_numero_mas_alto(dataset):
    conservan, cambios = N.planificar(dataset / "ana")
    assert conservan == 2
    assert [(o.name, d.name) for o, d in cambios] == [
        ("ana_0002.png", "ana_0003.png"),
        ("beto__S01E01__t0010000.jpg", "ana_0004.jpg"),
        ("videoframe_5.png", "ana_0005.png"),
    ]


# --- Vista previa y aplicación -------------------------------------------------------------------------------

def test_la_vista_previa_no_toca_nada(dataset, monkeypatch, capsys):
    antes = listado(dataset)
    ejecutar(monkeypatch, "--dataset", str(dataset))
    salida = capsys.readouterr().out
    assert "beto__S01E01__t0010000.jpg  ->  ana_0004.jpg" in salida
    assert "Se borran 3 variantes" in salida
    assert listado(dataset) == antes
    assert not (dataset / N.REGISTRO_RENOMBRES).exists()


def test_aplicar_borra_variantes_renombra_y_registra(dataset, monkeypatch):
    cajas_antes = (dataset / N.REGISTRO_CAJAS).read_bytes()
    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")

    assert listado(dataset) == {
        "ana": ["_PENDIENTE.txt", "ana_0001.jpg", "ana_0002.jpg", "ana_0003.png", "ana_0004.jpg", "ana_0005.png"],
        "beto": ["beto_0001.jpg"],
    }
    assert filas_renombres(dataset) == [
        ("ana", "ana_0002.png", "ana_0003.png"),
        ("ana", "beto__S01E01__t0010000.jpg", "ana_0004.jpg"),
        ("ana", "videoframe_5.png", "ana_0005.png"),
        ("beto", "beto__S01E01__t0012000.jpg", "beto_0001.jpg"),
    ]
    assert N.leer_nombres_anteriores(dataset)["ana_0004.jpg"] == "beto__S01E01__t0010000.jpg"
    # Los registros de las variantes se borran (se regeneran) y la carpeta de revisión se vacía
    for registro in (G.REGISTRO, G.REGISTRO_PARECIDAS, G.REGISTRO_GEMELAS):
        assert not (dataset / registro).exists()
    assert not (dataset.parent / G.NOMBRE_REVISION).exists()
    # Las cajas cambian de nombre, conservan el resto y los saltos de línea; la versión anterior queda al lado
    cajas = (dataset / N.REGISTRO_CAJAS).read_bytes()
    assert cajas.count(b"\r\n") == len(CAJAS)
    assert [json.loads(linea)["archivo"] for linea in cajas.decode("utf-8").splitlines()] == [
        "ana_0001.jpg", "ana_0004.jpg", "beto_0001.jpg"]
    assert json.loads(cajas.decode("utf-8").splitlines()[1])["cajas"] == CAJAS[1]["cajas"]
    assert (dataset / (N.REGISTRO_CAJAS + N.SUFIJO_RESPALDO)).read_bytes() == cajas_antes
    assert G.originales_sin_formato(G.carpetas_personaje(dataset)) == []


def test_volver_a_aplicar_solo_numera_las_nuevas(dataset, monkeypatch):
    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")
    antes, filas = listado(dataset), filas_renombres(dataset)
    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")
    assert listado(dataset) == antes and filas_renombres(dataset) == filas

    # Un fotograma exportado después recibe el número siguiente
    imagen(dataset / "beto" / "beto__S02E02__t0000500.jpg", 40)
    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")
    assert listado(dataset)["beto"] == ["beto_0001.jpg", "beto_0002.jpg"]
    assert filas_renombres(dataset)[-1] == ("beto", "beto__S02E02__t0000500.jpg", "beto_0002.jpg")


def test_una_ejecucion_interrumpida_se_completa(dataset, monkeypatch):
    # Se anotó el renombre pero se cortó antes de hacerlo
    with open(dataset / N.REGISTRO_RENOMBRES, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(N.CAMPOS_RENOMBRES)
        escritor.writerow(["beto", "beto__S01E01__t0012000.jpg", "beto_0001.jpg"])
    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")
    assert listado(dataset)["beto"] == ["beto_0001.jpg"]
    assert [f for f in filas_renombres(dataset) if f[0] == "beto"] == [
        ("beto", "beto__S01E01__t0012000.jpg", "beto_0001.jpg")]


# --- Generación con la nomenclatura --------------------------------------------------------------------------

def test_la_generacion_exige_el_formato_y_sus_variantes_lo_cumplen(dataset, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generar_imagenes_alteradas.py", "--dataset", str(dataset), "--fraccion", "0.5"])
    with pytest.raises(SystemExit, match="renombrar_imagenes.py --aplicar"):
        G.main()
    assert (dataset / "ana" / "ana_0001__alterada.jpg").exists()  # se detuvo sin borrar nada

    ejecutar(monkeypatch, "--dataset", str(dataset), "--aplicar")
    monkeypatch.setattr(sys, "argv", ["generar_imagenes_alteradas.py", "--dataset", str(dataset), "--fraccion", "0.5"])
    G.main()

    generadas = 0
    for carpeta, archivos in listado(dataset).items():
        patron = re.compile(rf"{carpeta}_\d{{4,}}(__alterada|__parecida)?\.(jpg|png)")
        for nombre in archivos:
            if Path(nombre).suffix.lower() in G.EXTENSIONES_VALIDAS:
                assert patron.fullmatch(nombre), nombre
                generadas += G.es_generado(Path(nombre))
    assert generadas >= 2  # la mitad de las originales de cada personaje recibe su alterada
