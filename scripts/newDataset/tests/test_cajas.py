"""Pruebas de las cajas de detectar_cajas.py y de su geometría en generar_imagenes_alteradas.py.

No necesitan GPU, modelos ni el dataset:
    .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detectar_cajas as D  # noqa: E402
import generar_imagenes_alteradas as G  # noqa: E402


class GiroFijo:
    """Hace que girar() use exactamente este ángulo (con su signo)."""

    def __init__(self, angulo):
        self.angulo = angulo

    def uniform(self, *_):
        return abs(self.angulo)

    def choice(self, _):
        return 1.0 if self.angulo > 0 else -1.0


def degradado(ancho, alto):
    x = np.linspace(0, 255, ancho, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, alto, dtype=np.float32)[:, None, None]
    canales = [np.broadcast_to(x, (alto, ancho, 1)), np.broadcast_to(y, (alto, ancho, 1)),
               np.broadcast_to((x + y) / 2, (alto, ancho, 1))]
    return Image.fromarray(np.concatenate(canales, axis=2).astype(np.uint8), "RGB")


# Salida de girar() antes de separar recorte_del_giro: la variante tiene que seguir siendo la misma
HASHES_GIRAR = {
    (160, 90, 1): "2840237b82b8f453b74866297cfbfa46ee313f9853595d801e8b9ceb0ef23c0c",
    (90, 160, 2): "397e6cfde6994e36ebe8f576b0ed748e9c46b6f14afd27395424a9a48b7fe74d",
    (640, 480, 3): "6e5d5404094037acdf5c4bf14c5cedd7a6b4124ee770b361301f7346f09932cc",
    (37, 41, 4): "dd9c83f04b77b9ae31c00abe323b99dd58110fad295186b7ea80468c349bdab8",
    (3, 3, 5): "a9bc6f5e6b784e8a37312fc000196c4048cf6d1a2ed6f7bf6fa32ced2873a6fd",
}


@pytest.mark.parametrize("ancho, alto, semilla", list(HASHES_GIRAR))
def test_girar_no_cambia_byte_a_byte(ancho, alto, semilla):
    girada, _ = G.girar(degradado(ancho, alto), np.random.default_rng(semilla))
    assert hashlib.sha256(girada.tobytes()).hexdigest() == HASHES_GIRAR[(ancho, alto, semilla)]


@pytest.mark.parametrize("ancho, alto", [(640, 480), (300, 500), (960, 720)])
@pytest.mark.parametrize("relativa", [(0.1, 0.1, 0.31, 0.62), (0.62, 0.2, 0.95, 0.92), (0.0, 0.0, 1.0, 1.0),
                                      (0.4, 0.4, 0.6, 0.6)])
@pytest.mark.parametrize("angulo", [10.0, -10.0, 15.5, -15.5, 20.0, -20.0])
def test_caja_tras_giro_sigue_al_personaje(ancho, alto, relativa, angulo):
    # Un rectángulo rojo hace de personaje: su caja en la variante es la que envuelve a lo rojo
    caja = (round(relativa[0] * ancho), round(relativa[1] * alto), round(relativa[2] * ancho), round(relativa[3] * alto))
    imagen = Image.new("RGB", (ancho, alto), (128, 128, 128))
    ImageDraw.Draw(imagen).rectangle((caja[0], caja[1], caja[2] - 1, caja[3] - 1), fill=(255, 0, 0))
    variante, aplicado = G.girar(imagen, GiroFijo(angulo))
    pixeles = np.asarray(variante).astype(int)
    filas, columnas = np.nonzero((pixeles[..., 0] > 200) & (pixeles[..., 1] < 60))
    real = (columnas.min(), filas.min(), columnas.max() + 1, filas.max() + 1)
    calculada = G.caja_tras_giro(caja, ancho, alto, aplicado)
    tolerancia = max(4, 0.01 * max(ancho, alto))
    assert all(abs(c - r) <= tolerancia for c, r in zip(calculada, real)), (real, calculada)


def test_caja_tras_giro_fuera_de_la_variante():
    # Una esquina que el giro deja fuera del recorte
    assert G.caja_tras_giro((0, 0, 20, 20), 640, 480, 20.0) is None
    # Sin recorte posible (imagen diminuta), la caja solo gira
    assert G.recorte_del_giro(3, 3, 18.0) is None
    assert G.caja_tras_giro((0, 0, 3, 3), 3, 3, 18.0) is not None


@pytest.mark.parametrize("caja, ancho, alto", [
    ((100, 50, 200, 400), 640, 480),     # alta: el cuadrado cabe
    ((0, 0, 50, 50), 640, 480),          # en la esquina: se desplaza hacia dentro
    ((590, 430, 640, 480), 640, 480),    # en la otra esquina
    ((20, 100, 620, 300), 640, 480),     # más ancha que el alto de la imagen
    ((100, 10, 200, 470), 300, 500),     # alta en una imagen vertical
    ((0, 0, 640, 480), 640, 480),        # toda la imagen
    ((2, 2, 36, 40), 38, 42),            # imagen diminuta
])
def test_recorte_cuadrado(caja, ancho, alto):
    x1, y1, x2, y2 = D.recorte_cuadrado(caja, ancho, alto)
    assert all(isinstance(v, int) for v in (x1, y1, x2, y2))
    assert 0 <= x1 < x2 <= ancho and 0 <= y1 < y2 <= alto
    assert x1 <= caja[0] + 1 and y1 <= caja[1] + 1 and x2 >= caja[2] - 1 and y2 >= caja[3] - 1
    lado = max(caja[2] - caja[0], caja[3] - caja[1]) * (1 + 2 * D.MARGEN)
    if lado <= min(ancho, alto):
        assert x2 - x1 == y2 - y1  # cuadrado, con el margen
        assert x2 - x1 >= round(lado) - 1


def test_filtrar_cajas_deja_una_por_personaje():
    cajas = [
        (100, 100, 200, 300),   # personaje A
        (102, 98, 201, 302),    # A otra vez, con otra palabra del texto y menos puntaje
        (300, 100, 400, 300),   # personaje B
        (90, 90, 410, 310),     # grupo: encierra a A y a B
        (500, 100, 510, 300),   # menos de LADO_MINIMO de ancho
        (420, 150, 480, 320),   # personaje C, que se sale un poco de la imagen
        (430, 160, 470, 200),   # D en brazos de C: C contiene a una sola, no es de grupo
    ]
    puntajes = [0.8, 0.5, 0.7, 0.9, 0.9, 0.6, 0.4]
    resultado = D.filtrar_cajas(cajas, puntajes, 640, 300)
    assert [p for _, p in resultado] == [0.8, 0.7, 0.6, 0.4]
    assert resultado[2][0] == (420.0, 150.0, 480.0, 300.0)  # limitada a la imagen
    muchas = [(10 * i, 0, 10 * i + 20, 50) for i in range(0, 60, 3)]
    assert len(D.filtrar_cajas(muchas, [0.5] * len(muchas), 640, 300)) == D.MAXIMO_CAJAS


def test_elegir_la_caja_del_personaje_de_la_carpeta():
    chica = {"caja": (0, 0, 50, 50), "p_carpeta": 0.01}
    grande = {"caja": (0, 0, 300, 300), "p_carpeta": 0.02}
    buena = {"caja": (0, 0, 40, 40), "p_carpeta": 0.60}
    assert D.elegir([]) == (None, "sin_detecciones")
    assert D.elegir([chica, buena, grande]) == (1, "cnn")
    # Nadie lo reconoce: entra la imagen completa, no la caja más grande
    assert D.elegir([chica, grande]) == (None, "nadie_lo_reconoce")
    # Con la CNN de recortes, manda su puntuación
    con_recortes = {**chica, "recortes": {"nombre": "lou", "confianza": 0.7, "p_carpeta": 0.7}}
    assert D.elegir([con_recortes, buena]) == (0, "cnn")


def test_rotulo_con_nombre_y_confianza():
    assert D.rotulo("jasper_beardly", 0.923) == "Jasper Beardly · 92 %"
    assert D.rotulo("kang_kodos", 0.5) == "Kang y Kodos · 50 %"
    assert D.rotulo("krusty_the_clown", 0.996) == "Krusty the Clown · 100 %"


def registro_de_prueba(cajas, p_completa=0.3):
    return {"personaje": "lou", "archivo": "lou__S01E01__t0001000.jpg", "ancho": 640, "alto": 480, "pasada": 1,
            "config": "x", "completa": {"nombre": "chief_wiggum", "confianza": 0.4, "p_carpeta": p_completa},
            "cajas": cajas}


def test_filas_de_la_original_y_de_sus_variantes():
    cajas = [
        {"caja": [300, 100, 420, 400], "puntaje": 0.5, "recorte": list(D.recorte_cuadrado((300, 100, 420, 400), 640, 480)),
         "nombre": "lou", "confianza": 0.7, "p_carpeta": 0.7},
        {"caja": [50, 120, 160, 420], "puntaje": 0.6, "recorte": list(D.recorte_cuadrado((50, 120, 160, 420), 640, 480)),
         "nombre": "chief_wiggum", "confianza": 0.9, "p_carpeta": 0.01},
    ]
    original = D.fila_original(registro_de_prueba(cajas))
    assert (original["fuente"], original["motivo"], original["nombre"]) == ("caja", "cnn", "lou")
    assert (original["caja_x1"], original["recorte_x1"]) == (300, cajas[0]["recorte"][0])

    variante = D.fila_variante(original, "lou__S01E01__t0001000__alterada.jpg", "alterada", -15.0)
    esperada = G.caja_tras_giro((300, 100, 420, 400), 640, 480, -15.0)
    assert variante["tipo"] == "alterada" and variante["fuente"] == "caja"
    assert [variante[f"caja_{e}"] for e in ("x1", "y1", "x2", "y2")] == [round(v, 1) for v in esperada]
    recorte = [variante[f"recorte_{e}"] for e in ("x1", "y1", "x2", "y2")]
    assert recorte == list(D.recorte_cuadrado(esperada, 640, 480))

    # Si la original entra completa, sus variantes también
    completa = D.fila_original(registro_de_prueba([], p_completa=0.9))
    assert (completa["fuente"], completa["motivo"]) == ("completa", "sin_detecciones")
    assert (completa["recorte_x2"], completa["recorte_y2"]) == (640, 480)
    assert D.fila_variante(completa, "otra.jpg", "parecida", 12.0)["fuente"] == "completa"

    # Una caja que el giro deja fuera: la variante entra completa
    esquina = {"caja": [0, 0, 20, 20], "puntaje": 0.5, "recorte": [0, 0, 24, 24], "nombre": "lou", "confianza": 0.5,
               "p_carpeta": 0.5}
    fuera = D.fila_variante(D.fila_original(registro_de_prueba([esquina])), "v.jpg", "alterada", 20.0)
    assert (fuera["fuente"], fuera["motivo"], fuera["recorte_x2"]) == ("completa", "fuera_de_la_variante", 640)


def test_registros_con_variantes(tmp_path):
    caja = {"caja": [300, 100, 420, 400], "puntaje": 0.5, "recorte": list(D.recorte_cuadrado((300, 100, 420, 400), 640, 480)),
            "nombre": "lou", "confianza": 0.7, "p_carpeta": 0.7}
    original = registro_de_prueba([caja])
    carpeta = tmp_path / "lou"
    carpeta.mkdir()
    alterada = original["archivo"].replace(".jpg", G.MARCA_ALTERADA + ".jpg")
    (carpeta / alterada).write_bytes(b"")
    base = {"rotacion": "-15.0", "color": "", "parametro_color": "", "textura": "", "parametro_textura": ""}
    G.escribir_registro(tmp_path / G.REGISTRO, G.CAMPOS_REGISTRO, [
        {"personaje": "lou", "original": original["archivo"], "alterada": alterada, **base},
        # Una variante que se borró del dataset no aparece, y una sin su original en la caché queda fuera
        {"personaje": "lou", "original": original["archivo"], "alterada": "borrada__alterada.jpg", **base},
        {"personaje": "lou", "original": "otra.jpg", "alterada": "otra__alterada.jpg", **base},
    ])
    (carpeta / "otra__alterada.jpg").write_bytes(b"")
    filas, candidatas = D.construir_registros(tmp_path, {("lou", original["archivo"]): original})
    assert [(f["archivo"], f["tipo"], f["fuente"]) for f in filas] == [
        (original["archivo"], "original", "caja"), (alterada, "alterada", "caja")]
    assert [(c["archivo"], c["elegida"]) for c in candidatas] == [(original["archivo"], True)]


def test_caja_segura_para_la_cnn_de_recortes():
    def caja(p):
        return {"caja": [0, 0, 50, 50], "puntaje": 0.5, "recorte": [0, 0, 60, 60], "nombre": "lou", "confianza": p,
                "p_carpeta": p}

    assert D.caja_segura(registro_de_prueba([])) is None
    # Una sola caja: sirve si la CNN ve al personaje en ella o en la imagen completa
    assert D.caja_segura(registro_de_prueba([caja(0.01)], p_completa=0.30)) == 0
    assert D.caja_segura(registro_de_prueba([caja(0.01)], p_completa=0.01)) is None
    # Varias: solo una ganadora clara
    assert D.caja_segura(registro_de_prueba([caja(0.1), caja(0.8)])) == 1
    assert D.caja_segura(registro_de_prueba([caja(0.6), caja(0.3)])) is None  # menos del triple
    assert D.caja_segura(registro_de_prueba([caja(0.4), caja(0.01)])) is None  # bajo UMBRAL_SEGURA
    # Se elige con la CNN final aunque ya haya puntuación de la CNN de recortes: no se refuerza a sí misma
    con_recortes = {**caja(0.1), "recortes": {"nombre": "lou", "confianza": 0.9, "p_carpeta": 0.9}}
    assert D.caja_segura(registro_de_prueba([con_recortes, caja(0.8)])) == 1


def test_cache_ignora_una_linea_cortada(tmp_path):
    ruta = tmp_path / D.CACHE
    primero = registro_de_prueba([])
    segundo = {**primero, "pasada": 2}
    ruta.write_text(json.dumps(primero) + "\n" + json.dumps(segundo) + "\n" + '{"personaje": "lo', encoding="utf-8")
    registros = D.leer_cache(ruta)
    assert list(registros) == [("lou", primero["archivo"])]
    assert registros[("lou", primero["archivo"])]["pasada"] == 2  # la última línea manda
