"""Pruebas de detectar_cajas.py y del giro de generar_imagenes_alteradas.py.

No necesitan GPU, modelos ni el dataset:
    .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detectar_cajas as D  # noqa: E402
import generar_imagenes_alteradas as G  # noqa: E402


def degradado(ancho, alto):
    x = np.linspace(0, 255, ancho, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, alto, dtype=np.float32)[:, None, None]
    canales = [np.broadcast_to(x, (alto, ancho, 1)), np.broadcast_to(y, (alto, ancho, 1)),
               np.broadcast_to((x + y) / 2, (alto, ancho, 1))]
    return Image.fromarray(np.concatenate(canales, axis=2).astype(np.uint8), "RGB")


# Salida de girar(): las variantes del dataset tienen que seguir siendo las mismas, byte a byte
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


# --- Geometría -------------------------------------------------------------------------------------------

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


def test_rotulo_con_nombre_y_confianza():
    assert D.rotulo("jasper_beardly", 0.923) == "Jasper Beardly · 92 %"
    assert D.rotulo("kang_kodos", 0.5) == "Kang y Kodos · 50 %"
    assert D.rotulo("krusty_the_clown", 0.996) == "Krusty the Clown · 100 %"


# --- El modelo y su entrada ------------------------------------------------------------------------------

def modelo_falso(carpeta, nombre="mlp_final", representacion="color", con_entrada=True, **extra):
    """Un .keras vacío (no se carga) con su .json y, si se pide, su _entrada.py."""
    ruta = carpeta / f"{nombre}.keras"
    ruta.write_bytes(b"")
    metadatos = {"configuracion": "Base", "representacion": representacion, "alto": 8, "ancho": 6,
                 "clases": ["bart_simpson", "homer_simpson", "lisa_simpson"], **extra}
    ruta.with_suffix(".json").write_text(json.dumps(metadatos), encoding="utf-8")
    if con_entrada:
        (carpeta / f"{nombre}{D.SUFIJO_ENTRADA}").write_text(
            "import numpy as np\n"
            "def preparar(lote):\n"
            "    return np.asarray(lote, dtype=np.float32).mean(axis=(1, 2))\n",
            encoding="utf-8",
        )
    return ruta, metadatos


def test_resolver_modelo(tmp_path):
    ruta, _ = modelo_falso(tmp_path)
    assert D.resolver_modelo(str(ruta)) == ruta
    with pytest.raises(SystemExit):
        D.resolver_modelo(str(tmp_path / "no_existe.keras"))
    with pytest.raises(SystemExit):
        D.resolver_modelo(str(ruta.with_suffix(".json")))
    ruta.with_suffix(".json").unlink()
    with pytest.raises(SystemExit):  # sin su .json no sirve
        D.resolver_modelo(str(ruta))


def test_cargar_preparacion_usa_el_archivo_del_notebook(tmp_path):
    ruta, metadatos = modelo_falso(tmp_path)
    preparar = D.cargar_preparacion(ruta, metadatos)
    lote = np.full((2, 8, 6, 3), 10, dtype=np.uint8)
    assert preparar(lote).shape == (2, 3)


def test_cargar_preparacion_sin_archivo(tmp_path):
    # La CNN recibe la imagen RGB tal cual: no necesita _entrada.py
    ruta, metadatos = modelo_falso(tmp_path, "cnn_final", representacion="rgb", con_entrada=False)
    lote = np.zeros((1, 8, 6, 3), dtype=np.uint8)
    assert D.cargar_preparacion(ruta, metadatos)(lote) is not None
    # El MLP con descriptores sí: sin él no se puede repetir su entrada
    ruta, metadatos = modelo_falso(tmp_path, "mlp_final", representacion="color", con_entrada=False)
    with pytest.raises(SystemExit):
        D.cargar_preparacion(ruta, metadatos)


def test_redimensionar_como_el_notebook():
    lote = D.redimensionar([degradado(64, 48), degradado(20, 30).convert("L")], (6, 8))
    assert lote.shape == (2, 8, 6, 3) and lote.dtype == np.uint8
    esperado = np.asarray(degradado(64, 48).resize((6, 8), Image.Resampling.BILINEAR), dtype=np.uint8)
    assert np.array_equal(lote[0], esperado)


def test_nombrar_cada_caja_con_su_personaje():
    clases = ["bart_simpson", "homer_simpson", "lisa_simpson"]
    imagen = degradado(640, 480)
    cajas = [((100, 50, 200, 400), 0.8), ((400, 100, 500, 300), 0.6)]
    vistos = []

    def clasificar(recortes):
        vistos.extend(r.size for r in recortes)
        return np.array([[0.1, 0.7, 0.2], [0.05, 0.15, 0.8], [0.92, 0.05, 0.03]], dtype=np.float32)

    filas = D.nombrar(imagen, cajas, clasificar, clases)
    assert vistos[0] == (640, 480)  # primero la imagen completa
    assert [f["caja"] for f in filas] == [0, 1, 2]
    assert [f["fuente"] for f in filas] == ["imagen_completa", "detector", "detector"]
    assert filas[0]["rotulo"] == "Homer Simpson · 70 %" and filas[0]["puntaje"] == ""
    assert filas[1]["rotulo"] == "Lisa Simpson · 80 %" and filas[1]["puntaje"] == 0.8
    assert filas[2]["personaje"] == "bart_simpson" and filas[2]["confianza"] == 0.92
    assert filas[1]["candidatas"] == "Homer Simpson · 15 %, Bart Simpson · 5 %"
    assert (filas[1]["x1"], filas[1]["y2"]) == (100.0, 400.0)
    assert [filas[1][f"recorte_{e}"] for e in ("x1", "y1", "x2", "y2")] == list(D.recorte_cuadrado(cajas[0][0], 640, 480))
    assert set(filas[0]) | {"imagen", "carpeta", "ancho", "alto"} == set(D.CAMPOS)


def test_nombrar_sin_cajas_deja_la_imagen_completa():
    filas = D.nombrar(degradado(64, 48), [], lambda r: np.array([[0.2, 0.3, 0.5]]), ["a", "b", "c"])
    assert len(filas) == 1 and filas[0]["fuente"] == "imagen_completa"


def test_dibujar_marca_la_carpeta():
    imagen = degradado(1280, 720)
    filas = [
        {"fuente": "imagen_completa", "personaje": "homer_simpson", "rotulo": "Homer Simpson · 70 %",
         "x1": 0, "y1": 0, "x2": 1280, "y2": 720},
        {"fuente": "detector", "personaje": "lisa_simpson", "rotulo": "Lisa Simpson · 80 %",
         "x1": 100, "y1": 100, "x2": 400, "y2": 600},
    ]
    dibujada = D.dibujar(imagen, filas, "lisa_simpson")
    assert max(dibujada.size) == D.LADO_MAXIMO_IMAGEN
    pixel = dibujada.getpixel((round(100 * D.LADO_MAXIMO_IMAGEN / 1280), 300))
    assert pixel == D.VERDE
    sin_cajas = D.dibujar(imagen, filas[:1], "lisa_simpson")
    assert sin_cajas.getpixel((sin_cajas.size[0] // 2, sin_cajas.size[1] - 3)) == D.NARANJO


# --- Qué imágenes ----------------------------------------------------------------------------------------

def dataset_falso(raiz):
    for personaje, archivos in {"bart_simpson": ["b1.jpg", "b2.jpg", "b3.jpg", "b1__alterada.jpg"],
                                "lisa_simpson": ["l1.jpg", "l2.png", "l2__parecida.png", "notas.txt"]}.items():
        carpeta = raiz / personaje
        carpeta.mkdir(parents=True)
        for archivo in archivos:
            (carpeta / archivo).write_bytes(b"")
    return raiz


def test_reunir_imagenes_salta_las_variantes(tmp_path):
    dataset = dataset_falso(tmp_path / "ds")
    rutas = D.reunir_imagenes([dataset, dataset / "bart_simpson" / "b1.jpg"])
    assert [r.name for r in rutas] == ["b1.jpg", "b2.jpg", "b3.jpg", "l1.jpg", "l2.png"]
    with pytest.raises(SystemExit):
        D.reunir_imagenes([dataset / "lisa_simpson" / "notas.txt"])


def test_imagenes_del_dataset_con_la_particion(tmp_path):
    dataset = dataset_falso(tmp_path / "ds")
    ruta = tmp_path / "mlp_final_particion.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["personaje", "archivo", "conjunto"])
        escritor.writerows([("bart_simpson", "b1.jpg", "train"), ("bart_simpson", "b2.jpg", "test"),
                            ("bart_simpson", "b3.jpg", "test"), ("lisa_simpson", "l1.jpg", "val"),
                            ("lisa_simpson", "l2.png", "test")])
    particion = D.leer_particion(ruta)
    nombres = [r.name for r in D.imagenes_del_dataset(dataset, particion, "test", None, 42)]
    assert nombres == ["b2.jpg", "b3.jpg", "l2.png"]
    assert len(D.imagenes_del_dataset(dataset, particion, "test", 1, 42)) == 2
    assert len(D.imagenes_del_dataset(dataset, particion, "todos", None, 42)) == 5
    assert len(D.imagenes_del_dataset(dataset, None, "test", None, 42)) == 5  # sin partición, todas
    solo_lisa = D.imagenes_del_dataset(dataset, particion, "test", None, 42, ["lisa_simpson"])
    assert [r.name for r in solo_lisa] == ["l2.png"]
    with pytest.raises(SystemExit):
        D.imagenes_del_dataset(dataset, particion, "test", None, 42, ["moe_szyslak"])


def test_carpeta_y_destino(tmp_path):
    clases = ["bart_simpson"]
    assert D.carpeta_de(Path("ds/bart_simpson/b1.jpg"), clases) == "bart_simpson"
    assert D.carpeta_de(Path("descargas/foto.png"), clases) == ""
    usados = set()
    primero = D.destino_imagen(tmp_path, Path("a/x/foto.png"), usados)
    segundo = D.destino_imagen(tmp_path, Path("b/x/foto.jpg"), usados)
    assert primero == tmp_path / "x" / "foto.jpg" and segundo == tmp_path / "x" / "foto_2.jpg"


def test_argumentos():
    with pytest.raises(SystemExit):
        D.leer_argumentos(["--modelo", "cnn"])  # ni imágenes ni --dataset
    argumentos = D.leer_argumentos(["--modelo", "mlp", "--dataset", "--muestra", "3"])
    assert argumentos.dataset is not None and argumentos.muestra == 3 and argumentos.conjunto == "test"
