"""Pruebas de detectar_ruido_etiquetas.py.

No necesitan GPU, modelos ni el dataset: la prueba de punta a punta usa un dataset de juguete y un modelo falso.
    .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detectar_cajas as D  # noqa: E402
import detectar_ruido_etiquetas as R  # noqa: E402

CLASES = ["ana", "beto", "carla"]


# --- Escenas -----------------------------------------------------------------------------------------------

def test_leer_fotograma():
    assert R.leer_fotograma("lou__S02E15__t0675633.jpg") == ("lou", "S02E15", 675633)
    assert R.leer_fotograma("homer_simpson_0001.jpg") is None


def test_en_escena_respeta_la_ventana_y_el_nombre_del_archivo():
    pares = [
        ("ana", "ana__S01E01__t0010000.jpg"),
        ("beto", "beto__S01E01__t0012000.jpg"),   # 2 s después: misma escena
        ("carla", "carla__S01E01__t0020000.jpg"),  # 10 s después: otra escena
        ("carla", "carla__S01E02__t0010000.jpg"),  # otro episodio
    ]
    lineas = R.lineas_de_tiempo(pares)
    assert R.en_escena("ana", "ana__S01E01__t0010000.jpg", lineas, set(CLASES)) == {"beto"}
    # Un fotograma movido a la carpeta de su protagonista conserva el nombre de otro personaje del encuadre
    assert R.en_escena("ana", "carla__S01E02__t0010000.jpg", lineas, set(CLASES)) == {"carla"}
    assert R.en_escena("ana", "ana_0001.jpg", lineas, set(CLASES)) == set()


def test_leer_gemelas_en_los_dos_sentidos(tmp_path):
    with open(tmp_path / "_gemelas.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["personaje", "original", "otro_personaje", "otro_original"])
        escritor.writerow(["ana", "a.jpg", "beto", "b.jpg"])
        escritor.writerow(["ana", "a2.jpg", "desconocido", "x.jpg"])
    gemelas = R.leer_gemelas(tmp_path, set(CLASES))
    assert gemelas == {("ana", "a.jpg"): {"beto"}, ("beto", "b.jpg"): {"ana"}}
    assert R.leer_gemelas(tmp_path / "no_existe", set(CLASES)) == {}


# --- Aprendizaje confiado ------------------------------------------------------------------------------------

def test_umbrales_por_clase_usa_solo_las_imagenes_fuera_de_muestra():
    probabilidades = np.array([
        [0.9, 0.1, 0.0],  # ana, train
        [0.5, 0.5, 0.0],  # ana, val
        [0.7, 0.3, 0.0],  # ana, test
        [0.2, 0.8, 0.0],  # beto, train: la única de beto
        [0.0, 0.4, 0.6],  # carla, val
    ])
    etiquetas = np.array([0, 0, 0, 1, 2])
    fuera = np.array([False, True, True, False, True])
    umbrales = R.umbrales_por_clase(probabilidades, etiquetas, 3, fuera)
    assert umbrales == pytest.approx([0.6, 0.8, 0.6])  # beto no tiene fuera de muestra: usa la suya
    assert R.umbrales_por_clase(probabilidades, etiquetas, 3) == pytest.approx([0.7, 0.8, 0.6])


UMBRALES = np.array([0.6, 0.6, 0.6])


@pytest.mark.parametrize("probabilidades, escena, esperado", [
    ([0.70, 0.20, 0.10], set(), None),            # el modelo apoya a la carpeta
    ([0.05, 0.90, 0.05], set(), R.OTRO),          # nombra con seguridad a otro
    ([0.05, 0.90, 0.05], {"beto"}, R.ESCENA),     # ese otro está en la escena
    ([0.40, 0.55, 0.05], set(), None),            # duda entre dos: no alcanza el umbral de beto
    ([0.08, 0.08, 0.08], set(), R.DUDOSA),        # nadie pasa de 0,10 (no suman 1: da igual aquí)
    ([0.15, 0.15, 0.15], set(), None),            # nadie convence, pero no tan poco como para marcarla
])
def test_clasificar(probabilidades, escena, esperado):
    motivo, candidata, puntaje = R.clasificar(np.array(probabilidades), 0, UMBRALES, CLASES, escena)
    assert motivo == esperado
    if esperado in (R.OTRO, R.ESCENA):
        assert candidata == 1 and puntaje == pytest.approx(0.85)
    if esperado is None:
        assert candidata is None and puntaje == 0.0


def test_tasas_de_confusion():
    probabilidades = np.array([
        [0.9, 0.1, 0.0],  # ana como ana
        [0.2, 0.8, 0.0],  # ana como beto
        [0.6, 0.3, 0.1],  # ana como ana
        [0.1, 0.1, 0.8],  # ana como carla
        [0.0, 1.0, 0.0],  # beto como beto
    ])
    tasas = R.tasas_de_confusion(probabilidades, np.array([0, 0, 0, 0, 1]), 3)
    assert tasas[0] == pytest.approx([0.0, 0.25, 0.25])
    assert tasas[1] == pytest.approx([0.0, 0.0, 0.0])
    assert tasas[2] == pytest.approx([0.0, 0.0, 0.0])  # carla no tiene imágenes


def test_clasificar_separa_los_personajes_que_el_modelo_suele_confundir():
    probabilidades = np.array([0.05, 0.90, 0.05])
    confusiones = np.array([[0.0, 0.09, 0.01], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert R.clasificar(probabilidades, 0, UMBRALES, CLASES, set(), confusiones=confusiones)[0] == R.PARECIDO
    assert R.clasificar(probabilidades, 0, UMBRALES, CLASES, set(), confusiones=confusiones,
                        confusion_habitual=0.10)[0] == R.OTRO
    # Si el otro está en la escena, eso pesa más que el parecido
    assert R.clasificar(probabilidades, 0, UMBRALES, CLASES, {"beto"}, confusiones=confusiones)[0] == R.ESCENA


def test_clasificar_exige_la_confianza_minima_aunque_el_umbral_sea_bajo():
    umbrales = np.array([0.6, 0.3, 0.6])
    probabilidades = np.array([0.2, 0.45, 0.35])
    assert R.clasificar(probabilidades, 0, umbrales, CLASES, set())[0] is None
    assert R.clasificar(probabilidades, 0, umbrales, CLASES, set(), confianza_minima=0.4)[0] == R.OTRO


# --- Caché ---------------------------------------------------------------------------------------------------

def test_cache_ida_y_vuelta_y_firma(tmp_path):
    ruta = tmp_path / "probabilidades.npz"
    cache = {"x/a.jpg": np.array([0.1, 0.9], dtype=np.float32), "y/b.jpg": np.array([0.7, 0.3], dtype=np.float32)}
    R.guardar_cache(ruta, "firma-1", cache)
    leida = R.leer_cache(ruta, "firma-1")
    assert set(leida) == set(cache)
    assert np.allclose(leida["x/a.jpg"], cache["x/a.jpg"])
    assert R.leer_cache(ruta, "otra firma") == {}
    assert R.leer_cache(tmp_path / "no_existe.npz", "firma-1") == {}


# --- De punta a punta, con un modelo falso -------------------------------------------------------------------

COLORES = {"ana": (220, 30, 30), "beto": (30, 220, 30), "carla": (30, 30, 220)}


class _Tensor:
    def __init__(self, arreglo):
        self.arreglo = arreglo

    def to(self, _dispositivo):
        return self.arreglo


class _Torch:
    @staticmethod
    def as_tensor(arreglo):
        return _Tensor(arreglo)


class _Red:
    """Reconoce a cada personaje por su color: el canal dominante. Una imagen gris le da lo mismo."""

    def predict(self, lote, verbose=0, batch_size=None):
        medias = np.asarray(lote, dtype=np.float64).mean(axis=(1, 2))
        dominante = medias - medias.mean(axis=1, keepdims=True)
        exponentes = np.exp(dominante / 10)
        return exponentes / exponentes.sum(axis=1, keepdims=True)


class _Modelo:
    def __init__(self, ruta, usar_cpu=False):
        self.tamano = (16, 16)
        self.preparar = lambda lote: np.asarray(lote, dtype=np.uint8)
        self.red, self.torch, self.dispositivo = _Red(), _Torch(), "cpu"

    def descripcion(self):
        return "modelo falso"


def _dataset_de_juguete(tmp_path):
    """Tres personajes de un color cada uno, más una imagen verde (de beto) en la carpeta de ana y una gris en la
    de carla. La partición manda dos de cada personaje a validación y test."""
    dataset = tmp_path / "dataset"
    particion = []
    for personaje, color in COLORES.items():
        carpeta = dataset / personaje
        carpeta.mkdir(parents=True)
        for numero in range(6):
            nombre = f"{personaje}_{numero:04d}.jpg"
            Image.new("RGB", (40, 30), color).save(carpeta / nombre)
            particion.append((personaje, nombre, "test" if numero < 2 else "train"))
    Image.new("RGB", (40, 30), COLORES["beto"]).save(dataset / "ana" / "ana_ruido.jpg")
    particion.append(("ana", "ana_ruido.jpg", "train"))
    Image.new("RGB", (40, 30), (128, 128, 128)).save(dataset / "carla" / "carla_gris.jpg")
    particion.append(("carla", "carla_gris.jpg", "train"))

    modelos = tmp_path / "modelos"
    modelos.mkdir()
    ruta_modelo = modelos / "falso.keras"
    ruta_modelo.write_bytes(b"no es un modelo")
    ruta_modelo.with_suffix(".json").write_text(
        json.dumps({"clases": CLASES, "alto": 16, "ancho": 16, "representacion": "rgb"}), encoding="utf-8")
    with open(modelos / "falso_particion.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["personaje", "archivo", "conjunto"])
        escritor.writerows(particion)
    return dataset, ruta_modelo


def test_main_marca_el_ruido_del_dataset_de_juguete(tmp_path, monkeypatch):
    dataset, ruta_modelo = _dataset_de_juguete(tmp_path)
    monkeypatch.setattr(D, "resolver_modelo", lambda texto: ruta_modelo)
    monkeypatch.setattr(D, "Modelo", _Modelo)
    salida = tmp_path / "salida"

    argumentos = ["--dataset", str(dataset), "--salida", str(salida), "--umbral-dudosa", "0.4"]
    assert R.main(argumentos) == 0

    with open(salida / "ruido.csv", newline="", encoding="utf-8") as archivo:
        marcadas = {fila["archivo"]: fila for fila in csv.DictReader(archivo)}
    assert set(marcadas) == {"ana_ruido.jpg", "carla_gris.jpg"}
    assert marcadas["ana_ruido.jpg"]["motivo"] == R.OTRO
    assert marcadas["ana_ruido.jpg"]["candidata"] == "beto"
    assert marcadas["ana_ruido.jpg"]["rotulo"].startswith("Beto · ")
    assert marcadas["carla_gris.jpg"]["motivo"] == R.DUDOSA

    resumen = json.loads((salida / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["imagenes_revisadas"] == 20 and resumen["marcadas"] == 2
    assert (salida / "hojas" / "ana.jpg").is_file() and not (salida / "hojas" / "beto.jpg").exists()

    # Segunda ejecución: todo sale de la caché y da lo mismo, también filtrando por personaje
    monkeypatch.setattr(_Red, "predict", lambda *a, **k: pytest.fail("no debía volver a predecir"))
    assert R.main(argumentos + ["--personajes", "ana", "--sin-hojas"]) == 0
    with open(salida / "ruido.csv", newline="", encoding="utf-8") as archivo:
        assert [fila["archivo"] for fila in csv.DictReader(archivo)] == ["ana_ruido.jpg"]
