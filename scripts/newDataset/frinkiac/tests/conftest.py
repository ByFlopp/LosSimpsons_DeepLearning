import csv
import io
import sys
import zlib
from pathlib import Path

import boto3
import numpy as np
import pytest
from moto import mock_aws
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from almacen import Almacen, AlmacenLocal  # noqa: E402
from fuentes import NoEncontrado  # noqa: E402

BUCKET = "frinkiac-pruebas"


def imagen_ruido(semilla: int, lado=(160, 120)) -> bytes:
    """JPEG de ruido: dos semillas distintas dan imágenes nada parecidas entre sí."""
    rng = np.random.default_rng(semilla)
    datos = rng.integers(0, 256, (lado[1], lado[0], 3), dtype=np.uint8)
    salida = io.BytesIO()
    Image.fromarray(datos).save(salida, format="JPEG", quality=90)
    return salida.getvalue()


class FrinkiacFalso:
    """Doble de ClienteFrinkiac. Cada episodio tiene subtítulos de 2 s y fotogramas cada 200 ms."""

    def __init__(self, subtitulos: dict[str, list[dict]], imagen=None):
        self.subs = subtitulos
        self.imagen_de = imagen or (lambda ep, ts: imagen_ruido(zlib.crc32(f"{ep}/{ts}".encode())))
        self.descargas: list[tuple[str, int]] = []

    def buscar(self, texto):
        return [{"Episode": ep, "Timestamp": s["StartTimestamp"] + 100} for ep, lista in self.subs.items()
                for s in lista if texto.split()[0] in s["Content"].lower()]

    def subtitulos(self, episodio, ts):
        return self.subs.get(episodio, [])

    def fotogramas(self, episodio, ts, antes, despues):
        inicio = (ts - antes) // 200 * 200
        return [{"Episode": episodio, "Timestamp": t} for t in range(inicio, ts + despues + 1, 200)]

    def imagen(self, episodio, ts):
        if episodio not in self.subs:
            raise NoEncontrado(episodio)
        self.descargas.append((episodio, ts))
        return self.imagen_de(episodio, ts)


def escribir_guiones(carpeta: Path, lineas: list[tuple[int, str, int, str]]) -> tuple[Path, Path]:
    """lineas: (id, personaje, ts, texto), todas en el episodio S01E01 (id 1 de Kaggle)."""
    episodios = carpeta / "simpsons_episodes.csv"
    with open(episodios, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["id", "title", "number_in_season", "season"])
        escritor.writerow([1, "Piloto", 1, 1])
    guiones = carpeta / "simpsons_script_lines.csv"
    with open(guiones, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["id", "episode_id", "number", "raw_text", "timestamp_in_ms", "speaking_line",
                           "character_id", "location_id", "raw_character_text", "raw_location_text",
                           "spoken_words", "normalized_text", "word_count"])
        for i, personaje, ts, texto in lineas:
            escritor.writerow([i, 1, i, texto, ts, "true", 1, 1, personaje, "Casa", texto, texto.lower(),
                               len(texto.split())])
        escritor.writerow([999, 1, 999, "(Ruido)", 1, "false", 1, 1, "Louie", "Casa", "", "", 0])
    return guiones, episodios


def subtitulos_para(lineas: list[tuple[int, str, int, str]]) -> dict[str, list[dict]]:
    return {"S01E01": [{"StartTimestamp": ts, "EndTimestamp": ts + 2000, "Content": texto}
                       for _, _, ts, texto in lineas]}


@pytest.fixture
def aws(monkeypatch):
    for variable, valor in (("AWS_ACCESS_KEY_ID", "x"), ("AWS_SECRET_ACCESS_KEY", "x"),
                            ("AWS_DEFAULT_REGION", "us-east-1"), ("AWS_REGION", "us-east-1")):
        monkeypatch.setenv(variable, valor)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield s3


@pytest.fixture(params=["s3", "local"])
def almacen(request, tmp_path):
    """Cada prueba que usa el almacén corre dos veces: contra S3 (moto) y contra una carpeta local."""
    if request.param == "s3":
        return Almacen(request.getfixturevalue("aws"), BUCKET)
    return AlmacenLocal(tmp_path / "almacen_local")


@pytest.fixture
def opcion_almacen(almacen):
    """Las opciones de la CLI que apuntan al mismo almacén de la prueba."""
    if isinstance(almacen, AlmacenLocal):
        return ["--local", str(almacen.raiz)]
    return ["--bucket", BUCKET]
