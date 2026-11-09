"""Pruebas de scripts/newDataset/mlp_robusto.py.

Usan un MLP de píxeles mínimo con la misma estructura que R1 y un dataset de juguete escrito en tmp_path: no
necesitan GPU, modelos guardados ni el dataset real. Corren en .venv-gpu:
    .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_mlp_robusto.py -q
"""

import csv
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")

import mlp_robusto as M  # noqa: E402

ALTO = ANCHO = 8
CLASES = 3


def mlp_minimo():
    keras.utils.set_random_seed(5)
    return M.construir_mlp_pixeles(ALTO, ANCHO, CLASES, neuronas=(16, 8), nombre="mlp_minimo")


def lote(n=6, semilla=0):
    return torch.as_tensor(np.random.default_rng(semilla).integers(0, 256, (n, ALTO * ANCHO * 3), dtype=np.uint8))


def test_la_red_es_la_de_r1():
    modelo = mlp_minimo()
    assert [c.name for c in modelo.layers] == ["escalado", "oculta_1", "norm_1", "dropout_1", "oculta_2", "norm_2",
                                                "dropout_2", "salida"]
    assert modelo.input_shape == (None, ALTO * ANCHO * 3) and modelo.output_shape == (None, CLASES)


def test_aumentar_conserva_forma_y_rango():
    generador = torch.Generator().manual_seed(1)
    x = M.aumentar(lote(), ALTO, ANCHO, desplazamiento=2, ruido=0.05, generador=generador)
    assert x.shape == (6, ALTO * ANCHO * 3) and x.dtype == torch.float32
    assert float(x.min()) >= 0 and float(x.max()) <= 255


def test_sin_desplazamiento_ni_ruido_solo_voltea():
    original = lote()
    x = M.aumentar(original, ALTO, ANCHO, desplazamiento=0, ruido=0.0, generador=torch.Generator().manual_seed(2))
    imagenes = original.float().reshape(6, ALTO, ANCHO, 3)
    for i, fila in enumerate(x.reshape(6, ALTO, ANCHO, 3)):
        assert torch.equal(fila, imagenes[i]) or torch.equal(fila, imagenes[i].flip(1))


def test_el_desplazamiento_es_un_corrimiento_entero_sin_envolver_los_bordes():
    """Con desplazamiento 1, cada imagen es la original corrida a lo sumo un píxel, con el borde repetido."""
    original = lote(n=20, semilla=3)
    x = M.aumentar(original, ALTO, ANCHO, desplazamiento=1, generador=torch.Generator().manual_seed(4))
    imagenes = original.float().reshape(20, ALTO, ANCHO, 3)
    for i, fila in enumerate(x.reshape(20, ALTO, ANCHO, 3)):
        candidatas = []
        for base in (imagenes[i], imagenes[i].flip(1)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    filas = (torch.arange(ALTO) + dy).clamp(0, ALTO - 1)
                    columnas = (torch.arange(ANCHO) + dx).clamp(0, ANCHO - 1)
                    candidatas.append(base[filas][:, columnas])
        assert any(torch.allclose(fila, c, atol=1e-3) for c in candidatas)


def test_pgd_respeta_el_radio_y_sube_la_perdida():
    modelo = mlp_minimo()
    x = lote(n=8).float()
    objetivo = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    eps = 0.5
    atacada = M.pgd_l2(modelo, x, objetivo, eps=eps, pasos=5)
    normas = (atacada - x).norm(dim=1)
    assert float(normas.max()) <= eps * 255.0 * (1 + 1e-4)
    assert float(atacada.min()) >= 0 and float(atacada.max()) <= 255

    def perdida(entrada):
        with torch.no_grad():
            return float(torch.nn.functional.nll_loss(torch.log(modelo(entrada, training=False).clamp_min(1e-7)),
                                                      objetivo))

    assert perdida(atacada) > perdida(x)
    # Con eps 0 no cambia nada y los pesos no acumulan gradiente
    assert torch.equal(M.pgd_l2(modelo, x, objetivo, eps=0.0), x)
    assert all(p.grad is None for p in modelo.parameters())


def test_entrenar_robusto_corre_y_restaura_la_mejor_epoca(tmp_path):
    modelo = mlp_minimo()
    X = lote(n=48, semilla=6)
    y = torch.as_tensor(np.arange(48) % CLASES)
    ruta = tmp_path / "robusto.keras"
    historial = M.entrenar_robusto(modelo, X, y, X[:12], y[:12], ALTO, ANCHO, eps=0.2, pasos_pgd=2,
                                   desplazamiento=1, epocas=3, batch=16, pesos_por_clase={0: 1.0, 1: 2.0, 2: 0.5},
                                   ruta_modelo=ruta, verbose=0)
    assert len(historial["loss"]) == 3 and 1 <= historial["mejor_epoca"] <= 3
    assert ruta.is_file()
    guardado = keras.models.load_model(ruta, compile=False)
    with torch.no_grad():
        np.testing.assert_allclose(np.asarray(guardado(X[:4].float(), training=False)),
                                   np.asarray(modelo(X[:4].float(), training=False)), atol=1e-5)
    # Acepta también etiquetas one-hot, como las del notebook
    M.entrenar_robusto(mlp_minimo(), X, torch.nn.functional.one_hot(y, CLASES).float(), X[:12],
                       torch.nn.functional.one_hot(y[:12], CLASES).float(), ALTO, ANCHO, eps=0.0, epocas=1, batch=16,
                       verbose=0)


def test_pesos_raiz_como_el_notebook():
    y = np.array([0] * 8 + [1] * 2)
    pesos = M.pesos_raiz(y, 2)
    balanceados = len(y) / (2 * np.bincount(y))
    esperados = np.sqrt(balanceados) * len(y) / (np.sqrt(balanceados) * np.bincount(y)).sum()
    np.testing.assert_allclose(pesos, esperados)
    # El peso medio por imagen de entrenamiento es 1
    assert (pesos[y]).mean() == pytest.approx(1.0)


def test_el_indice_sigue_la_particion_del_notebook(tmp_path):
    """Originales con su conjunto, parecidas en lugar de su original, alteradas en el conjunto de su original, y
    fuera lo que no está en la partición."""
    from PIL import Image

    dataset = tmp_path / "simpsons_dataset"
    for personaje, archivos in {
        "bart_simpson": ["b1.jpg", "b1__alterada.jpg", "b2.jpg", "b2__parecida.jpg", "b3.jpg"],
        "homer_simpson": ["h1.png", "h1__alterada.png"],
    }.items():
        (dataset / personaje).mkdir(parents=True)
        for archivo in archivos:
            Image.new("RGB", (10, 6), (10, 20, 30)).save(dataset / personaje / archivo)
    (dataset / "_gemelas.csv").write_text("x", encoding="utf-8")
    particion = tmp_path / "particion.csv"
    with open(particion, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["personaje", "archivo", "conjunto"])
        escritor.writerows([["bart_simpson", "b1.jpg", "train"], ["bart_simpson", "b2.jpg", "val"],
                            ["homer_simpson", "h1.png", "test"]])
    filas = [(ruta.name, personaje, conjunto) for ruta, personaje, conjunto in M.indice_dataset(dataset, particion)]
    assert filas == [("b1.jpg", "bart_simpson", "train"), ("b1__alterada.jpg", "bart_simpson", "train"),
                     ("b2__parecida.jpg", "bart_simpson", "val"), ("h1.png", "homer_simpson", "test"),
                     ("h1__alterada.png", "homer_simpson", "test")]
    datos = M.cargar_pixeles(dataset, particion, ["bart_simpson", "homer_simpson"], ALTO, ANCHO)
    assert datos["train"][0].shape == (2, ALTO * ANCHO * 3) and datos["train"][0].dtype == np.uint8
    assert list(datos["test"][1]) == [1, 1] and list(datos["val"][1]) == [0]
