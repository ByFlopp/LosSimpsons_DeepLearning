"""Pruebas de scripts/newDataset/pipeline_comun, el código común de los notebooks del MLP y de la CNN.

Usan modelos diminutos y datos de juguete en la CPU: no necesitan GPU, modelos guardados ni el dataset real. Corren
en .venv-gpu:
    .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_pipeline_comun.py -q
"""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")

from pipeline_comun import auditoria, datos, entrenamiento, evaluacion, experimentacion  # noqa: E402


def mlp(l2=0.0, entradas=8, clases=3):
    keras.utils.set_random_seed(0)
    regularizador = keras.regularizers.L2(l2) if l2 else None
    return keras.Sequential([
        keras.layers.Input((entradas,)),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizador),
        keras.layers.Dense(clases, activation="softmax", kernel_regularizer=regularizador),
    ])


def datos_de_juguete(n=256, entradas=8, clases=3, semilla=0):
    generador = torch.Generator().manual_seed(semilla)
    X = torch.randn(n, entradas, generator=generador)
    y = torch.nn.functional.one_hot(X[:, :clases].argmax(1), clases).float()
    return X, y


# --- Fuente de cada imagen ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("archivo", "fuente"),
    [
        ("homer_simpson__S05E12__t123456.jpg", datos.FUENTE_EPISODIO),
        ("homer_simpson_0001.jpg", datos.FUENTE_KAGGLE),
        ("bart_simpson__s10e03__t5.png", datos.FUENTE_EPISODIO),
        ("Captura de pantalla 2026-10-14 183529.png", datos.FUENTE_KAGGLE),
    ],
)
def test_fuente_de_la_imagen_sale_del_nombre(archivo, fuente):
    assert datos.fuente_imagen(archivo) == fuente


def test_leer_fotograma():
    assert datos.leer_fotograma("lisa_simpson__S05E12__t1500.jpg") == ("lisa_simpson", "S05E12", 1500)
    assert datos.leer_fotograma("lisa_simpson_12.jpg") is None


# --- Entrenamiento -----------------------------------------------------------------------------------------


def test_la_penalizacion_l2_es_lambda_por_la_suma_de_cuadrados_y_tiene_gradiente():
    modelo = mlp(l2=1e-2)
    penalizacion = entrenamiento.penalizacion(modelo)
    esperada = sum(1e-2 * float((capa.kernel.value.detach() ** 2).sum()) for capa in modelo.layers)
    assert penalizacion.requires_grad
    assert float(penalizacion.detach()) == pytest.approx(esperada, rel=1e-5)
    assert entrenamiento.penalizacion(mlp(l2=0.0)) is None


def test_entrenar_con_l2_achica_los_pesos():
    X, y = datos_de_juguete()
    normas = {}
    for l2 in (0.0, 1.0):
        modelo = mlp(l2=l2)
        entrenamiento.entrenar(modelo, X, y, X[:64], y[:64], epocas=30, batch=32, paciencia=100, verbose=0)
        normas[l2] = float(modelo.layers[0].kernel.value.norm())
    assert normas[1.0] < 0.6 * normas[0.0]


def test_motivo_de_parada_tope():
    X, y = datos_de_juguete()
    historial = entrenamiento.entrenar(mlp(), X, y, X[:64], y[:64], epocas=3, batch=32, paciencia=100, verbose=0)
    assert historial.motivo_parada == entrenamiento.PARADA_TOPE
    assert len(historial.history["loss"]) == 3
    assert historial.epocas_maximas == 3
    assert historial.mejor_epoca == int(np.argmin(historial.history["val_loss"])) + 1


def test_motivo_de_parada_paciencia():
    X, y = datos_de_juguete()
    # La validación tiene las etiquetas cambiadas: cuanto mejor aprende train, peor le va a validación
    y_valida = torch.roll(y[:64], shifts=1, dims=1)
    historial = entrenamiento.entrenar(mlp(), X, y, X[:64], y_valida, epocas=50, batch=32, paciencia=3, verbose=0)
    assert historial.motivo_parada == entrenamiento.PARADA_PACIENCIA
    assert len(historial.history["loss"]) == historial.mejor_epoca + 3


def test_aumento_conserva_forma_y_rango_y_cambia_la_imagen():
    generador = torch.Generator().manual_seed(0)
    lote = torch.randint(0, 256, (4, 16, 16, 3), dtype=torch.uint8, generator=generador)
    aumentado = entrenamiento.aumentar_imagenes(lote, generador)
    assert aumentado.shape == lote.shape
    assert aumentado.dtype == torch.float32
    assert float(aumentado.min()) >= 0 and float(aumentado.max()) <= 255
    assert not torch.allclose(aumentado, lote.float())


def test_aumento_sin_transformaciones_es_la_identidad():
    generador = torch.Generator().manual_seed(0)
    lote = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8, generator=generador)
    # Sin giro, sin zoom y sin contraste solo queda el volteo: cada imagen es la original o su espejo
    aumentado = entrenamiento.aumentar_imagenes(lote, generador, giro_maximo=0, zoom_maximo=0, contraste_maximo=0)
    for original, resultado in zip(lote.float(), aumentado):
        assert torch.allclose(resultado, original, atol=1e-3) or torch.allclose(resultado, original.flip(1), atol=1e-3)


def test_el_historial_se_guarda_y_se_lee_igual(tmp_path):
    X, y = datos_de_juguete()
    historial = entrenamiento.entrenar(mlp(), X, y, X[:64], y[:64], epocas=2, batch=32, verbose=0)
    entrenamiento.guardar_historial(tmp_path, "g1", historial, experimento="G1 - sin dropout",
                                    resultado={"val_f1_macro": np.float32(0.5), "batchnorm": np.bool_(True)})
    leido = entrenamiento.cargar_historiales(tmp_path)["g1"]
    assert leido["history"] == historial.history
    assert leido["mejor_epoca"] == historial.mejor_epoca
    assert leido["motivo_parada"] == historial.motivo_parada
    assert leido["experimento"] == "G1 - sin dropout"
    assert leido["resultado"]["batchnorm"] is True


def test_efecto_parada_temprana_compara_la_mejor_con_la_ultima():
    historial = {"history": {"loss": [1.0, 0.8, 0.6], "accuracy": [0.4, 0.6, 0.8],
                             "val_loss": [1.0, 0.7, 0.9], "val_accuracy": [0.4, 0.55, 0.5]}}
    tabla = entrenamiento.efecto_parada_temprana(historial)
    assert list(tabla["epoca"]) == [2, 3]
    assert list(tabla["brecha_train_val"]) == pytest.approx([0.05, 0.3])


# --- Métricas y auditoría ----------------------------------------------------------------------------------


def test_metricas_perfectas_y_al_azar():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfectas = np.eye(3)[y] * 0.9 + 0.1 / 3
    metricas = evaluacion.evaluar_metricas(y, perfectas)
    assert metricas["accuracy"] == 1.0
    assert metricas["f1_macro"] == 1.0
    assert metricas["roc_auc_macro"] == pytest.approx(1.0)
    constantes = np.full((6, 3), 1 / 3)
    assert evaluacion.evaluar_metricas(y, constantes)["roc_auc_macro"] == pytest.approx(0.5)


def test_roc_auc_coincide_con_sklearn():
    from sklearn.metrics import roc_auc_score

    generador = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 25)
    probabilidades = generador.dirichlet(np.ones(4), size=100)
    esperado = roc_auc_score(y, probabilidades, multi_class="ovr", average="macro")
    assert evaluacion.evaluar_metricas(y, probabilidades)["roc_auc_macro"] == pytest.approx(esperado)


def test_ece_cero_si_la_confianza_es_la_accuracy():
    # 10 imágenes con confianza 0,8 de las que acierta 8: calibración perfecta
    y = np.array([0] * 8 + [1] * 2)
    probabilidades = np.tile([0.8, 0.2], (10, 1))
    _, ece = auditoria.calibracion(y, probabilidades, graficar=False)
    assert ece == pytest.approx(0.0)


def test_ece_de_un_modelo_sobreconfiado():
    # Confianza 0,9 y acierta la mitad: el ECE es 0,4
    y = np.array([0] * 5 + [1] * 5)
    probabilidades = np.tile([0.9, 0.1], (10, 1))
    _, ece = auditoria.calibracion(y, probabilidades, graficar=False)
    assert ece == pytest.approx(0.4)


def test_umbral_de_revision():
    confianza = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    acierto = np.array([True, True, True, True, False, False])
    umbral = auditoria.elegir_umbral(confianza, acierto, objetivo=0.9)
    assert umbral == pytest.approx(0.8)
    aplicado = auditoria.aplicar_umbral(confianza, acierto, umbral)
    assert aplicado["cobertura"] == pytest.approx(4 / 6)
    assert aplicado["accuracy_aceptadas"] == 1.0
    assert aplicado["accuracy_a_revision"] == 0.0


def test_sesgos_por_subgrupo():
    y = np.array([0, 0, 1, 1, 2, 2])
    predicciones = np.array([0, 0, 1, 0, 2, 1])
    grupos = {"fuente": np.array(["Kaggle", "Kaggle", "Kaggle", "episodio", "episodio", "episodio"])}
    tabla, brechas = auditoria.sesgos_por_subgrupo(y, predicciones, grupos)
    kaggle = tabla[tabla["grupo"] == "Kaggle"].iloc[0]
    episodio = tabla[tabla["grupo"] == "episodio"].iloc[0]
    assert kaggle["accuracy"] == 1.0
    assert episodio["accuracy"] == pytest.approx(1 / 3, abs=1e-4)  # la tabla redondea a cuatro decimales
    assert brechas.iloc[0]["brecha_accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_grupos_de_auditoria():
    clases = ["bart_simpson", "moe_szyslak", "ned_flanders"]
    tabla = pd.DataFrame({
        "etiqueta": [0, 1, 2],
        "fuente": [datos.FUENTE_KAGGLE, datos.FUENTE_EPISODIO, datos.FUENTE_KAGGLE],
        "alterada": [False, True, False],
        "parecida": [False, False, True],
    })
    grupos = auditoria.grupos_de_auditoria(tabla, clases, conteos_train=np.array([10, 100, 1000]))
    assert list(grupos["familia Simpson"]) == ["familia Simpson", "resto de personajes", "resto de personajes"]
    assert list(grupos["tipo de imagen"]) == ["original", "alterada", "parecida"]
    assert grupos["tamaño de la clase"][0].startswith("pequeñas")
    assert grupos["tamaño de la clase"][2].startswith("grandes")


# --- Búsqueda de hiperparámetros ---------------------------------------------------------------------------


ESPACIO = {"lr": [1e-3, 5e-4], "dropout": [0.2, 0.3, 0.5], "l2": [0.0, 1e-4]}


def test_grid_recorre_todas_las_combinaciones():
    configuraciones = experimentacion.generar_configuraciones(ESPACIO, "grid")
    assert len(configuraciones) == 12
    assert len({tuple(c.items()) for c in configuraciones}) == 12


def test_random_search_es_reproducible_y_sin_repetir():
    primera = experimentacion.generar_configuraciones(ESPACIO, "aleatoria", n_configuraciones=5, semilla=42)
    segunda = experimentacion.generar_configuraciones(ESPACIO, "aleatoria", n_configuraciones=5, semilla=42)
    otra = experimentacion.generar_configuraciones(ESPACIO, "aleatoria", n_configuraciones=5, semilla=7)
    assert primera == segunda
    assert primera != otra
    assert len({tuple(c.items()) for c in primera}) == 5


def test_optimizar_modelo_entrena_cada_configuracion_y_ordena_por_f1():
    llamadas = []

    def entrenar_configuracion(nombre, lr, dropout, fijo):
        llamadas.append((nombre, lr, dropout, fijo))
        historial = entrenamiento.Historial()
        historial.history = {"loss": [1.0], "val_loss": [1.0]}
        resultado = {"experimento": nombre, "val_f1_macro": lr * 100 + dropout, "val_roc_auc": 0.9, "epocas": 1,
                     "parada": "tope", "segundos": 0.0}
        return object(), historial, resultado

    tabla, modelos, historiales = experimentacion.optimizar_modelo(
        entrenar_configuracion, {"lr": [1e-3, 5e-4], "dropout": [0.2, 0.5]}, estrategia="grid", prefijo="B",
        fijos={"fijo": "sí"},
    )
    assert len(llamadas) == 4
    assert all(fijo == "sí" for *_, fijo in llamadas)
    assert list(tabla["val_f1_macro"]) == sorted(tabla["val_f1_macro"], reverse=True)
    assert tabla.iloc[0]["factores"] == {"lr": 1e-3, "dropout": 0.5}
    assert set(modelos) == set(historiales) == set(tabla["experimento"])


def test_combinar_mejoras():
    experimentos = [{"nombre": "E4", "lr": 1e-4}, {"nombre": "E6", "dropout": 0.5}, {"nombre": "E3", "act": "tanh"}]
    resultados = {"E4": {"val_f1_macro": 0.6}, "E6": {"val_f1_macro": 0.7}, "E3": {"val_f1_macro": 0.4}}
    factores, componentes = experimentacion.combinar_mejoras(experimentos, resultados, f1_macro_base=0.5)
    assert factores == {"lr": 1e-4, "dropout": 0.5}
    assert componentes == ["E4", "E6"]
    factores, _ = experimentacion.combinar_mejoras(experimentos, resultados, f1_macro_base=0.65)
    assert factores is None


def test_tabla_impacto_lee_el_signo_segun_la_accion():
    historia = {"loss": [1, 0.5], "accuracy": [0.5, 0.9], "val_loss": [1, 0.8], "val_accuracy": [0.5, 0.7]}
    historia_g1 = {"loss": [1, 0.4], "accuracy": [0.5, 0.95], "val_loss": [1, 0.9], "val_accuracy": [0.5, 0.65]}
    tabla = pd.DataFrame([
        {"experimento": "Base", "val_f1_macro": 0.70, "mejor_epoca": 2, "parada": "paciencia"},
        {"experimento": "G1", "val_f1_macro": 0.66, "mejor_epoca": 2, "parada": "paciencia"},
        {"experimento": "G3", "val_f1_macro": 0.71, "mejor_epoca": 2, "parada": "paciencia"},
    ])
    tecnicas = [
        {"tecnica": "Dropout", "experimento": "G1", "accion": "quita", "en_final": True},
        {"tecnica": "L2", "experimento": "G3", "accion": "agrega", "en_final": False},
    ]
    impacto = experimentacion.tabla_impacto(tabla, {"Base": historia, "G1": historia_g1, "G3": historia}, "Base",
                                            tecnicas)
    dropout = impacto[impacto["técnica"] == "Dropout"].iloc[0]
    l2 = impacto[impacto["técnica"] == "L2"].iloc[0]
    assert dropout["efecto de la técnica"] == pytest.approx(0.04)
    assert "ayuda" in dropout["lectura"]
    assert l2["efecto de la técnica"] == pytest.approx(0.01)
    assert dropout["en el modelo final"] == "sí"
