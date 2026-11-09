"""Pruebas de scripts/newDataset/recrear_personajes.py.

Usan una CNN y un MLP mínimos armados aquí, con la misma estructura de capas que los de los notebooks: no
necesitan GPU, modelos guardados ni el dataset. Necesitan Keras y PyTorch, así que corren en .venv-gpu:
    .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_recrear_personajes.py -q
"""

import os
import re
import sys
from pathlib import Path

# Sin GPU y con el backend de los notebooks, antes de importar Keras o PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

keras = pytest.importorskip("keras")
torch = pytest.importorskip("torch")
from keras import layers  # noqa: E402

import recrear_personajes as R  # noqa: E402

ALTO = ANCHO = 8
CLASES = 3


def _estadisticas_al_azar(modelo, semilla):
    """BatchNorm con medias y varianzas guardadas distintas de 0 y 1, como las de un modelo entrenado."""
    azar = np.random.default_rng(semilla)
    for capa in modelo.layers:
        if isinstance(capa, layers.BatchNormalization):
            n = capa.moving_mean.shape[0]
            capa.moving_mean.assign(azar.normal(0, 0.5, n).astype("float32"))
            capa.moving_variance.assign(azar.uniform(0.5, 2.0, n).astype("float32"))
    return modelo


def cnn_minima():
    keras.utils.set_random_seed(1)
    modelo = keras.Sequential([
        layers.Input(shape=(ALTO, ANCHO, 3), name="entrada"),
        layers.Rescaling(1.0 / 255, name="escalado"),
        layers.Conv2D(4, 3, padding="same", use_bias=False, name="conv_1"),
        layers.BatchNormalization(name="norm_1"),
        layers.Activation("relu", name="activacion_1"),
        layers.MaxPooling2D(2, name="pooling_1"),
        layers.GlobalAveragePooling2D(name="pooling_global"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(6, activation="relu", name="densa"),
        layers.BatchNormalization(name="norm_densa"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(CLASES, activation="softmax", name="salida"),
    ], name="cnn_minima")
    return _estadisticas_al_azar(modelo, 1)


def mlp_minimo():
    keras.utils.set_random_seed(2)
    modelo = keras.Sequential([
        layers.Input(shape=(ALTO * ANCHO * 3,), name="entrada"),
        layers.Rescaling(1.0 / 255, name="escalado"),
        layers.Dense(16, activation="relu", name="oculta_1"),
        layers.BatchNormalization(name="norm_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(8, activation="relu", name="oculta_2"),
        layers.BatchNormalization(name="norm_2"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(CLASES, activation="softmax", name="salida"),
    ], name="mlp_minimo")
    return _estadisticas_al_azar(modelo, 2)


def mlp_descriptores():
    return keras.Sequential([
        layers.Input(shape=(10,), name="entrada"),
        layers.Normalization(mean=np.zeros(10), variance=np.ones(10), name="estandarizacion"),
        layers.Dense(4, activation="relu", name="oculta_1"),
        layers.BatchNormalization(name="norm_1"),
        layers.Dense(CLASES, activation="softmax", name="salida"),
    ], name="mlp_descriptores")


def imagenes(n=5, semilla=0):
    return np.random.default_rng(semilla).integers(0, 256, (n, ALTO, ANCHO, 3), dtype=np.uint8)


def en_01(lote):
    return torch.as_tensor(lote).float() / 255.0


def salida_del_modelo(modelo, lote):
    entrada = lote.reshape(len(lote), -1) if len(modelo.input_shape) == 2 else lote
    with torch.no_grad():
        return np.asarray(modelo(torch.as_tensor(entrada).float(), training=False).detach())


@pytest.mark.parametrize("construir, normas", [(cnn_minima, ["norm_1", "norm_densa"]), (mlp_minimo, ["norm_1", "norm_2"])])
def test_logits_coherentes_con_la_salida_del_modelo(construir, normas):
    modelo = construir()
    lote = imagenes()
    logits, estadisticas = R.recorrer(modelo, en_01(lote))
    assert [e.capa for e in estadisticas] == normas
    np.testing.assert_allclose(torch.softmax(logits, dim=-1).detach().numpy(), salida_del_modelo(modelo, lote),
                               atol=1e-5)


def test_el_aplanado_del_mlp_es_reshape_n_menos_1():
    lote = imagenes()
    np.testing.assert_array_equal(R.aplanar(torch.as_tensor(lote)).numpy(), lote.reshape(len(lote), -1))
    # Y es el orden con que el MLP recibe la imagen: con otro orden los logits no coincidirían
    modelo = mlp_minimo()
    logits, _ = R.recorrer(modelo, en_01(lote))
    otro_orden = lote.transpose(0, 3, 1, 2)
    assert not np.allclose(torch.softmax(logits, -1).detach().numpy(), salida_del_modelo(modelo, otro_orden.reshape(5, -1)),
                           atol=1e-4)


def test_los_pesos_quedan_sin_gradiente():
    modelo = cnn_minima()
    x = en_01(imagenes()).requires_grad_(True)
    logits, _ = R.recorrer(modelo, x)
    logits.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert all(not v.value.requires_grad and v.value.grad is None for v in modelo.weights)


@pytest.mark.parametrize("construir", [cnn_minima, mlp_minimo])
def test_perdida_bn_vale_cero_si_las_estadisticas_coinciden(construir):
    modelo = construir()
    x = en_01(imagenes(n=16))
    # Cada BatchNorm guarda la estadística que mide en este lote; se fija en orden, porque cambiar una
    # cambia la entrada de las siguientes
    for _ in range(len([c for c in modelo.layers if isinstance(c, layers.BatchNormalization)])):
        _, estadisticas = R.recorrer(modelo, x)
        pendiente = next(e for e in estadisticas
                         if not torch.allclose(e.media, e.media_bn) or not torch.allclose(e.varianza, e.varianza_bn))
        capa = modelo.get_layer(pendiente.capa)
        capa.moving_mean.assign(pendiente.media.detach().numpy())
        capa.moving_variance.assign(pendiente.varianza.detach().numpy())
    _, estadisticas = R.recorrer(modelo, x)
    assert float(R.perdida_bn(estadisticas)) == pytest.approx(0.0, abs=1e-5)
    # Y no vale cero si no coinciden
    _, otras = R.recorrer(modelo, en_01(imagenes(n=16, semilla=7)) * 0.2)
    assert float(R.perdida_bn(otras)) > 0.01


def test_la_primera_bn_pesa_mas():
    modelo = cnn_minima()
    _, estadisticas = R.recorrer(modelo, en_01(imagenes()))
    primera = float(R.perdida_bn(estadisticas[:1], multiplicador_primera=1.0))
    assert float(R.perdida_bn(estadisticas, 10.0)) - float(R.perdida_bn(estadisticas, 1.0)) == pytest.approx(9 * primera, rel=1e-4)


@pytest.mark.parametrize("construir", [cnn_minima, mlp_minimo])
def test_unos_pocos_pasos_suben_la_clase_objetivo(construir):
    modelo = construir()
    objetivos = [0, 1, 2]
    # Sin el término de BatchNorm: en una red tan pequeña, sus estadísticas al azar tiran en otra dirección
    hiper = R.Hiperparametros(pasos=60, lr=0.1, alfa_bn=0.0, desplazamiento=1)
    recreacion = R.recrear(modelo, objetivos, (ALTO, ANCHO), hiper, semilla=3, dispositivo="cpu")
    assert sorted(recreacion.instantaneas) == [0, 50, 60]
    assert recreacion.final.dtype == np.uint8 and recreacion.final.shape == (3, ALTO, ANCHO, 3)
    filas = np.arange(len(objetivos))
    antes = salida_del_modelo(modelo, recreacion.instantaneas[0])[filas, objetivos]
    despues = salida_del_modelo(modelo, recreacion.final)[filas, objetivos]
    # Una red tan pequeña se satura pronto: basta con que la clase objetivo suba y la entropía baje
    assert despues.mean() > antes.mean() + 0.01
    assert recreacion.historial[-1]["entropia"] < 0.95 * recreacion.historial[0]["entropia"]


def test_misma_semilla_mismo_ruido_inicial():
    modelo = cnn_minima()
    hiper = R.Hiperparametros(pasos=2, desplazamiento=1)
    a = R.recrear(modelo, [0, 1], (ALTO, ANCHO), hiper, semilla=5, dispositivo="cpu")
    b = R.recrear(modelo, [0, 1], (ALTO, ANCHO), hiper, semilla=5, dispositivo="cpu")
    np.testing.assert_array_equal(a.instantaneas[0], b.instantaneas[0])
    np.testing.assert_array_equal(a.final, b.final)


def test_rechaza_un_mlp_de_descriptores():
    modelo = mlp_descriptores()
    assert R.es_descriptores(modelo)
    with pytest.raises(ValueError, match="Normalization"):
        R.recorrer(modelo, torch.rand(2, 10))


def test_los_rotulos_tienen_el_formato_esperado():
    lineas = R.rotulos_casilla(("homer_simpson", 0.97), [("bart_simpson", 0.123), ("kang_kodos", 0.5)])
    assert lineas == ["propia: Homer Simpson · 97 %", "otras: Bart Simpson · 12 %", "otras: Kang y Kodos · 50 %"]
    for linea in lineas:
        assert re.fullmatch(r"(propia|otras): \S.* · \d{1,3} %", linea)


def test_el_juicio_compara_por_nombre_de_clase():
    class RedFija:
        clases = ["bart_simpson", "homer_simpson", "lisa_simpson"]

        def probabilidades(self, _):
            return np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]], dtype=np.float32)

    juicio = R.juzgar(RedFija(), np.zeros((2, ALTO, ANCHO, 3), np.uint8), ["homer_simpson", "lisa_simpson"])
    assert juicio["prediccion"] == ["homer_simpson", "bart_simpson"]
    assert list(juicio["posicion"]) == [1, 3]
    assert R.fila_matriz(juicio) == {"top1": 1, "top3": 2, "n": 2, "confianza": pytest.approx(0.4, abs=1e-4),
                                     "posicion_media": 2.0}


def test_generadores_por_defecto_son_todos_los_modelos():
    assert R.leer_argumentos([]).generadores == list(R.ESPECIFICACIONES)
    assert R.leer_argumentos(["--modelos", "cnn,mlp"]).generadores == ["cnn", "mlp"]
    assert R.leer_argumentos(["--generadores", "mlp"]).generadores == ["mlp"]


def test_generadores_fuera_de_modelos_se_rechazan():
    with pytest.raises(SystemExit):
        R.leer_argumentos(["--modelos", "cnn", "--generadores", "mlp"])
    with pytest.raises(SystemExit):
        R.leer_argumentos(["--generadores", ","])


def test_las_figuras_van_a_la_carpeta_de_quien_genera():
    from types import SimpleNamespace

    mlp = SimpleNamespace(especificacion=R.ESPECIFICACIONES["mlp"])
    cnn = SimpleNamespace(especificacion=R.ESPECIFICACIONES["cnn"])
    recortes = SimpleNamespace(especificacion=R.ESPECIFICACIONES["cnn_recortes"])
    assert R.carpeta_comun([mlp]) == R.SALIDA_MLP
    assert R.carpeta_comun([cnn, recortes]) == R.SALIDA_CNN
    assert R.carpeta_comun([cnn, mlp]) == R.SALIDA_CNN
    assert R.carpeta_comun([mlp], Path("otra")) == Path("otra")


def test_main_solo_genera_con_los_generadores(tmp_path, monkeypatch):
    """Con --generadores mlp, la CNN juzga pero no recrea: no deja carpeta ni fila en la matriz."""
    clases = ["bart_simpson", "homer_simpson", "lisa_simpson"]
    modelos = {"cnn": cnn_minima(), "mlp": mlp_minimo()}

    def resolver(especificacion, _usar_cpu):
        modelo = modelos[especificacion.identificador]
        metadatos = {"clases": clases, "alto": ALTO, "ancho": ANCHO}
        ruta = tmp_path / "modelos" / f"{especificacion.identificador}.keras"
        return R.Red(especificacion, ruta, modelo, metadatos, "cpu", "0" * 64), []

    monkeypatch.setattr(R, "resolver", resolver)
    salida = tmp_path / "salida"
    resultado = R.main(["--modelos", "cnn,mlp", "--generadores", "mlp", "--pasos", "2", "--semilla", "1",
                        "--desplazamiento", "1", "--salida", str(salida), "--cpu"])
    assert resultado["generadores"] == ["mlp"]
    assert [r["id"] for r in resultado["redes"]] == ["cnn", "mlp"]
    assert list(resultado["matriz"]) == ["mlp"]
    assert sorted(resultado["matriz"]["mlp"]) == ["cnn", "mlp"]
    assert list(resultado["instantaneas"]) == ["mlp"] and list(resultado["salidas"]) == ["mlp"]
    assert (salida / "mlp" / "lamina.png").is_file() and not (salida / "cnn").exists()
    assert (salida / "comparacion.png").is_file() and (salida / "evolucion.png").is_file()


def test_titulos_nombran_a_la_red_que_genera():
    jueces = ["cnn", "cnn_recortes", "mlp"]
    assert R.titulo_lamina("mlp", "mlp_r1.keras", 64, jueces) == (
        "Los 64 personajes recreados solo desde el MLP de píxeles (mlp_r1.keras)\n"
        "propia: lo que dice el MLP de píxeles · otras: lo que dicen CNN fotograma y CNN recortes · "
        "verde si nombra al personaje")
    solo_mlp = R.titulo_comparacion(["mlp"], jueces, 12)
    assert solo_mlp.startswith("12 personajes recreados desde el MLP de píxeles\n")
    assert "diapositiva" not in solo_mlp
    assert R.titulo_comparacion(["cnn", "cnn_recortes"], jueces, 12).startswith(
        "12 personajes recreados desde cada red, una fila por red\n")
    assert R.titulo_evolucion(["mlp"]) == "Del ruido a la recreación: lo que dice el MLP de píxeles en cada paso"
    assert R.titulo_evolucion(["cnn", "cnn_recortes"]).endswith("lo que dice la red que la genera en cada paso")
    # Sin otras redes que juzguen, la leyenda no menciona «otras»
    assert "otras" not in R.titulo_lamina("mlp", "mlp_r1.keras", 64, ["mlp"])


def test_metadatos_alternativos_solo_aportan_clases_y_resolucion(tmp_path):
    """Un experimento sin .json toma del modelo final solo las clases y la resolución, no su configuración."""
    import json

    mlp_minimo().save(tmp_path / "mlp_r1.keras")
    (tmp_path / "mlp_final.json").write_text(json.dumps({
        "clases": ["bart_simpson", "homer_simpson", "lisa_simpson"], "alto": ALTO, "ancho": ANCHO,
        "configuracion": "C1 - combinación de mejoras", "val_accuracy": 0.54,
    }), encoding="utf-8")
    _, metadatos, _ = R.cargar(tmp_path / "mlp_r1.keras", ("mlp_final.json",), usar_cpu=True)
    assert metadatos == {"clases": ["bart_simpson", "homer_simpson", "lisa_simpson"], "alto": ALTO, "ancho": ANCHO,
                         "metadatos_de": "mlp_final.json"}


def test_correlacion_de_pesos_distingue_una_red_afinada():
    a = cnn_minima()
    b = cnn_minima()  # misma semilla: los mismos pesos, como una red afinada desde la otra
    keras.utils.set_random_seed(99)
    c = keras.models.clone_model(a)  # misma arquitectura, otra inicialización
    assert R.correlacion_pesos(a, b) == pytest.approx(1.0)
    assert abs(R.correlacion_pesos(a, c)) < R.CORRELACION_DEPENDIENTE
    assert R.correlacion_pesos(a, mlp_minimo()) is None


# --- Hiperparámetros por red, BatchNorm normalizada, escalas y reinicios ---------------------------------

def _fijar_estadisticas_del_lote(modelo, x):
    """Cada BatchNorm guarda la media y la varianza que mide en x; en orden, porque cambiar una cambia las
    siguientes."""
    for _ in range(len([c for c in modelo.layers if isinstance(c, layers.BatchNormalization)])):
        _, estadisticas = R.recorrer(modelo, x)
        pendiente = next((e for e in estadisticas
                          if not torch.allclose(e.media, e.media_bn) or not torch.allclose(e.varianza, e.varianza_bn)),
                         None)
        if pendiente is None:
            break
        capa = modelo.get_layer(pendiente.capa)
        capa.moving_mean.assign(pendiente.media.detach().numpy())
        capa.moving_variance.assign(pendiente.varianza.detach().numpy())


@pytest.mark.parametrize("construir", [cnn_minima, mlp_minimo])
def test_perdida_bn_normalizada_vale_cero_si_las_estadisticas_coinciden(construir):
    modelo = construir()
    x = en_01(imagenes(n=16))
    _fijar_estadisticas_del_lote(modelo, x)
    _, estadisticas = R.recorrer(modelo, x)
    assert float(R.perdida_bn(estadisticas, normalizada=True)) == pytest.approx(0.0, abs=1e-4)
    _, otras = R.recorrer(modelo, en_01(imagenes(n=16, semilla=7)) * 0.2)
    assert float(R.perdida_bn(otras, normalizada=True)) > 0.01


def test_la_bn_normalizada_no_depende_de_la_escala_de_la_capa():
    """La misma desviación relativa pesa lo mismo en una neurona con activaciones de 1 que en una de 1.000."""
    def estadistica(escala):
        uno = torch.ones(4)
        return R.Estadistica("norm", 1.5 * escala * uno, 2.0 * escala ** 2 * uno, escala * uno, escala ** 2 * uno)

    chica = float(R.perdida_bn([estadistica(1.0)], 1.0, normalizada=True))
    grande = float(R.perdida_bn([estadistica(1000.0)], 1.0, normalizada=True))
    assert grande == pytest.approx(chica, rel=1e-2)
    # Sin normalizar, la neurona grande domina por un factor de miles
    assert float(R.perdida_bn([estadistica(1000.0)], 1.0)) > 1000 * float(R.perdida_bn([estadistica(1.0)], 1.0))


def test_la_clave_de_cache_de_las_cnn_no_cambia():
    """Con los valores por defecto, la clave es la de antes de agregar los campos del MLP: las recreaciones de las
    CNN que ya estaban en caché siguen valiendo."""
    antes = {"pasos": 2000, "lr": 0.05, "alfa_bn": 0.01, "alfa_tv": 1e-3, "alfa_l2": 1e-5, "desplazamiento": 16,
             "multiplicador_primera_bn": 10.0, "beta1": 0.5, "beta2": 0.9}
    assert R.clave_hiper(R.Hiperparametros()) == antes
    assert R.clave_hiper(R.ESPECIFICACIONES["cnn"].hiper) == antes
    assert R.clave_hiper(R.ESPECIFICACIONES["cnn_recortes"].hiper) == antes
    # Un campo nuevo distinto de su valor por defecto sí entra
    assert R.clave_hiper(R.Hiperparametros(reinicios=4)) == {**antes, "reinicios": 4}


def test_los_flags_solo_reemplazan_lo_que_se_paso():
    from dataclasses import replace

    sin_flags = R.leer_argumentos([])
    assert R.hiper_de(R.ESPECIFICACIONES["cnn"], sin_flags) == R.Hiperparametros()
    assert R.hiper_de(R.ESPECIFICACIONES["mlp"], sin_flags) == R.HIPER_MLP
    con_flags = R.leer_argumentos(["--pasos", "5", "--escalas", "ninguna", "--no-volteo", "--reinicios", "2"])
    assert R.hiper_de(R.ESPECIFICACIONES["mlp_robusto"], con_flags) == replace(
        R.HIPER_MLP_ROBUSTO, pasos=5, escalas=(), volteo=False, reinicios=2)
    assert R.leer_argumentos(["--escalas", "16,32,64"]).escalas == (16, 32, 64)


def test_las_etapas_reparten_los_pasos_de_lo_grueso_a_lo_fino():
    assert R.etapas(R.Hiperparametros(pasos=10), (8, 8)) == [(8, 8, 10)]
    assert R.etapas(R.Hiperparametros(pasos=10, escalas=(2, 4, 8)), (8, 8)) == [(2, 2, 3), (4, 4, 3), (8, 8, 4)]
    # Las escalas mayores que el modelo no se usan; si no queda ninguna, es una sola etapa
    assert R.etapas(R.Hiperparametros(pasos=10, escalas=(4, 16)), (8, 8)) == [(4, 4, 10)]
    assert R.etapas(R.Hiperparametros(pasos=10, escalas=(16, 32)), (8, 8)) == [(8, 8, 10)]
    with pytest.raises(ValueError):
        R.etapas(R.Hiperparametros(pasos=10, escalas=(8, 4)), (8, 8))


@pytest.mark.parametrize("construir", [cnn_minima, mlp_minimo])
def test_con_escalas_la_recreacion_sale_a_la_resolucion_del_modelo(construir):
    modelo = construir()
    hiper = R.Hiperparametros(pasos=9, lr=0.1, alfa_bn=0.0, desplazamiento=0, volteo=False, escalas=(2, 4, 8),
                              bn_normalizada=True)
    recreacion = R.recrear(modelo, [0, 1, 2], (ALTO, ANCHO), hiper, semilla=3, dispositivo="cpu")
    assert sorted(recreacion.instantaneas) == [0, 9]
    assert recreacion.final.dtype == np.uint8 and recreacion.final.shape == (3, ALTO, ANCHO, 3)
    # El punto de partida es ruido de 2 × 2 ampliado: mucho más suave que ruido a la resolución del modelo
    inicial = recreacion.instantaneas[0].astype(float)
    assert np.abs(np.diff(inicial, axis=1)).mean() < 0.5 * np.abs(np.diff(imagenes(3).astype(float), axis=1)).mean()


def test_los_reinicios_los_elige_la_red_que_genera():
    # 2 reinicios de 3 personajes: el reinicio k del personaje i está en k·3 + i
    p_objetivo = np.array([0.1, 0.9, 0.4,
                           0.5, 0.2, 0.3])
    assert list(R.indices_elegidos(p_objetivo, 3, 2)) == [3, 1, 2]

    class RedFija:
        clases = ["bart_simpson", "homer_simpson"]

        def probabilidades(self, imagenes):
            # Reconoce a Homer cuanto más clara es la imagen y a Bart cuanto más oscura
            brillo = imagenes.reshape(len(imagenes), -1).mean(axis=1) / 255.0
            return np.stack([1 - brillo, brillo], axis=1)

    oscura, clara = np.zeros((ALTO, ANCHO, 3), np.uint8), np.full((ALTO, ANCHO, 3), 255, np.uint8)
    # Reinicio 0: Bart claro, Homer oscuro (mal); reinicio 1: Bart oscuro, Homer claro (bien)
    final = np.stack([clara, oscura, oscura, clara])
    recreacion = R.Recreacion(final, {0: final.copy(), 5: final.copy()})
    elegida = R.elegir_reinicios(recreacion, RedFija(), ["bart_simpson", "homer_simpson"], 2)
    np.testing.assert_array_equal(elegida.final, np.stack([oscura, clara]))
    assert sorted(elegida.instantaneas) == [0, 5] and elegida.instantaneas[0].shape == (2, ALTO, ANCHO, 3)
    # Con un solo reinicio no cambia nada
    assert R.elegir_reinicios(recreacion, RedFija(), ["bart_simpson"] * 4, 1) is recreacion


def test_main_usa_los_hiperparametros_de_cada_red(tmp_path, monkeypatch):
    """El MLP genera con HIPER_MLP (reinicios incluidos) y el JSON lo registra."""
    import json

    clases = ["bart_simpson", "homer_simpson", "lisa_simpson"]
    modelos = {"cnn": cnn_minima(), "mlp": mlp_minimo()}

    def resolver(especificacion, _usar_cpu):
        metadatos = {"clases": clases, "alto": ALTO, "ancho": ANCHO}
        ruta = tmp_path / "modelos" / f"{especificacion.identificador}.keras"
        return R.Red(especificacion, ruta, modelos[especificacion.identificador], metadatos, "cpu", "0" * 64), []

    monkeypatch.setattr(R, "resolver", resolver)
    salida = tmp_path / "salida"
    resultado = R.main(["--modelos", "cnn,mlp", "--generadores", "mlp", "--pasos", "3", "--semilla", "1",
                        "--salida", str(salida), "--cpu"])
    assert resultado["hiperparametros"]["mlp"]["reinicios"] == R.HIPER_MLP.reinicios
    assert resultado["hiperparametros"]["mlp"]["pasos"] == 3
    registro = json.loads((salida / "mlp" / "recreacion.json").read_text(encoding="utf-8"))
    assert registro["hiperparametros"]["lote"] == len(clases) * R.HIPER_MLP.reinicios
    assert registro["matriz"]["mlp"]["n"] == len(clases)
    assert "posicion_media" in registro["matriz"]["cnn"]
