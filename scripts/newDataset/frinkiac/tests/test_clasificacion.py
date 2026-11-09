"""Clasificación con la CNN de por_revisar/: reparto en seguras, dudosas y ruido, revisión de las dudosas y
traslado de la decisión al flujo de `pnpm frinkiac`. La CNN se reemplaza por un doble: no hace falta torch."""

import json

import numpy as np
import pytest
from PIL import Image

import ampliar_con_frinkiac
import clasificacion as C
from ampliar_con_frinkiac import main
from clasificar_con_cnn import clasificar, vistas
from test_flujo import entorno, imagenes  # noqa: F401  (entorno es una fixture)

CLASES = ["homer_simpson", "louie"]
LADO = 32
# Probabilidad de louie (fotograma completo, mejor recorte) para las 6 imágenes, en orden de nombre
PLAN = [(0.95, 0.10), (0.30, 0.92), (0.50, 0.60), (0.10, 0.04), (0.01, 0.02), (0.02, 0.01)]
GRUPOS_PLAN = [C.SEGURAS, C.SEGURAS, C.DUDOSAS, C.DUDOSAS, C.RUIDO, C.RUIDO]


def cnn_falsa(plan):
    """Doble de la CNN: devuelve para cada fotograma, en orden, las probabilidades del plan."""
    llamadas = []

    def predecir_lote(lote):
        assert lote.dtype == np.uint8 and lote.shape[1:] == (LADO, LADO, 3)
        inicio = sum(llamadas)
        llamadas.append(len(lote) // 4)
        filas = []
        for p_completa, p_recorte in plan[inicio:inicio + len(lote) // 4]:
            for p in (p_completa, 0.0, p_recorte, 0.0):  # completo, izquierda, centro, derecha
                filas.append([1 - p, p])
        return np.array(filas)

    return predecir_lote


def clasificar_louie(ampliacion, plan=PLAN):
    return clasificar(ampliacion / "por_revisar", CLASES, cnn_falsa(plan), LADO, LADO, C.Umbrales())


def test_grupos_segun_umbrales():
    u = C.Umbrales()
    assert C.grupo_de(0.80, 0.0, u) == C.SEGURAS
    assert C.grupo_de(0.79, 0.90, u) == C.SEGURAS
    assert C.grupo_de(0.79, 0.89, u) == C.DUDOSAS
    assert C.grupo_de(0.05, 0.0, u) == C.DUDOSAS
    assert C.grupo_de(0.0, 0.049, u) == C.RUIDO
    with pytest.raises(ValueError):
        C.Umbrales(segura=0.8, segura_recorte=0.9, ruido=0.9).validar()


def test_vistas_completo_y_tres_recortes(tmp_path):
    # El rojo crece hacia la derecha: el recorte izquierdo es el menos rojo y el derecho, el más
    x = np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (90, 1))
    ruta = tmp_path / "ancha.jpg"
    Image.fromarray(np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)).save(ruta, quality=95)
    v = vistas(ruta, 16, 16)
    assert v.shape == (4, 16, 16, 3) and v.dtype == np.uint8
    rojo = v[..., 0].mean(axis=(1, 2))
    assert rojo[1] < rojo[2] < rojo[3]
    assert abs(rojo[0] - rojo[2]) < 10  # el fotograma completo y el centro tienen el mismo promedio

    vertical = tmp_path / "vertical.jpg"
    Image.new("RGB", (50, 120), "white").save(vertical)
    assert vistas(vertical, 16, 16).shape == (4, 16, 16, 3)


def test_mismos_nombres_que_el_flujo():
    assert C.MARCA_PENDIENTE == ampliar_con_frinkiac.MARCA_PENDIENTE
    assert C.REGISTRO_REVISION == ampliar_con_frinkiac.REGISTRO_REVISION
    assert C.CARPETA in ampliar_con_frinkiac.GENERADO


def test_flujo_con_clasificacion_y_revision_de_dudosas(entorno, capsys):  # noqa: F811
    comando, dataset, ampliacion = entorno
    revision = ampliacion / "por_revisar" / "louie"
    raiz = ampliacion / "clasificacion"

    assert main(comando) == 0
    primeras = imagenes(revision)
    assert len(primeras) == 6
    assert "pnpm frinkiac clasificar" in capsys.readouterr().out

    # La CNN reparte: por_revisar/ no cambia y cada imagen queda en un solo grupo
    hechos = clasificar_louie(ampliacion)
    assert hechos["louie"] == {C.SEGURAS: 2, C.DUDOSAS: 2, C.RUIDO: 2}
    for nombre, grupo in zip(primeras, GRUPOS_PLAN):
        assert (raiz / grupo / "louie" / nombre).exists()
    assert imagenes(revision) == primeras and (revision / "_PENDIENTE.txt").exists()
    assert (raiz / C.DUDOSAS / "louie" / C.MARCA_PENDIENTE).exists()
    registro = C.leer_registro(raiz)
    assert [registro[("louie", n)]["grupo"] for n in primeras] == GRUPOS_PLAN
    # Volver a clasificar no repite nada
    assert clasificar_louie(ampliacion) == {}

    # Con las dudosas sin revisar no se aplica nada
    assert main(comando) == 0
    assert "faltan revisar 2 dudosas" in capsys.readouterr().out
    assert not (ampliacion / "revision" / "louie.json").exists()
    assert imagenes(revision) == primeras

    # Revisión: se borra una dudosa, se rescata una de ruido y se termina la carpeta
    (raiz / C.DUDOSAS / "louie" / primeras[3]).unlink()
    (raiz / C.RUIDO / "louie" / primeras[5]).rename(raiz / C.SEGURAS / "louie" / primeras[5])
    (raiz / C.DUDOSAS / "louie" / C.MARCA_PENDIENTE).unlink()
    assert main(comando) == 0
    assert "clasificación de la CNN lista (4 seguras o dudosas que quedan, 2 de ruido o borradas)" in \
        capsys.readouterr().out
    decision = json.loads((ampliacion / "revision" / "louie.json").read_text())
    assert decision["aprobadas"] == sorted(primeras[i] for i in (0, 1, 2, 5))
    assert decision["rechazadas"] == sorted(primeras[i] for i in (3, 4))
    for grupo in C.GRUPOS:  # lo trasladado sale de clasificacion/
        assert not (raiz / grupo / "louie").exists()

    # Las 2 rechazadas se reponen con fotogramas nuevos, que se clasifican como cualquier otro
    nuevas = imagenes(revision)
    assert len(nuevas) == 2 and not set(nuevas) & set(primeras)
    assert clasificar_louie(ampliacion, [(0.99, 0.99)] * 2)["louie"] == {C.SEGURAS: 2}
    assert main(comando) == 0
    assert len(imagenes(ampliacion / "final" / "louie")) == 6

    # limpiar también borra la clasificación
    assert main(["limpiar", "--local", str(ampliacion), "--dataset", str(dataset), "--si"]) == 0
    assert not raiz.exists()


def test_imagenes_sin_clasificar_o_revision_a_mano_no_se_tocan(entorno):  # noqa: F811
    comando, _, ampliacion = entorno
    por_revisar = ampliacion / "por_revisar"
    revision = por_revisar / "louie"
    assert main(comando) == 0
    clasificar_louie(ampliacion, [(0.01, 0.01)] * 6)  # todo ruido: no hay dudosas que revisar

    # Llega una imagen nueva mientras tanto: hasta clasificarla no se traslada nada
    descargadas = json.loads((revision / C.REGISTRO_REVISION).read_text())
    nueva = "louie__S01E01__t0099999.jpg"
    (revision / nueva).write_bytes((revision / imagenes(revision)[0]).read_bytes())
    descargadas["archivos"].append(nueva)
    (revision / C.REGISTRO_REVISION).write_text(json.dumps(descargadas))
    assert [(s.estado, s.faltan) for s in C.sincronizar(por_revisar)] == [("sin_clasificar", 1)]
    assert len(imagenes(revision)) == 7

    # Si el equipo terminó la revisión a mano en por_revisar/, la clasificación no la pisa
    (revision / "_PENDIENTE.txt").unlink()
    assert C.sincronizar(por_revisar) == []
    assert len(imagenes(revision)) == 7
