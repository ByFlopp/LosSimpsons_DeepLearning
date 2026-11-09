"""Pruebas de scripts/newDataset/barrido_recreacion.py (sin GPU ni modelos):
    .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_barrido_recreacion.py -q
"""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("keras")

import barrido_recreacion as B  # noqa: E402
import recrear_personajes as R  # noqa: E402

CLASES = sorted(set(R.PERSONAJES_F1) | set(R.PERSONAJES_EVOLUCION) | {f"personaje_{i:02d}" for i in range(40)})


def test_los_personajes_de_ajuste_no_salen_en_las_figuras_y_son_siempre_los_mismos():
    ajuste = B.personajes_de_ajuste(CLASES)
    assert len(ajuste) == B.N_AJUSTE == len(set(ajuste))
    assert not set(ajuste) & (set(R.PERSONAJES_F1) | set(R.PERSONAJES_EVOLUCION))
    assert ajuste == B.personajes_de_ajuste(CLASES) == sorted(ajuste)


def test_cada_configuracion_es_un_juego_de_argumentos_valido():
    for nombre_rejilla, rejilla in B.REJILLAS.items():
        todas = B.configuraciones(rejilla)
        assert len(todas) == len({B.nombre_configuracion(c) for c in todas})
        for configuracion in todas:
            argv = B.argumentos_main("mlp", configuracion, ["bart_simpson"], 10, Path("salida"))
            args = R.leer_argumentos(argv)
            hiper = R.hiper_de(R.ESPECIFICACIONES["mlp"], args)
            assert args.sin_cache and hiper.bn_normalizada and hiper.pasos == 10
            assert hiper.volteo == configuracion["volteo"]
            assert hiper.desplazamiento == configuracion["desplazamiento"]
            assert hiper.escalas == R._escalas(configuracion["escalas"]), nombre_rejilla


def test_la_matriz_de_un_subconjunto_sale_del_detalle():
    detalle = [
        {"personaje": "bart_simpson", "jueces": {"cnn": {"posicion": 1, "p_objetivo": 0.8},
                                                 "mlp": {"posicion": 1, "p_objetivo": 0.9}}},
        {"personaje": "homer_simpson", "jueces": {"cnn": {"posicion": 5, "p_objetivo": 0.1},
                                                  "mlp": {"posicion": 1, "p_objetivo": 0.7}}},
        {"personaje": "lisa_simpson", "jueces": {"cnn": {"posicion": 40, "p_objetivo": 0.0},
                                                 "mlp": {"posicion": 2, "p_objetivo": 0.3}}},
    ]
    matriz = B.matriz_de(detalle, ["homer_simpson", "lisa_simpson"])
    assert matriz["cnn"] == {"top1": 0, "top3": 0, "n": 2, "confianza": 0.05, "posicion_media": 22.5}
    assert matriz["mlp"]["top1"] == 1 and matriz["mlp"]["top3"] == 2
    assert B.matriz_de(detalle, []) == {}


def test_el_orden_prefiere_reconocerse_y_despues_acercarse_para_las_cnn():
    filas = [
        {"configuracion": "a", "top1_propio": 15, "posicion_cnn": 10.0, "confianza_cnn": 0.2},
        {"configuracion": "b", "top1_propio": 16, "posicion_cnn": 20.0, "confianza_cnn": 0.1},
        {"configuracion": "c", "top1_propio": 16, "posicion_cnn": 12.0, "confianza_cnn": 0.1},
    ]
    assert [f["configuracion"] for f in sorted(filas, key=B.orden)] == ["c", "b", "a"]
