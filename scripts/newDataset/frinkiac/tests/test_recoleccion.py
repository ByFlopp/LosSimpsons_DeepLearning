"""Recolección de punta a punta con S3 simulado (moto) y un doble de Frinkiac."""

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

import estado as E
import inventario as inv

from comun import nombre_fotograma
from conftest import FrinkiacFalso, escribir_guiones, imagen_ruido, subtitulos_para
from estado import Estado
from fuentes import leer_episodios, leer_lineas
from recolector import AGOTADO, COMPLETO, POR_REVISAR, Config, Recolector

PERSONAJE = "louie"
LINEAS = [(i, "Louie", 10_000 * i, f"linea{i} dice algo importante") for i in range(1, 13)]


# --- Preparación ---------------------------------------------------------------------------------

@pytest.fixture
def dataset(tmp_path):
    """louie con 3 originales y varias sintéticas que no deben contar."""
    carpeta = tmp_path / "dataset" / PERSONAJE
    carpeta.mkdir(parents=True)
    for nombre, semilla in (("louie_0001.jpg", 1), ("louie_0002.jpg", 2), ("louie__S01E01__t0010000.jpg", 3)):
        (carpeta / nombre).write_bytes(imagen_ruido(semilla))
    for nombre, semilla in (("louie_0001__alterada.jpg", 4), ("louie__S01E01__t0010000__parecida.jpg", 5),
                            ("louie_0002__alterada.jpg.tmp", 6)):
        (carpeta / nombre).write_bytes(imagen_ruido(semilla))
    (tmp_path / "dataset" / "_alteraciones.csv").write_text("personaje,original\n")
    otro = tmp_path / "dataset" / "homer_simpson"
    otro.mkdir()
    for i in range(12):
        (otro / f"homer_{i:04d}.jpg").write_bytes(imagen_ruido(100 + i))
    return tmp_path / "dataset"


@pytest.fixture
def entorno(tmp_path, dataset, almacen):
    inventario = inv.construir(dataset, objetivo=10, progreso=lambda *_: None)
    guiones, episodios = escribir_guiones(tmp_path, LINEAS)
    lineas = leer_lineas(guiones, leer_episodios(episodios), {PERSONAJE: ["Louie"]})

    def nuevo(almacen_usado=None, cliente=None, margen=0.0):
        return Recolector(almacen_usado or almacen, cliente or FrinkiacFalso(subtitulos_para(LINEAS)),
                          Estado(tmp_path / "datos" / "estado.sqlite"), inventario, lineas, tmp_path / "datos",
                          [PERSONAJE], Config(objetivo=10, margen=margen, respaldo_cada_s=0))
    return nuevo, inventario


def objetos(almacen, prefijo):
    return sorted(k for k in almacen.listar(prefijo) if k.endswith(".jpg"))


def revisar(almacen, aprobadas=(), rechazadas=()):
    almacen.escribir_json(almacen.clave("revision", f"{PERSONAJE}.json"),
                          {"aprobadas": sorted(aprobadas), "rechazadas": sorted(rechazadas)})


def nombres(claves):
    return [c.rsplit("/", 1)[-1] for c in claves]


# --- Pruebas -------------------------------------------------------------------------------------

def test_inventario_solo_cuenta_originales(dataset, tmp_path):
    inventario = inv.construir(dataset, objetivo=10, progreso=lambda *_: None)
    assert inventario.conteos == {PERSONAJE: 3, "homer_simpson": 12}
    assert inventario.claves[PERSONAJE] == {("S01E01", 10000)}
    assert inventario.miniaturas[PERSONAJE].shape == (3, 768)
    assert "homer_simpson" not in inventario.miniaturas  # ya tiene el objetivo
    assert inventario.por_debajo() == [PERSONAJE]
    ruta = tmp_path / "inv.npz"
    inv.guardar(inventario, ruta)
    cargado = inv.cargar(ruta)
    assert cargado.conteos == inventario.conteos and cargado.claves == inventario.claves
    assert np.allclose(cargado.miniaturas[PERSONAJE], inventario.miniaturas[PERSONAJE], atol=1e-3)


def test_recolecta_hasta_la_meta_sin_repetir(entorno, almacen):
    nuevo, _ = entorno
    recolector = nuevo()
    [ev] = recolector.vuelta()
    assert ev.estado == POR_REVISAR
    subidas = objetos(almacen, "staging/louie/")
    assert len(subidas) == 10 - 3  # meta exacta: originales + nuevas = objetivo
    # el fotograma que ya está en el dataset nunca se vuelve a bajar
    assert nombre_fotograma(PERSONAJE, "S01E01", 10000) not in nombres(subidas)
    assert ("S01E01", 10000) not in recolector.cliente.descargas
    shas = [hashlib.sha256(almacen.leer_bytes(c)).hexdigest() for c in subidas]
    assert len(set(shas)) == len(shas)
    manifiesto, _ = almacen.leer_json("staging/louie/_manifiesto.json")
    assert len(manifiesto["vigentes"]) == 7
    assert almacen.cabecera("estado/ultimo.sqlite") is not None
    # una segunda vuelta no hace nada nuevo
    [ev2] = recolector.vuelta()
    assert objetos(almacen, "staging/louie/") == subidas and ev2.estado == POR_REVISAR


class Caida(BaseException):
    """Simula que el proceso muere (kill -9): no la atrapa ningún except Exception."""


class AlmacenQueSeCae:
    """Envuelve cualquier almacén (S3 o local) y simula una caída en la n-ésima subida."""

    def __init__(self, base, caer_en: int, antes_de_subir: bool):
        self._base = base
        self.caer_en, self.antes, self.subidas = caer_en, antes_de_subir, 0

    def __getattr__(self, nombre):
        return getattr(self._base, nombre)

    def subir_fotograma(self, *args, **kwargs):
        self.subidas += 1
        if self.subidas == self.caer_en and self.antes:
            raise Caida()
        resultado = self._base.subir_fotograma(*args, **kwargs)
        if self.subidas == self.caer_en:
            raise Caida()  # el almacén ya confirmó, la base todavía no lo sabe
        return resultado


@pytest.mark.parametrize("antes_de_subir", [False, True])
def test_caida_a_mitad_no_pierde_ni_repite(entorno, almacen, antes_de_subir):
    nuevo, _ = entorno
    with pytest.raises(Caida):
        nuevo(AlmacenQueSeCae(almacen, caer_en=4, antes_de_subir=antes_de_subir)).vuelta()
    assert len(objetos(almacen, "staging/louie/")) == (3 if antes_de_subir else 4)
    reanudado = nuevo()
    reanudado.reconciliar()
    [ev] = reanudado.vuelta()
    subidas = objetos(almacen, "staging/louie/")
    assert ev.estado == POR_REVISAR and len(subidas) == 7
    assert {nombre_fotograma(PERSONAJE, e, t) for e, t in reanudado.db.claves_subidas(PERSONAJE)} == set(nombres(subidas))
    assert reanudado.db.contar(PERSONAJE).get(E.DESCARGADA, 0) == 0


def test_reconstruye_desde_s3_si_se_pierde_la_base(entorno, almacen, tmp_path):
    nuevo, _ = entorno
    primero = nuevo()
    primero.vuelta()
    primero.db.cerrar()
    for archivo in (tmp_path / "datos").glob("estado.sqlite*"):
        archivo.unlink()
    reconstruido = nuevo()
    assert reconstruido.reconciliar() == {PERSONAJE: 7}
    [ev] = reconstruido.vuelta()
    assert ev.estado == POR_REVISAR and len(objetos(almacen, "staging/louie/")) == 7


def test_descarta_fotogramas_casi_identicos(entorno, almacen):
    nuevo, _ = entorno

    def toma(ts: int) -> int:
        return ts // 10_000

    def casi_igual(ep, ts):
        """Todos los fotogramas de una misma toma son la misma imagen con un ruido mínimo distinto."""
        imagen = np.asarray(Image.open(io.BytesIO(imagen_ruido(1000 + toma(ts)))), np.int16)
        ruido = np.random.default_rng(ts).integers(-3, 4, imagen.shape)
        salida = io.BytesIO()
        Image.fromarray(np.clip(imagen + ruido, 0, 255).astype(np.uint8)).save(salida, format="JPEG", quality=90)
        return salida.getvalue()

    recolector = nuevo(cliente=FrinkiacFalso(subtitulos_para(LINEAS), casi_igual))
    [ev] = recolector.vuelta()
    subidas = nombres(objetos(almacen, "staging/louie/"))
    tomas = {toma(int(n.split("__t")[1][:7])) for n in subidas}
    assert len(tomas) == len(subidas)  # una sola imagen por toma: las demás eran casi idénticas
    assert recolector.db.contar(PERSONAJE).get(E.DUPLICADA, 0) > 0
    assert ev.estado == POR_REVISAR


def test_revision_repone_rechazadas_y_completa_final(entorno, almacen):
    nuevo, _ = entorno
    recolector = nuevo()
    recolector.vuelta()
    primeras = nombres(objetos(almacen, "staging/louie/"))
    revisar(almacen, aprobadas=primeras[2:], rechazadas=primeras[:2])
    [ev] = recolector.vuelta()
    assert ev.estado == POR_REVISAR and ev.pendientes_revision == 2
    todas = nombres(objetos(almacen, "staging/louie/"))
    assert len(todas) == 9 and set(primeras) < set(todas)  # nada se borra de staging
    nuevas = sorted(set(todas) - set(primeras))
    revisar(almacen, aprobadas=primeras[2:] + nuevas, rechazadas=primeras[:2])
    [ev] = recolector.vuelta()
    assert ev.estado == COMPLETO
    final = nombres(objetos(almacen, "final/louie/"))
    assert len(final) == 7 and not set(final) & set(primeras[:2])
    resumen, _ = almacen.leer_json("final/louie/_RESUMEN.json")
    assert resumen["completo"] and resumen["total"] == 10
    # repetir la vuelta no duplica nada en final/
    recolector.vuelta()
    assert nombres(objetos(almacen, "final/louie/")) == final


def test_personaje_sin_lineas_suficientes_queda_agotado(tmp_path, dataset, almacen):
    # Una línea da como mucho 3 fotogramas por pasada (3 pasadas): no alcanza para 30
    inventario = inv.construir(dataset, objetivo=30, progreso=lambda *_: None)
    pocas = LINEAS[:1]
    guiones, episodios = escribir_guiones(tmp_path, pocas)
    lineas = leer_lineas(guiones, leer_episodios(episodios), {PERSONAJE: ["Louie"]})
    recolector = Recolector(almacen, FrinkiacFalso(subtitulos_para(pocas)), Estado(tmp_path / "e.sqlite"),
                            inventario, lineas, tmp_path / "d", [PERSONAJE], Config(objetivo=30, margen=0))
    [ev] = recolector.vuelta()
    assert ev.agotado and ev.estado == POR_REVISAR  # sus pocas imágenes igual se revisan
    assert 0 < ev.subidas <= 9
    revisar(almacen, aprobadas=nombres(objetos(almacen, "staging/louie/")))
    [ev] = recolector.vuelta()
    assert ev.estado == AGOTADO and len(objetos(almacen, "final/louie/")) == ev.aprobadas > 0
