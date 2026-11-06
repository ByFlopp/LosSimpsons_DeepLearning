"""El flujo completo de `pnpm frinkiac` en modo local, como lo usa el equipo: recolectar, revisar a
mano, volver a ejecutar, completar y exportar. Frinkiac se reemplaza por el doble; no hay red."""

import json
import zipfile

import pytest
import responses

import ampliar_con_frinkiac
import guiones as G
from ampliar_con_frinkiac import main
from conftest import FrinkiacFalso, escribir_guiones, imagen_ruido, subtitulos_para

LINEAS = [(i, "Louie", 10_000 * i, f"linea{i} dice algo") for i in range(1, 8)]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    # Misma distribución que en el PC: el dataset y la ampliación lado a lado
    raiz = tmp_path / "simpsons_dataset"
    dataset = raiz / "simpsons_dataset"
    for personaje, cantidad in (("louie", 3), ("homer_simpson", 9)):
        (dataset / personaje).mkdir(parents=True)
        for i in range(cantidad):
            (dataset / personaje / f"{personaje}_{i:04d}.jpg").write_bytes(imagen_ruido(hash(personaje) % 999 + i))
    (dataset / "louie" / "louie_0000__alterada.jpg").write_bytes(imagen_ruido(4242))  # sintética: no cuenta
    lineas, episodios = escribir_guiones(tmp_path, LINEAS)
    alias = tmp_path / "alias.json"
    alias.write_text(json.dumps({"louie": ["Louie"]}))
    monkeypatch.setattr(ampliar_con_frinkiac, "ClienteFrinkiac", lambda **_: FrinkiacFalso(subtitulos_para(LINEAS)))
    ampliacion = raiz / "frinkiac_ampliacion"
    comando = ["flujo", "--local", str(ampliacion), "--dataset", str(dataset), "--guiones", str(lineas),
               "--episodios", str(episodios), "--alias", str(alias), "--objetivo", "9", "--margen", "0",
               "--sin-issue"]
    return comando, dataset, ampliacion


def imagenes(carpeta):
    return sorted(p.name for p in carpeta.glob("*.jpg"))


def test_flujo_completo_con_revision_manual(entorno, capsys):
    comando, dataset, ampliacion = entorno
    revision = ampliacion / "por_revisar" / "louie"
    antes = imagenes(dataset / "louie")

    # 1.ª ejecución: recolecta lo que falta (9 - 3 originales = 6) y lo deja listo para revisar
    assert main(comando) == 0
    assert (ampliacion / "LEEME.txt").exists()
    assert len(imagenes(ampliacion / "staging" / "louie")) == 6
    primeras = imagenes(revision)
    assert len(primeras) == 6 and (revision / "_PENDIENTE.txt").exists()
    assert "@DiegoPyLL" in capsys.readouterr().out

    # Revisión a medias (sigue el _PENDIENTE.txt): no se aplica nada aunque falten imágenes
    (revision / primeras[0]).unlink()
    assert main(comando) == 0
    assert not (ampliacion / "revision" / "louie.json").exists()

    # Revisión terminada: se rechazan 2 y se borra la marca
    (revision / primeras[1]).unlink()
    (revision / "_PENDIENTE.txt").unlink()
    assert main(comando) == 0
    decision = json.loads((ampliacion / "revision" / "louie.json").read_text(encoding="utf-8"))
    assert sorted(decision["rechazadas"]) == primeras[:2] and len(decision["aprobadas"]) == 4
    # se repusieron las 2 rechazadas y solo esas quedan para revisar
    assert len(imagenes(ampliacion / "staging" / "louie")) == 8
    repuestas = imagenes(revision)
    assert len(repuestas) == 2 and not set(repuestas) & set(primeras)

    # Se aprueban las repuestas: el personaje queda completo en final/ y el dataset sigue intacto
    (revision / "_PENDIENTE.txt").unlink()
    assert main(comando) == 0
    final = imagenes(ampliacion / "final" / "louie")
    assert len(final) == 6 and not set(final) & set(primeras[:2])
    assert imagenes(dataset / "louie") == antes
    assert not revision.exists()

    # Exportar copia al dataset solo las aprobadas, sin tocar lo que ya estaba
    exportar = ["exportar", "--local", str(ampliacion), "--dataset", str(dataset)]
    assert main(exportar) == 0
    assert imagenes(dataset / "louie") == sorted(antes + final)

    # Volver a ejecutar todo después de exportar no cuenta dos veces las exportadas ni trae más
    assert main(comando) == 0
    assert len(imagenes(ampliacion / "staging" / "louie")) == 8
    assert imagenes(ampliacion / "final" / "louie") == final


def test_flujo_rechaza_almacen_dentro_del_dataset(entorno):
    comando, dataset, _ = entorno
    comando = [*comando]
    comando[comando.index("--local") + 1] = str(dataset / "frinkiac")
    assert main(comando) == 2


# --- Guiones -----------------------------------------------------------------------------------------------

def test_guiones_se_encuentran_dentro_de_un_zip_de_kaggle(tmp_path):
    lineas, episodios = escribir_guiones(tmp_path, LINEAS)
    descargas = tmp_path / "Downloads"
    descargas.mkdir()
    with zipfile.ZipFile(descargas / "archive.zip", "w") as zip_:
        zip_.write(lineas, G.LINEAS)
        zip_.write(episodios, G.EPISODIOS)
        zip_.writestr("simpsons_characters.csv", "id,name\n")
    obtenidos = G.obtener(tmp_path / "copia", [descargas])
    assert obtenidos.completo and "zip" in obtenidos.fuente
    assert obtenidos.lineas.read_bytes() == lineas.read_bytes()


@responses.activate
def test_sin_guiones_ni_kaggle_usa_tidytuesday_y_avisa(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "sin_kaggle"))
    lineas, episodios = escribir_guiones(tmp_path, LINEAS)
    responses.get(G.URL_TIDYTUESDAY + G.LINEAS, body=lineas.read_bytes())
    responses.get(G.URL_TIDYTUESDAY + G.EPISODIOS, body=episodios.read_bytes())
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    obtenidos = G.obtener(tmp_path / "copia", [vacia])
    assert not obtenidos.completo and "TidyTuesday" in obtenidos.fuente
    # si después aparece el guion completo, se prefiere ese
    (vacia / G.LINEAS).write_bytes(lineas.read_bytes())
    (vacia / G.EPISODIOS).write_bytes(episodios.read_bytes())
    assert G.obtener(tmp_path / "copia", [vacia]).completo
