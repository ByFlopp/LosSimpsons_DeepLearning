"""Renombra las imágenes originales del dataset a <personaje>_<número>.

Uso:
    .venv-gpu/Scripts/python.exe scripts/newDataset/renombrar_imagenes.py              vista previa: no toca nada
    .venv-gpu/Scripts/python.exe scripts/newDataset/renombrar_imagenes.py --aplicar
    .venv-gpu/Scripts/python.exe scripts/newDataset/generar_imagenes_alteradas.py      después: regenera las variantes

El dataset mezcla nombres de Kaggle (bart_simpson_0001.jpg), de Frinkiac (apu_nahasapeemapetilon__S21E21__t0659367.jpg,
a veces en la carpeta de otro personaje) y otros (videoframe_102047.png, capturas de pantalla). Este script deja cada
original con el nombre de su carpeta y un número:

    homer_simpson/homer_simpson_0001.jpg                          ya lo cumple: no cambia
    homer_simpson/apu_nahasapeemapetilon__S21E21__t0659367.jpg -> homer_simpson/homer_simpson_2294.jpg

Las que ya tienen el formato conservan su número; si dos lo comparten (la misma imagen en .jpg y en .png), la primera
en orden alfabético lo conserva. Las demás, en orden alfabético, siguen desde el número más alto de la carpeta, con al
menos 4 dígitos. Volver a ejecutarlo solo numera las nuevas (por ejemplo, después de exportar de Frinkiac).

Con --aplicar:
1. Borra las variantes __alterada y __parecida, sus registros y la carpeta de revisión, igual que
   generar_imagenes_alteradas.py --limpiar. Ese script las vuelve a generar, ya con el nombre nuevo de su original.
2. Renombra las originales y anota cada cambio en _renombres.csv (personaje, antiguo, nuevo) antes de hacerlo: si se
   interrumpe, al volver a ejecutarlo termina los pendientes. El nombre antiguo guarda el episodio y el instante de
   los fotogramas, que leen la métrica de escenas y la recolección de Frinkiac.
3. Actualiza el nombre de cada imagen en _cajas_detecciones.jsonl, si existe, y deja la versión anterior en
   _cajas_detecciones.jsonl.antes_de_renombrar.

La ruta del dataset sale de --dataset, de la variable SIMPSONS_DATASET o, si no hay ninguna, de
~/Desktop/simpsons_dataset; la de revisión, de --revision o gemelas_por_revisar junto al dataset.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import generar_imagenes_alteradas as G  # noqa: E402

REGISTRO_RENOMBRES = "_renombres.csv"
CAMPOS_RENOMBRES = ("personaje", "antiguo", "nuevo")
REGISTRO_CAJAS = "_cajas_detecciones.jsonl"
SUFIJO_RESPALDO = ".antes_de_renombrar"
DIGITOS = 4


# --- Registro de renombres ---------------------------------------------------------------------------------------

def leer_renombres(dataset: Path) -> list[dict]:
    """Filas de _renombres.csv (personaje, antiguo, nuevo), o ninguna si no existe."""
    ruta = Path(dataset) / REGISTRO_RENOMBRES
    if not ruta.is_file():
        return []
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def leer_nombres_anteriores(dataset: Path) -> dict[str, str]:
    """{nombre nuevo: nombre antiguo}. El nuevo empieza por su carpeta, así que es único en todo el dataset."""
    return {fila["nuevo"]: fila["antiguo"] for fila in leer_renombres(dataset)}


def completar_pendientes(dataset: Path) -> int:
    """Termina los renombres anotados que una ejecución interrumpida no llegó a hacer."""
    hechos = 0
    for fila in leer_renombres(dataset):
        antiguo = dataset / fila["personaje"] / fila["antiguo"]
        nuevo = dataset / fila["personaje"] / fila["nuevo"]
        if antiguo.is_file() and not nuevo.exists():
            os.rename(antiguo, nuevo)
            hechos += 1
    return hechos


# --- Plan ----------------------------------------------------------------------------------------------------------

def planificar(carpeta: Path) -> tuple[int, list[tuple[Path, Path]]]:
    """(originales que conservan su nombre, [(original, destino)]) de una carpeta de personaje."""
    con_numero, sin_formato = G.numeradas(carpeta)
    siguiente = max(con_numero.values(), default=0)
    cambios = []
    for ruta in sin_formato:
        siguiente += 1
        cambios.append((ruta, carpeta / f"{carpeta.name}_{siguiente:0{DIGITOS}d}{ruta.suffix.lower()}"))
    return len(con_numero), cambios


def variantes_de(carpetas: list[Path]) -> int:
    return sum(1 for carpeta in carpetas for ruta in carpeta.iterdir() if ruta.is_file() and G.es_generado(ruta))


def mostrar_plan(planes: dict[str, tuple[int, list]], variantes: int, ejemplos: int) -> None:
    ancho = max(len(nombre) for nombre in planes)
    print(f"\n{'personaje':<{ancho}}  originales  conservan  se_renombran")
    for nombre, (conservan, cambios) in planes.items():
        print(f"{nombre:<{ancho}}  {conservan + len(cambios):>10}  {conservan:>9}  {len(cambios):>12}")
        for origen, destino in cambios[:ejemplos]:
            print(f"    {origen.name}  ->  {destino.name}")
    conservan = sum(c for c, _ in planes.values())
    renombres = sum(len(cambios) for _, cambios in planes.values())
    print(f"\nTotal: {conservan + renombres} originales, {conservan} conservan su nombre y {renombres} se renombran. "
          f"Se borran {variantes} variantes __alterada/__parecida, que se regeneran después con los nombres nuevos.")


# --- Aplicar -------------------------------------------------------------------------------------------------------

def renombrar(dataset: Path, planes: dict[str, tuple[int, list]]) -> int:
    """Anota cada cambio en _renombres.csv y luego lo hace: una interrupción deja, como mucho, renombres anotados
    sin hacer, que completar_pendientes() termina."""
    ruta = dataset / REGISTRO_RENOMBRES
    nuevo_registro = not ruta.is_file() or ruta.stat().st_size == 0
    hechos = 0
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS_RENOMBRES, lineterminator="\n")
        if nuevo_registro:
            escritor.writeheader()
        for nombre, (_, cambios) in planes.items():
            for origen, destino in cambios:
                if destino.exists():
                    sys.exit(f"{destino} ya existe: no se renombra {origen}")
                escritor.writerow({"personaje": nombre, "antiguo": origen.name, "nuevo": destino.name})
                archivo.flush()
                os.rename(origen, destino)
                hechos += 1
    return hechos


def actualizar_cajas(dataset: Path) -> int:
    """Cambia `archivo` en _cajas_detecciones.jsonl según _renombres.csv. Devuelve cuántas líneas cambió."""
    ruta = dataset / REGISTRO_CAJAS
    if not ruta.is_file():
        return 0
    mapa = {(fila["personaje"], fila["antiguo"]): fila["nuevo"] for fila in leer_renombres(dataset)}
    lineas, cambiadas = [], 0
    with open(ruta, newline="", encoding="utf-8") as archivo:
        for linea in archivo:
            contenido = linea.rstrip("\r\n")
            final = linea[len(contenido):]
            if contenido.strip():
                registro = json.loads(contenido)
                nuevo = mapa.get((registro.get("personaje"), registro.get("archivo")))
                if nuevo is not None:
                    registro["archivo"] = nuevo
                    contenido = json.dumps(registro, ensure_ascii=False)
                    cambiadas += 1
            lineas.append(contenido + final)
    if cambiadas:
        respaldo = ruta.with_name(ruta.name + SUFIJO_RESPALDO)
        if not respaldo.exists():
            shutil.copy2(ruta, respaldo)
        temporal = ruta.with_name(ruta.name + G.SUFIJO_TEMPORAL)
        with open(temporal, "w", newline="", encoding="utf-8") as archivo:
            archivo.writelines(lineas)
        os.replace(temporal, ruta)
    return cambiadas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=None, help="carpeta con una subcarpeta por personaje")
    parser.add_argument("--revision", type=Path, default=None,
                        help=f"carpeta de revisión de las gemelas (por defecto, {G.NOMBRE_REVISION} junto al dataset)")
    parser.add_argument("--aplicar", action="store_true",
                        help="borra las variantes y renombra; sin esta opción solo muestra la vista previa")
    parser.add_argument("--ejemplos", type=int, default=2, help="cambios de ejemplo por personaje en la vista previa")
    args = parser.parse_args()

    dataset = args.dataset or Path(os.environ.get("SIMPSONS_DATASET", G.RUTA_POR_DEFECTO))
    dataset = dataset.expanduser().resolve()
    revision = (args.revision or dataset.parent / G.NOMBRE_REVISION).expanduser().resolve()
    if revision == dataset or revision.is_relative_to(dataset):
        sys.exit("--revision debe quedar fuera del dataset: cada subcarpeta del dataset es un personaje")

    carpetas = G.validar(dataset)
    print(f"Dataset: {dataset}  ({len(carpetas)} personajes)")

    if not args.aplicar:
        planes = {carpeta.name: planificar(carpeta) for carpeta in carpetas}
        mostrar_plan(planes, variantes_de(carpetas), args.ejemplos)
        print("\nVista previa: no se cambió nada. Para aplicarlo, agrega --aplicar.")
        return

    pendientes = completar_pendientes(dataset)
    if pendientes:
        print(f"Completados {pendientes} renombres de una ejecución interrumpida")
    borrados = G.limpiar(dataset)
    borrados_revision = G.limpiar_revision(revision)
    print(f"Borradas {borrados} variantes y {borrados_revision} copias de la carpeta de revisión")

    planes = {carpeta.name: planificar(carpeta) for carpeta in carpetas}
    hechos = renombrar(dataset, planes)
    cajas = actualizar_cajas(dataset)
    restantes = G.originales_sin_formato(carpetas)
    print(f"Renombradas {hechos} originales (registro: {dataset / REGISTRO_RENOMBRES})")
    if cajas:
        print(f"Actualizadas {cajas} líneas de {REGISTRO_CAJAS}")
    if restantes:
        sys.exit(f"Quedaron {len(restantes)} originales sin el formato, por ejemplo {restantes[0]}")
    print("Ahora ejecuta scripts/newDataset/generar_imagenes_alteradas.py para regenerar las variantes.")


if __name__ == "__main__":
    main()
