"""Carga de datos, partición, escenas compartidas y análisis exploratorio.

Etapas «Carga de datos» y «Análisis exploratorio» del pipeline. Nada de este módulo mira los píxeles: trabaja con
las rutas, las carpetas y los nombres de archivo, así que la partición no puede depender de nada calculado sobre las
imágenes.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from sklearn.model_selection import train_test_split

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Sufijos con los que el script de alteraciones marca las imágenes que genera: una alterada es una variante extra
# que va al mismo conjunto que su original; una parecida reemplaza a su original
MARCA_ALTERADA = "__alterada"
MARCA_PARECIDA = "__parecida"

# Fotogramas de episodio: <personaje>__<episodio>__t<milisegundos>. El resto viene del dataset de Kaggle
PATRON_FOTOGRAMA = re.compile(r"^(?P<nombre>.+?)__(?P<episodio>S\d+E\d+)__t(?P<ms>\d+)$", re.IGNORECASE)
VENTANA_ESCENA_MS = 3000
FUENTE_EPISODIO = "fotograma de episodio"
FUENTE_KAGGLE = "Kaggle"


# --- Índice y partición ---------------------------------------------------------------------------------------


def construir_indice(directorio):
    """Recorre un directorio con una subcarpeta por personaje y devuelve un DataFrame con la ruta de cada imagen,
    su personaje y, para las alteradas, la original de la que provienen.

    Si el script reemplazó una original casi idéntica a otra por su versión __parecida, la fila de esa original
    apunta a la versión parecida. La etiqueta sale siempre de la carpeta, nunca del nombre del archivo."""
    registros = []

    for carpeta_clase in sorted(Path(directorio).iterdir()):
        if not carpeta_clase.is_dir() or carpeta_clase.name.startswith("."):
            continue

        archivos = sorted(carpeta_clase.iterdir())
        nombres = {ruta.name for ruta in archivos}
        for ruta in archivos:
            # Las parecidas no son imágenes nuevas: entran abajo, en el lugar de su original
            if ruta.suffix.lower() not in EXTENSIONES_VALIDAS or ruta.stem.endswith(MARCA_PARECIDA):
                continue
            alterada = ruta.stem.endswith(MARCA_ALTERADA)
            # Para una original, su origen es ella misma
            origen = ruta.stem.removesuffix(MARCA_ALTERADA) + ruta.suffix if alterada else ruta.name
            reemplazo = f"{ruta.stem}{MARCA_PARECIDA}{ruta.suffix}"
            parecida = not alterada and reemplazo in nombres
            registros.append(
                {
                    "ruta": ruta.with_name(reemplazo) if parecida else ruta,
                    "personaje": carpeta_clase.name,
                    "alterada": alterada,
                    "parecida": parecida,
                    "origen": origen,
                }
            )

    if not registros:
        raise FileNotFoundError(f"No se encontraron imagenes en: {directorio}")

    return pd.DataFrame(registros)


def resumir_indice(indice):
    """Originales, alteradas y parecidas de cada personaje, de mayor a menor. Imprime los totales y avisa si falta
    ejecutar el script de alteraciones."""
    resumen = (
        indice.groupby("personaje")
        .agg(
            originales=("alterada", lambda serie: int((~serie).sum())),
            alteradas=("alterada", "sum"),
            parecidas=("parecida", "sum"),
        )
        .astype(int)
        .sort_values("originales", ascending=False)
        .reset_index()
    )
    resumen["porcentaje"] = (resumen["originales"] / resumen["originales"].sum() * 100).round(2)

    n_originales = int(resumen["originales"].sum())
    n_alteradas = int(resumen["alteradas"].sum())
    n_parecidas = int(resumen["parecidas"].sum())
    print(f"Dataset: {n_originales} imagenes originales y {n_alteradas} alteradas en {len(resumen)} personajes")
    print(f"De las originales, {n_parecidas} eran casi identicas a otra y entran con su version filtrada (__parecida)")
    if n_alteradas == 0:
        print(
            "AVISO: no hay imagenes alteradas. Ejecuta antes "
            "scripts/newDataset/generar_imagenes_alteradas.py; el notebook funciona sin ellas, "
            "pero el entrenamiento no tendra variantes de color y textura."
        )
    return resumen


def fuente_imagen(nombre_archivo):
    """De dónde viene una imagen según su nombre: un fotograma de episodio (`nombre__SxxEyy__tN.jpg`, sumado por el
    equipo con la herramienta de Frinkiac) o el dataset de Kaggle (`nombre_N.jpg`)."""
    return FUENTE_EPISODIO if leer_fotograma(nombre_archivo) is not None else FUENTE_KAGGLE


def repartir(indice, proporciones, semilla, minimo_por_personaje):
    """Reparte las originales de cada personaje en train, validación y test (estratificado y con semilla), sin
    tope por personaje, y manda cada alterada al conjunto de su original.

    Devuelve `(datos, originales, clases)`: `datos` tiene una fila por imagen de trabajo con su `conjunto`, su
    `etiqueta` (el índice de su carpeta en `clases`) y su `fuente` (Kaggle o fotograma de episodio). Los personajes
    con menos de `minimo_por_personaje` originales no se pueden repartir en tres conjuntos: se listan y se
    excluyen."""
    originales = indice[~indice["alterada"]].reset_index(drop=True)
    alteradas = indice[indice["alterada"]].reset_index(drop=True)

    # Personajes con muy pocas originales para repartirlos en tres conjuntos
    conteo_originales = originales["personaje"].value_counts()
    excluidos = sorted(conteo_originales[conteo_originales < minimo_por_personaje].index)
    if excluidos:
        print(f"Personajes excluidos (menos de {minimo_por_personaje} originales): {', '.join(excluidos)}")
        originales = originales[~originales["personaje"].isin(excluidos)].reset_index(drop=True)
        alteradas = alteradas[~alteradas["personaje"].isin(excluidos)].reset_index(drop=True)

    # Reparto estratificado de las originales: primero se separa train, y el resto se divide entre validación y test
    fraccion_resto = 1 - proporciones["train"]
    indices_train, indices_resto = train_test_split(
        originales.index,
        test_size=fraccion_resto,
        stratify=originales["personaje"],
        random_state=semilla,
    )
    indices_val, indices_test = train_test_split(
        indices_resto,
        test_size=proporciones["test"] / fraccion_resto,
        stratify=originales.loc[indices_resto, "personaje"],
        random_state=semilla,
    )
    originales["conjunto"] = ""
    originales.loc[indices_train, "conjunto"] = "train"
    originales.loc[indices_val, "conjunto"] = "val"
    originales.loc[indices_test, "conjunto"] = "test"

    # Cada alterada va al mismo conjunto que su original. Solo se descarta una alterada cuya original ya no está
    conjunto_de_original = dict(zip(zip(originales["personaje"], originales["origen"]), originales["conjunto"]))
    alteradas["conjunto"] = [
        conjunto_de_original.get((personaje, origen), "descartada")
        for personaje, origen in zip(alteradas["personaje"], alteradas["origen"])
    ]

    datos = (
        pd.concat([originales, alteradas], ignore_index=True)
        .query("conjunto != 'descartada'")
        .sort_values(["personaje", "origen", "alterada"])
        .reset_index(drop=True)
    )
    clases = sorted(datos["personaje"].unique())
    clase_a_indice = {clase: indice_clase for indice_clase, clase in enumerate(clases)}
    # Etiqueta numérica de cada imagen: la de su carpeta
    datos["etiqueta"] = datos["personaje"].map(clase_a_indice)
    datos["fuente"] = datos["origen"].map(fuente_imagen)
    return datos, originales, clases


def contar(datos, filtro):
    """Imágenes por personaje de las filas de `datos` que cumplen `filtro` (una máscara booleana)."""
    return datos[filtro].groupby("personaje").size()


def tabla_reparto(datos, originales, conjuntos):
    """Imágenes de cada personaje en cada conjunto, con sus alteradas. Imprime los totales, el desbalance y el
    reparto de las originales."""
    reparto = (
        pd.DataFrame(
            {
                "disponibles": originales.groupby("personaje").size(),
                "train_originales": contar(datos, (datos["conjunto"] == "train") & ~datos["alterada"]),
                "train_alteradas": contar(datos, (datos["conjunto"] == "train") & datos["alterada"]),
                "val": contar(datos, datos["conjunto"] == "val"),
                "val_alteradas": contar(datos, (datos["conjunto"] == "val") & datos["alterada"]),
                "test": contar(datos, datos["conjunto"] == "test"),
                "test_alteradas": contar(datos, (datos["conjunto"] == "test") & datos["alterada"]),
            }
        )
        .fillna(0)
        .astype(int)
    )
    reparto["train_total"] = reparto["train_originales"] + reparto["train_alteradas"]
    reparto = reparto.sort_values("disponibles", ascending=False)

    conteo_originales = originales["personaje"].value_counts()
    totales = datos["conjunto"].value_counts()
    alteradas_por_conjunto = datos[datos["alterada"]].groupby("conjunto").size()
    print(
        "   |   ".join(
            f"{nombre}: {totales[nombre]} imagenes ({int(alteradas_por_conjunto.get(nombre, 0))} alteradas)"
            for nombre in conjuntos
        )
        + f"   |   personajes: {datos['personaje'].nunique()}"
    )
    print(
        f"Desbalance en train: {reparto['train_total'].max() / reparto['train_total'].min():.1f}:1"
        f"   (en el dataset completo: {conteo_originales.max() / conteo_originales.min():.1f}:1)"
    )
    parecidas_por_conjunto = datos[datos["parecida"]].groupby("conjunto").size()
    print(
        "Imagenes parecidas (version filtrada en lugar de la original): "
        + "   |   ".join(f"{nombre} {int(parecidas_por_conjunto.get(nombre, 0))}" for nombre in conjuntos)
    )
    n_por_conjunto = originales["conjunto"].value_counts()
    print(
        "Reparto de las originales: "
        + "   |   ".join(f"{nombre} {n_por_conjunto[nombre] / len(originales):.1%}" for nombre in conjuntos)
    )
    return reparto


def por_original(datos, mascaras, columna, conjuntos):
    """Cuántas imágenes marcadas en `columna` (alteradas o parecidas) hay por cada original en cada conjunto. Si
    los tres conjuntos tienen las mismas condiciones, la proporción es la misma en los tres."""
    es_alterada = datos["alterada"].to_numpy()
    return {
        nombre: float(datos.loc[mascaras[nombre], columna].sum()) / max(int((mascaras[nombre] & ~es_alterada).sum()), 1)
        for nombre in conjuntos
    }


def comprobar_particion(datos, originales, mascaras, clases, proporciones, conjuntos, indices, onehot, entradas,
                        fila_entrada, tolerancia=0.03):
    """Comprueba la partición y muestra la proporción de cada personaje en el dataset, validación y test.

    - `mascaras`, `indices`, `onehot` y `entradas` son diccionarios por conjunto: la máscara de filas de `datos`,
      las etiquetas como índice, las etiquetas one-hot y la entrada del modelo;
    - `fila_entrada` es la comprobación de la forma de la entrada, propia de cada modelo:
      `(comprobación, obtenido, esperado, correcto)`.

    Lanza un error si una imagen y su variante quedan en conjuntos distintos, si las condiciones (alteradas y
    parecidas por original) difieren más de `tolerancia` entre conjuntos o si la estratificación falla."""
    from .preprocesamiento import verificar

    numero_clases = len(clases)
    proporcion_dataset = originales["personaje"].value_counts(normalize=True)
    tabla = pd.DataFrame(
        {
            "personaje": clases,
            "dataset_%": [round(float(proporcion_dataset[clase]) * 100, 2) for clase in clases],
            "val_%": [round((indices["val"] == i).mean() * 100, 2) for i in range(numero_clases)],
            "test_%": [round((indices["test"] == i).mean() * 100, 2) for i in range(numero_clases)],
            "n_train": [int((indices["train"] == i).sum()) for i in range(numero_clases)],
            "n_val": [int((indices["val"] == i).sum()) for i in range(numero_clases)],
            "n_test": [int((indices["test"] == i).sum()) for i in range(numero_clases)],
        }
    )
    tabla["desviacion_%"] = tabla[["val_%", "test_%"]].sub(tabla["dataset_%"], axis=0).abs().max(axis=1).round(2)
    display(tabla)

    # Una imagen y sus variantes se identifican por (personaje, original de la que salen)
    origenes = {
        nombre: set(zip(datos.loc[mascaras[nombre], "personaje"], datos.loc[mascaras[nombre], "origen"]))
        for nombre in conjuntos
    }
    compartidas = (
        len(origenes["train"] & origenes["val"])
        + len(origenes["train"] & origenes["test"])
        + len(origenes["val"] & origenes["test"])
    )
    es_alterada = datos["alterada"].to_numpy()
    proporciones_obtenidas = {
        nombre: (mascaras[nombre] & ~es_alterada).sum() / (~es_alterada).sum() for nombre in conjuntos
    }
    clases_por_conjunto = [len(np.unique(indices[nombre])) for nombre in conjuntos]

    verificar(
        [
            (
                "Cada imagen en un solo conjunto",
                int(sum(mascara.sum() for mascara in mascaras.values())),
                len(datos),
                int(sum(mascara.sum() for mascara in mascaras.values())) == len(datos),
            ),
            ("Sin imágenes compartidas entre conjuntos", compartidas, 0, compartidas == 0),
            *[
                (
                    f"{nombre} por original en train / val / test",
                    " / ".join(f"{proporciones_tipo[conjunto]:.1%}" for conjunto in conjuntos),
                    f"iguales (± {tolerancia:.0%})",
                    all(abs(proporciones_tipo[conjunto] - proporciones_tipo["train"]) <= tolerancia
                        for conjunto in conjuntos),
                )
                for nombre, proporciones_tipo in (
                    ("Alteradas", por_original(datos, mascaras, "alterada", conjuntos)),
                    ("Parecidas", por_original(datos, mascaras, "parecida", conjuntos)),
                )
            ],
            (
                "Todas las originales repartidas (sin tope)",
                int((~es_alterada).sum()),
                len(originales),
                int((~es_alterada).sum()) == len(originales),
            ),
            (
                "Proporciones train / val / test de las originales",
                " / ".join(f"{proporciones_obtenidas[nombre]:.1%}" for nombre in conjuntos),
                " / ".join(f"{proporciones[nombre]:.0%}" for nombre in conjuntos),
                all(abs(proporciones_obtenidas[nombre] - proporciones[nombre]) < 0.005 for nombre in conjuntos),
            ),
            fila_entrada,
            *[
                (f"X e y alineados en {nombre}", len(onehot[nombre]), len(entradas[nombre]),
                 len(onehot[nombre]) == len(entradas[nombre]))
                for nombre in conjuntos
            ],
            (
                "One-hot coherente con índices",
                "y_val.argmax(axis=1)",
                "y_val_idx",
                bool(np.array_equal(np.argmax(onehot["val"], axis=1), indices["val"])),
            ),
            (
                "Todas las clases en train / val / test",
                " / ".join(str(n) for n in clases_por_conjunto),
                numero_clases,
                all(n == numero_clases for n in clases_por_conjunto),
            ),
            (
                "Estratificación de val y test (desviación máxima)",
                f"{tabla['desviacion_%'].max():.2f} %",
                "< 1 %",
                bool(tabla["desviacion_%"].max() < 1.0),
            ),
        ],
        "La partición",
    )
    return tabla


# --- Escenas con varios personajes ---------------------------------------------------------------------------


def leer_fotograma(nombre_archivo):
    """(personaje del nombre, episodio, ms) de un fotograma de episodio, o None si no lo es."""
    coincidencia = PATRON_FOTOGRAMA.match(Path(nombre_archivo).stem)
    if coincidencia is None:
        return None
    return coincidencia["nombre"], coincidencia["episodio"].upper(), int(coincidencia["ms"])


def construir_lineas_tiempo(tabla):
    """Fotogramas de cada episodio ordenados por instante: {episodio: (instantes, personajes)}."""
    por_episodio = {}
    for personaje, origen in zip(tabla["personaje"], tabla["origen"]):
        fotograma = leer_fotograma(origen)
        if fotograma is not None:
            por_episodio.setdefault(fotograma[1], []).append((fotograma[2], personaje))
    lineas = {}
    for episodio, fotogramas in por_episodio.items():
        fotogramas.sort()
        lineas[episodio] = (np.array([ms for ms, _ in fotogramas]), [p for _, p in fotogramas])
    return lineas


def personajes_en_escena(personaje, origen, lineas_tiempo, clases_validas, ventana_ms=VENTANA_ESCENA_MS):
    """Personajes que probablemente acompañan al de la carpeta en esa imagen (sin incluirlo): el del nombre del
    fotograma, si es otro, y los de los fotogramas del mismo episodio a menos de `ventana_ms`."""
    fotograma = leer_fotograma(origen)
    if fotograma is None:
        return frozenset()
    nombre, episodio, ms = fotograma
    acompanantes = {nombre} if nombre != personaje and nombre in clases_validas else set()
    if episodio in lineas_tiempo:
        instantes, personajes = lineas_tiempo[episodio]
        desde = np.searchsorted(instantes, ms - ventana_ms, side="left")
        hasta = np.searchsorted(instantes, ms + ventana_ms, side="right")
        acompanantes.update(otro for otro in personajes[desde:hasta] if otro != personaje)
    return frozenset(acompanantes)


def leer_gemelas(dataset_base, clases_validas):
    """Gemelas entre personajes: la misma escena guardada en la carpeta de otro personaje, según el registro
    `_gemelas.csv` del script de alteraciones. Devuelve `({(personaje, original): {(otro, su_original)}}, pares)`."""
    registro_gemelas = Path(dataset_base) / "_gemelas.csv"
    gemelas = {}
    if not registro_gemelas.exists():
        print("No hay registro de gemelas entre personajes: ejecuta antes el script de alteraciones.")
        return gemelas, 0
    pares_gemelas = pd.read_csv(registro_gemelas)
    pares_gemelas = pares_gemelas[
        pares_gemelas["personaje"].isin(clases_validas) & pares_gemelas["otro_personaje"].isin(clases_validas)
    ]
    for fila in pares_gemelas.itertuples():
        gemelas.setdefault((fila.personaje, fila.original), set()).add((fila.otro_personaje, fila.otro_original))
        gemelas.setdefault((fila.otro_personaje, fila.otro_original), set()).add((fila.personaje, fila.original))
    return gemelas, len(pares_gemelas)


def personajes_gemelos(personaje, origen, gemelas):
    """Personajes en cuya carpeta hay una imagen casi idéntica a esta."""
    return frozenset(otro for otro, _ in gemelas.get((personaje, origen), ()))


def anotar_escenas(datos, indice, dataset_base, clases_validas, conjuntos):
    """Agrega a `datos` la columna `escena`: los otros personajes que probablemente aparecen en cada imagen, por la
    escena del episodio (±3 s) y por las gemelas en la carpeta de otro personaje.

    La escena no cambia el acierto: solo alimenta la métrica de referencia. Muestra el resumen por conjunto y las
    parejas que más comparten escena, y devuelve `(escena_por_imagen, pares_gemelas)`."""
    gemelas, n_pares_gemelas = leer_gemelas(dataset_base, clases_validas)
    # Línea de tiempo con todos los fotogramas originales del dataset
    originales_indexados = indice[~indice["alterada"] & indice["personaje"].isin(clases_validas)]
    lineas_tiempo = construir_lineas_tiempo(originales_indexados)
    escena = [
        personajes_en_escena(p, o, lineas_tiempo, clases_validas) | personajes_gemelos(p, o, gemelas)
        for p, o in zip(datos["personaje"], datos["origen"])
    ]
    datos["escena"] = [", ".join(sorted(acompanantes)) for acompanantes in escena]

    es_fotograma = np.array([leer_fotograma(origen) is not None for origen in datos["origen"]])
    con_gemela = np.array(
        [bool(personajes_gemelos(p, o, gemelas)) for p, o in zip(datos["personaje"], datos["origen"])]
    )
    con_escena = np.array([bool(acompanantes) for acompanantes in escena])

    display(
        pd.DataFrame(
            [
                (
                    nombre,
                    int(mascara.sum()),
                    int(es_fotograma[mascara].sum()),
                    int(con_gemela[mascara].sum()),
                    int(con_escena[mascara].sum()),
                )
                for nombre in conjuntos
                for mascara in [(datos["conjunto"] == nombre).to_numpy()]
            ],
            columns=["conjunto", "imagenes", "fotogramas", "con_gemela", "escena_compartida"],
        )
    )
    # Parejas de personajes que más comparten escena (solo originales)
    display(
        pd.DataFrame(
            [
                (personaje, otro)
                for personaje, acompanantes, alterada in zip(datos["personaje"], escena, datos["alterada"])
                if not alterada
                for otro in acompanantes
            ],
            columns=["personaje", "acompanante"],
        )
        .value_counts()
        .head(10)
        .reset_index(name="imagenes")
    )

    if not all(p not in a for p, a in zip(datos["personaje"], escena)):
        raise ValueError("Alguna imagen aparece acompañada por el personaje de su propia carpeta.")

    print(
        f"Fotogramas de episodio: {int(es_fotograma.sum())} de {len(datos)} imagenes"
        f"   |   con gemela en otra carpeta: {int(con_gemela.sum())} ({n_pares_gemelas} pares)"
        f"   |   con escena compartida: {int(con_escena.sum())}"
    )
    return escena, n_pares_gemelas


# --- Análisis exploratorio -----------------------------------------------------------------------------------


def medir_dimensiones(datos):
    """Agrega a `datos` el ancho y el alto originales de cada imagen (sin cargar los píxeles) y devuelve su
    resumen estadístico."""
    dimensiones = []
    for ruta in datos["ruta"]:
        with Image.open(ruta) as img:
            dimensiones.append(img.size)  # (ancho, alto)

    datos["ancho"] = [ancho for ancho, _ in dimensiones]
    datos["alto"] = [alto for _, alto in dimensiones]

    resumen_dimensiones = datos[["ancho", "alto"]].describe().T
    resumen_dimensiones["mediana"] = datos[["ancho", "alto"]].median()
    return resumen_dimensiones[["count", "min", "25%", "mediana", "75%", "max", "mean", "std"]].round(1)


def analizar_dataset(datos, clases, conjuntos, semilla, ejemplos=12):
    """Análisis exploratorio del dataset de trabajo, en cuatro partes:

    1. cuántas imágenes tiene cada personaje, separadas por fuente (Kaggle o fotograma de episodio);
    2. la fuente en cada conjunto, porque es un posible sesgo que audita «Auditoría del modelo»;
    3. las dimensiones originales de las imágenes;
    4. una muestra de imágenes al azar.

    Devuelve un diccionario con las tablas: `por_personaje`, `por_fuente` y `dimensiones`."""
    originales = datos[~datos["alterada"]]

    # 1. Imágenes originales por personaje y por fuente
    por_personaje = (
        originales.groupby(["personaje", "fuente"]).size().unstack(fill_value=0).reindex(clases, fill_value=0)
    )
    for fuente in (FUENTE_KAGGLE, FUENTE_EPISODIO):
        if fuente not in por_personaje:
            por_personaje[fuente] = 0
    por_personaje = por_personaje[[FUENTE_KAGGLE, FUENTE_EPISODIO]]
    por_personaje["total"] = por_personaje.sum(axis=1)
    por_personaje["% episodio"] = (por_personaje[FUENTE_EPISODIO] / por_personaje["total"] * 100).round(1)
    por_personaje = por_personaje.sort_values("total")

    figura, eje = plt.subplots(figsize=(10, 14))
    eje.barh(por_personaje.index, por_personaje[FUENTE_KAGGLE], color="#8fb8de", label=FUENTE_KAGGLE)
    eje.barh(
        por_personaje.index,
        por_personaje[FUENTE_EPISODIO],
        left=por_personaje[FUENTE_KAGGLE],
        color="#d95f02",
        label=FUENTE_EPISODIO,
    )
    eje.axvline(por_personaje["total"].mean(), color="gray", linestyle="--", linewidth=1,
                label=f"media = {por_personaje['total'].mean():.0f}")
    eje.set_xlabel("Imágenes originales")
    eje.set_title("Imágenes originales por personaje y por fuente", fontsize=11)
    eje.tick_params(axis="y", labelsize=7)
    eje.legend()
    eje.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()

    # 2. La fuente en cada conjunto: si una fuente se concentrara en test, las métricas medirían otra cosa
    por_fuente = pd.crosstab(datos["conjunto"], datos["fuente"]).reindex(list(conjuntos))
    por_fuente["% episodio"] = (por_fuente.get(FUENTE_EPISODIO, 0) / por_fuente.sum(axis=1) * 100).round(1)
    display(por_fuente)

    # 3. Dimensiones originales
    dimensiones = medir_dimensiones(datos)
    display(dimensiones)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4))
    for eje, fuente in zip(ejes, (FUENTE_KAGGLE, FUENTE_EPISODIO)):
        seleccion = datos[(datos["fuente"] == fuente) & ~datos["alterada"]]
        eje.scatter(seleccion["ancho"], seleccion["alto"], s=4, alpha=0.3)
        eje.set_xscale("log")
        eje.set_yscale("log")
        eje.set_xlabel("ancho (px)")
        eje.set_ylabel("alto (px)")
        eje.set_title(f"{fuente}: {len(seleccion):,} originales", fontsize=10)
        eje.grid(alpha=0.3)
    figura.suptitle("Dimensiones originales de las imágenes, por fuente", fontsize=11)
    plt.tight_layout()
    plt.show()

    # 4. Muestra al azar
    generador = np.random.default_rng(semilla)
    muestra = generador.choice(originales.index.to_numpy(), size=min(ejemplos, len(originales)), replace=False)
    columnas = 6
    filas = int(np.ceil(len(muestra) / columnas))
    figura, ejes = plt.subplots(filas, columnas, figsize=(2.3 * columnas, 2.7 * filas), layout="constrained")
    ejes = np.ravel(ejes)
    for eje, indice in zip(ejes, muestra):
        fila = datos.loc[indice]
        with Image.open(fila["ruta"]) as img:
            eje.imshow(img.convert("RGB"))
        eje.set_xticks([])
        eje.set_yticks([])
        eje.set_xlabel(f"{fila['personaje'].replace('_', ' ')}\n{fila['fuente']}", fontsize=7)
    for eje in ejes[len(muestra):]:
        eje.axis("off")
    figura.suptitle("Muestra al azar de imágenes originales", fontsize=12, fontweight="bold")
    plt.show()

    fraccion_episodio = (originales["fuente"] == FUENTE_EPISODIO).mean()
    print(
        f"Originales: {len(originales):,}   |   fotogramas de episodio: {fraccion_episodio:.1%}"
        f"   |   personajes: {len(clases)}"
        f"   |   de {int(por_personaje['total'].min())} a {int(por_personaje['total'].max())} por personaje"
    )
    return {"por_personaje": por_personaje, "por_fuente": por_fuente, "dimensiones": dimensiones}
