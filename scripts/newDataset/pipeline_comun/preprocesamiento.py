"""Preprocesamiento: redimensionado, normalización, codificación de las etiquetas y pesos por clase.

Etapa «Preprocesamiento» del pipeline. Cada paso tiene su función de comprobación, que muestra una tabla
`Comprobacion / Obtenido / Esperado / Correcto` y detiene el notebook si algo no se cumple.
"""

import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

VALOR_MAXIMO_PIXEL = 255.0
COLUMNAS_COMPROBACION = ["Comprobacion", "Obtenido", "Esperado", "Correcto"]


def verificar(filas, que):
    """Muestra una tabla de comprobaciones `(comprobación, obtenido, esperado, correcto)` y lanza un error si
    alguna no se cumple. `que` nombra lo que se comprueba, para el mensaje."""
    tabla = pd.DataFrame(filas, columns=COLUMNAS_COMPROBACION)
    display(tabla)
    if not tabla["Correcto"].all():
        raise ValueError(f"{que} no cumple alguna de las comprobaciones.")
    print(f"{que}: verificado correctamente.")
    return tabla


def marco(eje, color="#999999", grosor=1.2):
    """Quita las marcas de los ejes de una imagen y le pone un marco de color."""
    eje.set_xticks([])
    eje.set_yticks([])
    for lado in eje.spines.values():
        lado.set_edgecolor(color)
        lado.set_linewidth(grosor)


# --- Redimensionado ------------------------------------------------------------------------------------------


def cargar_y_redimensionar(ruta, tamano):
    """Abre una imagen, la convierte a RGB y la redimensiona completa al tamaño fijo `(ancho, alto)`."""
    with Image.open(ruta) as img:
        return np.asarray(img.convert("RGB").resize(tamano, Image.Resampling.BILINEAR), dtype=np.uint8)


def cargar_imagenes(rutas, tamano):
    """Matriz `(n_imagenes, alto, ancho, 3)` en uint8 con cada imagen completa y redimensionada. Decodificar y
    redimensionar libera el GIL, así que varios hilos cargan en paralelo. Devuelve `(imagenes, segundos)`."""
    inicio = time.perf_counter()
    imagenes = np.stack(
        Parallel(n_jobs=-1, prefer="threads")(delayed(cargar_y_redimensionar)(ruta, tamano) for ruta in rutas)
    )
    return imagenes, time.perf_counter() - inicio


def comprobar_redimensionado(imagenes, etiquetas, alto, ancho):
    """Forma, tipo y rango de la matriz de imágenes redimensionadas."""
    return verificar(
        [
            ("Número de imágenes", imagenes.shape[0], len(etiquetas), imagenes.shape[0] == len(etiquetas)),
            ("Alto", imagenes.shape[1], alto, imagenes.shape[1] == alto),
            ("Ancho", imagenes.shape[2], ancho, imagenes.shape[2] == ancho),
            ("Canales", imagenes.shape[3], 3, imagenes.shape[3] == 3),
            ("Tipo de dato", str(imagenes.dtype), "uint8", imagenes.dtype == np.uint8),
            ("Valor mínimo", int(imagenes.min()), ">= 0", imagenes.min() >= 0),
            ("Valor máximo", int(imagenes.max()), "<= 255", imagenes.max() <= 255),
        ],
        "El redimensionado",
    )


def comparar_redimensionado(datos, imagenes, alto, ancho, semilla=42, n=5):
    """Original (arriba) frente a lo que recibe la red (abajo), para una muestra aleatoria."""
    generador = np.random.default_rng(semilla)
    muestra = generador.choice(len(datos), size=n, replace=False)

    figura, ejes = plt.subplots(2, len(muestra), figsize=(2.4 * len(muestra), 5.8), layout="constrained")
    for columna, indice in enumerate(muestra):
        fila = datos.iloc[indice]
        with Image.open(fila["ruta"]) as img:
            original = img.convert("RGB")
        ejes[0, columna].imshow(original)
        ejes[1, columna].imshow(imagenes[indice])
        # El texto va debajo de la imagen para que no invada la fila de arriba
        ejes[0, columna].set_xlabel(
            f"{fila['personaje'].replace('_', ' ')}\n{original.size[0]}x{original.size[1]}",
            fontsize=8,
            labelpad=6,
            linespacing=1.5,
        )
        marco(ejes[0, columna])
        marco(ejes[1, columna])

    ejes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold", labelpad=8)
    ejes[1, 0].set_ylabel(f"{ancho}x{alto}", fontsize=10, fontweight="bold", labelpad=8)
    figura.suptitle("Original (arriba) vs lo que recibe la red (abajo)", fontsize=13, fontweight="bold")
    plt.show()


# --- Normalización -------------------------------------------------------------------------------------------


def normalizar_muestra(imagenes, tamano_muestra=512):
    """Escala a [0, 1] una muestra de imágenes. El escalado real lo hace la capa Rescaling (CNN) o la
    estandarización (MLP) dentro del modelo, ya sobre la GPU: esta muestra solo sirve para comprobarlo."""
    tamano_muestra = min(tamano_muestra, len(imagenes))
    imagenes_norm = imagenes[:tamano_muestra].astype(np.float32) / VALOR_MAXIMO_PIXEL
    print(f"imagenes      : {imagenes.shape}  dtype={imagenes.dtype}  ->  {imagenes.nbytes / 1024 ** 3:.2f} GB")
    print(f"imagenes_norm : {imagenes_norm.shape}  dtype={imagenes_norm.dtype}  (muestra de comprobacion)")
    print(f"rango: [{imagenes_norm.min():.3f}, {imagenes_norm.max():.3f}]")
    print(f"media: {imagenes_norm.mean():.4f}  desviacion: {imagenes_norm.std():.4f}")
    print(
        f"\nEl conjunto completo en float32 ocuparia {imagenes.nbytes * 4 / 1024 ** 3:.2f} GB;"
        f" en uint8 ocupa {imagenes.nbytes / 1024 ** 3:.2f} GB y cabe entero en la VRAM."
    )
    return imagenes_norm


def comprobar_normalizacion(imagenes, imagenes_norm, indice_prueba=0):
    """Forma, tipo, rango y reversibilidad del escalado a [0, 1]."""
    return verificar(
        [
            (
                "Forma conservada (sobre la muestra)",
                str(imagenes_norm.shape),
                str(imagenes[: len(imagenes_norm)].shape),
                imagenes_norm.shape == imagenes[: len(imagenes_norm)].shape,
            ),
            ("Tipo de dato", str(imagenes_norm.dtype), "float32", imagenes_norm.dtype == np.float32),
            ("Valor mínimo", round(float(imagenes_norm.min()), 4), ">= 0.0", imagenes_norm.min() >= 0.0),
            ("Valor máximo", round(float(imagenes_norm.max()), 4), "<= 1.0", imagenes_norm.max() <= 1.0),
            (
                "Sin NaN ni infinitos",
                bool(np.isfinite(imagenes_norm).all()),
                True,
                bool(np.isfinite(imagenes_norm).all()),
            ),
            (
                "Escalado exacto (x255 == original)",
                "imagenes_norm * 255",
                "imagenes",
                np.allclose(
                    imagenes_norm[indice_prueba] * VALOR_MAXIMO_PIXEL, imagenes[indice_prueba].astype(np.float32)
                ),
            ),
        ],
        "La normalización",
    )


def mostrar_normalizacion(imagenes, imagenes_norm, personaje, indice_prueba=0):
    """La imagen normalizada debe verse idéntica a la original: solo cambia la escala numérica."""
    figura, ejes = plt.subplots(1, 2, figsize=(6.4, 3.6), layout="constrained")
    for eje, (imagen, etiqueta) in zip(
        ejes, ((imagenes[indice_prueba], "uint8 [0, 255]"), (imagenes_norm[indice_prueba], "float32 [0, 1]"))
    ):
        eje.imshow(imagen)
        marco(eje)
        eje.set_xlabel(etiqueta, fontsize=9, labelpad=6)
    figura.suptitle(f"Personaje: {personaje.replace('_', ' ')}", fontsize=12, fontweight="bold")
    plt.show()


# --- Imágenes alteradas y parecidas --------------------------------------------------------------------------


def mostrar_alteradas(datos, dataset_base, semilla, n=6):
    """Parejas original / alterada del conjunto de entrenamiento, con la transformación que aplicó el script."""
    alteradas_train = datos[datos["alterada"] & (datos["conjunto"] == "train")]
    if alteradas_train.empty:
        print("No hay imagenes alteradas en entrenamiento: ejecuta antes el script de alteraciones.")
        return

    transformaciones = {}
    registro_alteraciones = Path(dataset_base) / "_alteraciones.csv"
    if registro_alteraciones.exists():
        registro_script = pd.read_csv(registro_alteraciones)
        con_giro = "rotacion" in registro_script.columns
        transformaciones = {
            (fila.personaje, fila.alterada): (f"giro {fila.rotacion:+.0f}° + " if con_giro else "")
            + f"{fila.color} + {fila.textura}"
            for fila in registro_script.itertuples()
        }

    muestra = alteradas_train.sample(n=min(n, len(alteradas_train)), random_state=semilla)
    figura, ejes = plt.subplots(
        2, len(muestra), figsize=(2.4 * len(muestra), 5.6), layout="constrained", squeeze=False
    )
    for columna, fila in enumerate(muestra.itertuples()):
        for eje, ruta in ((ejes[0, columna], fila.ruta.with_name(fila.origen)), (ejes[1, columna], fila.ruta)):
            with Image.open(ruta) as img:
                eje.imshow(img.convert("RGB"))
            eje.set_xticks([])
            eje.set_yticks([])
        ejes[0, columna].set_title(fila.personaje.replace("_", " "), fontsize=8)
        ejes[1, columna].set_xlabel(transformaciones.get((fila.personaje, fila.ruta.name), "alterada"), fontsize=8)

    ejes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    ejes[1, 0].set_ylabel("Alterada", fontsize=10, fontweight="bold")
    figura.suptitle("Variantes del script: giro, cambio de color y filtro de textura", fontsize=12, fontweight="bold")
    plt.show()

    fraccion = len(alteradas_train) / int((datos["conjunto"] == "train").sum())
    print(f"Imagenes alteradas en entrenamiento: {len(alteradas_train)} ({fraccion:.1%} del conjunto)")


def mostrar_parecidas(dataset_base, semilla, n=5):
    """Grupos de imágenes parecidas: la representante queda intacta y cada casi idéntica entra al notebook con su
    versión filtrada."""
    registro_parecidas = Path(dataset_base) / "_parecidas.csv"
    if not registro_parecidas.exists():
        print("No hay registro de imagenes parecidas: ejecuta antes el script de alteraciones.")
        return
    parecidas_script = pd.read_csv(registro_parecidas)
    print(
        f"Imagenes parecidas reemplazadas: {len(parecidas_script)} en {parecidas_script['grupo'].nunique()} grupos"
        f"   |   similitud mediana con su gemela: {parecidas_script['similitud_original'].median():.2f} antes"
        f" y {parecidas_script['similitud_final'].median():.2f} despues del filtro"
    )
    muestra = parecidas_script.sample(n=min(n, len(parecidas_script)), random_state=semilla)
    figura, ejes = plt.subplots(3, len(muestra), figsize=(2.4 * len(muestra), 8.2), layout="constrained",
                                squeeze=False)
    for columna, fila in enumerate(muestra.itertuples()):
        carpeta = Path(dataset_base) / fila.personaje
        paneles = (
            (fila.representante, fila.personaje.replace("_", " ")),
            (fila.original, f"similitud {fila.similitud_original:.2f}"),
            (
                fila.parecida,
                f"giro {fila.rotacion:+.0f}° + {fila.color} + {fila.textura}\nsimilitud {fila.similitud_final:.2f}",
            ),
        )
        for numero_fila, (nombre_archivo, texto) in enumerate(paneles):
            eje = ejes[numero_fila, columna]
            with Image.open(carpeta / nombre_archivo) as img:
                eje.imshow(img.convert("RGB"))
            eje.set_xticks([])
            eje.set_yticks([])
            eje.set_xlabel(texto, fontsize=7)

    ejes[0, 0].set_ylabel("Intacta", fontsize=10, fontweight="bold")
    ejes[1, 0].set_ylabel("Casi idéntica", fontsize=10, fontweight="bold")
    ejes[2, 0].set_ylabel("Entra al notebook", fontsize=10, fontweight="bold")
    figura.suptitle(
        "Imágenes parecidas: una queda intacta y su gemela entra con un filtro", fontsize=12, fontweight="bold"
    )
    plt.show()


# --- Etiquetas ----------------------------------------------------------------------------------------------


def codificar_one_hot(y, numero_clases):
    """One-hot: la fila i de la matriz identidad es el vector con un 1 en la posición i. Equivale a
    `keras.utils.to_categorical(y, numero_clases)`."""
    return np.eye(numero_clases, dtype=np.float32)[y]


def comprobar_codificacion(y_onehot, y, numero_ejemplos, clases):
    """Forma, tipo y coherencia del one-hot con los índices enteros y con la lista de clases."""
    numero_clases = len(clases)
    return verificar(
        [
            ("Número de ejemplos", y_onehot.shape[0], numero_ejemplos, y_onehot.shape[0] == numero_ejemplos),
            ("Columnas = número de clases", y_onehot.shape[1], numero_clases, y_onehot.shape[1] == numero_clases),
            ("Tipo de dato", str(y_onehot.dtype), "float32", y_onehot.dtype == np.float32),
            (
                "Un único 1 por fila",
                float(y_onehot.sum(axis=1).min()),
                1.0,
                bool(np.all(y_onehot.sum(axis=1) == 1.0)),
            ),
            (
                "Solo valores 0 o 1",
                sorted(np.unique(y_onehot).tolist()),
                [0.0, 1.0],
                bool(np.isin(np.unique(y_onehot), [0.0, 1.0]).all()),
            ),
            (
                "Coincide con el índice entero",
                "y_onehot.argmax(axis=1)",
                "y",
                bool(np.array_equal(y_onehot.argmax(axis=1), y)),
            ),
            (
                "Todas las clases presentes",
                int((y_onehot.sum(axis=0) > 0).sum()),
                numero_clases,
                int((y_onehot.sum(axis=0) > 0).sum()) == numero_clases,
            ),
        ],
        "La codificación de etiquetas",
    )


def tabla_de_clases(y, clases):
    """Diccionario de clases: índice, nombre y ejemplos disponibles."""
    tabla = pd.DataFrame(
        {
            "indice": range(len(clases)),
            "personaje": clases,
            "imagenes": [int((y == indice).sum()) for indice in range(len(clases))],
        }
    )
    tabla["porcentaje"] = (tabla["imagenes"] / len(y) * 100).round(2)
    return tabla


# --- Desbalance de clases ------------------------------------------------------------------------------------


def calcular_pesos_por_clase(y_train_idx, numero_clases):
    """Pesos por clase calculados SOLO con las etiquetas de entrenamiento, en tres esquemas:

    - `ninguno`: todas las clases pesan lo mismo;
    - `balanceado`: inversamente proporcionales a la frecuencia (`compute_class_weight("balanced")`);
    - `raiz`: la raíz cuadrada del balanceado, reescalada para que el peso medio por imagen sea 1 y la pérdida
      conserve su escala. Los personajes pequeños pesan más sin que el modelo deje de reconocer a los frecuentes.

    Devuelve `(pesos_por_esquema, tabla)`: un diccionario `{esquema: {indice_de_clase: peso} o None}` y una tabla
    con los pesos de cada personaje."""
    pesos_balanceados = compute_class_weight(
        class_weight="balanced", classes=np.arange(numero_clases), y=y_train_idx
    )
    conteos_train = np.bincount(y_train_idx, minlength=numero_clases)
    pesos_raiz = np.sqrt(pesos_balanceados)
    pesos_raiz = pesos_raiz * len(y_train_idx) / (pesos_raiz * conteos_train).sum()
    pesos_por_esquema = {
        "ninguno": None,
        "raiz": {indice: float(peso) for indice, peso in enumerate(pesos_raiz)},
        "balanceado": {indice: float(peso) for indice, peso in enumerate(pesos_balanceados)},
    }
    tabla = pd.DataFrame(
        {
            "indice": range(numero_clases),
            "n_train": conteos_train,
            "porcentaje": (conteos_train / conteos_train.sum() * 100).round(2),
            "peso_balanceado": np.round(pesos_balanceados, 3),
            "peso_raiz": np.round(pesos_raiz, 3),
        }
    )
    return pesos_por_esquema, tabla


def comprobar_pesos(pesos, conteos_train, clases, n_train, n_etiquetas):
    """Un peso positivo por clase, mayor para la minoritaria e inverso a la frecuencia."""
    return verificar(
        [
            ("Un peso por clase", len(pesos), len(clases), len(pesos) == len(clases)),
            ("Todos los pesos positivos", round(float(pesos.min()), 3), "> 0", bool((pesos > 0).all())),
            (
                "Mayor peso a la clase minoritaria",
                clases[int(pesos.argmax())],
                clases[int(conteos_train.argmin())],
                int(pesos.argmax()) == int(conteos_train.argmin()),
            ),
            (
                "Menor peso a la clase mayoritaria",
                clases[int(pesos.argmin())],
                clases[int(conteos_train.argmax())],
                int(pesos.argmin()) == int(conteos_train.argmax()),
            ),
            (
                "Peso inverso a la frecuencia",
                "corr(peso, n_train)",
                "negativa",
                bool(float(np.corrcoef(pesos, conteos_train)[0, 1]) < 0),
            ),
            ("Calculados solo con train", n_etiquetas, n_train, n_etiquetas == n_train),
        ],
        "Los pesos por clase",
    )


def graficar_desbalance(conteos_train, pesos, clases):
    """Imágenes de entrenamiento por personaje, coloreadas según el peso asignado."""
    figura, eje = plt.subplots(figsize=(10, 14))
    orden = np.argsort(conteos_train)
    eje.barh(
        [clases[i] for i in orden],
        conteos_train[orden],
        color=matplotlib.colormaps["viridis"](pesos[orden] / pesos.max()),
    )
    eje.set_xlabel("Número de imágenes en entrenamiento")
    eje.set_title("Desbalance de clases (color = peso asignado; más claro = mayor peso)", fontsize=10)
    eje.tick_params(axis="y", labelsize=7)
    eje.axvline(conteos_train.mean(), color="red", linestyle="--", linewidth=1,
                label=f"media = {conteos_train.mean():.0f}")
    eje.legend()
    plt.tight_layout()
    plt.show()
