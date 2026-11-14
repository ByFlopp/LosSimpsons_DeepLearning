"""Recorre el dataset personaje por personaje y marca las imágenes cuya etiqueta probablemente está mal.

La etiqueta de cada imagen es su carpeta. Si una imagen de la carpeta de Barney solo muestra a Bart, o la de Otto es
una hoja de papel, el modelo aprende de ella como si fuera de ese personaje y la evaluación la cuenta como error:
es ruido en las etiquetas. Este script lo busca con el modelo final ya entrenado (por defecto la CNN), sin anotar
nada a mano y sin tocar el dataset: solo escribe un informe y hojas de contacto para mirar.

Método: aprendizaje confiado (confident learning; Northcutt, Jiang y Chuang, 2021).

1. El modelo da la probabilidad de cada personaje para cada imagen original del dataset.
2. El umbral de cada personaje es la probabilidad media que el modelo le da a sus propias imágenes de validación y
   test, que no vio al entrenar: cuánta seguridad suele tener con ese personaje.
3. Una imagen se marca si el modelo apoya a su carpeta menos que ese umbral y además, de más a menos sospechosa:
     otro personaje               nombra a otro con al menos su umbral y CONFIANZA_MINIMA, ese otro no figura en la
                                  escena de la imagen y el modelo casi nunca confunde a los dos (en validación y
                                  test, menos de CONFUSION_HABITUAL): la carpeta probablemente está mal;
     otro personaje de la escena  igual, pero el otro sí figura en la escena del episodio (±3 s) o tiene una gemela:
                                  la imagen probablemente muestra a los dos, y el modelo nombró al otro;
     personaje parecido           igual, pero el modelo confunde a menudo a los dos personajes (Selma y Patty, el
                                  abuelo y Homero): más probablemente es un error del modelo que de la carpeta;
     sin personaje claro          no nombra a nadie con más de UMBRAL_DUDOSA: puede no haber ningún personaje (una
                                  hoja, un objeto, un paisaje), o ser una imagen difícil. Es la señal más débil.

En las imágenes de entrenamiento el modelo ya vio su etiqueta, así que tiende a darle la razón a la carpeta: ahí se
marcan menos, pero cada marca pesa más, porque el modelo contradice lo que le enseñaron. La columna `conjunto` lo
indica.

Deja en --salida (por defecto ruido_<modelo>/ junto al modelo, fuera de Git):
    ruido.csv                   las imágenes marcadas, por personaje y de más a menos sospechosa
    resumen_por_personaje.csv   imágenes, marcadas y destino más frecuente de cada personaje
    imagenes.csv                todas las imágenes revisadas, con su predicción y su probabilidad
    hojas/<personaje>.jpg       las marcadas de cada personaje, cada una con el personaje que nombra el modelo y su
                                confianza («Bart Simpson · 99 %») y la probabilidad de la carpeta
    resumen.json                modelo, umbrales y conteos
    probabilidades.npz          caché: con el mismo modelo, una segunda ejecución no vuelve a leer las imágenes

Uso, desde la raíz del repositorio (entorno .venv-gpu):
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_ruido_etiquetas.py
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_ruido_etiquetas.py --personajes otto_mann,barney_gumble
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_ruido_etiquetas.py --modelo mlp --conjunto test

Pruebas (sin GPU): .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("KERAS_BACKEND", "torch")

CARPETA = Path(__file__).resolve().parent
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import detectar_cajas as D  # noqa: E402
import generar_imagenes_alteradas as G  # noqa: E402

OTRO = "otro personaje"
ESCENA = "otro personaje de la escena"
PARECIDO = "personaje parecido"
DUDOSA = "sin personaje claro"
# Orden de las marcas en el informe y en las hojas: primero las que más probablemente son etiquetas equivocadas
PRIORIDAD = {OTRO: 0, ESCENA: 1, PARECIDO: 2, DUDOSA: 3}
COLORES = {OTRO: (192, 57, 43), ESCENA: (214, 137, 16), PARECIDO: (31, 119, 180), DUDOSA: (110, 110, 110)}

# Para marcar «otro personaje», el modelo tiene que nombrarlo al menos con esta probabilidad, además de su umbral
CONFIANZA_MINIMA = 0.50
# «Sin personaje claro»: ningún personaje supera esta probabilidad. Con 0,20 marcaba sobre todo imágenes difíciles en
# las que el personaje sí está; las hojas y los objetos sin personaje quedan por debajo de 0,10
UMBRAL_DUDOSA = 0.10
# «Personaje parecido»: el modelo nombra al otro en al menos esta fracción de las imágenes de validación y test de la
# carpeta (el abuelo como Homero, un 9 %; Selma como Patty, un 40 %)
CONFUSION_HABITUAL = 0.05
FUERA_DE_MUESTRA = ("val", "test")
MAXIMO_POR_HOJA = 24
LOTE = 256
VENTANA_ESCENA_MS = 3000
# Fotogramas de episodio: <personaje>__<episodio>__t<milisegundos>, como en el notebook
PATRON_FOTOGRAMA = re.compile(r"^(?P<nombre>.+?)__(?P<episodio>S\d+E\d+)__t(?P<ms>\d+)$", re.IGNORECASE)

CAMPOS_IMAGEN = (
    "personaje", "archivo", "conjunto", "prediccion", "confianza", "p_carpeta", "umbral_carpeta", "motivo",
    "candidata", "p_candidata", "confusion_habitual", "puntaje", "escena", "rotulo", "ruta",
)
CAMPOS_RESUMEN = (
    "personaje", "imagenes", "marcadas", "porcentaje", OTRO, ESCENA, PARECIDO, DUDOSA, "umbral",
    "destino_mas_frecuente",
)


# --- Escenas: quién más aparece en cada imagen, como en «Escenas con varios personajes» del notebook --------------

def leer_fotograma(nombre_archivo: str):
    """(personaje del nombre, episodio, ms) de un fotograma de episodio, o None si no lo es."""
    coincidencia = PATRON_FOTOGRAMA.match(Path(nombre_archivo).stem)
    if coincidencia is None:
        return None
    return coincidencia["nombre"], coincidencia["episodio"].upper(), int(coincidencia["ms"])


def lineas_de_tiempo(pares) -> dict:
    """{episodio: (instantes ordenados, personajes)} a partir de pares (personaje de la carpeta, archivo)."""
    por_episodio = {}
    for personaje, archivo in pares:
        fotograma = leer_fotograma(archivo)
        if fotograma is not None:
            por_episodio.setdefault(fotograma[1], []).append((fotograma[2], personaje))
    lineas = {}
    for episodio, fotogramas in por_episodio.items():
        fotogramas.sort()
        lineas[episodio] = (np.array([ms for ms, _ in fotogramas]), [p for _, p in fotogramas])
    return lineas


def en_escena(personaje: str, archivo: str, lineas: dict, clases: set, ventana_ms: int = VENTANA_ESCENA_MS) -> set:
    """Otros personajes que probablemente aparecen en la imagen: el del nombre del fotograma, si es otro, y los que
    tienen un fotograma del mismo episodio a menos de `ventana_ms`."""
    fotograma = leer_fotograma(archivo)
    if fotograma is None:
        return set()
    nombre, episodio, ms = fotograma
    otros = {nombre} if nombre != personaje and nombre in clases else set()
    if episodio in lineas:
        instantes, personajes = lineas[episodio]
        desde = np.searchsorted(instantes, ms - ventana_ms, side="left")
        hasta = np.searchsorted(instantes, ms + ventana_ms, side="right")
        otros.update(otro for otro in personajes[desde:hasta] if otro != personaje)
    return otros


def leer_gemelas(dataset: Path, clases: set) -> dict:
    """{(personaje, archivo): {otros personajes}} según `_gemelas.csv`: la misma escena guardada en dos carpetas."""
    ruta = dataset / "_gemelas.csv"
    gemelas = {}
    if not ruta.is_file():
        return gemelas
    with open(ruta, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["personaje"] in clases and fila["otro_personaje"] in clases:
                gemelas.setdefault((fila["personaje"], fila["original"]), set()).add(fila["otro_personaje"])
                gemelas.setdefault((fila["otro_personaje"], fila["otro_original"]), set()).add(fila["personaje"])
    return gemelas


# --- Aprendizaje confiado --------------------------------------------------------------------------------------

def umbrales_por_clase(probabilidades: np.ndarray, etiquetas: np.ndarray, numero_clases: int,
                       fuera_de_muestra: np.ndarray | None = None) -> np.ndarray:
    """Umbral de cada clase: la probabilidad media que el modelo le da a sus propias imágenes. Si se indica
    `fuera_de_muestra`, solo cuentan esas imágenes (las que el modelo no vio al entrenar); una clase sin ninguna usa
    todas las suyas."""
    umbrales = np.zeros(numero_clases, dtype=np.float64)
    for clase in range(numero_clases):
        de_la_clase = etiquetas == clase
        mascara = de_la_clase if fuera_de_muestra is None else de_la_clase & fuera_de_muestra
        if not mascara.any():
            mascara = de_la_clase
        umbrales[clase] = float(probabilidades[mascara, clase].mean()) if mascara.any() else 1.0
    return umbrales


def tasas_de_confusion(probabilidades: np.ndarray, etiquetas: np.ndarray, numero_clases: int) -> np.ndarray:
    """(clases, clases): en qué fracción de las imágenes de cada clase (filas) el modelo nombra a cada otra
    (columnas). La diagonal queda en cero."""
    tasas = np.zeros((numero_clases, numero_clases), dtype=np.float64)
    np.add.at(tasas, (etiquetas, probabilidades.argmax(axis=1)), 1.0)
    totales = tasas.sum(axis=1, keepdims=True)
    tasas = np.divide(tasas, totales, out=np.zeros_like(tasas), where=totales > 0)
    np.fill_diagonal(tasas, 0.0)
    return tasas


def clasificar(probabilidades: np.ndarray, etiqueta: int, umbrales: np.ndarray, clases: list[str], escena: set,
               confianza_minima: float = CONFIANZA_MINIMA, umbral_dudosa: float = UMBRAL_DUDOSA,
               confusiones: np.ndarray | None = None, confusion_habitual: float = CONFUSION_HABITUAL):
    """(motivo, candidata, puntaje) de una imagen, o (None, None, 0.0) si no se marca.

    Se marca solo si la probabilidad de la carpeta queda bajo su umbral. Entonces, si algún otro personaje alcanza
    su propio umbral y `confianza_minima`, la candidata es el más probable de ellos y el puntaje, cuánto supera a la
    carpeta; el motivo es ESCENA si la candidata figura en `escena`, PARECIDO si el modelo la confunde con la carpeta
    en al menos `confusion_habitual` de los casos según `confusiones` (ver tasas_de_confusion) y OTRO en los demás.
    Si ningún personaje pasa de `umbral_dudosa`, el motivo es DUDOSA y el puntaje, 1 menos la mayor probabilidad."""
    if probabilidades[etiqueta] >= umbrales[etiqueta]:
        return None, None, 0.0
    otros = [
        clase for clase in range(len(clases))
        if clase != etiqueta and probabilidades[clase] >= max(umbrales[clase], confianza_minima)
    ]
    if otros:
        candidata = max(otros, key=lambda clase: probabilidades[clase])
        if clases[candidata] in escena:
            motivo = ESCENA
        elif confusiones is not None and confusiones[etiqueta, candidata] >= confusion_habitual:
            motivo = PARECIDO
        else:
            motivo = OTRO
        return motivo, candidata, float(probabilidades[candidata] - probabilidades[etiqueta])
    if probabilidades.max() < umbral_dudosa:
        return DUDOSA, int(probabilidades.argmax()), float(1.0 - probabilidades.max())
    return None, None, 0.0


# --- Probabilidades del modelo, con caché ------------------------------------------------------------------------

def firma(ruta_modelo: Path) -> str:
    """Identifica el archivo del modelo: si se vuelve a entrenar, la caché deja de valer."""
    estado = ruta_modelo.stat()
    return f"{ruta_modelo.resolve()}|{estado.st_size}|{estado.st_mtime_ns}"


def leer_cache(ruta: Path, firma_modelo: str) -> dict:
    """{ruta de la imagen: probabilidades} de una ejecución anterior con el mismo modelo, o {}."""
    if not ruta.is_file():
        return {}
    with np.load(ruta, allow_pickle=False) as datos:
        if str(datos["firma"]) != firma_modelo:
            return {}
        return dict(zip(datos["rutas"].tolist(), datos["probabilidades"]))


def guardar_cache(ruta: Path, firma_modelo: str, cache: dict) -> None:
    rutas = sorted(cache)
    np.savez_compressed(ruta, firma=np.array(firma_modelo), rutas=np.array(rutas),
                        probabilidades=np.stack([cache[r] for r in rutas]).astype(np.float32))


def calcular_probabilidades(modelo, rutas: list[Path], cache: dict, lote: int = LOTE) -> tuple[dict, list[str]]:
    """Agrega a `cache` las probabilidades de las imágenes que faltan, leídas y redimensionadas en paralelo como en
    el notebook. Devuelve (cache, imágenes que no se pudieron abrir)."""
    faltan = [ruta for ruta in rutas if str(ruta) not in cache]
    errores = []
    if not faltan:
        return cache, errores

    def leer(ruta: Path):
        try:
            return np.array(G.abrir_rgb(ruta).resize(modelo.tamano, Image.Resampling.BILINEAR), dtype=np.uint8)
        except Exception:  # una imagen dañada no detiene la revisión: queda informada
            return None

    inicio = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as hilos:
        for desde in range(0, len(faltan), lote):
            tramo = faltan[desde:desde + lote]
            leidas = list(hilos.map(leer, tramo))
            validas = [(ruta, imagen) for ruta, imagen in zip(tramo, leidas) if imagen is not None]
            errores += [str(ruta) for ruta, imagen in zip(tramo, leidas) if imagen is None]
            if validas:
                entrada = np.ascontiguousarray(modelo.preparar(np.stack([imagen for _, imagen in validas])))
                salida = modelo.red.predict(modelo.torch.as_tensor(entrada).to(modelo.dispositivo), verbose=0,
                                            batch_size=lote)
                for (ruta, _), fila in zip(validas, np.asarray(salida, dtype=np.float32)):
                    cache[str(ruta)] = fila
            hechas = min(desde + lote, len(faltan))
            if hechas % (lote * 20) == 0 or hechas == len(faltan):
                ritmo = hechas / max(time.time() - inicio, 1e-6)
                print(f"  {hechas:6d}/{len(faltan)} imágenes  {ritmo:5.0f} img/s  "
                      f"faltan ~{(len(faltan) - hechas) / max(ritmo, 1e-6) / 60:.0f} min", flush=True)
    return cache, errores


# --- Hojas de contacto -----------------------------------------------------------------------------------------

def miniatura(ruta: Path, lado: int) -> Image.Image:
    """La imagen reducida para caber en un cuadrado de `lado`, centrada sobre fondo gris."""
    imagen = G.abrir_rgb(ruta)
    imagen.thumbnail((lado, lado), Image.Resampling.BILINEAR)
    lienzo = Image.new("RGB", (lado, lado), (225, 225, 225))
    lienzo.paste(imagen, ((lado - imagen.size[0]) // 2, (lado - imagen.size[1]) // 2))
    return lienzo


def hoja_de_contacto(personaje: str, filas: list[dict], total: int, destino: Path, columnas: int = 4,
                     lado: int = 260) -> None:
    """Las imágenes marcadas de un personaje en una hoja. Cada una lleva un marco del color de su motivo, el
    personaje que nombra el modelo con su confianza y la probabilidad que le da a la carpeta."""
    if not filas:
        return
    fuente, fuente_titulo = D._fuente(14), D._fuente(18)
    alto_texto = 64
    ancho_celda, alto_celda = lado + 12, lado + alto_texto + 12
    numero_filas = -(-len(filas) // columnas)
    hoja = Image.new("RGB", (columnas * ancho_celda + 12, numero_filas * alto_celda + 76), (250, 250, 250))
    dibujo = ImageDraw.Draw(hoja)
    dibujo.text((12, 10), f"{D.nombre_legible(personaje)} (carpeta {personaje}): {len(filas)} de {total} imágenes "
                          "marcadas como posible ruido", fill=(0, 0, 0), font=fuente_titulo)
    dibujo.text((12, 38), "rojo = otro personaje   ámbar = otro personaje de la escena   azul = personaje parecido   "
                          "gris = sin personaje claro",
                fill=(80, 80, 80), font=fuente)
    for k, fila in enumerate(filas):
        x, y = 12 + (k % columnas) * ancho_celda, 72 + (k // columnas) * alto_celda
        color = COLORES[fila["motivo"]]
        try:
            hoja.paste(miniatura(Path(fila["ruta"]), lado), (x, y))
        except Exception:
            dibujo.text((x + 8, y + lado // 2), "no se pudo abrir", fill=(0, 0, 0), font=fuente)
        dibujo.rectangle((x - 3, y - 3, x + lado + 2, y + lado + 2), outline=color, width=4)
        lineas = (
            fila["rotulo"],
            f"carpeta: {D.rotulo(personaje, float(fila['p_carpeta']))}",
            f"{fila['conjunto']} · {fila['archivo'][:34]}",
        )
        for numero, texto in enumerate(lineas):
            dibujo.text((x, y + lado + 6 + numero * 19), texto, fill=color if numero == 0 else (40, 40, 40),
                        font=fuente)
    destino.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(destino, quality=85)


# --- Programa --------------------------------------------------------------------------------------------------

def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="cnn", help="mlp, cnn o la ruta de un .keras con su .json al lado "
                                                        "(por defecto %(default)s)")
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("SIMPSONS_DATASET", str(G.RUTA_POR_DEFECTO))),
                        help="carpeta del dataset (por defecto SIMPSONS_DATASET o %(default)s)")
    parser.add_argument("--personajes", type=lambda t: [p.strip() for p in t.split(",") if p.strip()],
                        help="solo estos personajes, separados por comas")
    parser.add_argument("--conjunto", choices=("todos", "train", "val", "test"), default="todos",
                        help="qué imágenes revisar según la partición del notebook (por defecto %(default)s)")
    parser.add_argument("--confianza-minima", type=float, default=CONFIANZA_MINIMA,
                        help="probabilidad mínima del otro personaje para marcar (%(default)s)")
    parser.add_argument("--umbral-dudosa", type=float, default=UMBRAL_DUDOSA,
                        help="por debajo de esta probabilidad máxima, «sin personaje claro» (%(default)s)")
    parser.add_argument("--confusion-habitual", type=float, default=CONFUSION_HABITUAL,
                        help="desde esta tasa de confusión en validación y test, «personaje parecido» (%(default)s)")
    parser.add_argument("--maximo-por-hoja", type=int, default=MAXIMO_POR_HOJA,
                        help="imágenes por hoja de contacto (%(default)s)")
    parser.add_argument("--salida", type=Path, help="carpeta de resultados (por defecto ruido_<modelo>/ junto al modelo)")
    parser.add_argument("--sin-hojas", action="store_true", help="solo escribe los CSV y el resumen")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU (muy lento)")
    argumentos = parser.parse_args(argv)
    if not all(0 < valor <= 1 for valor in (argumentos.confianza_minima, argumentos.umbral_dudosa,
                                             argumentos.confusion_habitual)):
        parser.error("--confianza-minima, --umbral-dudosa y --confusion-habitual deben estar entre 0 y 1")
    return argumentos


def escribir_csv(ruta: Path, campos, filas) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos, lineterminator="\n", extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)


def main(argv=None) -> int:
    args = leer_argumentos(argv)
    ruta_modelo = D.resolver_modelo(args.modelo)
    metadatos = D.leer_metadatos(ruta_modelo)
    clases = list(metadatos["clases"])
    clase_a_indice = {clase: indice for indice, clase in enumerate(clases)}

    carpetas = [c for c in G.validar(args.dataset) if c.name in clase_a_indice]
    if args.personajes:
        desconocidos = sorted(set(args.personajes) - {c.name for c in carpetas})
        if desconocidos:
            raise SystemExit(f"No hay carpetas de personajes del modelo para: {', '.join(desconocidos)}")
    ruta_particion = D.ruta_hermana(ruta_modelo, metadatos, "particion", D.SUFIJO_PARTICION)
    particion = D.leer_particion(ruta_particion) if ruta_particion.is_file() else None
    if particion is None:
        print(f"No hay {ruta_particion.name}: los umbrales se calculan con todas las imágenes, también las que el "
              "modelo vio al entrenar.")

    # Todas las originales del dataset: las que se revisan y las de validación y test de todos los personajes, que
    # fijan los umbrales
    todas = [(carpeta.name, ruta) for carpeta in carpetas for ruta in G.originales_de(carpeta)]
    conjunto_de = {(p, r.name): (particion or {}).get((p, r.name), "sin partición") for p, r in todas}
    revisar = [
        (p, r) for p, r in todas
        if (not args.personajes or p in args.personajes) and (args.conjunto == "todos" or conjunto_de[(p, r.name)] == args.conjunto)
    ]
    para_umbrales = [(p, r) for p, r in todas if particion is None or conjunto_de[(p, r.name)] in FUERA_DE_MUESTRA]
    if not revisar:
        raise SystemExit("No hay imágenes que revisar con esos filtros.")

    salida = args.salida or ruta_modelo.with_name(f"ruido_{ruta_modelo.stem}")
    salida.mkdir(parents=True, exist_ok=True)
    ruta_cache = salida / "probabilidades.npz"
    firma_modelo = firma(ruta_modelo)

    modelo = D.Modelo(ruta_modelo, args.cpu)
    print(f"Modelo : {ruta_modelo}  |  {modelo.descripcion()}")
    print(f"Revisa : {len(revisar)} originales de {len({p for p, _ in revisar})} personajes "
          f"(conjunto {args.conjunto})  ->  {salida}")
    cache = leer_cache(ruta_cache, firma_modelo)
    necesarias = list(dict.fromkeys(r for _, r in revisar + para_umbrales))
    print(f"Probabilidades: {sum(str(r) in cache for r in necesarias)} en caché, "
          f"{sum(str(r) not in cache for r in necesarias)} por calcular\n")
    inicio = time.time()
    cache, errores = calcular_probabilidades(modelo, necesarias, cache)
    guardar_cache(ruta_cache, firma_modelo, cache)
    for ruta in errores:
        print(f"  No se pudo abrir {ruta}")

    # Umbral de cada personaje y confusiones habituales del modelo, con las imágenes que no vio al entrenar
    pares_umbral = [(p, r) for p, r in para_umbrales if str(r) in cache]
    probabilidades_umbral = np.stack([cache[str(r)] for _, r in pares_umbral])
    etiquetas_umbral = np.array([clase_a_indice[p] for p, _ in pares_umbral])
    umbrales = umbrales_por_clase(probabilidades_umbral, etiquetas_umbral, len(clases))
    confusiones = tasas_de_confusion(probabilidades_umbral, etiquetas_umbral, len(clases))

    nombres_clases = set(clases)
    lineas = lineas_de_tiempo((p, r.name) for p, r in todas)
    gemelas = leer_gemelas(args.dataset, nombres_clases)

    filas, por_personaje = [], {}
    for personaje, ruta in revisar:
        if str(ruta) not in cache:
            continue
        probabilidades = cache[str(ruta)]
        etiqueta = clase_a_indice[personaje]
        escena = en_escena(personaje, ruta.name, lineas, nombres_clases) | gemelas.get((personaje, ruta.name), set())
        motivo, candidata, puntaje = clasificar(probabilidades, etiqueta, umbrales, clases, escena,
                                                args.confianza_minima, args.umbral_dudosa, confusiones,
                                                args.confusion_habitual)
        prediccion = int(probabilidades.argmax())
        fila = {
            "personaje": personaje,
            "archivo": ruta.name,
            "conjunto": conjunto_de[(personaje, ruta.name)],
            "prediccion": clases[prediccion],
            "confianza": round(float(probabilidades[prediccion]), 4),
            "p_carpeta": round(float(probabilidades[etiqueta]), 4),
            "umbral_carpeta": round(float(umbrales[etiqueta]), 4),
            "motivo": motivo or "",
            "candidata": clases[candidata] if candidata is not None else "",
            "p_candidata": round(float(probabilidades[candidata]), 4) if candidata is not None else "",
            "confusion_habitual": round(float(confusiones[etiqueta, candidata]), 4) if candidata is not None else "",
            "puntaje": round(puntaje, 4),
            "escena": ", ".join(sorted(escena)),
            "rotulo": D.rotulo(clases[prediccion], probabilidades[prediccion]),
            "ruta": str(ruta),
        }
        filas.append(fila)
        por_personaje.setdefault(personaje, []).append(fila)

    marcadas = sorted(
        (fila for fila in filas if fila["motivo"]),
        key=lambda f: (f["personaje"], PRIORIDAD[f["motivo"]], -f["puntaje"]),
    )
    escribir_csv(salida / "imagenes.csv", CAMPOS_IMAGEN, filas)
    escribir_csv(salida / "ruido.csv", CAMPOS_IMAGEN, marcadas)

    # Resumen por personaje, de más a menos marcado en proporción
    resumen = []
    for personaje, suyas in por_personaje.items():
        suyas_marcadas = [f for f in suyas if f["motivo"]]
        motivos = Counter(f["motivo"] for f in suyas_marcadas)
        destinos = Counter(f["candidata"] for f in suyas_marcadas if f["motivo"] in (OTRO, ESCENA, PARECIDO))
        destino, veces = destinos.most_common(1)[0] if destinos else ("", 0)
        resumen.append({
            "personaje": personaje,
            "imagenes": len(suyas),
            "marcadas": len(suyas_marcadas),
            "porcentaje": round(100 * len(suyas_marcadas) / len(suyas), 1),
            OTRO: motivos[OTRO],
            ESCENA: motivos[ESCENA],
            PARECIDO: motivos[PARECIDO],
            DUDOSA: motivos[DUDOSA],
            "umbral": round(float(umbrales[clase_a_indice[personaje]]), 3),
            "destino_mas_frecuente": f"{destino} ({veces})" if destino else "",
        })
    # De más a menos «otro personaje», la señal más directa de ruido
    resumen.sort(key=lambda f: (-f[OTRO], -f["porcentaje"], f["personaje"]))
    escribir_csv(salida / "resumen_por_personaje.csv", CAMPOS_RESUMEN, resumen)

    if not args.sin_hojas:
        for personaje, suyas in por_personaje.items():
            suyas_marcadas = [f for f in marcadas if f["personaje"] == personaje][:args.maximo_por_hoja]
            hoja_de_contacto(personaje, suyas_marcadas, len(suyas), salida / "hojas" / f"{personaje}.jpg")

    print(f"{'personaje':28s} {'imágenes':>8s} {'marcadas':>8s} {'%':>5s} {'otro':>5s} {'escena':>6s} "
          f"{'parecido':>8s} {'dudosa':>6s}  destino más frecuente")
    for fila in resumen:
        print(f"{fila['personaje']:28s} {fila['imagenes']:8d} {fila['marcadas']:8d} {fila['porcentaje']:5.1f} "
              f"{fila[OTRO]:5d} {fila[ESCENA]:6d} {fila[PARECIDO]:8d} {fila[DUDOSA]:6d}  "
              f"{fila['destino_mas_frecuente']}")

    por_motivo = Counter(f["motivo"] for f in marcadas)
    por_conjunto = Counter(f["conjunto"] for f in marcadas)
    revisadas_por_conjunto = Counter(f["conjunto"] for f in filas)
    informe = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": str(ruta_modelo),
        "configuracion": metadatos.get("configuracion"),
        "dataset": str(args.dataset),
        "conjunto": args.conjunto,
        "metodo": "aprendizaje confiado: umbral por personaje = probabilidad media sobre sus imágenes de "
                  + ("validación y test" if particion is not None else "todo el dataset"),
        "confianza_minima": args.confianza_minima,
        "umbral_dudosa": args.umbral_dudosa,
        "confusion_habitual": args.confusion_habitual,
        "imagenes_revisadas": len(filas),
        "imagenes_con_error": len(errores),
        "marcadas": len(marcadas),
        "marcadas_por_motivo": {motivo: por_motivo[motivo] for motivo in PRIORIDAD},
        "marcadas_por_conjunto": {
            conjunto: {"revisadas": revisadas_por_conjunto[conjunto], "marcadas": por_conjunto[conjunto]}
            for conjunto in sorted(revisadas_por_conjunto)
        },
        "umbrales": {clase: round(float(umbral), 4) for clase, umbral in zip(clases, umbrales)},
        "segundos": round(time.time() - inicio, 1),
    }
    (salida / "resumen.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(marcadas)} de {len(filas)} imágenes marcadas ({len(marcadas) / max(len(filas), 1):.1%}): "
          + ", ".join(f"{por_motivo[m]} {m}" for m in PRIORIDAD))
    for conjunto, valores in informe["marcadas_por_conjunto"].items():
        print(f"  {conjunto:13s} {valores['marcadas']:5d} de {valores['revisadas']}")
    print(f"Imágenes marcadas: {salida / 'ruido.csv'}")
    if not args.sin_hojas:
        print(f"Hojas de contacto: {salida / 'hojas'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
