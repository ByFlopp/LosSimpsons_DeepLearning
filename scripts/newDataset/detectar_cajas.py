"""Encuentra a cada personaje en una imagen y lo nombra con un modelo ya entrenado: el MLP o la CNN.

Es el paso que va DESPUÉS de entrenar, y va aparte: los notebooks entrenan con la imagen completa y no
usan este script. Al terminar, cada notebook deja su modelo final en models_gpu/newDataset/ con todo lo
que hace falta para usarlo fuera del notebook:

    <nombre>.keras            la red entrenada
    <nombre>.json             clases, resolución y representación de la entrada
    <nombre>_entrada.py       cómo se prepara la entrada (píxeles o descriptores); lo escribe el notebook
    <nombre>_particion.csv    a qué conjunto (train, val o test) fue cada imagen original del dataset

    --modelo mlp   PMC/models_gpu/newDataset/mlp_final.*
    --modelo cnn   CNN/models_gpu/newDataset/cnn_final.*

Para cada imagen hay dos etapas, sin anotar nada a mano:

1. Dónde. Grounding DINO (IDEA-Research/grounding-dino-base, un detector zero-shot de Hugging Face)
   busca personajes animados. No sabe quién es quién: las palabras con que se le pregunta (TEXTO) solo
   sirven para buscar y nunca se muestran. Si no encuentra a nadie, busca otra vez con un umbral más bajo.
2. Quién. Cada caja se recorta ampliada un 10 % por lado y llevada a un cuadrado, se redimensiona a la
   resolución del modelo con interpolación bilineal y se prepara como en el entrenamiento
   (<nombre>_entrada.py). El modelo la nombra con su personaje más probable y su confianza:
   "Jasper Beardly · 92 %". También nombra la imagen completa, que es lo que vio al entrenar.

El modelo aprendió con fotogramas completos, así que en un recorte pierde el escenario en que se apoyaba:
el nombre de una caja es menos seguro que el de la imagen completa, y la confianza lo refleja.

Deja en --salida (por defecto, cajas_<nombre>/ junto al modelo):
    cajas.csv        una fila por caja, con el personaje, la confianza y las otras candidatas; la caja 0 es
                     la imagen completa
    imagenes/        cada imagen con sus cajas y sus rótulos
    hojas/           con --dataset, una hoja de contacto por personaje
    resumen.json     modelo, detector, umbrales y conteos

Con --dataset solo toma originales (no las variantes __alterada ni __parecida) y, si existe
<nombre>_particion.csv, solo las de test (--conjunto), que el modelo no vio al entrenar.

Uso, desde la raíz del repositorio (entorno .venv-gpu con requirements-cajas.txt):
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --modelo cnn foto.jpg carpeta/
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --modelo mlp --dataset --muestra 5
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --modelo ruta/a/modelo.keras img.png
Se puede detener con Ctrl+C: cajas.csv queda con las imágenes ya procesadas.

Pruebas (sin GPU): .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Antes de que se importe huggingface_hub: sin telemetría y sin el aviso de enlaces simbólicos de Windows
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# El backend de Keras se fija antes de importarlo: los modelos se entrenaron con Keras sobre PyTorch
os.environ.setdefault("KERAS_BACKEND", "torch")

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[1]
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import generar_imagenes_alteradas as G  # noqa: E402

MODELOS = {
    "mlp": RAIZ_REPO / "PMC" / "models_gpu" / "newDataset" / "mlp_final.keras",
    "cnn": RAIZ_REPO / "CNN" / "models_gpu" / "newDataset" / "cnn_final.keras",
}
SUFIJO_ENTRADA = "_entrada.py"
SUFIJO_PARTICION = "_particion.csv"
# Representaciones cuya entrada es la imagen RGB tal cual: la CNN escala y pasa a grises dentro del modelo
SIN_PREPARACION = ("rgb", "gris")

DETECTOR = "IDEA-Research/grounding-dino-base"
# Solo para buscar: Grounding DINO no distingue personajes. "baby" encuentra a Maggie y "alien", a Kang y Kodos
TEXTO = "cartoon character. person. baby. alien."
# Lado corto con que el detector ve la imagen (su procesador usa 800). 640 está dentro de las escalas con que
# se entrenó, la mediana del dataset es 640 x 480 y en la RTX 4060 baja el tiempo por imagen un 30 %
LADO_DETECTOR = 640
UMBRAL_CAJA = 0.30
UMBRAL_TEXTO = 0.25
# Si con UMBRAL_CAJA no aparece nadie, se busca otra vez con este
UMBRAL_SEGUNDA_PASADA = 0.20
# El detector repite la misma caja con cada palabra del texto: se queda la de mayor puntaje
IOU_NMS = 0.6
LADO_MINIMO = 16
MAXIMO_CAJAS = 10
# Una caja "contiene" a otra si cubre al menos este tanto de su área. La que contiene a dos o más es una
# caja de grupo: varios personajes que ya tienen su propia caja
CONTENCION = 0.9
# Lo que entra al modelo es la caja ampliada este tanto por lado y llevada a un cuadrado
MARGEN = 0.10
CANDIDATAS = 3
MUESTRAS_HOJA = 12
LADO_MAXIMO_IMAGEN = 960

CAMPOS = (
    "imagen", "carpeta", "ancho", "alto", "caja", "fuente", "x1", "y1", "x2", "y2",
    "recorte_x1", "recorte_y1", "recorte_x2", "recorte_y2", "puntaje", "personaje", "confianza", "rotulo",
    "candidatas",
)

# Nombres que no salen bien de la carpeta con title()
NOMBRES = {
    "kang_kodos": "Kang y Kodos",
    "krusty_the_clown": "Krusty the Clown",
    "captain_horatio_mccallister": "Captain Horatio McCallister",
    "dr_julius_hibbert": "Dr. Julius Hibbert",
    "dr_nick_riviera": "Dr. Nick Riviera",
    "rev_timothy_lovejoy": "Rev. Timothy Lovejoy",
}

Caja = tuple[float, float, float, float]


# --- Rótulos ---------------------------------------------------------------------------------------------

def nombre_legible(personaje: str) -> str:
    """'jasper_beardly' -> 'Jasper Beardly'."""
    return NOMBRES.get(personaje, personaje.replace("_", " ").title())


def rotulo(personaje: str, confianza: float) -> str:
    """El rótulo de una caja: el personaje y la confianza del modelo, 'Jasper Beardly · 92 %'."""
    return f"{nombre_legible(personaje)} · {round(100 * float(confianza))} %"


# --- Geometría -------------------------------------------------------------------------------------------

def area(caja: Caja) -> float:
    return max(0.0, caja[2] - caja[0]) * max(0.0, caja[3] - caja[1])


def interseccion(a: Caja, b: Caja) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Caja, b: Caja) -> float:
    comun = interseccion(a, b)
    union = area(a) + area(b) - comun
    return comun / union if union > 0 else 0.0


def filtrar_cajas(cajas: list, puntajes: list, ancho: int, alto: int) -> list[tuple[Caja, float]]:
    """Deja una caja por personaje: las limita a la imagen, descarta las de menos de LADO_MINIMO, quita las
    repetidas (NMS sin clase) y las de grupo, y se queda con las MAXIMO_CAJAS de mayor puntaje. Devuelve
    [(caja, puntaje)] de mayor a menor puntaje."""
    orden = []
    for caja, puntaje in zip(cajas, puntajes):
        x1, y1, x2, y2 = (float(v) for v in caja)
        caja = (min(max(x1, 0.0), ancho), min(max(y1, 0.0), alto), min(max(x2, 0.0), ancho), min(max(y2, 0.0), alto))
        if caja[2] - caja[0] >= LADO_MINIMO and caja[3] - caja[1] >= LADO_MINIMO:
            orden.append((caja, float(puntaje)))
    orden.sort(key=lambda par: -par[1])
    elegidas = []
    for caja, puntaje in orden:
        if all(iou(caja, otra) < IOU_NMS for otra, _ in elegidas):
            elegidas.append((caja, puntaje))
    sin_grupo = [
        (caja, puntaje)
        for i, (caja, puntaje) in enumerate(elegidas)
        if sum(1 for j, (otra, _) in enumerate(elegidas) if j != i and interseccion(caja, otra) >= CONTENCION * area(otra)) < 2
    ]
    return sin_grupo[:MAXIMO_CAJAS]


def recorte_cuadrado(caja: Caja, ancho: int, alto: int, margen: float = MARGEN) -> tuple[int, int, int, int]:
    """Lo que entra al modelo por cada caja: la caja ampliada un `margen` por lado y llevada a un cuadrado
    con su mismo centro, desplazado para quedar dentro de la imagen. Si no cabe un cuadrado (una caja casi
    tan alta o tan ancha como la imagen), es el rectángulo más parecido a un cuadrado que la contiene.
    Devuelve (x1, y1, x2, y2) en píxeles enteros."""
    x1, y1, x2, y2 = caja
    necesario_x = min((x2 - x1) * (1 + 2 * margen), ancho)
    necesario_y = min((y2 - y1) * (1 + 2 * margen), alto)
    lado = max(necesario_x, necesario_y)
    tramo_x, tramo_y = min(lado, ancho), min(lado, alto)
    # Si un eje topa con la imagen, el otro se acerca a su largo sin dejar de contener a la caja
    if tramo_y < lado:
        tramo_x = min(ancho, max(necesario_x, tramo_y))
    if tramo_x < lado:
        tramo_y = min(alto, max(necesario_y, tramo_x))

    def ubicar(centro: float, tramo: float, limite: int) -> tuple[int, int]:
        tramo = max(1, min(limite, round(tramo)))
        inicio = round(min(max(centro - tramo / 2, 0), limite - tramo))
        return inicio, inicio + tramo

    rx1, rx2 = ubicar((x1 + x2) / 2, tramo_x, ancho)
    ry1, ry2 = ubicar((y1 + y2) / 2, tramo_y, alto)
    return rx1, ry1, rx2, ry2


# --- El modelo entrenado ---------------------------------------------------------------------------------

def resolver_modelo(texto: str) -> Path:
    """'mlp', 'cnn' o la ruta de un .keras con su .json al lado."""
    ruta = MODELOS.get(texto.lower(), Path(texto))
    if ruta.suffix != ".keras":
        raise SystemExit(f"--modelo debe ser mlp, cnn o un archivo .keras, no {texto!r}")
    faltan = [str(r) for r in (ruta, ruta.with_suffix(".json")) if not r.is_file()]
    if faltan:
        raise SystemExit("Falta " + " y ".join(faltan) + ". Ejecuta antes el notebook del modelo hasta la celda "
                         "«Selección del modelo final», que guarda el modelo y sus archivos.")
    return ruta


def leer_metadatos(ruta_modelo: Path) -> dict:
    metadatos = json.loads(ruta_modelo.with_suffix(".json").read_text(encoding="utf-8"))
    faltan = [clave for clave in ("clases", "alto", "ancho") if clave not in metadatos]
    if faltan:
        raise SystemExit(f"{ruta_modelo.with_suffix('.json')} no trae {', '.join(faltan)}")
    return metadatos


def ruta_hermana(ruta_modelo: Path, metadatos: dict, clave: str, sufijo: str) -> Path:
    """Un archivo del modelo: el que nombra su .json o, si no, <nombre><sufijo> junto al .keras."""
    return ruta_modelo.with_name(metadatos.get(clave) or ruta_modelo.stem + sufijo)


def cargar_preparacion(ruta_modelo: Path, metadatos: dict):
    """La función preparar(lote) de <nombre>_entrada.py, que el notebook escribe con las mismas funciones que
    usó para entrenar: (n, alto, ancho, 3) en uint8 -> la entrada del modelo. Sin ese archivo solo se aceptan
    los modelos que reciben la imagen RGB tal cual (la CNN)."""
    ruta = ruta_hermana(ruta_modelo, metadatos, "preparacion", SUFIJO_ENTRADA)
    if ruta.is_file():
        especificacion = importlib.util.spec_from_file_location(ruta.stem, ruta)
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        return modulo.preparar
    representacion = metadatos.get("representacion", "rgb")
    if representacion in SIN_PREPARACION:
        return lambda lote: np.asarray(lote, dtype=np.uint8)
    raise SystemExit(f"El modelo usa la representación {representacion!r} y falta {ruta}: vuelve a ejecutar la "
                     "celda «Selección del modelo final» del notebook, que lo genera.")


def redimensionar(recortes: list[Image.Image], tamano: tuple[int, int]) -> np.ndarray:
    """(n, alto, ancho, 3) en uint8: el mismo redimensionado bilineal del notebook. np.array y no np.asarray:
    la copia es escribible, y PyTorch avisa si recibe la memoria de solo lectura de PIL."""
    return np.stack([np.array(r.convert("RGB").resize(tamano, Image.Resampling.BILINEAR), dtype=np.uint8)
                     for r in recortes])


class Modelo:
    """El modelo final de un notebook, listo para nombrar recortes."""

    def __init__(self, ruta_modelo: Path, usar_cpu: bool = False):
        self.ruta = ruta_modelo
        self.metadatos = leer_metadatos(ruta_modelo)
        self.clases = list(self.metadatos["clases"])
        self.tamano = (int(self.metadatos["ancho"]), int(self.metadatos["alto"]))
        self.preparar = cargar_preparacion(ruta_modelo, self.metadatos)
        if usar_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import keras
        import torch

        self.torch = torch
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        # compile=False: solo se predice. El entrenamiento usa un optimizador de PyTorch, así que el de Keras
        # guardado en el archivo está vacío
        self.red = keras.models.load_model(ruta_modelo, compile=False).to(self.dispositivo)

    def descripcion(self) -> str:
        datos = self.metadatos
        return (f"{datos.get('modelo', self.ruta.stem).upper()} {datos.get('configuracion', '')} "
                f"({datos.get('representacion', 'rgb')}, {self.tamano[1]}x{self.tamano[0]}, {self.dispositivo})")

    def __call__(self, recortes: list[Image.Image]) -> np.ndarray:
        """(n, clases): probabilidades de cada recorte."""
        entrada = np.ascontiguousarray(self.preparar(redimensionar(recortes, self.tamano)))
        salida = self.red.predict(self.torch.as_tensor(entrada).to(self.dispositivo), verbose=0, batch_size=256)
        return np.asarray(salida, dtype=np.float32)


# --- El detector -----------------------------------------------------------------------------------------

class Detector:
    """Grounding DINO: encuentra personajes animados sin saber quiénes son. Se usa de a una imagen: en una
    GPU de 8 GB, un lote de 4 ya no cabe y Windows lo pasa a la RAM, decenas de veces más lenta."""

    def __init__(self, nombre: str = DETECTOR, usar_cpu: bool = False, lado: int = LADO_DETECTOR):
        if usar_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.utils import logging as registro_transformers

        registro_transformers.set_verbosity_error()
        registro_transformers.disable_progress_bar()
        self.torch = torch
        self.nombre = nombre
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.procesador = AutoProcessor.from_pretrained(nombre)
        self.modelo = AutoModelForZeroShotObjectDetection.from_pretrained(nombre).to(self.dispositivo).eval()
        self.tamano = {"shortest_edge": lado, "longest_edge": round(lado * 1333 / 800)}

    def __call__(self, imagen: Image.Image, umbral: float = UMBRAL_CAJA, entradas=None) -> list[tuple[Caja, float]]:
        """Cajas de una imagen, ya filtradas (filtrar_cajas), en sus píxeles."""
        torch = self.torch
        if entradas is None:
            entradas = self.procesador(images=[imagen], text=[TEXTO], return_tensors="pt", size=self.tamano)
        entradas = entradas.to(self.dispositivo)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dispositivo == "cuda"):
            salidas = self.modelo(**entradas)
        resultado = self.procesador.post_process_grounded_object_detection(
            salidas, entradas.input_ids, threshold=umbral, text_threshold=UMBRAL_TEXTO,
            target_sizes=[imagen.size[::-1]],
        )[0]
        return filtrar_cajas(resultado["boxes"].float().cpu().tolist(), resultado["scores"].float().cpu().tolist(),
                             *imagen.size)

    def buscar(self, imagen: Image.Image, umbral: float = UMBRAL_CAJA) -> list[tuple[Caja, float]]:
        """Primero con `umbral`; si no aparece nadie, otra vez con UMBRAL_SEGUNDA_PASADA."""
        entradas = self.procesador(images=[imagen], text=[TEXTO], return_tensors="pt", size=self.tamano)
        cajas = self(imagen, umbral, entradas)
        if not cajas and umbral > UMBRAL_SEGUNDA_PASADA:
            cajas = self(imagen, UMBRAL_SEGUNDA_PASADA, entradas)
        return cajas


# --- Nombrar cada caja -----------------------------------------------------------------------------------

def _columnas(prefijo: str, valores) -> dict:
    return {f"{prefijo}{eje}": valor for eje, valor in zip(("x1", "y1", "x2", "y2"), valores)}


def nombrar(imagen: Image.Image, cajas: list[tuple[Caja, float]], clasificar, clases: list[str],
            margen: float = MARGEN, candidatas: int = CANDIDATAS) -> list[dict]:
    """Una fila por caja con el nombre que le da el modelo. La fila 0 es la imagen completa; las demás, las
    cajas del detector en su orden. `clasificar` recibe una lista de imágenes y devuelve (n, clases)."""
    ancho, alto = imagen.size
    recortes = [(0, 0, ancho, alto)] + [recorte_cuadrado(caja, ancho, alto, margen) for caja, _ in cajas]
    probabilidades = clasificar([imagen if i == 0 else imagen.crop(r) for i, r in enumerate(recortes)])
    filas = []
    for numero, (recorte, p) in enumerate(zip(recortes, probabilidades)):
        orden = np.argsort(-p)[:candidatas]
        mejor = int(orden[0])
        caja, puntaje = ((0, 0, ancho, alto), "") if numero == 0 else (cajas[numero - 1][0], cajas[numero - 1][1])
        filas.append({
            "caja": numero,
            "fuente": "imagen_completa" if numero == 0 else "detector",
            **_columnas("", [round(float(v), 1) for v in caja]),
            **_columnas("recorte_", recorte),
            "puntaje": puntaje if puntaje == "" else round(float(puntaje), 3),
            "personaje": clases[mejor],
            "confianza": round(float(p[mejor]), 4),
            "rotulo": rotulo(clases[mejor], p[mejor]),
            "candidatas": ", ".join(rotulo(clases[int(i)], p[int(i)]) for i in orden[1:]),
        })
    return filas


# --- Qué imágenes ----------------------------------------------------------------------------------------

def es_imagen(ruta: Path) -> bool:
    return ruta.is_file() and ruta.suffix.lower() in G.EXTENSIONES_VALIDAS


def reunir_imagenes(entradas: list[Path]) -> list[Path]:
    """Las imágenes indicadas: cada archivo tal cual y, de cada carpeta, sus imágenes en todas las subcarpetas.
    En las carpetas se saltan las variantes __alterada y __parecida del script de alteraciones."""
    rutas = []
    for entrada in entradas:
        if entrada.is_dir():
            rutas.extend(r for r in sorted(entrada.rglob("*")) if es_imagen(r) and not G.es_generado(r))
        elif es_imagen(entrada):
            rutas.append(entrada)
        else:
            raise SystemExit(f"No es una imagen ni una carpeta: {entrada}")
    vistas, unicas = set(), []
    for ruta in rutas:
        clave = ruta.resolve()
        if clave not in vistas:
            vistas.add(clave)
            unicas.append(ruta)
    return unicas


def leer_particion(ruta: Path) -> dict[tuple[str, str], str]:
    """{(personaje, archivo original): conjunto} del <nombre>_particion.csv que deja el notebook."""
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return {(fila["personaje"], fila["archivo"]): fila["conjunto"] for fila in csv.DictReader(archivo)}


def imagenes_del_dataset(dataset: Path, particion: dict | None, conjunto: str, muestra: int | None,
                         semilla: int, personajes: list[str] | None = None) -> list[Path]:
    """Las originales del dataset que se van a procesar: las del `conjunto` según la partición del notebook
    (todas si conjunto es 'todos' o no hay partición), y como mucho `muestra` por personaje, al azar."""
    carpetas = G.validar(dataset)
    if personajes:
        desconocidos = sorted(set(personajes) - {c.name for c in carpetas})
        if desconocidos:
            raise SystemExit(f"No hay carpetas para: {', '.join(desconocidos)}")
        carpetas = [c for c in carpetas if c.name in set(personajes)]
    seleccion = []
    for carpeta in carpetas:
        rutas = G.originales_de(carpeta)
        if particion is not None and conjunto != "todos":
            rutas = [r for r in rutas if particion.get((carpeta.name, r.name)) == conjunto]
        if muestra is not None and len(rutas) > muestra:
            azar = random.Random(f"{semilla}-{carpeta.name}")
            rutas = sorted(azar.sample(rutas, muestra))
        seleccion.extend(rutas)
    return seleccion


def carpeta_de(ruta: Path, clases: list[str]) -> str:
    """El personaje de la carpeta de la imagen, si la carpeta lleva el nombre de una clase; si no, ''."""
    return ruta.parent.name if ruta.parent.name in clases else ""


def destino_imagen(carpeta: Path, ruta: Path, usados: set) -> Path:
    """imagenes/<carpeta de origen>/<nombre>.jpg, sin pisar otra imagen con el mismo nombre."""
    destino = carpeta / ruta.parent.name / f"{ruta.stem}.jpg"
    numero = 2
    while destino in usados:
        destino = carpeta / ruta.parent.name / f"{ruta.stem}_{numero}.jpg"
        numero += 1
    usados.add(destino)
    return destino


# --- Imágenes con sus cajas ------------------------------------------------------------------------------

VERDE, AZUL, NARANJO = (27, 127, 59), (31, 119, 180), (214, 137, 16)


def _fuente(tamano: int):
    for nombre in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nombre, tamano)
        except OSError:
            continue
    return ImageFont.load_default()


def _rotular_caja(dibujo: ImageDraw.ImageDraw, caja, texto: str, color, fuente, grosor: int) -> None:
    dibujo.rectangle(caja, outline=color, width=grosor)
    izquierda, arriba, derecha, abajo = dibujo.textbbox((0, 0), texto, font=fuente)
    alto_texto = abajo - arriba + 4
    y = caja[1] - alto_texto if caja[1] >= alto_texto else caja[1]
    dibujo.rectangle((caja[0], y, caja[0] + derecha - izquierda + 6, y + alto_texto), fill=color)
    dibujo.text((caja[0] + 3, y + 2 - arriba), texto, fill=(255, 255, 255), font=fuente)


def dibujar(imagen: Image.Image, filas: list[dict], carpeta: str = "", lado_maximo: int = LADO_MAXIMO_IMAGEN) -> Image.Image:
    """La imagen con cada caja y su rótulo: verde si nombra al personaje de la carpeta, azul si nombra a otro.
    Si el detector no encontró a nadie, un marco naranjo con el nombre de la imagen completa."""
    escala = min(1.0, lado_maximo / max(imagen.size))
    lienzo = imagen.convert("RGB").resize((max(1, round(imagen.size[0] * escala)), max(1, round(imagen.size[1] * escala))),
                                          Image.Resampling.BILINEAR)
    dibujo = ImageDraw.Draw(lienzo)
    fuente = _fuente(max(12, round(lienzo.size[0] / 45)))
    grosor = max(2, round(lienzo.size[0] / 250))
    cajas = [fila for fila in filas if fila["fuente"] == "detector"]
    # Las que nombran al personaje de la carpeta se dibujan encima
    for fila in sorted(cajas, key=lambda f: f["personaje"] == carpeta):
        caja = tuple(fila[eje] * escala for eje in ("x1", "y1", "x2", "y2"))
        color = VERDE if carpeta and fila["personaje"] == carpeta else AZUL
        _rotular_caja(dibujo, caja, fila["rotulo"], color, fuente, grosor)
    if not cajas:
        completa = filas[0]
        _rotular_caja(dibujo, (1, 1, lienzo.size[0] - 2, lienzo.size[1] - 2),
                      f"{completa['rotulo']} (imagen completa)", NARANJO, fuente, grosor + 1)
    return lienzo


def hoja_de_contacto(personaje: str, miniaturas: list[Image.Image], destino: Path, columnas: int = 4) -> None:
    """Hasta MUESTRAS_HOJA imágenes de un personaje, con sus cajas, en una sola hoja."""
    if not miniaturas:
        return
    miniaturas = [m.resize((360, max(1, round(m.size[1] * 360 / m.size[0]))), Image.Resampling.BILINEAR)
                  for m in miniaturas]
    ancho_celda = max(m.size[0] for m in miniaturas) + 8
    alto_celda = max(m.size[1] for m in miniaturas) + 8
    filas = -(-len(miniaturas) // columnas)
    hoja = Image.new("RGB", (columnas * ancho_celda + 8, filas * alto_celda + 40), (245, 245, 245))
    ImageDraw.Draw(hoja).text(
        (10, 10), f"{nombre_legible(personaje)} (carpeta {personaje}): verde = el modelo nombra al personaje de la "
                  "carpeta, azul = otro personaje, naranjo = sin cajas", fill=(0, 0, 0), font=_fuente(16))
    for k, miniatura in enumerate(miniaturas):
        hoja.paste(miniatura, (8 + (k % columnas) * ancho_celda, 40 + (k // columnas) * alto_celda))
    destino.parent.mkdir(parents=True, exist_ok=True)
    hoja.save(destino, quality=85)


# --- Programa --------------------------------------------------------------------------------------------

def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("imagenes", nargs="*", type=Path, help="imágenes o carpetas con imágenes")
    parser.add_argument("--modelo", required=True,
                        help="mlp, cnn o la ruta de un .keras con su .json al lado")
    parser.add_argument("--dataset", nargs="?", type=Path, const=Path(os.environ.get("SIMPSONS_DATASET", str(G.RUTA_POR_DEFECTO))),
                        help="procesa originales del dataset (por defecto SIMPSONS_DATASET o %(const)s)")
    parser.add_argument("--conjunto", choices=("test", "val", "train", "todos"), default="test",
                        help="con --dataset, qué conjunto de la partición del notebook (por defecto %(default)s)")
    parser.add_argument("--muestra", type=int, help="con --dataset, como mucho N imágenes por personaje")
    parser.add_argument("--personajes", type=lambda t: [p.strip() for p in t.split(",") if p.strip()],
                        help="con --dataset, solo estos personajes, separados por comas")
    parser.add_argument("--salida", type=Path, help="carpeta de resultados (por defecto cajas_<modelo>/ junto al modelo)")
    parser.add_argument("--detector", default=DETECTOR, help="modelo de Hugging Face (por defecto %(default)s)")
    parser.add_argument("--umbral", type=float, default=UMBRAL_CAJA, help="puntaje mínimo de una caja (%(default)s)")
    parser.add_argument("--margen", type=float, default=MARGEN, help="ampliación de cada caja por lado (%(default)s)")
    parser.add_argument("--sin-imagenes", action="store_true", help="solo escribe cajas.csv y resumen.json")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU (muy lento)")
    parser.add_argument("--semilla", type=int, default=G.SEMILLA, help="para la muestra y las hojas (%(default)s)")
    argumentos = parser.parse_args(argv)
    if not argumentos.imagenes and argumentos.dataset is None:
        parser.error("indica imágenes o carpetas, o --dataset")
    if argumentos.muestra is not None and argumentos.muestra < 1:
        parser.error("--muestra debe ser al menos 1")
    return argumentos


def main(argv=None) -> int:
    args = leer_argumentos(argv)
    ruta_modelo = resolver_modelo(args.modelo)
    metadatos = leer_metadatos(ruta_modelo)
    clases = list(metadatos["clases"])

    rutas = reunir_imagenes(args.imagenes)
    if args.dataset is not None:
        ruta_particion = ruta_hermana(ruta_modelo, metadatos, "particion", SUFIJO_PARTICION)
        particion = leer_particion(ruta_particion) if ruta_particion.is_file() else None
        if particion is None and args.conjunto != "todos":
            print(f"No hay {ruta_particion.name}: se toman las imágenes de todos los conjuntos, también las que el "
                  "modelo vio al entrenar.")
        rutas += imagenes_del_dataset(args.dataset, particion, args.conjunto, args.muestra, args.semilla,
                                      args.personajes)
    if not rutas:
        raise SystemExit("No hay imágenes que procesar.")

    salida = args.salida or ruta_modelo.with_name(f"cajas_{ruta_modelo.stem}")
    carpeta_imagenes = salida / "imagenes"
    salida.mkdir(parents=True, exist_ok=True)

    modelo = Modelo(ruta_modelo, args.cpu)
    print(f"Modelo  : {ruta_modelo}  |  {modelo.descripcion()}")
    print(f"Detector: {args.detector} (lado {LADO_DETECTOR}, umbral {args.umbral}, segunda pasada "
          f"{UMBRAL_SEGUNDA_PASADA})")
    print(f"Imágenes: {len(rutas)}  ->  {salida}\n")
    detector = Detector(args.detector, args.cpu)

    inicio = time.time()
    procesadas, errores, total_cajas, sin_cajas = 0, 0, 0, 0
    con_carpeta, carpeta_en_caja, carpeta_en_completa = 0, 0, 0
    por_personaje, usados = {}, set()
    detenido = False
    with open(salida / "cajas.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        try:
            for numero, ruta in enumerate(rutas, start=1):
                try:
                    imagen = G.abrir_rgb(ruta)
                except Exception as error:  # una imagen dañada no detiene la corrida: queda informada
                    errores += 1
                    print(f"  No se pudo abrir {ruta}: {error}", flush=True)
                    continue
                carpeta = carpeta_de(ruta, clases)
                filas = nombrar(imagen, detector.buscar(imagen, args.umbral), modelo, clases, args.margen)
                base = {"imagen": str(ruta), "carpeta": carpeta, "ancho": imagen.size[0], "alto": imagen.size[1]}
                escritor.writerows({**base, **fila} for fila in filas)
                archivo.flush()

                cajas = filas[1:]
                procesadas += 1
                total_cajas += len(cajas)
                sin_cajas += not cajas
                if carpeta:
                    con_carpeta += 1
                    carpeta_en_caja += any(f["personaje"] == carpeta for f in cajas)
                    carpeta_en_completa += filas[0]["personaje"] == carpeta
                if not args.sin_imagenes:
                    dibujada = dibujar(imagen, filas, carpeta)
                    destino = destino_imagen(carpeta_imagenes, ruta, usados)
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    dibujada.save(destino, quality=88)
                    if args.dataset is not None and carpeta:
                        lista = por_personaje.setdefault(carpeta, [])
                        if len(lista) < MUESTRAS_HOJA:
                            lista.append(dibujada)

                if len(rutas) <= 20:
                    nombres = ", ".join(f["rotulo"] for f in cajas) or f"sin cajas; imagen completa: {filas[0]['rotulo']}"
                    print(f"  {ruta.name}: {nombres}", flush=True)
                elif numero % 50 == 0 or numero == len(rutas):
                    ritmo = numero / max(time.time() - inicio, 1e-6)
                    faltan = (len(rutas) - numero) / max(ritmo, 1e-6)
                    print(f"  {numero:6d}/{len(rutas)} imágenes  {ritmo:4.1f} img/s  faltan ~{faltan / 60:.0f} min",
                          flush=True)
        except KeyboardInterrupt:
            detenido = True
            print("\nDetenido: cajas.csv queda con las imágenes ya procesadas.")

    for personaje, miniaturas in por_personaje.items():
        hoja_de_contacto(personaje, miniaturas, salida / "hojas" / f"{personaje}.jpg")

    resumen = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": str(ruta_modelo),
        "configuracion": metadatos.get("configuracion"),
        "representacion": metadatos.get("representacion", "rgb"),
        "resolucion": f"{metadatos['alto']}x{metadatos['ancho']}",
        "detector": args.detector,
        "texto_detector": TEXTO,
        "lado_detector": LADO_DETECTOR,
        "umbral_caja": args.umbral,
        "umbral_segunda_pasada": UMBRAL_SEGUNDA_PASADA,
        "margen": args.margen,
        "dataset": None if args.dataset is None else str(args.dataset),
        "conjunto": None if args.dataset is None else args.conjunto,
        "imagenes": procesadas,
        "imagenes_con_error": errores,
        "imagenes_sin_cajas": sin_cajas,
        "cajas": total_cajas,
        "cajas_por_imagen": round(total_cajas / max(procesadas, 1), 2),
        # Solo en las imágenes cuya carpeta es un personaje: si alguna caja o la imagen completa lo nombra
        "imagenes_con_carpeta": con_carpeta,
        "carpeta_nombrada_en_alguna_caja": round(carpeta_en_caja / con_carpeta, 4) if con_carpeta else None,
        "carpeta_nombrada_en_imagen_completa": round(carpeta_en_completa / con_carpeta, 4) if con_carpeta else None,
        "segundos": round(time.time() - inicio, 1),
        "completo": not detenido,
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{procesadas} imágenes, {total_cajas} cajas ({resumen['cajas_por_imagen']} por imagen), "
          f"{sin_cajas} sin cajas, {errores} con error, en {resumen['segundos']:.0f} s")
    if con_carpeta:
        print(f"En {con_carpeta} imágenes con carpeta de personaje: alguna caja lo nombra en el "
              f"{resumen['carpeta_nombrada_en_alguna_caja']:.1%}; la imagen completa, en el "
              f"{resumen['carpeta_nombrada_en_imagen_completa']:.1%}")
    print(f"Resultados: {salida / 'cajas.csv'}")
    if not args.sin_imagenes:
        print(f"Imágenes con sus cajas: {carpeta_imagenes}")
    return 3 if detenido else 0


if __name__ == "__main__":
    sys.exit(main())
