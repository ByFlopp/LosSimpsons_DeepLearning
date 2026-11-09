"""Detecta a cada personaje en las imágenes del dataset y deja su caja para los notebooks de GPU de la CNN y
del MLP.

Cada caja sale de dos etapas:

1. Dónde. Grounding DINO (IDEA-Research/grounding-dino-base, un detector zero-shot de Hugging Face)
   busca personajes animados en el fotograma. No necesita cajas anotadas, pero no sabe quién es quién:
   las palabras con que se le pregunta (TEXTO) solo sirven para buscar y nunca se muestran.
2. Quién. Cada caja se recorta como la verá el notebook, y una CNN del proyecto, que conoce a los 64
   personajes, la rotula con su personaje más probable y su confianza: "Jasper Beardly · 92 %". Primero
   lo hace la CNN final (CNN/models_gpu/newDataset/cnn_final.keras); después, una CNN de recortes que
   se entrena con las cajas sin dudas (más abajo) las vuelve a rotular todas.

De las cajas de cada imagen, la del personaje de la carpeta es la que entra al notebook. La etiqueta
sigue siendo la carpeta; p_carpeta es la probabilidad que la CNN le da al personaje de la carpeta:

    sin cajas                                           -> imagen completa  (sin_detecciones)
    la de mayor p_carpeta, si es >= UMBRAL_ASIGNACION   -> esa caja         (cnn)
    ninguna caja lo reconoce                            -> imagen completa  (nadie_lo_reconoce)

Las imágenes que quedan en imagen completa pasan una segunda vez por el detector, con un umbral más bajo.
Si aun así nadie reconoce al personaje, la imagen entra completa, como antes de las cajas: en el piloto,
tomar la caja más grande se equivocaba (Helen Lovejoy terminaba en la caja del reverendo).

La CNN final se entrenó con fotogramas completos y nombra mal los recortes, porque se apoya en el
escenario: en el piloto, Jasper Beardly salía como "Moe Szyslak · 11 %" aunque la caja era la suya, y
en la carpeta de Jasper, Edna Krabappel salía como "Jasper Beardly". Afinarla con recortes no le quitaba
la costumbre. Por eso, sin anotar nada (autoentrenamiento), se entrena desde cero, con la misma
arquitectura, una CNN que solo ve recortes: los sin dudas (la única caja de la imagen, o una que la CNN
final reconoce claramente mejor que las demás), cada uno con el personaje de su carpeta. Con ella se
vuelven a rotular y a elegir todas las cajas. Queda en CNN/models_gpu/newDataset/cajas/maestra_recortes.keras
y solo sirve para anotar: los modelos de los notebooks se entrenan aparte, únicamente con las imágenes
de entrenamiento.

Solo pasan por el detector las originales. Las variantes __alterada y __parecida que genera
generar_imagenes_alteradas.py son la original girada, recortada y devuelta a su tamaño: su caja se
calcula con ese mismo giro (caja_tras_giro), que queda en _alteraciones.csv y _parecidas.csv.

Deja en la carpeta del dataset:
    _cajas.csv                 una fila por imagen que usa el notebook (original o variante)
    _cajas_candidatas.csv      todas las cajas de cada original, con su personaje y su confianza
    _cajas_detecciones.jsonl   caché: cada ejecución solo procesa las originales nuevas
y junto al dataset, nunca dentro, cajas_por_revisar/<personaje>.jpg: hojas de contacto para mirar las
cajas con su rótulo, sin anotar nada.

Orden: generar_imagenes_alteradas.py, este script y después los notebooks. Si cambia el dataset, se
vuelven a ejecutar los dos scripts: este solo detecta lo nuevo.

Uso, desde la raíz del repositorio (entorno .venv-gpu con requirements-cajas.txt):
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --muestra 5     piloto
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py                 todo el dataset
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --personajes lou,louie --rehacer
    .venv-gpu/Scripts/python.exe scripts/newDataset/detectar_cajas.py --reentrenar    vuelve a entrenar la CNN de recortes
Se puede detener con Ctrl+C: lo detectado queda en la caché y la próxima ejecución sigue desde ahí.

Pruebas (sin GPU): .venv-frinkiac/Scripts/python.exe -m pytest scripts/newDataset/tests -q
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Antes de que se importe huggingface_hub: sin telemetría y sin el aviso de enlaces simbólicos de Windows
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[1]
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import generar_imagenes_alteradas as G  # noqa: E402

PREDECIR_IMAGEN = RAIZ_REPO / "CNN" / "scripts" / "newDataset" / "predecir_imagen.py"

REGISTRO_CAJAS = "_cajas.csv"
REGISTRO_CANDIDATAS = "_cajas_candidatas.csv"
CACHE = "_cajas_detecciones.jsonl"
NOMBRE_GALERIA = "cajas_por_revisar"
TEXTO_MARCA_GALERIA = (
    "Esta carpeta la genera scripts/newDataset/detectar_cajas.py.\n"
    "Cada imagen es una hoja de contacto de un personaje: la caja verde es la que entra al notebook y las\n"
    "azules son los demás personajes de la imagen, cada una con el nombre y la confianza que les da la CNN.\n"
    "Un marco naranjo indica que la imagen entra completa. Solo sirve para mirar: no hay que anotar nada.\n"
    "Cada ejecución del script vacía esta carpeta y la vuelve a llenar.\n"
)

DETECTOR = "IDEA-Research/grounding-dino-base"
# Solo para buscar: Grounding DINO no distingue personajes. "baby" encuentra a Maggie y "alien", a Kang y Kodos
TEXTO = "cartoon character. person. baby. alien."
# Lado corto con que el detector ve la imagen (su procesador usa 800). 640 está dentro de las escalas con que
# se entrenó, la mediana del dataset es 640 x 480 y en la RTX 4060 baja el tiempo por imagen un 30 %
LADO_DETECTOR = 640
UMBRAL_CAJA = 0.30
UMBRAL_TEXTO = 0.25
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
# p_carpeta mínima para que una caja sea la del personaje de la carpeta
UMBRAL_ASIGNACION = 0.05
COBERTURA_MINIMA = 0.90
# Originales que se cargan juntas: mientras la GPU detecta un grupo, otros hilos preparan el siguiente
IMAGENES_POR_GRUPO = 16
MUESTRAS_GALERIA = 12

CAMPOS_CAJAS = (
    "personaje", "archivo", "original", "tipo", "fuente", "motivo", "ancho", "alto",
    "caja_x1", "caja_y1", "caja_x2", "caja_y2", "recorte_x1", "recorte_y1", "recorte_x2", "recorte_y2",
    "puntaje", "nombre", "confianza", "p_carpeta", "p_completa", "n_cajas",
)
CAMPOS_CANDIDATAS = (
    "personaje", "archivo", "caja_x1", "caja_y1", "caja_x2", "caja_y2",
    "recorte_x1", "recorte_y1", "recorte_x2", "recorte_y2", "puntaje", "nombre", "confianza", "p_carpeta",
    "elegida",
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


def puntuacion(caja: dict) -> dict:
    """Nombre, confianza y p_carpeta de una caja: los de la CNN de recortes si ya los tiene, y si no, los
    de la CNN final."""
    return caja.get("recortes") or caja


def elegir(cajas: list[dict]) -> tuple[int | None, str]:
    """Cuál caja es el personaje de la carpeta: (índice o None si entra la imagen completa, motivo)."""
    if not cajas:
        return None, "sin_detecciones"
    mejor = max(range(len(cajas)), key=lambda i: puntuacion(cajas[i])["p_carpeta"])
    if puntuacion(cajas[mejor])["p_carpeta"] >= UMBRAL_ASIGNACION:
        return mejor, "cnn"
    return None, "nadie_lo_reconoce"


# --- Modelos ---------------------------------------------------------------------------------------------

class Detector:
    """Grounding DINO: encuentra personajes animados sin saber quiénes son. Se usa de a una imagen: en una
    GPU de 8 GB, un lote de 4 ya no cabe y Windows lo pasa a la RAM, decenas de veces más lenta."""

    def __init__(self, nombre: str = DETECTOR, dispositivo: str | None = None, lado: int = LADO_DETECTOR):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.utils import logging as registro_transformers

        registro_transformers.set_verbosity_error()
        registro_transformers.disable_progress_bar()
        self.torch = torch
        self.nombre = nombre
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.procesador = AutoProcessor.from_pretrained(nombre)
        self.modelo = AutoModelForZeroShotObjectDetection.from_pretrained(nombre).to(self.dispositivo).eval()
        self.tamano = {"shortest_edge": lado, "longest_edge": round(lado * 1333 / 800)}

    def preparar(self, imagen: Image.Image):
        """Pre-procesamiento en la CPU: se puede hacer en otro hilo mientras la GPU detecta."""
        return self.procesador(images=[imagen], text=[TEXTO], return_tensors="pt", size=self.tamano)

    def __call__(self, imagen: Image.Image, umbral: float = UMBRAL_CAJA, entradas=None) -> list[tuple[Caja, float]]:
        """Cajas de una imagen, ya filtradas (filtrar_cajas), en sus píxeles."""
        torch = self.torch
        entradas = (entradas if entradas is not None else self.preparar(imagen)).to(self.dispositivo)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dispositivo == "cuda"):
            salidas = self.modelo(**entradas)
        resultado = self.procesador.post_process_grounded_object_detection(
            salidas, entradas.input_ids, threshold=umbral, text_threshold=UMBRAL_TEXTO,
            target_sizes=[imagen.size[::-1]],
        )[0]
        return filtrar_cajas(resultado["boxes"].float().cpu().tolist(), resultado["scores"].float().cpu().tolist(),
                             *imagen.size)


_DETECTOR = None


def detectar(imagenes: list[Image.Image], umbral: float = UMBRAL_CAJA) -> list[list[tuple[Caja, float]]]:
    """Cajas de cada imagen con el detector por defecto, que se carga la primera vez (los notebooks la usan
    para probar el modelo con una imagen cualquiera)."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return [_DETECTOR(imagen, umbral) for imagen in imagenes]


class Maestra:
    """Una CNN del proyecto que rotula cada caja con su personaje: la CNN final, entrenada con la imagen
    completa, o la CNN de recortes. Se carga con el mismo código de CNN/scripts/newDataset/predecir_imagen.py."""

    def __init__(self, ruta_modelo: Path | None = None, usar_cpu: bool = False):
        especificacion = importlib.util.spec_from_file_location("predecir_imagen", PREDECIR_IMAGEN)
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        self.modelo, metadatos, self.dispositivo = modulo.cargar_modelo(ruta_modelo, usar_cpu)
        self.ruta = Path(ruta_modelo) if ruta_modelo else next(r for r in modulo.MODELOS if r.exists())
        if metadatos.get("representacion", "rgb") != "rgb":
            raise SystemExit(f"La CNN usa la representación {metadatos['representacion']!r}; este script solo "
                             "prepara imágenes RGB.")
        import torch

        self.torch = torch
        self.clases = list(metadatos["clases"])
        self.tamano = (metadatos["ancho"], metadatos["alto"])
        self.configuracion = metadatos["configuracion"]

    @classmethod
    def con_modelo(cls, plantilla: "Maestra", modelo, configuracion: str) -> "Maestra":
        """Otra maestra con las mismas clases y el mismo pre-procesamiento, pero con este modelo."""
        nueva = cls.__new__(cls)
        nueva.__dict__.update(plantilla.__dict__)
        nueva.modelo, nueva.configuracion = modelo, configuracion
        return nueva

    def predecir(self, lote: np.ndarray) -> np.ndarray:
        """(n, clases) para un lote (n, alto, ancho, 3) en uint8; el escalado lo hace el modelo."""
        salida = self.modelo.predict(self.torch.as_tensor(lote).to(self.dispositivo), verbose=0, batch_size=256)
        return np.asarray(salida, dtype=np.float32)

    def __call__(self, recortes: list[Image.Image]) -> np.ndarray:
        """(n, clases): mismo pre-procesamiento que el notebook, RGB y redimensionado bilineal en uint8."""
        return self.predecir(np.stack([np.array(r.resize(self.tamano, Image.Resampling.BILINEAR), dtype=np.uint8)
                                       for r in recortes]))


# --- Detección de las originales -------------------------------------------------------------------------

def configuracion(detector: str, maestra_configuracion: str) -> str:
    """Todo lo que cambia el contenido de la caché: si cambia, las originales se vuelven a procesar."""
    return "|".join(str(v) for v in (
        detector, LADO_DETECTOR, TEXTO, UMBRAL_CAJA, UMBRAL_TEXTO, UMBRAL_SEGUNDA_PASADA, IOU_NMS, LADO_MINIMO,
        MAXIMO_CAJAS, CONTENCION, MARGEN, UMBRAL_ASIGNACION, maestra_configuracion,
    ))


def leer_cache(ruta: Path) -> dict[tuple[str, str], dict]:
    """La última línea de cada original manda; una línea cortada por un apagón se ignora."""
    registros = {}
    if ruta.exists():
        with open(ruta, encoding="utf-8") as archivo:
            for linea in archivo:
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                registros[(registro["personaje"], registro["archivo"])] = registro
    return registros


def _abrir(ruta: Path, detector: Detector):
    try:
        imagen = G.abrir_rgb(ruta)
    except Exception as error:  # una imagen dañada no detiene la corrida: queda informada
        return None, error
    return imagen, detector.preparar(imagen)


def _rotular(imagen: Image.Image, cajas: list[tuple[Caja, float]], personaje: str, maestra: Maestra) -> tuple[dict, list[dict]]:
    """La CNN rotula la imagen completa y cada caja; devuelve (completa, cajas) como quedan en la caché."""
    recortes = [recorte_cuadrado(caja, *imagen.size) for caja, _ in cajas]
    probabilidades = maestra([imagen] + [imagen.crop(r) for r in recortes])
    indice = maestra.clases.index(personaje)

    def resumen(p):
        mejor = int(p.argmax())
        return {"nombre": maestra.clases[mejor], "confianza": round(float(p[mejor]), 4),
                "p_carpeta": round(float(p[indice]), 4)}

    completa = resumen(probabilidades[0])
    filas = [
        {"caja": [round(v, 1) for v in caja], "puntaje": round(puntaje, 3), "recorte": list(recorte), **resumen(p)}
        for (caja, puntaje), recorte, p in zip(cajas, recortes, probabilidades[1:])
    ]
    return completa, filas


def detectar_originales(pendientes: list[tuple[str, Path]], detector: Detector, maestra: Maestra, ruta_cache: Path,
                        registros: dict, config: str) -> None:
    """Procesa las originales pendientes y anota cada grupo en la caché apenas termina."""
    grupos = [pendientes[i:i + IMAGENES_POR_GRUPO] for i in range(0, len(pendientes), IMAGENES_POR_GRUPO)]
    inicio, hechas, errores = time.time(), 0, 0
    with ThreadPoolExecutor(4) as hilos, open(ruta_cache, "a", encoding="utf-8") as cache:
        futuros = [hilos.submit(_abrir, ruta, detector) for _, ruta in grupos[0]] if grupos else []
        for numero, grupo in enumerate(grupos):
            cargadas = [futuro.result() for futuro in futuros]
            if numero + 1 < len(grupos):
                futuros = [hilos.submit(_abrir, ruta, detector) for _, ruta in grupos[numero + 1]]
            for (personaje, ruta), (imagen, entradas) in zip(grupo, cargadas):
                if imagen is None:
                    errores += 1
                    print(f"  No se pudo abrir {ruta}: {entradas}", flush=True)
                    continue
                pasada = 1
                cajas = detector(imagen, UMBRAL_CAJA, entradas)
                completa, filas = _rotular(imagen, cajas, personaje, maestra)
                if elegir(filas)[0] is None:
                    # Segunda pasada, más permisiva, para las que entrarían completas
                    pasada = 2
                    cajas = detector(imagen, UMBRAL_SEGUNDA_PASADA, entradas)
                    completa, filas = _rotular(imagen, cajas, personaje, maestra)
                registro = {"personaje": personaje, "archivo": ruta.name, "ancho": imagen.size[0],
                            "alto": imagen.size[1], "pasada": pasada, "config": config, "completa": completa,
                            "cajas": filas}
                registros[(personaje, ruta.name)] = registro
                cache.write(json.dumps(registro, ensure_ascii=False) + "\n")
                hechas += 1
            cache.flush()
            if (numero + 1) % 10 == 0 or numero + 1 == len(grupos):
                segundos = time.time() - inicio
                ritmo = hechas / max(segundos, 1e-6)
                faltan = (len(pendientes) - hechas - errores) / max(ritmo, 1e-6)
                print(f"  {hechas + errores:6d}/{len(pendientes)} originales  {ritmo:4.1f} img/s  "
                      f"faltan ~{faltan / 3600:.0f} h {faltan % 3600 / 60:02.0f} min", flush=True)
        os.fsync(cache.fileno())
    if errores:
        print(f"  {errores} imágenes no se pudieron abrir: quedan fuera de _cajas.csv.")


# --- CNN de recortes -------------------------------------------------------------------------------------

RUTA_RECORTES = RAIZ_REPO / "CNN" / "models_gpu" / "newDataset" / "cajas" / "maestra_recortes.keras"
# Una caja sin dudas entre varias: la CNN final le da al personaje al menos UMBRAL_SEGURA, y al menos
# VENTAJA_SEGURA veces lo que le da en la segunda caja
UMBRAL_SEGURA = 0.50
VENTAJA_SEGURA = 3.0
MINIMO_SEGURAS = 1000
# La receta de los notebooks: Adam, ReduceLROnPlateau a la mitad, parada temprana con los mejores pesos,
# suavizado de etiquetas, pesos por clase y volteo horizontal
EPOCAS_RECORTES = 40
PACIENCIA_RECORTES = 6
PACIENCIA_LR_RECORTES = 3
TASA_RECORTES = 1e-3
TASA_MINIMA_RECORTES = 1e-5
LOTE_RECORTES = 128
SUAVIZADO_RECORTES = 0.1
VALIDACION_RECORTES = 0.10


def caja_segura(registro: dict) -> int | None:
    """La caja que es, sin dudas, el personaje de la carpeta según la CNN final, o None. Es la única caja de
    la imagen (si la CNN ve al personaje en ella o en la imagen completa), o una que la CNN reconoce
    claramente mejor que las demás. Son los ejemplos con que se entrena la CNN de recortes; los elige la CNN
    final y no la de recortes, para que esta no se refuerce a sí misma."""
    cajas = registro["cajas"]
    if not cajas:
        return None
    if len(cajas) == 1:
        visible = max(cajas[0]["p_carpeta"], registro["completa"]["p_carpeta"]) >= UMBRAL_ASIGNACION
        return 0 if visible else None
    orden = sorted(range(len(cajas)), key=lambda i: -cajas[i]["p_carpeta"])
    mejor, segunda = cajas[orden[0]]["p_carpeta"], cajas[orden[1]]["p_carpeta"]
    return orden[0] if mejor >= UMBRAL_SEGURA and mejor >= VENTAJA_SEGURA * segunda else None


def _recortes(dataset: Path, registro: dict, indices: list[int], tamano: tuple[int, int]) -> list[np.ndarray]:
    imagen = G.abrir_rgb(dataset / registro["personaje"] / registro["archivo"])
    return [np.array(imagen.crop(tuple(registro["cajas"][i]["recorte"])).resize(tamano, Image.Resampling.BILINEAR),
                     dtype=np.uint8) for i in indices]


def entrenar_recortes(maestra: Maestra, dataset: Path, registros: dict, semilla: int):
    """Entrena desde cero, con la arquitectura de la CNN final, una CNN que solo ve los recortes sin dudas,
    cada uno con el personaje de su carpeta. Deja un 10 % de cada personaje aparte para validar. Devuelve
    (modelo, resumen), o None si no hay MINIMO_SEGURAS recortes."""
    import keras

    torch = maestra.torch
    seguras = [(registro, indice) for registro in registros.values() if (indice := caja_segura(registro)) is not None]
    if len(seguras) < MINIMO_SEGURAS:
        print(f"\nCNN de recortes: solo hay {len(seguras)} recortes sin dudas (hacen falta {MINIMO_SEGURAS}); las "
              "cajas quedan con los nombres de la CNN final.")
        return None
    inicio = time.time()
    print(f"\nEntrenando desde cero la CNN de recortes con {len(seguras)} recortes sin dudas de {len(registros)} "
          "originales...", flush=True)
    with ThreadPoolExecutor(8) as hilos:
        X = np.stack(list(hilos.map(lambda par: _recortes(dataset, par[0], [par[1]], maestra.tamano)[0], seguras)))
    y = np.array([maestra.clases.index(registro["personaje"]) for registro, _ in seguras], dtype=np.int64)
    print(f"  recortes cargados en {time.time() - inicio:.0f} s", flush=True)
    azar = np.random.default_rng(semilla)
    validacion = np.zeros(len(y), dtype=bool)
    for clase in np.unique(y):
        filas = np.flatnonzero(y == clase)
        if len(filas) >= 2:
            azar.shuffle(filas)
            validacion[filas[:max(1, round(VALIDACION_RECORTES * len(filas)))]] = True
    n_clases = len(maestra.clases)
    conteo = np.bincount(y[~validacion], minlength=n_clases)
    pesos = np.where(conteo > 0, np.sqrt(conteo.sum() / (n_clases * np.maximum(conteo, 1))), 1.0)

    dispositivo = maestra.dispositivo
    X_entrena, y_entrena = (torch.as_tensor(a).to(dispositivo) for a in (X[~validacion], y[~validacion]))
    X_valida, y_valida = (torch.as_tensor(a).to(dispositivo) for a in (X[validacion], y[validacion]))
    pesos = torch.as_tensor(pesos, dtype=torch.float32, device=dispositivo)
    del X

    def evaluar(modelo):
        modelo.train(False)
        with torch.no_grad():
            p = torch.cat([modelo(X_valida[i:i + 512], training=False) for i in range(0, len(X_valida), 512)])
        filas = torch.arange(len(y_valida), device=dispositivo)
        perdida = float(-torch.log(p[filas, y_valida].clamp_min(1e-7)).mean())
        return perdida, float((p.argmax(1) == y_valida).float().mean())

    _, acierto_final = evaluar(maestra.modelo)
    print(f"  referencia: la CNN final, que se entrenó con estos mismos fotogramas completos, nombra bien el "
          f"{acierto_final:.1%} de los {len(y_valida)} recortes de validación", flush=True)
    keras.utils.set_random_seed(semilla)
    # Misma arquitectura, pesos nuevos: nada de lo que la CNN final aprendió del escenario
    modelo = keras.models.clone_model(maestra.modelo).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_RECORTES)
    generador = torch.Generator(device=dispositivo)
    generador.manual_seed(semilla)
    mejor_perdida, mejor_epoca, mejor_acierto, mejores_pesos = float("inf"), 0, 0.0, None
    sin_mejora = sin_mejora_lr = 0
    for epoca in range(1, EPOCAS_RECORTES + 1):
        inicio_epoca = time.time()
        modelo.train(True)
        orden = torch.randperm(len(X_entrena), device=dispositivo, generator=generador)
        for desde in range(0, len(X_entrena), LOTE_RECORTES):
            indices = orden[desde:desde + LOTE_RECORTES]
            lote, objetivo = X_entrena[indices], y_entrena[indices]
            # Volteo horizontal de la mitad del lote: un personaje que mira al otro lado sigue siendo él
            voltear = torch.rand(len(indices), device=dispositivo, generator=generador) < 0.5
            lote = torch.where(voltear[:, None, None, None], lote.flip(2), lote)
            probabilidades = modelo(lote, training=True)
            suave = (torch.nn.functional.one_hot(objetivo, n_clases).float() * (1 - SUAVIZADO_RECORTES)
                     + SUAVIZADO_RECORTES / n_clases)
            perdida = (-(suave * torch.log(probabilidades.clamp_min(1e-7))).sum(1) * pesos[objetivo]).mean()
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()
        perdida_valida, acierto_valida = evaluar(modelo)
        print(f"  época {epoca:2d}: pérdida de validación {perdida_valida:.4f}, nombra bien el {acierto_valida:.1%}"
              f"  ({time.time() - inicio_epoca:.0f} s)", flush=True)
        if perdida_valida < mejor_perdida - 1e-6:
            mejor_perdida, mejor_epoca, mejor_acierto = perdida_valida, epoca, acierto_valida
            mejores_pesos = {nombre: t.detach().cpu().clone() for nombre, t in modelo.state_dict().items()}
            sin_mejora = sin_mejora_lr = 0
            continue
        sin_mejora += 1
        sin_mejora_lr += 1
        if sin_mejora >= PACIENCIA_RECORTES:
            break
        if sin_mejora_lr >= PACIENCIA_LR_RECORTES:
            for grupo in optimizador.param_groups:
                grupo["lr"] = max(grupo["lr"] / 2, TASA_MINIMA_RECORTES)
            sin_mejora_lr = 0
            print(f"  la tasa de aprendizaje baja a {optimizador.param_groups[0]['lr']:.1e}", flush=True)
    modelo.load_state_dict({nombre: t.to(dispositivo) for nombre, t in mejores_pesos.items()})
    modelo.train(False)
    del X_entrena, y_entrena, X_valida, y_valida
    if dispositivo == "cuda":
        torch.cuda.empty_cache()
    resumen = {"ejemplos": len(y), "validacion": int(validacion.sum()), "mejor_epoca": mejor_epoca,
               "accuracy_validacion": round(mejor_acierto, 4),
               "accuracy_validacion_cnn_final": round(acierto_final, 4), "segundos": round(time.time() - inicio, 1)}
    print(f"  la CNN de recortes nombra bien el {mejor_acierto:.1%} de los recortes de validación, que no vio "
          f"(mejor época {mejor_epoca}, {resumen['segundos']:.0f} s)")
    return modelo, resumen


def huella(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def obtener_recortes(maestra: Maestra, dataset: Path, registros: dict, guardar: bool, reentrenar: bool, semilla: int,
                     usar_cpu: bool) -> tuple[Maestra | None, str | None]:
    """La CNN de recortes: la guardada en RUTA_RECORTES, o una nueva. Solo se guarda cuando se entrenó con
    todo el dataset; la de un piloto queda en memoria."""
    if RUTA_RECORTES.exists() and not reentrenar:
        metadatos = json.loads(RUTA_RECORTES.with_suffix(".json").read_text(encoding="utf-8"))
        if "recortes" in metadatos:
            return Maestra(RUTA_RECORTES, usar_cpu), huella(RUTA_RECORTES)
        print(f"\n{RUTA_RECORTES.name} es de una versión anterior del script (afinada desde la CNN final): se "
              "entrena de nuevo.")
    resultado = entrenar_recortes(maestra, dataset, registros, semilla)
    if resultado is None:
        return None, None
    modelo, resumen = resultado
    configuracion = f"Recortes ({resumen['ejemplos']} sin dudas), desde cero: {maestra.configuracion}"
    if not guardar:
        return Maestra.con_modelo(maestra, modelo, configuracion), "piloto"
    RUTA_RECORTES.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(RUTA_RECORTES)
    metadatos = json.loads(maestra.ruta.with_suffix(".json").read_text(encoding="utf-8"))
    metadatos.update({"configuracion": configuracion, "recortes": resumen})
    RUTA_RECORTES.with_suffix(".json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"  guardada en {RUTA_RECORTES}")
    return Maestra(RUTA_RECORTES, usar_cpu), huella(RUTA_RECORTES)


def rotular_con_recortes(recortes: Maestra, identificador: str, dataset: Path, registros: dict) -> None:
    """Vuelve a nombrar todas las cajas con la CNN de recortes (queda en cada caja como 'recortes')."""
    pendientes = [registro for registro in registros.values() if registro.get("recortes_id") != identificador]
    if not pendientes:
        return
    print(f"\nRotulando con la CNN de recortes las cajas de {len(pendientes)} originales...", flush=True)
    inicio = time.time()

    def recortes_de(registro):
        indices = list(range(len(registro["cajas"])))
        return _recortes(dataset, registro, indices, recortes.tamano) if indices else []

    with ThreadPoolExecutor(8) as hilos:
        for desde in range(0, len(pendientes), 256):
            grupo = pendientes[desde:desde + 256]
            listas = list(hilos.map(recortes_de, grupo))
            planos = [recorte for lista in listas for recorte in lista]
            probabilidades = recortes.predecir(np.stack(planos)) if planos else None
            k = 0
            for registro in grupo:
                indice = recortes.clases.index(registro["personaje"])
                for caja in registro["cajas"]:
                    p = probabilidades[k]
                    k += 1
                    mejor = int(p.argmax())
                    caja.pop("afinada", None)  # de la versión anterior del script
                    caja["recortes"] = {"nombre": recortes.clases[mejor], "confianza": round(float(p[mejor]), 4),
                                        "p_carpeta": round(float(p[indice]), 4)}
                registro.pop("afinada_id", None)
                registro["recortes_id"] = identificador
    print(f"  listo en {time.time() - inicio:.0f} s")


# --- Salidas ---------------------------------------------------------------------------------------------

def _fila_base(personaje: str, archivo: str, original: str, tipo: str, registro: dict) -> dict:
    return {"personaje": personaje, "archivo": archivo, "original": original, "tipo": tipo,
            "ancho": registro["ancho"], "alto": registro["alto"], "p_completa": registro["completa"]["p_carpeta"],
            "n_cajas": len(registro["cajas"])}


def fila_original(registro: dict) -> dict:
    """La fila de _cajas.csv de una original, con la caja elegida o la imagen completa."""
    indice, motivo = elegir(registro["cajas"])
    fila = _fila_base(registro["personaje"], registro["archivo"], registro["archivo"], "original", registro)
    fila["motivo"] = motivo
    if indice is None:
        completa = registro["completa"]
        return {**fila, "fuente": "completa", "recorte_x1": 0, "recorte_y1": 0, "recorte_x2": registro["ancho"],
                "recorte_y2": registro["alto"], "nombre": completa["nombre"], "confianza": completa["confianza"],
                "p_carpeta": completa["p_carpeta"]}
    caja = registro["cajas"][indice]
    nombre = puntuacion(caja)
    return {**fila, "fuente": "caja", **_columnas("caja", caja["caja"]), **_columnas("recorte", caja["recorte"]),
            "puntaje": caja["puntaje"], "nombre": nombre["nombre"], "confianza": nombre["confianza"],
            "p_carpeta": nombre["p_carpeta"]}


def _columnas(prefijo: str, valores) -> dict:
    return {f"{prefijo}_{eje}": valor for eje, valor in zip(("x1", "y1", "x2", "y2"), valores)}


def fila_variante(fila: dict, archivo: str, tipo: str, rotacion: float) -> dict:
    """La fila de una variante: la caja de su original, llevada a través del mismo giro."""
    variante = {**fila, "archivo": archivo, "tipo": tipo}
    if fila["fuente"] != "caja":
        return variante
    ancho, alto = fila["ancho"], fila["alto"]
    caja = G.caja_tras_giro(tuple(fila[f"caja_{eje}"] for eje in ("x1", "y1", "x2", "y2")), ancho, alto, rotacion)
    if caja is None:  # el personaje quedó fuera del recorte del giro
        return {**variante, "fuente": "completa", "motivo": "fuera_de_la_variante",
                **_columnas("caja", ("", "", "", "")), **_columnas("recorte", (0, 0, ancho, alto))}
    return {**variante, **_columnas("caja", [round(v, 1) for v in caja]),
            **_columnas("recorte", recorte_cuadrado(caja, ancho, alto))}


def _leer_csv(ruta: Path) -> list[dict]:
    if not ruta.exists():
        return []
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def construir_registros(dataset: Path, registros: dict) -> tuple[list[dict], list[dict]]:
    """Las filas de _cajas.csv (originales y variantes) y de _cajas_candidatas.csv."""
    filas, candidatas = [], []
    por_original = {}
    for (personaje, archivo), registro in sorted(registros.items()):
        fila = fila_original(registro)
        filas.append(fila)
        por_original[(personaje, archivo)] = fila
        elegida, _ = elegir(registro["cajas"])
        for i, caja in enumerate(registro["cajas"]):
            nombre = puntuacion(caja)
            candidatas.append({"personaje": personaje, "archivo": archivo, **_columnas("caja", caja["caja"]),
                               **_columnas("recorte", caja["recorte"]), "puntaje": caja["puntaje"],
                               "nombre": nombre["nombre"], "confianza": nombre["confianza"],
                               "p_carpeta": nombre["p_carpeta"], "elegida": i == elegida})
    sin_original = 0
    for registro, columna, tipo in ((G.REGISTRO, "alterada", "alterada"), (G.REGISTRO_PARECIDAS, "parecida", "parecida")):
        for variante in _leer_csv(dataset / registro):
            if not (dataset / variante["personaje"] / variante[columna]).is_file():
                continue  # una variante que se borró del dataset: el notebook tampoco la ve
            fila = por_original.get((variante["personaje"], variante["original"]))
            if fila is None:
                sin_original += 1
                continue
            filas.append(fila_variante(fila, variante[columna], tipo, float(variante["rotacion"])))
    if sin_original:
        print(f"  {sin_original} variantes sin su original en la caché: quedan fuera de _cajas.csv.")
    filas.sort(key=lambda f: (f["personaje"], f["archivo"]))
    return filas, candidatas


def cobertura(registros: dict) -> list[dict]:
    """Por personaje, cuántas originales entran con caja y por qué entran las demás completas."""
    por_personaje = {}
    for (personaje, _), registro in registros.items():
        _, motivo = elegir(registro["cajas"])
        por_personaje.setdefault(personaje, Counter())[motivo] += 1
    tabla = []
    for personaje, motivos in sorted(por_personaje.items()):
        total = sum(motivos.values())
        con_caja = motivos["cnn"]
        tabla.append({"personaje": personaje, "originales": total, "con_caja": con_caja,
                      "proporcion": con_caja / total, **motivos})
    return tabla


def imprimir_cobertura(tabla: list[dict]) -> list[str]:
    """Imprime la cobertura de menor a mayor y devuelve los personajes bajo COBERTURA_MINIMA."""
    motivos = ("cnn", "nadie_lo_reconoce", "sin_detecciones")
    print(f"\n{'personaje':<30}{'originales':>11}{'con caja':>10}{'':>3}" + "".join(f"{m:>19}" for m in motivos))
    bajos = []
    for fila in sorted(tabla, key=lambda f: f["proporcion"]):
        marca = " <" if fila["proporcion"] < COBERTURA_MINIMA else ""
        if marca:
            bajos.append(fila["personaje"])
        print(f"{fila['personaje']:<30}{fila['originales']:>11}{fila['proporcion']:>10.1%}{marca:>3}"
              + "".join(f"{fila.get(m, 0):>19}" for m in motivos))
    total = sum(f["originales"] for f in tabla)
    con_caja = sum(f["con_caja"] for f in tabla)
    print(f"\nTotal: {con_caja} de {total} originales con caja ({con_caja / max(total, 1):.1%}) en {len(tabla)} personajes.")
    if bajos:
        print(f"Bajo {COBERTURA_MINIMA:.0%} con caja ({len(bajos)}): {', '.join(bajos)}. Sus imágenes sin caja "
              "entran completas, como antes; mira sus hojas en cajas_por_revisar/.")
    else:
        print(f"Los {len(tabla)} personajes tienen al menos {COBERTURA_MINIMA:.0%} de sus originales con caja.")
    return bajos


# --- Hojas de contacto -----------------------------------------------------------------------------------

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


def miniatura_rotulada(ruta: Path, registro: dict, ancho_miniatura: int = 360) -> Image.Image:
    """El fotograma con todas sus cajas: la elegida en verde y las demás en azul, cada una con el nombre y la
    confianza que le da la CNN. Si entra completo, un marco naranjo con el rótulo de la imagen completa."""
    imagen = G.abrir_rgb(ruta)
    escala = ancho_miniatura / imagen.size[0]
    imagen = imagen.resize((ancho_miniatura, max(1, round(imagen.size[1] * escala))), Image.Resampling.BILINEAR)
    dibujo = ImageDraw.Draw(imagen)
    fuente = _fuente(13)
    elegida, motivo = elegir(registro["cajas"])
    orden = sorted(range(len(registro["cajas"])), key=lambda i: i == elegida)  # la elegida se dibuja encima
    for i in orden:
        caja = registro["cajas"][i]
        x1, y1, x2, y2 = (v * escala for v in caja["caja"])
        nombre = puntuacion(caja)
        _rotular_caja(dibujo, (x1, y1, x2, y2), rotulo(nombre["nombre"], nombre["confianza"]),
                      VERDE if i == elegida else AZUL, fuente, 3 if i == elegida else 2)
    if elegida is None:
        completa = registro["completa"]
        _rotular_caja(dibujo, (1, 1, imagen.size[0] - 2, imagen.size[1] - 2),
                      f"{rotulo(completa['nombre'], completa['confianza'])} (imagen completa)", NARANJO, fuente, 4)
    return imagen


def hoja_de_contacto(dataset: Path, personaje: str, registros: list[dict], destino: Path, columnas: int = 4) -> None:
    fuente = _fuente(13)
    miniaturas = []
    for registro in registros:
        miniatura = miniatura_rotulada(dataset / personaje / registro["archivo"], registro)
        _, motivo = elegir(registro["cajas"])
        pie = Image.new("RGB", (miniatura.size[0], 20), (255, 255, 255))
        ImageDraw.Draw(pie).text((4, 3), f"{registro['archivo']}  |  {motivo}", fill=(60, 60, 60), font=fuente)
        con_pie = Image.new("RGB", (miniatura.size[0], miniatura.size[1] + 20), (255, 255, 255))
        con_pie.paste(miniatura, (0, 0))
        con_pie.paste(pie, (0, miniatura.size[1]))
        miniaturas.append(con_pie)
    if not miniaturas:
        return
    ancho_celda = max(m.size[0] for m in miniaturas) + 8
    alto_celda = max(m.size[1] for m in miniaturas) + 8
    filas = -(-len(miniaturas) // columnas)
    hoja = Image.new("RGB", (columnas * ancho_celda + 8, filas * alto_celda + 40), (245, 245, 245))
    ImageDraw.Draw(hoja).text(
        (10, 10), f"{nombre_legible(personaje)} (carpeta {personaje}): verde = la caja que entra al notebook, "
                  "azul = otros personajes, naranjo = imagen completa", fill=(0, 0, 0), font=_fuente(16))
    for k, miniatura in enumerate(miniaturas):
        hoja.paste(miniatura, (8 + (k % columnas) * ancho_celda, 40 + (k // columnas) * alto_celda))
    hoja.save(destino, quality=85)


def escribir_galerias(dataset: Path, registros: dict, semilla: int) -> Path:
    """Una hoja por personaje en cajas_por_revisar/, junto al dataset. Primero las imágenes más difíciles
    (sin caja de la CNN o con varias cajas) y el resto al azar."""
    carpeta = dataset.parent / NOMBRE_GALERIA
    G.limpiar_revision(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / G.MARCA_REVISION).write_text(TEXTO_MARCA_GALERIA, encoding="utf-8")
    por_personaje = {}
    for (personaje, _), registro in sorted(registros.items()):
        por_personaje.setdefault(personaje, []).append(registro)
    for personaje, lista in por_personaje.items():
        azar = random.Random(f"{semilla}-{personaje}")
        azar.shuffle(lista)
        es_dificil = [elegir(r["cajas"])[1] != "cnn" or len(r["cajas"]) > 1 for r in lista]
        dificiles = [r for r, dificil in zip(lista, es_dificil) if dificil]
        resto = [r for r, dificil in zip(lista, es_dificil) if not dificil]
        mitad = MUESTRAS_GALERIA // 2
        muestra = dificiles[:mitad] + resto[:MUESTRAS_GALERIA - min(mitad, len(dificiles))]
        muestra += dificiles[mitad:mitad + MUESTRAS_GALERIA - len(muestra)]
        hoja_de_contacto(dataset, personaje, muestra[:MUESTRAS_GALERIA], carpeta / f"{personaje}.jpg")
    return carpeta


# --- Programa --------------------------------------------------------------------------------------------

def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("SIMPSONS_DATASET", str(G.RUTA_POR_DEFECTO))),
                        help="carpeta con una subcarpeta por personaje (por defecto %(default)s)")
    parser.add_argument("--muestra", type=int, help="piloto: solo N originales por personaje; no escribe _cajas.csv")
    parser.add_argument("--personajes", type=lambda t: [p.strip() for p in t.split(",") if p.strip()],
                        help="solo estos personajes, separados por comas")
    parser.add_argument("--detector", default=DETECTOR, help="modelo de Hugging Face (por defecto %(default)s)")
    parser.add_argument("--modelo", type=Path, help="CNN que rotula las cajas (.keras con su .json al lado)")
    parser.add_argument("--rehacer", action="store_true", help="vuelve a detectar aunque estén en la caché")
    parser.add_argument("--reentrenar", action="store_true", help="vuelve a entrenar la CNN de recortes")
    parser.add_argument("--sin-recortes", action="store_true",
                        help="no entrena la CNN de recortes: rotula solo con la CNN final")
    parser.add_argument("--sin-galerias", action="store_true", help="no genera las hojas de cajas_por_revisar/")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU (muy lento)")
    parser.add_argument("--semilla", type=int, default=G.SEMILLA, help="para la muestra y las hojas (%(default)s)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = leer_argumentos(argv)
    dataset = args.dataset
    carpetas = G.validar(dataset)
    todas = [(carpeta.name, ruta) for carpeta in carpetas for ruta in G.originales_de(carpeta)]
    seleccion = todas
    if args.personajes:
        desconocidos = sorted(set(args.personajes) - {c.name for c in carpetas})
        if desconocidos:
            raise SystemExit(f"No hay carpetas para: {', '.join(desconocidos)}")
        seleccion = [(p, r) for p, r in seleccion if p in set(args.personajes)]
    if args.muestra:
        azar = random.Random(args.semilla)
        por_personaje = {}
        for personaje, ruta in seleccion:
            por_personaje.setdefault(personaje, []).append(ruta)
        seleccion = [(p, r) for p, rutas in por_personaje.items() for r in sorted(azar.sample(rutas, min(args.muestra, len(rutas))))]

    ruta_cache = dataset / CACHE
    registros = leer_cache(ruta_cache)
    maestra = Maestra(args.modelo, args.cpu)
    faltan_clases = sorted({p for p, _ in seleccion} - set(maestra.clases))
    if faltan_clases:
        raise SystemExit(f"La CNN no conoce a: {', '.join(faltan_clases)}. Entrénala con esos personajes antes.")
    config = configuracion(args.detector, maestra.configuracion)
    pendientes = [(p, r) for p, r in seleccion if args.rehacer or registros.get((p, r.name), {}).get("config") != config]
    print(f"Dataset: {dataset}")
    print(f"Detector: {args.detector} (lado {LADO_DETECTOR}, umbral {UMBRAL_CAJA}) | rotula: CNN "
          f"{maestra.configuracion} ({maestra.dispositivo})")
    print(f"Originales: {len(seleccion)} seleccionadas, {len(seleccion) - len(pendientes)} ya en la caché, "
          f"{len(pendientes)} por detectar.")
    if pendientes:
        detector = Detector(args.detector, "cpu" if args.cpu else None)
        try:
            detectar_originales(pendientes, detector, maestra, ruta_cache, registros, config)
        except KeyboardInterrupt:
            print("\nDetenido. Lo detectado quedó en la caché; vuelve a ejecutar para seguir.")
            return 3
        # El detector ya no hace falta: su memoria de la GPU queda libre para entrenar la CNN de recortes
        del detector
        if maestra.dispositivo == "cuda":
            maestra.torch.cuda.empty_cache()

    elegidos = {(p, r.name) for p, r in seleccion}
    actuales = {k: v for k, v in registros.items() if k in elegidos and v.get("config") == config}
    completos = {k: v for k, v in registros.items() if v.get("config") == config}
    faltan = [(p, r) for p, r in todas if (p, r.name) not in completos]
    # Los registros se escriben solo con todo el dataset detectado, y fuera del piloto
    final = not args.muestra and not faltan
    alcance = {(p, r.name): completos[(p, r.name)] for p, r in todas} if final else actuales
    if not args.sin_recortes:
        recortes, identificador = obtener_recortes(maestra, dataset, alcance, final, args.reentrenar, args.semilla,
                                                   args.cpu)
        if recortes is not None:
            rotular_con_recortes(recortes, identificador, dataset, alcance)
    bajos = imprimir_cobertura(cobertura(alcance))
    if not args.sin_galerias:
        carpeta = escribir_galerias(dataset, alcance, args.semilla)
        print(f"Hojas de contacto: {carpeta}")

    if args.muestra:
        print("\nPiloto: no se escribe _cajas.csv. Ejecuta sin --muestra para procesar todo el dataset.")
        return 0
    if faltan:
        print(f"\n_cajas.csv no se escribe: faltan {len(faltan)} originales por detectar. Ejecuta sin --personajes.")
        return 0
    vigentes = alcance
    filas, candidatas = construir_registros(dataset, vigentes)
    G.escribir_registro(dataset / REGISTRO_CAJAS, CAMPOS_CAJAS, filas)
    G.escribir_registro(dataset / REGISTRO_CANDIDATAS, CAMPOS_CANDIDATAS, candidatas)
    # La caché queda con una línea por original vigente: las de configuraciones anteriores sobran
    temporal = ruta_cache.with_name(ruta_cache.name + G.SUFIJO_TEMPORAL)
    with open(temporal, "w", encoding="utf-8") as archivo:
        for registro in vigentes.values():
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    os.replace(temporal, ruta_cache)
    tipos = Counter((f["tipo"], f["fuente"]) for f in filas)
    print(f"\n{dataset / REGISTRO_CAJAS}: {len(filas)} imágenes "
          + ", ".join(f"{tipo} {fuente} {n}" for (tipo, fuente), n in sorted(tipos.items())))
    print(f"{dataset / REGISTRO_CANDIDATAS}: {len(candidatas)} cajas en {len(vigentes)} originales.")
    if bajos:
        print("Revisa las hojas de los personajes marcados con '<' antes de entrenar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
