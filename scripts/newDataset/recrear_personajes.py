"""Recrea a cada personaje solo desde un modelo ya entrenado y comprueba si las otras redes lo reconocen.

La técnica es DeepInversion (Yin et al., 2020). Se parte de ruido y se optimizan los píxeles de un lote con una
imagen por personaje; los pesos del modelo no cambian. La pérdida tiene tres partes:

1. Que cada imagen maximice la clase de su personaje: entropía cruzada sobre los logits de la capa `salida`.
2. Que la media y la varianza de la entrada de cada BatchNorm, medidas en el lote, coincidan con las que esa
   capa guardó al entrenar (moving_mean y moving_variance): el promedio del entrenamiento que quedó dentro del
   modelo. La primera BatchNorm pesa 10 veces más.
3. Que la imagen sea suave: variación total y norma L2, más un desplazamiento y un volteo al azar en cada paso.

Punto de partida: 2000 pasos, lr 0,05, α_bn 0,01, α_tv 1e-3 (el 1e-4 inicial dejaba ruido; ver Hiperparametros) y
α_l2 1e-5.

No se lee ninguna imagen del dataset. Cada recreación se cuantiza después a uint8, como una imagen cualquiera, y
la clasifican todas las redes. Una red siempre reconoce lo que generó; lo que prueba el entrenamiento es que las
otras nombren al mismo personaje. Al azar acertarían 1 de 64 (1,6 %). Para saber si dos redes son de verdad
independientes, el script mide además la correlación de sus pesos: una red afinada desde otra comparte casi todo
lo que aprendió, y que reconozca las recreaciones de la otra no es una prueba independiente.

Modelos (--modelos). Todos ya entrenados: el script no entrena ni modifica ninguno.
    cnn            CNN/models_gpu/newDataset/cnn_final.keras                           CNN, fotograma completo
    cnn_recortes   CNN/models_gpu/newDataset/cajas/cnn_final.keras, o si no existe
                   cajas/maestra_recortes.keras                                        CNN, recortes
    mlp            PMC/models_gpu/newDataset/mlp_r1.keras, cajas/mlp_r1.keras o
                   mlp_final.keras si recibe píxeles                                   MLP, píxeles
Si falta alguno, se avisa y se sigue con los demás. Un MLP con capa Normalization recibe descriptores (los
histogramas del MLP final se calculan fuera de la red, sin gradiente hacia los píxeles) y no se puede recrear.

Generadores (--generadores). Las redes de --modelos que recrean personajes; por defecto, todas. Las demás solo
juzgan las recreaciones. Así cada notebook genera imágenes solo con su red: el del MLP usa --generadores mlp y el
de la CNN --generadores cnn,cnn_recortes, y las otras redes siguen haciendo la prueba cruzada.

Salidas. Por defecto, las de las CNN van a CNN/resultados/newDataset/recreaciones/ y las del MLP a
PMC/resultados/newDataset/recreaciones/; con --salida, todo va a esa carpeta. Solo las redes que generan dejan
salidas:
    <modelo>/<personaje>.png   la recreación de cada personaje, en uint8 y a la resolución del modelo
    <modelo>/lamina.png        las recreaciones en una lámina de 8 x 8, con lo que dice cada red
    <modelo>/recreacion.json   modelo y sha256, hiperparámetros, semilla, tiempo, filas de la matriz y detalle
    comparacion.png            los 12 personajes de la diapositiva de F1, una fila por red que genera
    evolucion.png              del ruido a la recreación de Homero, Marge, Bart y Lisa, por red que genera
comparacion.png y evolucion.png van a la carpeta de las redes que generan si todas son del mismo notebook (con
--generadores mlp, a la del MLP) y, si se mezclan, a la de las CNN.
Con --semilla, cada recreación queda en caché junto a su modelo (models_gpu/.../recreaciones/, ignorada por Git):
con el mismo modelo, los mismos hiperparámetros, la misma semilla y los mismos personajes se reutiliza, así los
dos notebooks muestran exactamente las mismas imágenes. --rehacer la vuelve a calcular.

Uso, desde la raíz del repositorio (entorno .venv-gpu):
    .venv-gpu/Scripts/python.exe scripts/newDataset/recrear_personajes.py --semilla 42
    .venv-gpu/Scripts/python.exe scripts/newDataset/recrear_personajes.py --generadores mlp --semilla 42
    .venv-gpu/Scripts/python.exe scripts/newDataset/recrear_personajes.py --pasos 300 \\
        --personajes homer_simpson,marge_simpson,bart_simpson,lisa_simpson --salida piloto/
Al terminar imprime líneas con un formato fijo, para que la presentación las lea:
    RECREACION generador=cnn juez=cnn_recortes top1=60/64 top3=63/64 confianza=0.873
    RECREACION_MODELO generador=cnn sha256=<64 hex> segundos=75.3 semilla=42 pasos=2000 personajes=64
    RECREACION_PASO generador=cnn paso=0 top1=1/64 confianza=0.016 mas_nombrado=sideshow_mel:40 confianza_prediccion=0.712
    RECREACION_PESOS a=cnn b=cnn_recortes correlacion=0.999

Pruebas (sin GPU): .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_recrear_personajes.py -q
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

# El backend se fija antes de importar Keras: los modelos se entrenaron con Keras sobre PyTorch
os.environ.setdefault("KERAS_BACKEND", "torch")

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[1]
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import detectar_cajas as D  # noqa: E402  (los rótulos «Homer Simpson · 97 %»)

PREDECIR_IMAGEN = RAIZ_REPO / "CNN" / "scripts" / "newDataset" / "predecir_imagen.py"
MODELOS_CNN = RAIZ_REPO / "CNN" / "models_gpu" / "newDataset"
MODELOS_MLP = RAIZ_REPO / "PMC" / "models_gpu" / "newDataset"
SALIDA_CNN = RAIZ_REPO / "CNN" / "resultados" / "newDataset" / "recreaciones"
SALIDA_MLP = RAIZ_REPO / "PMC" / "resultados" / "newDataset" / "recreaciones"
CARPETA_CACHE = "recreaciones"
VERSION_CACHE = 1

# Pasos en que se guarda una instantánea del lote, además del último
PASOS_INSTANTANEAS = (0, 50, 200, 500)
# Los de la figura de evolución: los cuatro de los mapas de activación del notebook de la CNN
PERSONAJES_EVOLUCION = ("homer_simpson", "marge_simpson", "bart_simpson", "lisa_simpson")
# Los doce de la diapositiva «Por personaje» (F1 en test) de CNN/scripts/newDataset/generar_presentacion.py
PERSONAJES_F1 = (
    "apu_nahasapeemapetilon", "comic_book_guy", "chief_wiggum", "sideshow_bob", "marge_simpson",
    "homer_simpson", "lisa_simpson", "bart_simpson", "martin_prince", "maggie_simpson",
    "rainier_wolfcastle", "disco_stu",
)
# Sobre esta correlación de pesos, dos redes comparten el origen (una se afinó desde la otra)
CORRELACION_DEPENDIENTE = 0.9
COLOR_ACIERTO = "#1b7f3b"
COLOR_ERROR = "#c0392b"
COLUMNAS_LAMINA = 8
# Épsilon de la BatchNorm de Keras: evita dividir por una varianza guardada nula (una neurona que nunca se activó)
EPSILON_BN = 1e-3


@dataclass
class Hiperparametros:
    pasos: int = 2000
    lr: float = 0.05
    alfa_bn: float = 0.01
    # El punto de partida era 1e-4. En el piloto (Homero, Marge, Bart y Lisa, 300 pasos, semilla 42) dejaba ruido
    # de alta frecuencia. Con los 64 personajes y 2000 pasos, 5e-3 lo borraba todo hasta dejar campos de color, y
    # 1e-3 deja formas: el pelo azul de Marge sobre caras amarillas, las puntas de Bart y de Lisa. Con los tres
    # valores las dos CNN nombran a los 64 personajes
    alfa_tv: float = 1e-3
    alfa_l2: float = 1e-5
    desplazamiento: int = 16
    multiplicador_primera_bn: float = 10.0
    beta1: float = 0.5
    beta2: float = 0.9
    # Agregados para el MLP. Con estos valores por defecto la recreación es exactamente la de antes, y la caché de
    # las CNN sigue valiendo (ver clave_hiper)
    # Volteo horizontal al azar en cada paso
    volteo: bool = True
    # BatchNorm en unidades de la varianza guardada: cada neurona pesa lo mismo, sea cual sea su escala
    bn_normalizada: bool = False
    # Resoluciones por las que pasa la imagen, de la más gruesa a la final (octavas); vacía, solo la del modelo
    escalas: tuple = ()
    # Imágenes por personaje en el lote: queda la que la red que genera reconoce con más probabilidad
    reinicios: int = 1


# Los campos que tenía Hiperparametros antes de los agregados para el MLP: siempre entran en la clave de la caché
CAMPOS_ORIGINALES = ("pasos", "lr", "alfa_bn", "alfa_tv", "alfa_l2", "desplazamiento", "multiplicador_primera_bn",
                     "beta1", "beta2")

# El MLP recibe la imagen aplanada: un peso por píxel y por canal, sin noción de vecindad ni de traslación.
# Con los hiperparámetros de la CNN reconocía 30 de sus 64 recreaciones (probabilidad media 0,18). Barrido del
# 09-11-2026 (scripts/newDataset/barrido_recreacion.py, 16 personajes de ajuste, 2000 pasos, 4 reinicios, 36
# configuraciones): con todas el MLP reconoce a los 16, y ninguna saca a las CNN del azar (posición media del
# personaje entre 28,7 y 39,1; al azar 32,5 ± 4,6 con 16 personajes; top-1 0 de 16 en todas). Como las diferencias
# para las CNN son ruido, decide la confianza del propio MLP: esta es la segunda en posición para las CNN (29,4;
# la primera, 28,7, con α_bn 0,1 y desplazamiento 4) y el MLP la reconoce con 0,99 en vez de 0,76
HIPER_MLP = Hiperparametros(
    pasos=3000,
    # Sin desplazamiento ni volteo: el MLP no es invariante a la traslación. Con los ±16 píxeles de la CNN cada
    # paso optimizaba otra alineación y se juzgaba sin desplazar
    desplazamiento=0,
    volteo=False,
    # Sin el término de BatchNorm. Sin normalizar aportaba 39,4 de los 43,8 de la pérdida y oscilaba entre 1.000 y
    # 100.000 de un paso a otro; normalizado, en el barrido 0,01 y 0,1 dejan al MLP menos seguro (0,58 a 0,95) sin
    # acercar las recreaciones a las CNN. Queda normalizado por si se vuelve a subir α_bn
    alfa_bn=0.0,
    bn_normalizada=True,
    # De 8 × 8 a 32 × 32, y de ahí ampliada a 128: lo que separa a los personajes para el MLP es la distribución del
    # color en el encuadre, no el detalle
    escalas=(8, 16, 32),
    reinicios=4,
)
# El MLP robusto se entrenó con desplazamientos y volteos: los de la recreación son los mismos que vio
HIPER_MLP_ROBUSTO = Hiperparametros(
    pasos=3000,
    desplazamiento=4,
    volteo=True,
    bn_normalizada=True,
    escalas=(16, 32, 64, 128),
    reinicios=4,
)


@dataclass(frozen=True)
class Especificacion:
    """Un modelo que se puede usar: dónde buscarlo, en orden, y dónde dejar sus salidas."""

    identificador: str
    nombre: str
    corto: str
    # Cómo se nombra la red dentro de una frase, en los títulos de las figuras
    en_frase: str
    candidatos: tuple
    salida: Path
    # Metadatos (clases y resolución) que se usan si el .keras no tiene su .json al lado: el modelo final del
    # mismo notebook, con las mismas clases en el mismo orden
    metadatos_alternativos: tuple = ()
    # Los hiperparámetros con que recrea esta red; los flags de la línea de comandos los sobrescriben
    hiper: Hiperparametros = field(default_factory=Hiperparametros)


ESPECIFICACIONES = {
    "cnn": Especificacion(
        "cnn", "CNN, fotograma completo", "CNN fotograma", "la CNN de fotograma completo",
        (MODELOS_CNN / "cnn_final.keras",), SALIDA_CNN,
    ),
    "cnn_recortes": Especificacion(
        "cnn_recortes", "CNN, recortes", "CNN recortes", "la CNN de recortes",
        (MODELOS_CNN / "cajas" / "cnn_final.keras", MODELOS_CNN / "cajas" / "maestra_recortes.keras"), SALIDA_CNN,
    ),
    "mlp": Especificacion(
        "mlp", "MLP, píxeles", "MLP píxeles", "el MLP de píxeles",
        (MODELOS_MLP / "mlp_r1.keras", MODELOS_MLP / "cajas" / "mlp_r1.keras", MODELOS_MLP / "mlp_final.keras"),
        SALIDA_MLP, ("mlp_final.json",), HIPER_MLP,
    ),
    "mlp_robusto": Especificacion(
        "mlp_robusto", "MLP robusto, píxeles", "MLP robusto", "el MLP robusto de píxeles",
        (MODELOS_MLP / "mlp_r1_robusto.keras",), SALIDA_MLP, ("mlp_final.json",), HIPER_MLP_ROBUSTO,
    ),
}


# --- Modelos ---------------------------------------------------------------------------------------------

def _predecir_imagen():
    especificacion = importlib.util.spec_from_file_location("predecir_imagen", PREDECIR_IMAGEN)
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def huella(ruta: Path) -> str:
    """sha256 del archivo del modelo: identifica exactamente qué pesos se usaron."""
    digesto = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digesto.update(bloque)
    return digesto.hexdigest()


def es_descriptores(modelo) -> bool:
    """Un modelo con capa Normalization recibe descriptores calculados fuera de la red, no píxeles."""
    import keras

    return any(isinstance(capa, keras.layers.Normalization) for capa in modelo.layers)


def cargar(ruta: Path, metadatos_alternativos: tuple = (), usar_cpu: bool = False):
    """(modelo, metadatos, dispositivo) con cargar_modelo de predecir_imagen.py. Si el .keras no tiene su .json
    al lado (los experimentos del MLP solo guardan el .keras), las clases salen del .json alternativo."""
    if ruta.with_suffix(".json").is_file():
        return _predecir_imagen().cargar_modelo(ruta, usar_cpu)
    alternativo = next((ruta.parent / nombre for nombre in metadatos_alternativos if (ruta.parent / nombre).is_file()), None)
    if alternativo is None:
        raise FileNotFoundError(f"{ruta} no tiene {ruta.with_suffix('.json').name} ni metadatos alternativos")
    # Los mismos pasos de cargar_modelo
    if usar_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import keras
    import torch

    modelo = keras.models.load_model(ruta, compile=False)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    # Solo las clases y la resolución, que son las mismas: la configuración, la entrada y las métricas del .json
    # alternativo son de otro modelo (mlp_final.json describe el MLP final, no el R1)
    completos = json.loads(alternativo.read_text(encoding="utf-8"))
    metadatos = {clave: completos[clave] for clave in ("clases", "alto", "ancho")}
    metadatos["metadatos_de"] = alternativo.name
    return modelo.to(dispositivo), metadatos, dispositivo


@dataclass
class Red:
    especificacion: Especificacion
    ruta: Path
    modelo: object
    metadatos: dict
    dispositivo: str
    sha256: str = ""

    @property
    def identificador(self) -> str:
        return self.especificacion.identificador

    @property
    def clases(self) -> list:
        return list(self.metadatos["clases"])

    @property
    def tamano(self) -> tuple:
        return int(self.metadatos["alto"]), int(self.metadatos["ancho"])

    @property
    def tipo(self) -> str:
        return "mlp" if len(self.modelo.input_shape) == 2 else "cnn"

    def probabilidades(self, imagenes: np.ndarray, lote: int = 256) -> np.ndarray:
        """(n, clases) para imágenes (n, alto, ancho, 3) en uint8, como una imagen cualquiera: si su tamaño no es
        el del modelo, se redimensionan con interpolación bilineal, como en el notebook."""
        import torch

        imagenes = redimensionar(np.asarray(imagenes, dtype=np.uint8), self.tamano)
        if self.tipo == "mlp":
            imagenes = imagenes.reshape(len(imagenes), -1)
        salidas = []
        with torch.no_grad():
            for inicio in range(0, len(imagenes), lote):
                entrada = torch.as_tensor(imagenes[inicio:inicio + lote]).to(self.dispositivo).float()
                salidas.append(np.asarray(self.modelo(entrada, training=False).detach().cpu(), dtype=np.float32))
        return np.concatenate(salidas)


def redimensionar(imagenes: np.ndarray, tamano: tuple) -> np.ndarray:
    from PIL import Image

    alto, ancho = tamano
    if imagenes.shape[1:3] == (alto, ancho):
        return imagenes
    return np.stack([np.array(Image.fromarray(i).resize((ancho, alto), Image.Resampling.BILINEAR), dtype=np.uint8)
                     for i in imagenes])


def resolver(especificacion: Especificacion, usar_cpu: bool) -> tuple:
    """(Red o None, avisos): el primer candidato que existe y recibe píxeles."""
    avisos = []
    for ruta in especificacion.candidatos:
        if not ruta.is_file():
            continue
        try:
            modelo, metadatos, dispositivo = cargar(ruta, especificacion.metadatos_alternativos, usar_cpu)
        except (FileNotFoundError, SystemExit) as error:
            avisos.append(f"{especificacion.identificador}: {ruta.name} no se pudo cargar ({error}).")
            continue
        if es_descriptores(modelo):
            avisos.append(f"{especificacion.identificador}: {ruta.name} recibe descriptores, no píxeles: no se puede "
                          "recrear.")
            continue
        return Red(especificacion, ruta, modelo, metadatos, dispositivo, huella(ruta)), avisos
    buscados = ", ".join(r.relative_to(RAIZ_REPO).as_posix() for r in especificacion.candidatos)
    avisos.append(f"{especificacion.identificador} ({especificacion.nombre}): no hay modelo que reciba píxeles en "
                  f"{buscados}; se sigue sin él.")
    return None, avisos


def comprobar_clases(redes: list) -> list:
    """Las clases de la primera red. Todas deben tener las mismas; si el orden cambia, se compara por nombre."""
    referencia = redes[0].clases
    avisos = []
    for red in redes[1:]:
        if set(red.clases) != set(referencia):
            raise SystemExit(f"{red.identificador} no tiene las mismas clases que {redes[0].identificador}: "
                             f"{sorted(set(red.clases) ^ set(referencia))}")
        if red.clases != referencia:
            avisos.append(f"{red.identificador} tiene las mismas clases en otro orden: se comparan por nombre.")
    return avisos


def correlacion_pesos(a, b) -> float | None:
    """Correlación entre los pesos de dos modelos con la misma arquitectura, o None si no la tienen. Cerca de 1,
    uno salió del otro; dos redes entrenadas desde cero, cada una con su inicialización, quedan cerca de 0."""
    pesos_a = {v.path: v for v in a.weights if v.path.endswith("kernel")}
    pesos_b = {v.path: v for v in b.weights if v.path.endswith("kernel")}
    formas_a = {p: tuple(v.shape) for p, v in pesos_a.items()}
    if not formas_a or formas_a != {p: tuple(v.shape) for p, v in pesos_b.items()}:
        return None
    comunes = sorted(pesos_a)
    vector_a = np.concatenate([np.asarray(pesos_a[p].numpy()).ravel() for p in comunes])
    vector_b = np.concatenate([np.asarray(pesos_b[p].numpy()).ravel() for p in comunes])
    return float(np.corrcoef(vector_a, vector_b)[0, 1])


# --- Recorrido y pérdida ---------------------------------------------------------------------------------

@dataclass
class Estadistica:
    """Media y varianza de la entrada de una BatchNorm en el lote, y las que la capa guardó al entrenar."""

    capa: str
    media: object
    varianza: object
    media_bn: object
    varianza_bn: object


def congelar(modelo) -> None:
    """Los pesos no reciben gradiente: solo se optimizan los píxeles."""
    for variable in modelo.weights:
        variable.value.requires_grad_(False)


def aplanar(x):
    """(n, alto, ancho, 3) -> (n, alto * ancho * 3) en orden C, como imagenes.reshape(N, -1) en el notebook."""
    return x.reshape(x.shape[0], -1)


def recorrer(modelo, x):
    """Pasa x, un lote (n, alto, ancho, 3) en [0, 1], capa por capa con training=False.

    Devuelve los logits de la capa `salida`, sin softmax, y la media y la varianza de la entrada de cada
    BatchNorm (en la CNN, norm_1 a norm_4 y norm_densa; en el MLP, norm_1 y norm_2). Al MLP le entrega la imagen
    aplanada. Las redes reciben la imagen en [0, 255] y la escalan con su capa Rescaling.
    """
    import keras

    if es_descriptores(modelo):
        raise ValueError(f"{modelo.name} tiene una capa Normalization: recibe descriptores calculados fuera de la "
                         "red, sin gradiente hacia los píxeles, y no se puede recrear.")
    congelar(modelo)
    h = x * 255.0
    if len(modelo.input_shape) == 2:
        h = aplanar(h)
    estadisticas = []
    for capa in modelo.layers:
        if isinstance(capa, keras.layers.BatchNormalization):
            # La BatchNorm normaliza el último eje: los canales en la CNN, las neuronas en el MLP
            ejes = tuple(range(h.ndim - 1))
            estadisticas.append(Estadistica(capa.name, h.mean(dim=ejes), h.var(dim=ejes, unbiased=False),
                                            capa.moving_mean.value, capa.moving_variance.value))
        if capa.name == "salida":
            return h @ capa.kernel.value + capa.bias.value, estadisticas
        h = capa(h, training=False)
    raise ValueError(f"{modelo.name} no tiene una capa llamada 'salida'.")


def perdida_bn(estadisticas: list, multiplicador_primera: float = 10.0, normalizada: bool = False):
    """Σ ‖μ − moving_mean‖₂ + ‖σ² − moving_variance‖₂, con la primera BatchNorm multiplicada.

    Normalizada, cada diferencia se mide en unidades de lo que la capa guardó:
    Σ ‖(μ − moving_mean) / √(moving_variance + ε)‖₂ + ‖(σ² − moving_variance) / (moving_variance + ε)‖₂. Así cada
    neurona pesa lo mismo sea cual sea su escala; en el MLP, cuyas activaciones llegan a miles, el término sin
    normalizar ahoga a la entropía cruzada."""
    import torch

    total = 0.0
    for posicion, estadistica in enumerate(estadisticas):
        if normalizada:
            escala = estadistica.varianza_bn + EPSILON_BN
            termino = (torch.linalg.vector_norm((estadistica.media - estadistica.media_bn) / torch.sqrt(escala))
                       + torch.linalg.vector_norm((estadistica.varianza - estadistica.varianza_bn) / escala))
        else:
            termino = (torch.linalg.vector_norm(estadistica.media - estadistica.media_bn)
                       + torch.linalg.vector_norm(estadistica.varianza - estadistica.varianza_bn))
        total = total + (multiplicador_primera if posicion == 0 else 1.0) * termino
    return total


def variacion_total(x):
    """Variación total L2 de DeepInversion: diferencias horizontales, verticales y en las dos diagonales."""
    import torch

    diferencias = (
        x[:, :, :-1] - x[:, :, 1:],
        x[:, :-1] - x[:, 1:],
        x[:, 1:, :-1] - x[:, :-1, 1:],
        x[:, :-1, :-1] - x[:, 1:, 1:],
    )
    return sum(torch.linalg.vector_norm(d) for d in diferencias)


def norma_l2(x):
    import torch

    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean()


def perdida(logits, objetivos, estadisticas, x, hiper: Hiperparametros):
    """Pérdida total y sus partes: entropía cruzada + α_bn·BN + α_tv·variación total + α_l2·L2."""
    import torch

    partes = {
        "entropia": torch.nn.functional.cross_entropy(logits, objetivos),
        "bn": perdida_bn(estadisticas, hiper.multiplicador_primera_bn, hiper.bn_normalizada),
        "tv": variacion_total(x),
        "l2": norma_l2(x),
    }
    total = (partes["entropia"] + hiper.alfa_bn * partes["bn"] + hiper.alfa_tv * partes["tv"]
             + hiper.alfa_l2 * partes["l2"])
    return total, partes


def tasa_coseno(paso: int, pasos: int, lr: float) -> float:
    return 0.5 * lr * (1 + math.cos(math.pi * paso / max(1, pasos)))


def cuantizar(x) -> np.ndarray:
    """Un lote en [0, 1] como imágenes uint8: lo que se guarda en PNG y lo que reciben los jueces."""
    import torch

    with torch.no_grad():
        return (x.detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


@dataclass
class Recreacion:
    final: np.ndarray
    instantaneas: dict
    historial: list = field(default_factory=list)
    segundos: float = 0.0
    desde_cache: bool = False


def etapas(hiper: Hiperparametros, tamano: tuple) -> list:
    """[(alto, ancho, pasos)] de cada etapa de la recreación. Sin escalas, una sola etapa a la resolución del modelo.
    Con escalas (alturas de menor a mayor), una etapa por escala y los pasos repartidos por igual; el resto va a la
    última. Si la última escala es menor que la del modelo, la imagen final se amplía desde ella. Las escalas
    mayores que la altura del modelo no se usan; si no queda ninguna, es una sola etapa."""
    alto, ancho = tamano
    lados = [int(lado) for lado in hiper.escalas]
    if lados != sorted(lados) or any(lado < 1 for lado in lados):
        raise ValueError(f"escalas debe ir de menor a mayor y ser positivas: {hiper.escalas}")
    lados = [lado for lado in lados if lado <= alto]
    if not lados:
        return [(alto, ancho, hiper.pasos)]
    if hiper.pasos < len(lados):
        raise ValueError(f"{hiper.pasos} pasos no alcanzan para {len(lados)} escalas")
    base = hiper.pasos // len(lados)
    pasos = [base] * (len(lados) - 1) + [hiper.pasos - base * (len(lados) - 1)]
    return [(lado, max(1, round(ancho * lado / alto)), p) for lado, p in zip(lados, pasos)]


def ampliar(z, tamano: tuple):
    """Un lote (n, alto, ancho, 3) llevado a `tamano` con interpolación bilineal; si ya lo tiene, el mismo tensor.
    Es una combinación convexa de píxeles: lo que está en [0, 1] sigue en [0, 1]."""
    import torch

    if tuple(z.shape[1:3]) == tuple(tamano):
        return z
    return torch.nn.functional.interpolate(z.permute(0, 3, 1, 2), size=tuple(tamano), mode="bilinear",
                                           align_corners=False).permute(0, 2, 3, 1)


def recrear(modelo, objetivos, tamano, hiper: Hiperparametros, semilla: int, dispositivo: str, avance=None):
    """Un lote de len(objetivos) imágenes desde ruido, una por clase objetivo.

    Adam con betas (beta1, beta2) y tasa coseno. En cada paso el lote se desplaza hasta `desplazamiento` píxeles
    y, si `volteo`, se voltea al azar; después del paso, los píxeles se recortan a [0, 1]. Con `escalas`, se
    optimiza primero una imagen pequeña que se amplía a la resolución del modelo, y en cada etapa se amplía la
    anterior y se reinician Adam y su tasa coseno: lo grueso se fija antes que el detalle. Guarda una instantánea
    en los pasos de PASOS_INSTANTANEAS y en el último.
    """
    import torch

    generador = torch.Generator().manual_seed(int(semilla))
    azar = random.Random(int(semilla))
    plan = etapas(hiper, tamano)
    z = torch.rand((len(objetivos), plan[0][0], plan[0][1], 3), generator=generador).to(dispositivo)
    z.requires_grad_(True)
    objetivo = torch.as_tensor(np.asarray(objetivos), dtype=torch.long, device=dispositivo)
    pasos_instantaneas = sorted({p for p in PASOS_INSTANTANEAS if p < hiper.pasos} | {hiper.pasos})
    instantaneas, historial = {}, []
    inicio = time.perf_counter()
    paso = 0
    for numero, (alto_etapa, ancho_etapa, pasos_etapa) in enumerate(plan):
        if numero > 0:
            with torch.no_grad():
                z = ampliar(z, (alto_etapa, ancho_etapa)).clamp(0, 1).contiguous()
            z.requires_grad_(True)
        optimizador = torch.optim.Adam([z], lr=hiper.lr, betas=(hiper.beta1, hiper.beta2), eps=1e-8)
        for paso_etapa in range(pasos_etapa):
            if paso in pasos_instantaneas:
                instantaneas[paso] = cuantizar(ampliar(z, tamano))
            for grupo in optimizador.param_groups:
                grupo["lr"] = tasa_coseno(paso_etapa, pasos_etapa, hiper.lr)
            movida = ampliar(z, tamano)
            if hiper.desplazamiento:
                movida = torch.roll(movida, shifts=(azar.randint(-hiper.desplazamiento, hiper.desplazamiento),
                                                    azar.randint(-hiper.desplazamiento, hiper.desplazamiento)),
                                    dims=(1, 2))
            if hiper.volteo and azar.random() < 0.5:
                movida = movida.flip(2)
            logits, estadisticas = recorrer(modelo, movida)
            total, partes = perdida(logits, objetivo, estadisticas, movida, hiper)
            optimizador.zero_grad(set_to_none=True)
            total.backward()
            optimizador.step()
            with torch.no_grad():
                z.clamp_(0, 1)
            if paso % 50 == 0 or paso == hiper.pasos - 1:
                historial.append({"paso": paso, "total": round(total.item(), 5),
                                  **{k: round(float(v.detach() if hasattr(v, "detach") else v), 5)
                                     for k, v in partes.items()}})
                if avance is not None:
                    avance(paso, historial[-1])
            paso += 1
    instantaneas[hiper.pasos] = cuantizar(ampliar(z, tamano))
    return Recreacion(instantaneas[hiper.pasos], instantaneas, historial, round(time.perf_counter() - inicio, 1))


def indices_elegidos(p_objetivo: np.ndarray, n_personajes: int, reinicios: int) -> np.ndarray:
    """Con un lote de reinicios × n_personajes recreaciones (el reinicio k del personaje i en k·n + i), el índice
    de la que la red asigna más probabilidad a su personaje, uno por personaje."""
    p = np.asarray(p_objetivo).reshape(reinicios, n_personajes)
    return p.argmax(axis=0) * n_personajes + np.arange(n_personajes)


def elegir_reinicios(recreacion: Recreacion, red, personajes: list, reinicios: int) -> Recreacion:
    """Se queda, de cada personaje, con el reinicio que mejor reconoce la red que lo generó. Solo esa red decide: si
    eligieran las otras, la prueba cruzada dejaría de ser independiente."""
    if reinicios <= 1:
        return recreacion
    p_objetivo = juzgar(red, recreacion.final, list(personajes) * reinicios)["p_objetivo"]
    indices = indices_elegidos(p_objetivo, len(personajes), reinicios)
    return Recreacion(recreacion.final[indices], {paso: lote[indices] for paso, lote in recreacion.instantaneas.items()},
                      recreacion.historial, recreacion.segundos, recreacion.desde_cache)


# --- Juicio de las recreaciones --------------------------------------------------------------------------

def juzgar(red: Red, imagenes: np.ndarray, personajes: list) -> dict:
    """Qué dice una red de cada recreación: su predicción, su confianza, la probabilidad del personaje objetivo
    y en qué posición lo pone (1 = su predicción)."""
    probabilidades = red.probabilidades(imagenes)
    clases = red.clases
    indice_objetivo = np.array([clases.index(p) for p in personajes])
    filas = np.arange(len(personajes))
    orden = np.argsort(-probabilidades, axis=1)
    return {
        "prediccion": [clases[i] for i in orden[:, 0]],
        "confianza": probabilidades[filas, orden[:, 0]],
        "p_objetivo": probabilidades[filas, indice_objetivo],
        "posicion": (orden == indice_objetivo[:, None]).argmax(axis=1) + 1,
    }


def fila_matriz(juicio: dict) -> dict:
    posicion = np.asarray(juicio["posicion"])
    return {
        "top1": int((posicion == 1).sum()),
        "top3": int((posicion <= 3).sum()),
        "n": int(len(posicion)),
        "confianza": round(float(np.mean(juicio["p_objetivo"])), 4),
        # En qué posición pone la red al personaje, en promedio (1 = siempre su predicción; al azar, la del medio).
        # Se mueve antes que el top-1: muestra si una recreación se acerca a su personaje aunque todavía no gane
        "posicion_media": round(float(np.mean(posicion)), 2),
    }


def resumen_instantaneas(propias: dict) -> list:
    """Por instantánea: cuántas recreaciones reconoce como su personaje la red que las genera, con qué
    probabilidad media para él, y qué personaje nombra más veces."""
    filas = []
    for paso in sorted(propias):
        juicio = propias[paso]
        nombres, veces = np.unique(juicio["prediccion"], return_counts=True)
        filas.append({
            **{"paso": int(paso)},
            **fila_matriz(juicio),
            "confianza_prediccion": round(float(np.mean(juicio["confianza"])), 4),
            "mas_nombrado": str(nombres[np.argmax(veces)]),
            "veces_mas_nombrado": int(veces.max()),
        })
    return filas


def rotulos_casilla(propia: tuple, otras: list) -> list:
    """Las líneas bajo cada recreación: lo que dice la red que la generó y lo que dicen las otras, con el
    personaje y la confianza: ['propia: Homer Simpson · 97 %', 'otras: Homer Simpson · 64 %']."""
    return [f"propia: {D.rotulo(*propia)}"] + [f"otras: {D.rotulo(*otra)}" for otra in otras]


# --- Figuras ---------------------------------------------------------------------------------------------

def _leyenda(generadores: list, jueces: list) -> str:
    """Segunda línea de los títulos: qué significa cada rótulo bajo las recreaciones."""
    if len(generadores) == 1:
        propia = f"lo que dice {ESPECIFICACIONES[generadores[0]].en_frase}"
        otras = [ESPECIFICACIONES[j].corto for j in jueces if j != generadores[0]]
        texto = f"propia: {propia}" + (f" · otras: lo que dicen {' y '.join(otras)}" if otras else "")
    else:
        texto = "propia: lo que dice la red que lo genera · otras: lo que dicen las demás redes"
    return f"{texto} · verde si nombra al personaje"


def titulo_lamina(generador: str, archivo: str, n_personajes: int, jueces: list) -> str:
    return (f"Los {n_personajes} personajes recreados solo desde {ESPECIFICACIONES[generador].en_frase} ({archivo})\n"
            f"{_leyenda([generador], jueces)}")


def titulo_comparacion(generadores: list, jueces: list, n_personajes: int) -> str:
    origen = (ESPECIFICACIONES[generadores[0]].en_frase if len(generadores) == 1
              else "cada red, una fila por red")
    return f"{n_personajes} personajes recreados desde {origen}\n{_leyenda(generadores, jueces)}"


def titulo_evolucion(generadores: list) -> str:
    quien = (ESPECIFICACIONES[generadores[0]].en_frase if len(generadores) == 1
             else "la red que la genera")
    return f"Del ruido a la recreación: lo que dice {quien} en cada paso"


def _figura(ancho: float, alto: float):
    # Figure sin pyplot: no depende del backend y no se muestra sola en el notebook
    from matplotlib.figure import Figure

    return Figure(figsize=(ancho, alto), dpi=110, layout="constrained")


def _casilla(eje, imagen, titulo: str, lineas: list, colores: list, tamano_texto: float = 6.2) -> None:
    eje.imshow(imagen, interpolation="nearest")
    eje.set_xticks([])
    eje.set_yticks([])
    eje.set_title(titulo, fontsize=tamano_texto + 0.8, fontweight="bold", pad=2)
    for posicion, (linea, color) in enumerate(zip(lineas, colores)):
        eje.text(0.5, -0.03 - 0.115 * posicion, linea, transform=eje.transAxes, ha="center", va="top",
                 fontsize=tamano_texto, color=color)
    color_marco = colores[0] if colores else "#888888"
    for lado in eje.spines.values():
        lado.set_edgecolor(color_marco)
        lado.set_linewidth(1.6)


def _textos(generador: str, jueces: list, juicios: dict, personaje: str, indice: int) -> tuple:
    """Rótulos y colores de la recreación `indice` del generador: verde si la red nombra al personaje."""
    otros = [j for j in jueces if j != generador]
    propia = juicios[generador]
    lineas = rotulos_casilla(
        (propia["prediccion"][indice], propia["confianza"][indice]),
        [(juicios[j]["prediccion"][indice], juicios[j]["confianza"][indice]) for j in otros],
    )
    colores = [COLOR_ACIERTO if juicios[j]["prediccion"][indice] == personaje else COLOR_ERROR
               for j in [generador] + otros]
    return lineas, colores


def lamina(red: Red, nombres: dict, personajes: list, imagenes: np.ndarray, juicios: dict, destino: Path) -> None:
    jueces = list(juicios)
    filas = math.ceil(len(personajes) / COLUMNAS_LAMINA)
    columnas = min(COLUMNAS_LAMINA, len(personajes))
    lineas_texto = len(jueces)
    figura = _figura(2.05 * columnas, (2.05 + 0.24 * lineas_texto) * filas + 0.7)
    ejes = np.atleast_1d(figura.subplots(filas, columnas, squeeze=False)).ravel()
    for indice, personaje in enumerate(personajes):
        lineas, colores = _textos(red.identificador, jueces, juicios, personaje, indice)
        _casilla(ejes[indice], imagenes[indice], D.nombre_legible(personaje), lineas, colores)
    for eje in ejes[len(personajes):]:
        eje.set_axis_off()
    figura.suptitle(titulo_lamina(red.identificador, red.ruta.name, len(personajes), jueces),
                    fontsize=11, fontweight="bold")
    figura.savefig(destino)


def comparacion(generadores: list, jueces: list, nombres: dict, personajes: list, recreaciones: dict, juicios: dict,
                destino: Path) -> None:
    """Una fila por red que genera; bajo cada recreación, lo que dicen todas las redes cargadas (`jueces`)."""
    elegidos = [p for p in PERSONAJES_F1 if p in personajes] or personajes[:12]
    figura = _figura(1.9 * len(elegidos) + 0.6, (1.9 + 0.24 * len(jueces)) * len(generadores) + 0.6)
    ejes = figura.subplots(len(generadores), len(elegidos), squeeze=False)
    for fila, red in enumerate(generadores):
        g = red.identificador
        for columna, personaje in enumerate(elegidos):
            indice = personajes.index(personaje)
            lineas, colores = _textos(g, jueces, juicios[g], personaje, indice)
            _casilla(ejes[fila, columna], recreaciones[g].final[indice], D.nombre_legible(personaje), lineas, colores)
        ejes[fila, 0].set_ylabel(nombres[g], fontsize=10, fontweight="bold")
    figura.suptitle(titulo_comparacion([r.identificador for r in generadores], jueces, len(elegidos)),
                    fontsize=12, fontweight="bold")
    figura.savefig(destino)


def evolucion(generadores: list, nombres: dict, personajes: list, recreaciones: dict, propias: dict,
              destino: Path) -> None:
    elegidos = [p for p in PERSONAJES_EVOLUCION if p in personajes] or personajes[:4]
    filas = [(red, p) for p in elegidos for red in generadores]
    pasos = sorted(recreaciones[generadores[0].identificador].instantaneas)
    figura = _figura(1.9 * len(pasos) + 0.9, 2.25 * len(filas) + 0.6)
    ejes = figura.subplots(len(filas), len(pasos), squeeze=False)
    for fila, (red, personaje) in enumerate(filas):
        g = red.identificador
        indice = personajes.index(personaje)
        for columna, paso in enumerate(pasos):
            prediccion, confianza = propias[g][paso]["prediccion"][indice], propias[g][paso]["confianza"][indice]
            color = COLOR_ACIERTO if prediccion == personaje else COLOR_ERROR
            titulo = f"paso {paso}" if paso != pasos[-1] else f"final (paso {paso})"
            _casilla(ejes[fila, columna], recreaciones[g].instantaneas[paso][indice], titulo,
                     [D.rotulo(prediccion, confianza)], [color])
        ejes[fila, 0].set_ylabel(f"{D.nombre_legible(personaje)}\n{nombres[g]}", fontsize=9, fontweight="bold")
    figura.suptitle(titulo_evolucion([r.identificador for r in generadores]), fontsize=12, fontweight="bold")
    figura.savefig(destino)


# --- Caché -----------------------------------------------------------------------------------------------

def clave_hiper(hiper: Hiperparametros) -> dict:
    """Los hiperparámetros que identifican una recreación en la caché: los originales siempre y los agregados
    después solo si no tienen su valor por defecto. Con los valores de siempre la clave es la misma que antes de
    agregarlos, así que las recreaciones de las CNN que ya estaban en caché siguen valiendo."""
    por_defecto = asdict(Hiperparametros())
    return {campo: valor for campo, valor in asdict(hiper).items()
            if campo in CAMPOS_ORIGINALES or valor != por_defecto[campo]}


def ruta_cache(red: Red, personajes: list, hiper: Hiperparametros, semilla: int) -> Path:
    clave = json.dumps({"modelo": red.sha256, "personajes": personajes, "hiper": clave_hiper(hiper), "semilla": semilla,
                        "instantaneas": PASOS_INSTANTANEAS, "version": VERSION_CACHE}, sort_keys=True)
    digesto = hashlib.sha256(clave.encode("utf-8")).hexdigest()[:16]
    return red.ruta.parent / CARPETA_CACHE / f"{red.ruta.stem}_{digesto}.npz"


def leer_cache(ruta: Path) -> Recreacion | None:
    if not ruta.is_file():
        return None
    with np.load(ruta, allow_pickle=False) as datos:
        pasos = [int(p) for p in datos["pasos"]]
        instantaneas = {p: datos["instantaneas"][i] for i, p in enumerate(pasos)}
        return Recreacion(instantaneas[pasos[-1]], instantaneas, json.loads(str(datos["historial"])),
                          float(datos["segundos"]), desde_cache=True)


def escribir_cache(ruta: Path, recreacion: Recreacion) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pasos = sorted(recreacion.instantaneas)
    temporal = ruta.with_name(ruta.stem + ".tmp.npz")
    np.savez_compressed(temporal, pasos=np.array(pasos), instantaneas=np.stack([recreacion.instantaneas[p] for p in pasos]),
                        historial=json.dumps(recreacion.historial), segundos=recreacion.segundos)
    os.replace(temporal, ruta)


# --- Programa --------------------------------------------------------------------------------------------

def _lista(texto: str) -> list:
    return [parte.strip() for parte in texto.split(",") if parte.strip()]


def _escalas(texto: str) -> tuple:
    """'16,32,64,128' -> (16, 32, 64, 128); 'ninguna' o '' -> (), una sola etapa a la resolución del modelo."""
    if texto.strip().lower() in ("", "ninguna"):
        return ()
    return tuple(int(parte) for parte in _lista(texto))


# Los flags que sobrescriben los hiperparámetros de cada red: campo de Hiperparametros -> flag
FLAGS_HIPER = {
    "pasos": "--pasos", "lr": "--lr", "alfa_bn": "--alfa-bn", "alfa_tv": "--alfa-tv", "alfa_l2": "--alfa-l2",
    "desplazamiento": "--desplazamiento", "volteo": "--volteo", "bn_normalizada": "--bn-normalizada",
    "escalas": "--escalas", "reinicios": "--reinicios",
}


def hiper_de(especificacion: Especificacion, args) -> Hiperparametros:
    """Los hiperparámetros de una red con los flags que se pasaron explícitamente encima."""
    cambios = {campo: getattr(args, campo) for campo in FLAGS_HIPER if getattr(args, campo, None) is not None}
    return replace(especificacion.hiper, **cambios)


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", type=_lista, default=list(ESPECIFICACIONES),
                        help=f"cuáles usar, separados por comas (por defecto {','.join(ESPECIFICACIONES)})")
    parser.add_argument("--generadores", type=_lista,
                        help="cuáles de --modelos recrean personajes, separados por comas (por defecto, todos); los "
                             "demás solo juzgan las recreaciones")
    parser.add_argument("--personajes", type=_lista, help="solo estos personajes, separados por comas")
    parser.add_argument("--semilla", type=int, help="semilla del ruido inicial y del desplazamiento; si falta, se "
                        "sortea y se guarda en el JSON")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU")
    parser.add_argument("--salida", type=Path, help="carpeta para todas las salidas (por defecto, la de cada red)")
    parser.add_argument("--rehacer", action="store_true", help="recalcula aunque esté en la caché")
    parser.add_argument("--sin-cache", action="store_true", help="ni lee ni escribe la caché (para los barridos)")
    hiper = parser.add_argument_group(
        "hiperparámetros", "Cada red tiene los suyos (CNN: Hiperparametros(); MLP: HIPER_MLP; MLP robusto: "
        "HIPER_MLP_ROBUSTO). Un flag explícito los reemplaza en todas las redes que generan.")
    hiper.add_argument("--pasos", type=int, help="pasos de Adam")
    hiper.add_argument("--lr", type=float, help="tasa de Adam")
    hiper.add_argument("--alfa-bn", type=float, help="peso de BatchNorm")
    hiper.add_argument("--alfa-tv", type=float, help="peso de la variación total")
    hiper.add_argument("--alfa-l2", type=float, help="peso de L2")
    hiper.add_argument("--desplazamiento", type=int, help="desplazamiento máximo en píxeles en cada paso")
    hiper.add_argument("--volteo", action=argparse.BooleanOptionalAction, help="volteo horizontal al azar en cada paso")
    hiper.add_argument("--bn-normalizada", action=argparse.BooleanOptionalAction,
                       help="BatchNorm en unidades de la varianza guardada")
    hiper.add_argument("--escalas", type=_escalas, help="resoluciones de menor a mayor, separadas por comas "
                       "(16,32,64,128), o 'ninguna' para optimizar solo a la del modelo")
    hiper.add_argument("--reinicios", type=int, help="imágenes por personaje; queda la que mejor reconoce su red")
    argumentos = parser.parse_args(argv)
    desconocidos = sorted(set(argumentos.modelos) - set(ESPECIFICACIONES))
    if desconocidos:
        parser.error(f"modelos desconocidos: {', '.join(desconocidos)} (hay {', '.join(ESPECIFICACIONES)})")
    if argumentos.generadores is None:
        argumentos.generadores = list(argumentos.modelos)
    fuera = sorted(set(argumentos.generadores) - set(argumentos.modelos))
    if fuera:
        parser.error(f"--generadores incluye {', '.join(fuera)}, que no está en --modelos ({','.join(argumentos.modelos)})")
    if not argumentos.generadores:
        parser.error("--generadores necesita al menos una red")
    if argumentos.pasos is not None and argumentos.pasos < 1:
        parser.error("--pasos debe ser al menos 1")
    if argumentos.reinicios is not None and argumentos.reinicios < 1:
        parser.error("--reinicios debe ser al menos 1")
    return argumentos


def _relativa(ruta: Path) -> str:
    try:
        return Path(ruta).resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return str(ruta)


def carpeta_comun(generadores: list, salida: Path | None = None) -> Path:
    """Dónde van comparacion.png y evolucion.png: --salida si se dio; si no, la carpeta de las redes que generan
    cuando todas son del mismo notebook (solo el MLP -> la del MLP) y, si se mezclan, la de las CNN. Así el
    notebook del MLP no reemplaza las figuras del de la CNN."""
    if salida is not None:
        return salida
    carpetas = {red.especificacion.salida for red in generadores}
    return carpetas.pop() if len(carpetas) == 1 else SALIDA_CNN


def main(argv=None) -> dict:
    """Recrea, juzga y guarda. Devuelve un resumen que los notebooks muestran como tabla."""
    args = leer_argumentos(argv)
    redes, avisos = [], []
    for identificador in args.modelos:
        red, mensajes = resolver(ESPECIFICACIONES[identificador], args.cpu)
        avisos += mensajes
        if red is not None:
            redes.append(red)
    for aviso in avisos:
        print(f"AVISO: {aviso}")
    if not redes:
        raise SystemExit("No hay ningún modelo que reciba píxeles: ejecuta antes el notebook de la CNN o del MLP.")
    generadores = [r for r in redes if r.identificador in args.generadores]
    if not generadores:
        raise SystemExit(f"Ninguna de las redes que generan ({', '.join(args.generadores)}) tiene un modelo que reciba "
                         "píxeles: ejecuta antes su notebook. No se recrea con las otras redes.")
    avisos_clases = comprobar_clases(redes)
    for aviso in avisos_clases:
        print(f"AVISO: {aviso}")
    clases = redes[0].clases
    personajes = list(args.personajes or clases)
    faltan = sorted(set(personajes) - set(clases))
    if faltan:
        raise SystemExit(f"Personajes desconocidos: {', '.join(faltan)}")
    semilla_sorteada = args.semilla is None
    semilla = random.SystemRandom().randrange(2 ** 31) if semilla_sorteada else args.semilla
    nombres = {r.identificador: r.especificacion.corto for r in redes}
    dispositivo = redes[0].dispositivo
    if dispositivo == "cuda":
        import torch

        torch.backends.cudnn.benchmark = True
    print(f"Redes: {', '.join(f'{r.identificador} ({_relativa(r.ruta)})' for r in redes)} | {dispositivo}")
    ids_generadores = {r.identificador for r in generadores}
    print(f"Generan: {', '.join(r.identificador for r in generadores)}"
          + (f" | solo juzgan: {', '.join(r.identificador for r in redes if r.identificador not in ids_generadores)}"
             if len(generadores) < len(redes) else ""))
    print(f"{len(personajes)} personajes, semilla {semilla}{' (sorteada)' if semilla_sorteada else ''}")
    # Cada red que genera con sus hiperparámetros, y encima los flags que se pasaron
    hipers = {red.identificador: hiper_de(red.especificacion, args) for red in generadores}
    for red in generadores:
        h = hipers[red.identificador]
        print(f"  {red.identificador:<13} {h.pasos} pasos, lr {h.lr}, α_bn {h.alfa_bn}"
              f"{' (normalizada)' if h.bn_normalizada else ''}, α_tv {h.alfa_tv}, desplazamiento {h.desplazamiento}, "
              f"volteo {'sí' if h.volteo else 'no'}, escalas {list(h.escalas) or 'ninguna'}, reinicios {h.reinicios}")

    # 1. Recreaciones de las redes que generan, desde la caché si ya se calcularon con todo igual
    recreaciones = {}
    for red in generadores:
        hiper = hipers[red.identificador]
        cache = ruta_cache(red, personajes, hiper, semilla)
        recreacion = None if (args.rehacer or args.sin_cache or semilla_sorteada) else leer_cache(cache)
        if recreacion is None:
            # Con reinicios, el lote tiene reinicios × personajes: el reinicio k del personaje i en k·n + i
            objetivos = [red.clases.index(p) for p in personajes] * hiper.reinicios

            def avance(paso, partes, red=red, hiper=hiper):
                if paso % 500 == 0 or paso == hiper.pasos - 1:
                    print(f"  {red.identificador:<13} paso {paso + 1:>5}/{hiper.pasos}  pérdida {partes['total']:.3f}  "
                          f"(entropía {partes['entropia']:.3f}, BN {partes['bn']:.2f}, TV {partes['tv']:.1f})")

            recreacion = recrear(red.modelo, objetivos, red.tamano, hiper, semilla, red.dispositivo, avance)
            recreacion = elegir_reinicios(recreacion, red, personajes, hiper.reinicios)
            if not (semilla_sorteada or args.sin_cache):
                escribir_cache(cache, recreacion)
        else:
            print(f"  {red.identificador:<13} desde la caché ({_relativa(cache)})")
        recreaciones[red.identificador] = recreacion

    # 2. Todas las redes cargadas juzgan las recreaciones de cada red que genera: la matriz generador x juez
    juicios = {g: {red.identificador: juzgar(red, recreaciones[g].final, personajes) for red in redes}
               for g in recreaciones}
    matriz = {g: {j: fila_matriz(juicio) for j, juicio in juicios[g].items()} for g in juicios}
    red_por_id = {r.identificador: r for r in redes}
    propias = {g: {paso: juzgar(red_por_id[g], imagenes, personajes)
                   for paso, imagenes in recreaciones[g].instantaneas.items()} for g in recreaciones}
    avance_por_paso = {g: resumen_instantaneas(propias[g]) for g in propias}
    # Solo los pares con al menos una red que genera: dicen si su prueba cruzada es independiente
    correlaciones = []
    for posicion, a in enumerate(redes):
        for b in redes[posicion + 1:]:
            if a.identificador not in ids_generadores and b.identificador not in ids_generadores:
                continue
            valor = correlacion_pesos(a.modelo, b.modelo)
            if valor is not None:
                correlaciones.append({"a": a.identificador, "b": b.identificador, "correlacion": round(valor, 4),
                                      "independientes": bool(abs(valor) < CORRELACION_DEPENDIENTE)})

    # 3. Salidas de cada red que genera
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    salidas = {}
    for red in generadores:
        g = red.identificador
        carpeta = (args.salida or red.especificacion.salida) / g
        carpeta.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for indice, personaje in enumerate(personajes):
            Image.fromarray(recreaciones[g].final[indice]).save(carpeta / f"{personaje}.png")
        lamina(red, nombres, personajes, recreaciones[g].final, juicios[g], carpeta / "lamina.png")
        detalle = [
            {
                "personaje": personaje,
                "archivo": f"{personaje}.png",
                "jueces": {
                    j: {
                        "prediccion": juicio["prediccion"][i],
                        "confianza": round(float(juicio["confianza"][i]), 4),
                        "p_objetivo": round(float(juicio["p_objetivo"][i]), 4),
                        "posicion": int(juicio["posicion"][i]),
                        "rotulo": D.rotulo(juicio["prediccion"][i], juicio["confianza"][i]),
                    }
                    for j, juicio in juicios[g].items()
                },
            }
            for i, personaje in enumerate(personajes)
        ]
        resumen = {
            "generador": {"id": g, "nombre": red.especificacion.nombre, "ruta": _relativa(red.ruta),
                          "sha256": red.sha256, "configuracion": red.metadatos.get("configuracion"),
                          "red": red.tipo, "alto": red.tamano[0], "ancho": red.tamano[1]},
            "jueces": [{"id": r.identificador, "nombre": r.especificacion.nombre, "ruta": _relativa(r.ruta),
                        "sha256": r.sha256} for r in redes],
            "tecnica": "DeepInversion (Yin et al., 2020)",
            "hiperparametros": {**asdict(hipers[g]), "lote": len(personajes) * hipers[g].reinicios,
                                "optimizador": "Adam, tasa coseno", "instantaneas": sorted(recreaciones[g].instantaneas)},
            "semilla": semilla,
            "semilla_sorteada": semilla_sorteada,
            "segundos": recreaciones[g].segundos,
            "desde_cache": recreaciones[g].desde_cache,
            "dispositivo": red.dispositivo,
            "fecha": fecha,
            "azar": round(1 / len(clases), 4),
            "matriz": matriz[g],
            "correlacion_pesos": [c for c in correlaciones if g in (c["a"], c["b"])],
            "instantaneas": avance_por_paso[g],
            "historial": recreaciones[g].historial,
            "detalle": detalle,
        }
        (carpeta / "recreacion.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        salidas[g] = {"carpeta": carpeta, "lamina": carpeta / "lamina.png", "json": carpeta / "recreacion.json"}

    # 4. Figuras de las redes que generan, en su carpeta (ver carpeta_comun)
    comun = carpeta_comun(generadores, args.salida)
    comun.mkdir(parents=True, exist_ok=True)
    comparacion(generadores, [r.identificador for r in redes], nombres, personajes, recreaciones, juicios,
                comun / "comparacion.png")
    evolucion(generadores, nombres, personajes, recreaciones, propias, comun / "evolucion.png")

    # 5. Líneas con formato fijo para la presentación
    print()
    for red in generadores:
        g = red.identificador
        print(f"RECREACION_MODELO generador={g} sha256={red.sha256} segundos={recreaciones[g].segundos:.1f} "
              f"semilla={semilla} pasos={hipers[g].pasos} personajes={len(personajes)}")
    for g in matriz:
        for j, fila in matriz[g].items():
            print(f"RECREACION generador={g} juez={j} top1={fila['top1']}/{fila['n']} top3={fila['top3']}/{fila['n']} "
                  f"confianza={fila['confianza']:.3f} posicion_media={fila['posicion_media']:.2f}")
    for g, filas in avance_por_paso.items():
        for fila in filas:
            print(f"RECREACION_PASO generador={g} paso={fila['paso']} top1={fila['top1']}/{fila['n']} "
                  f"confianza={fila['confianza']:.3f} mas_nombrado={fila['mas_nombrado']}:{fila['veces_mas_nombrado']} "
                  f"confianza_prediccion={fila['confianza_prediccion']:.3f}")
    for c in correlaciones:
        print(f"RECREACION_PESOS a={c['a']} b={c['b']} correlacion={c['correlacion']:.3f}")
    print(f"RECREACION_AZAR top1={1 / len(clases):.4f}")
    return {
        "redes": [{"id": r.identificador, "nombre": r.especificacion.nombre, "ruta": _relativa(r.ruta)} for r in redes],
        "generadores": [r.identificador for r in generadores],
        "avisos": avisos + avisos_clases,
        "personajes": personajes,
        "semilla": semilla,
        "hiperparametros": {g: asdict(h) for g, h in hipers.items()},
        "matriz": matriz,
        "instantaneas": avance_por_paso,
        "correlaciones": correlaciones,
        "salidas": salidas,
        "comparacion": comun / "comparacion.png",
        "evolucion": comun / "evolucion.png",
        "azar": 1 / len(clases),
    }


if __name__ == "__main__":
    main()
