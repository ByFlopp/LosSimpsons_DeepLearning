"""Genera la presentación de resultados de la CNN: CNN/presentacion_resultados_CNN.html.

Uso, desde la raíz del repositorio:
    .venv-gpu/Scripts/python.exe CNN/scripts/newDataset/generar_presentacion.py
    .venv-gpu/Scripts/python.exe CNN/scripts/newDataset/generar_presentacion.py --revision-64 6db8274

La presentación tiene el mismo diseño que la del MLP: los estilos, la barra superior, los controles del
modo «Presentar» y el script de navegación se copian de PMC/presentacion_resultados_PMC.html, así que un
cambio de diseño en esa presentación llega a esta al volver a generarla.

Las cifras no se escriben a mano. Salen de:

1. Los resúmenes de rendimiento de cada corrida: resultados/newDataset/GPU/*.json de la CNN y del MLP.
2. Las salidas guardadas del notebook de GPU de la CNN, que son las de la corrida a 128 x 128. Las cuatro
   celdas que se agregaron después se ejecutaron con el modelo final que guardó esa corrida.
3. Las salidas de la corrida a 64 x 64, que quedaron en el commit --revision-64 y se leen con git show:
   después el notebook se volvió a ejecutar a 32 x 32 y a 128 x 128, y de la de 32 x 32 solo queda su JSON.
4. Las salidas de los notebooks del MLP (corridas a 64 x 64 en GPU y a 32 x 32 en CPU), para el F1 por
   personaje, las confusiones, la brecha de entrenamiento y el costo de extremo a extremo.

Las figuras se extraen de las salidas del notebook (corrida a 128 x 128) y se embeben en el HTML: las
galerías en JPEG y las curvas en PNG, para que el archivo pese unos pocos MB y no decenas.

Los textos de cada diapositiva sí están en este script y describen las corridas del 19-10-2026. Antes de
escribir nada, el script comprueba los hechos de los que dependen esos textos (qué configuración gana a
cada resolución, qué experimentos no caben en la GPU, si C1 se entrenó) y se detiene con un aviso si
cambiaron: en ese caso hay que revisar las lecturas antes de volver a generar.
"""

import argparse
import base64
import html
import io
import json
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[3]
NOTEBOOK_CNN = "CNN/notebooks/newDataset/LosSimpsons_GPU_4060M.ipynb"
NOTEBOOK_MLP_GPU = "PMC/notebooks/newDataset/LosSimpsons_GPU_4060M.ipynb"
NOTEBOOK_MLP_CPU = "PMC/notebooks/newDataset/LosSimpsonsPMC.ipynb"
PRESENTACION_MLP = "PMC/presentacion_resultados_PMC.html"
SALIDA = "CNN/presentacion_resultados_CNN.html"
RESULTADOS_CNN = "CNN/resultados/newDataset/GPU"
RESULTADOS_MLP = "PMC/resultados/newDataset/GPU"
RESOLUCIONES = ("32x32", "64x64", "128x128")
REVISION_64 = "6db8274"
ANCHO_MAXIMO = 1200
CALIDAD_JPEG = 85
PIE = "¿Quién sale en la imagen? · CNN sobre Los Simpson"

# Celdas del notebook de GPU de la CNN cuyas salidas se usan. Los ids son los mismos en todas las corridas.
CELDAS = {
    "campo_receptivo": "cnn-ea5d2f1c",
    "pesos_primera_capa": "cnn-3295c76a",
    "entrenamiento_base": "cnn-8bf0b281",
    "curvas": "cnn-32da115a",
    "diagnostico_base": "cnn-a68c2d65",
    "experimentos": "cnn-26657d5f",
    "metricas_test": "cnn-687863b3",
    "por_clase": "cnn-8822db2d",
    "kpis": "cnn-42a0e188",
    "recorrido": "cnn-1385974e",
    "recorrido_varias": "cnn-recorrido-varias",
    "gradcam": "cnn-f2217062",
    "gradcam_clases": "cnn-gradcam-clases",
    "aciertos": "cnn-8ac5ce95",
    "errores": "cnn-8d809ece",
    "confusiones": "cnn-a418c228",
    "escenas": "cnn-217c80fd",
    "rendimiento": "cnn-e5f6a7b8",
}

EXPERIMENTOS = {
    "Base": "RGB, 4 bloques",
    "R1": "Escala de grises",
    "E1": "3 bloques (32, 64, 128)",
    "E2": "2 conv. por bloque",
    "E3": "Activación tanh",
    "E4": "Learning rate 1e-4",
    "E5": "Batch 512",
    "E6": "Dropout 0,5",
    "E7": "Cabeza con aplanado",
    "P1": "Sin pesos por clase",
    "P2": "Pesos balanceados",
    "S1": "Sin suavizado",
    "S2": "Sin etiquetas de escena",
    "A1": "Sin alteradas",
    "C1": "Combinación de mejoras",
}

NOMBRES = {
    "abraham_grampa_simpson": "Abuelo",
    "apu_nahasapeemapetilon": "Apu",
    "bart_simpson": "Bart",
    "carl_carlson": "Carl",
    "chief_wiggum": "Jefe Wiggum",
    "comic_book_guy": "Comic Book Guy",
    "disco_stu": "Disco Stu",
    "helen_lovejoy": "Helen Lovejoy",
    "homer_simpson": "Homero",
    "lenny_leonard": "Lenny",
    "lisa_simpson": "Lisa",
    "maggie_simpson": "Maggie",
    "marge_simpson": "Marge",
    "martin_prince": "Martin Prince",
    "patty_bouvier": "Patty",
    "rainier_wolfcastle": "Rainier Wolfcastle",
    "rev_timothy_lovejoy": "Rev. Lovejoy",
    "selma_bouvier": "Selma",
    "sideshow_bob": "Sideshow Bob",
}
FAMILIA = {"homer_simpson", "marge_simpson", "bart_simpson", "lisa_simpson", "maggie_simpson"}
# Los mismos doce personajes de la diapositiva «Por personaje» de la presentación del MLP
PERSONAJES_F1 = (
    "apu_nahasapeemapetilon", "comic_book_guy", "chief_wiggum", "sideshow_bob", "marge_simpson",
    "homer_simpson", "lisa_simpson", "bart_simpson", "martin_prince", "maggie_simpson",
    "rainier_wolfcastle", "disco_stu",
)
PARES_CONFUSION = (
    ("patty_bouvier", "selma_bouvier"), ("selma_bouvier", "patty_bouvier"),
    ("carl_carlson", "lenny_leonard"), ("helen_lovejoy", "rev_timothy_lovejoy"),
    ("abraham_grampa_simpson", "homer_simpson"), ("lisa_simpson", "bart_simpson"),
    ("bart_simpson", "lisa_simpson"), ("bart_simpson", "homer_simpson"),
)


# ---------------------------------------------------------------------------------------------------
# Formato de las cifras: coma decimal, punto de miles y redondeo hacia arriba en los empates
# ---------------------------------------------------------------------------------------------------

def num(valor, decimales=3):
    # round(…, 9) descarta el ruido de coma flotante: 0,7855 - 0,541 da 0,24449999…, que debe redondear a 24,5 puntos
    d = Decimal(str(round(valor, 9))).quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    signo = "−" if d < 0 else ""
    entero, _, fraccion = f"{abs(d):f}".partition(".")
    return signo + f"{int(entero):,}".replace(",", ".") + ("," + fraccion if fraccion else "")


def entero(valor):
    return num(valor, 0)


def en_palabras(n):
    palabras = ("cero", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez")
    return palabras[n] if 0 <= n < len(palabras) else entero(n)


def puntos(diferencia, decimales=1):
    """Diferencia entre dos proporciones, en puntos porcentuales."""
    return num(diferencia * 100, decimales)


def porcentaje(valor, decimales=1):
    return f"{num(valor * 100, decimales)} %"


def veces(valor, decimales=1):
    return f"{num(valor, decimales)}×"


def redondear(valor):
    return int(Decimal(str(valor)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def duracion(segundos):
    horas, resto = divmod(redondear(segundos), 3600)
    minutos, seg = divmod(resto, 60)
    if horas:
        return f"{horas} h {minutos:02d} min {seg:02d} s"
    return f"{minutos} min {seg:02d} s"


def duracion_corta(minutos):
    horas, resto = divmod(redondear(minutos), 60)
    return f"{horas} h {resto:02d} min" if horas else f"{resto} min"


def res(resolucion):
    lado = resolucion.split("x")[0]
    return f"{lado} × {lado}"


def esc(texto):
    return html.escape(str(texto), quote=True)


# ---------------------------------------------------------------------------------------------------
# Lectura de los datos
# ---------------------------------------------------------------------------------------------------

def leer_json(ruta):
    return json.loads((RAIZ / ruta).read_text(encoding="utf-8"))


def texto_salidas(celda):
    partes = []
    for salida in celda.get("outputs", []):
        if salida.get("output_type") == "stream":
            partes.append("".join(salida["text"]))
        elif "text/plain" in salida.get("data", {}):
            partes.append("".join(salida["data"]["text/plain"]))
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "\n".join(partes))


class Notebook:
    def __init__(self, contenido, nombre):
        self.nombre = nombre
        self.lista = contenido["cells"]
        self.celdas = {celda.get("id"): celda for celda in self.lista}

    def texto(self, clave):
        return texto_salidas(self.celdas[CELDAS[clave]])

    def imagenes(self, clave):
        celda = self.celdas[CELDAS[clave]]
        return [base64.b64decode(s["data"]["image/png"]) for s in celda.get("outputs", [])
                if "image/png" in s.get("data", {})]

    def buscar(self, patron):
        """Texto de la primera celda de código cuyas salidas coinciden con el patrón."""
        for celda in self.lista:
            if celda["cell_type"] == "code":
                texto = texto_salidas(celda)
                if re.search(patron, texto, re.M):
                    return texto
        raise LookupError(f"{self.nombre}: ninguna salida coincide con {patron!r}")


def leer_notebook(ruta, revision=None):
    if revision is None:
        contenido = (RAIZ / ruta).read_text(encoding="utf-8")
        nombre = ruta
    else:
        resultado = subprocess.run(["git", "show", f"{revision}:{ruta}"], cwd=RAIZ, capture_output=True)
        if resultado.returncode != 0:
            raise SystemExit(f"git show {revision}:{ruta} falló: {resultado.stderr.decode(errors='replace')}")
        contenido = resultado.stdout.decode("utf-8")
        nombre = f"{revision}:{ruta}"
    return Notebook(json.loads(contenido), nombre)


def valor(texto, patron, tipo=float):
    coincidencia = re.search(patron, texto, re.M)
    if not coincidencia:
        raise ValueError(f"no se encontró {patron!r}")
    return tipo(coincidencia.group(1).replace(",", ""))


def leer_experimentos(texto):
    """Métricas de validación de cada variante y cuántos factores superaron a la base."""
    experimentos = {}
    for codigo, nombre, linea in re.findall(r"Entrenando (\w+) - (.+?) \.\.\.\s*\n\s*(.*)", texto):
        if "no cabe" in linea:
            experimentos[codigo] = {"no_cabe": True}
            continue
        m = re.search(r"val_accuracy=([\d.]+)\s+val_f1_macro=([\d.]+)\s+epocas=(\d+)", linea)
        experimentos[codigo] = {"no_cabe": False, "acc": float(m[1]), "f1": float(m[2]), "epocas": int(m[3])}
    factores = valor(texto, r"C1 no se entrena: solo (\d+) factor", int)
    return experimentos, factores


def leer_costo_experimentos(texto):
    """Parámetros y segundos por época de cada configuración, de la tabla del resumen de rendimiento."""
    inicio = texto.index("experimento  epocas  parametros")
    tramo = texto[inicio:texto.index("Medicion", inicio)]
    filas, segunda = {}, {}
    for linea in tramo.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S.*?)\s{2,}(\d+)\s+(\d+)\s+([\d.]+)\s*\\?$", linea)
        if m:
            filas[int(m[1])] = {"codigo": m[2].split(" - ")[0], "parametros": int(m[4])}
            continue
        m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", linea)
        if m:
            segunda[int(m[1])] = float(m[2])
    return {f["codigo"]: {**f, "s_epoca": segunda[i]} for i, f in filas.items()}


def leer_reporte(texto):
    """Precision, recall y F1 por clase del classification_report."""
    filas = {}
    for m in re.finditer(r"^\s*([a-z_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", texto, re.M):
        filas[m[1]] = {"precision": float(m[2]), "recall": float(m[3]), "f1": float(m[4])}
    if len(filas) != 64:
        raise ValueError(f"el reporte por clase tiene {len(filas)} clases")
    return filas


def leer_confusiones(texto):
    """Pares (real, predicho) más frecuentes, con el porcentaje de la clase real."""
    primera, segunda = {}, {}
    for linea in texto.splitlines():
        m = re.match(r"^\s*(\d+)\s+([a-z_]+)\s+([a-z_]+)\s+(\d+)\s*\\?$", linea)
        if m:
            primera[int(m[1])] = (m[2], m[3], int(m[4]))
            continue
        m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$", linea)
        if m:
            segunda[int(m[1])] = (float(m[2]), int(m[3]), int(m[4]))
    pares = {}
    for i, (real, predicho, casos) in primera.items():
        pct, n_real, n_predicho = segunda[i]
        pares[(real, predicho)] = {"casos": casos, "pct": pct / 100, "n_real": n_real, "n_predicho": n_predicho}
    return pares


def imagenes_entrenamiento(*textos):
    """Imágenes de entrenamiento por personaje, de las tablas por clase y de las de confusiones."""
    n_train = {}

    def anotar(personaje, n):
        if n_train.setdefault(personaje, n) != n:
            raise ValueError(f"{personaje}: {n_train[personaje]} y {n} imágenes de entrenamiento")

    for texto in textos:
        patron = r"^\s*\d+\s+([a-z_]+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$"
        for m in re.finditer(patron, texto, re.M):
            anotar(m[1], int(m[3]))
        for (real, predicho), datos in leer_confusiones(texto).items():
            anotar(real, datos["n_real"])
            anotar(predicho, datos["n_predicho"])
    return n_train


def leer_registro_base(texto):
    epocas = {int(m[1]): {"acc": float(m[2]), "val_acc": float(m[3])}
              for m in re.finditer(r"Epoca\s+(\d+)/\d+\s+\S+\s+loss=\S+\s+acc=([\d.]+)\s+val_loss=\S+\s+val_acc=([\d.]+)", texto)}
    reducciones, ultima = [], None
    for linea in texto.splitlines():
        m = re.match(r"Epoca\s+(\d+)/", linea)
        if m:
            ultima = int(m[1])
        elif "ReduceLROnPlateau" in linea:
            reducciones.append(ultima)
    return {
        "epocas": epocas,
        "reducciones": reducciones,
        "vram_datos": valor(texto, r"VRAM ocupada por los datos: ([\d.]+) GB"),
        "vram_maxima": valor(texto, r"VRAM maxima usada: ([\d.]+) GB"),
    }


def leer_diagnostico(texto):
    return {
        "mejor_epoca": valor(texto, r"Mejor época \(val_loss\)\s+([\d.]+)", lambda v: int(float(v))),
        "perdida_val": valor(texto, r"Pérdida validación\s+([\d.]+)"),
        "acc_train": valor(texto, r"Accuracy entrenamiento\s+([\d.]+)"),
        "acc_val": valor(texto, r"Accuracy validación\s+([\d.]+)"),
        "f1_val": valor(texto, r"F1 macro validación\s+([\d.]+)"),
        "brecha": valor(texto, r"Brecha train - val \(accuracy\)\s+([\d.]+)"),
    }


def costo_extremo_a_extremo(texto):
    return valor(texto, r"^\s*4\s+1,000\s+([\d,]+)\s+SI\s*$")


def leer_cnn(nb, json_corrida):
    """Todo lo que la presentación usa de una corrida del notebook de la CNN."""
    exp, factores = leer_experimentos(nb.texto("experimentos"))
    diag = leer_diagnostico(nb.texto("diagnostico_base"))
    exp["Base"] = {"no_cabe": False, "acc": diag["acc_val"], "f1": diag["f1_val"], "epocas": 60}
    test = nb.texto("metricas_test")
    reporte = leer_reporte(nb.texto("por_clase"))
    gradcam = nb.texto("gradcam")
    datos = {
        "json": json_corrida,
        "experimentos": exp,
        "factores_c1": factores,
        "diagnostico": diag,
        "costo_exp": leer_costo_experimentos(nb.texto("rendimiento")),
        "reporte": reporte,
        "correlacion": valor(nb.texto("por_clase"), r"Correlacion entre imagenes de entrenamiento y F1 por clase: ([\d.]+)"),
        "confusiones": leer_confusiones(nb.texto("confusiones")),
        "absorbe_homero": valor(nb.texto("confusiones"), r"homer_simpson: (\d+) imagenes de otras clases", int),
        "mayoritaria": valor(test, r"Clase mayoritaria siempre\s+([\d.]+)"),
        "azar_veces": valor(test, r"Mejora sobre el azar: x([\d.]+)"),
        "aciertos_escena": valor(test, r"Aciertos por escena\s*:\s*(\d+)", int),
        "costo_kpi": costo_extremo_a_extremo(nb.texto("kpis")),
        "gradcam_10": valor(gradcam, r"10 % de píxeles más c.*?\s([\d.]+)\s*$"),
        "gradcam_centro_aciertos": valor(gradcam, r"mitad central \(aciertos\)\s+([\d.]+)"),
        "gradcam_centro_errores": valor(gradcam, r"mitad central \(errores\)\s+([\d.]+)"),
    }
    # El notebook debe ser el de la corrida de ese JSON
    macro = valor(nb.texto("por_clase"), r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)")
    if abs(macro - json_corrida["f1_macro_test"]) > 0.0006:
        raise SystemExit(f"{nb.nombre}: el F1 macro de sus salidas ({macro}) no es el del JSON "
                         f"{json_corrida['resolucion']} ({json_corrida['f1_macro_test']})")
    return datos


def texto_linea(texto, patron):
    """El grupo de la primera línea que coincide con el patrón, sin espacios a los lados."""
    coincidencia = re.search(patron, texto, re.M)
    if not coincidencia:
        raise ValueError(f"no se encontró {patron!r}")
    return coincidencia.group(1).strip()


def por_bloque(texto, etiqueta):
    """Valores de una línea «etiqueta: 1: v   |   2: v   |   …   |   cabeza: v», por bloque."""
    linea = re.search(rf"^{re.escape(etiqueta)}: (.*)$", texto, re.M)
    if not linea:
        raise ValueError(f"no se encontró la línea {etiqueta!r}")
    return {clave: float(numero) for clave, numero in re.findall(r"(\w+): (-?[\d.]+)", linea[1])}


def leer_detalle_128(nb):
    """Cifras que solo se usan de la corrida a 128 x 128, la de las salidas guardadas."""
    campo = nb.texto("campo_receptivo")
    bloques = [(int(m[1]), m[2]) for m in re.finditer(r"^\s*\d+\s+(\d)\s+\d+\s+\d+x\d+\s+\d+x\d+\s+(\d+)x\d+\s*\\?$", campo, re.M)]
    lados = [int(m[1]) for m in re.finditer(r"^\s*\d+\s+(\d+)\s*$", campo, re.M)]
    recorrido = nb.texto("recorrido")
    activos = [float(m[1]) for m in re.finditer(r"^\s*\d+\s+\d\s+activacion_\S+\s+\d+x\d+\s+\d+\s+([\d.]+)\s*\\?$", recorrido, re.M)]
    filtros = [float(m[1]) for m in re.finditer(r"^\s*\d+\s+([\d.]+)\s*$", recorrido, re.M)]
    prediccion = re.search(r"Prediccion: (\w+) con ([\d.]+) % de confianza", recorrido)
    escenas = nb.texto("escenas")
    errores = nb.texto("errores")
    aciertos = nb.texto("aciertos")
    pesos = re.search(r"^\s*\d+\s+128x128\s+(\d+)\s+(\d+)\s+(\d+)\s", nb.texto("pesos_primera_capa"), re.M)
    varias = nb.texto("recorrido_varias")
    clases_gradcam = nb.texto("gradcam_clases")
    return {
        "varias_correlacion": por_bloque(varias, "Correlacion entre las cuatro imagenes"),
        "varias_activos": por_bloque(varias, "Valores activos (%)"),
        "varias_confianza": [float(v) for v in re.findall(r"-> prediccion \w+ con ([\d.]+) % de confianza", varias)],
        "gradcam_clases_10": valor(clases_gradcam, r"más cálidos: media ([\d.]+)"),
        "gradcam_clases_centro": valor(clases_gradcam, r"mitad central: media ([\d.]+)"),
        "gradcam_clases_recall": valor(clases_gradcam, r"y su recall: (-?[\d.]+)"),
        "gradcam_concentrado": texto_linea(clases_gradcam, r"Calor mas concentrado: (.*)$").split(", "),
        "gradcam_disperso": texto_linea(clases_gradcam, r"Calor mas disperso\s*: (.*)$").split(", "),
        "gradcam_ejemplos": texto_linea(clases_gradcam, r"Ejemplos por tipo: (.*)$"),
        "campo_receptivo": [(b, lado, pct) for (b, lado), pct in zip(bloques, lados)],
        "activos": activos,
        "filtros_activos_ultimo": filtros[-1],
        "prediccion": (prediccion[1], float(prediccion[2])),
        "pesos_mlp_pixeles": int(pesos[2]),
        "pesos_conv_1": int(pesos[3]),
        "registro": leer_registro_base(nb.texto("entrenamiento_base")),
        "escenas_presentes": valor(escenas, r"con otro personaje presente.*?\s([\d.]+)\s*$", lambda v: int(float(v))),
        "escenas_distintas": valor(escenas, r"Predicciones distintas del personaje.*?\s([\d.]+)\s*$", lambda v: int(float(v))),
        "escenas_acc_estricta": valor(escenas, r"Accuracy estricta en imágenes con otro personaje\s+([\d.]+)"),
        "escenas_acc_oficial": valor(escenas, r"Accuracy oficial en imágenes con otro personaje\s+([\d.]+)"),
        "errores_total": valor(errores, r"Errores totales\s+([\d.]+)", lambda v: int(float(v))),
        "confianza_errores": valor(errores, r"Confianza media en errores\s+([\d.]+)"),
        "errores_090": valor(errores, r"Errores con confianza > 0\.90\s+([\d.]+)", lambda v: int(float(v))),
        "aciertos_total": valor(aciertos, r"Aciertos totales: (\d+) de", int),
        "confianza_aciertos": valor(aciertos, r"Confianza media en los aciertos: ([\d.]+)"),
    }


def leer_mlp(jsons):
    gpu = leer_notebook(NOTEBOOK_MLP_GPU)
    cpu = leer_notebook(NOTEBOOK_MLP_CPU)
    texto_clase = gpu.buscar(r"precision\s+recall\s+f1-score")
    macro = valor(texto_clase, r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)")
    if abs(macro - jsons["64x64"]["f1_macro_test"]) > 0.0006:
        raise SystemExit(f"{NOTEBOOK_MLP_GPU}: sus salidas ya no son las de la corrida de GPU a 64 x 64")
    return {
        "json": jsons,
        "reporte_64": leer_reporte(texto_clase),
        "correlacion_64": valor(gpu.buscar(r"Correlacion entre imagenes"), r"por clase: ([\d.]+)"),
        "confusiones_64": leer_confusiones(gpu.buscar(r"% de la clase real")),
        "brecha_64": valor(gpu.buscar(r"Brecha train - val \(accuracy\)"), r"Brecha train - val \(accuracy\)\s+([\d.]+)"),
        "acc_train_64": valor(gpu.buscar(r"Brecha train - val \(accuracy\)"), r"Accuracy entrenamiento\s+([\d.]+)"),
        "acc_val_64": valor(gpu.buscar(r"Brecha train - val \(accuracy\)"), r"Accuracy validación\s+([\d.]+)"),
        "costos_kpi": sorted(costo_extremo_a_extremo(nb.buscar(r"KPIs cumplidos")) for nb in (gpu, cpu)),
        "textos_n_train": [texto_clase, gpu.buscar(r"% de la clase real")],
    }


def comprobar_lecturas(cnn, mlp):
    """Los textos de las diapositivas dependen de estos hechos; si cambian, hay que reescribirlos."""
    problemas = []
    finales = {r: cnn[r]["json"]["configuracion_final"].split(" - ")[0] for r in RESOLUCIONES}
    if finales != {"32x32": "E2", "64x64": "E2", "128x128": "Base"}:
        problemas.append(f"configuraciones finales de la CNN: {finales}")
    exp128 = cnn["128x128"]["experimentos"]
    if not (exp128["E2"]["no_cabe"] and exp128["E5"]["no_cabe"]):
        problemas.append("a 128 x 128 E2 y E5 ya caben en la GPU")
    if cnn["128x128"]["factores_c1"] != 0 or cnn["64x64"]["factores_c1"] != 1:
        problemas.append("C1 cambió: los textos dicen que no se entrena en ninguna de las dos corridas")
    base = exp128["Base"]["f1"]
    if any(e["f1"] >= base for c, e in exp128.items() if c != "Base" and not e["no_cabe"]):
        problemas.append("alguna variante supera a la base a 128 x 128")
    finales_mlp = {r: mlp["json"][r]["configuracion_final"].split(" - ")[0] for r in RESOLUCIONES}
    if finales_mlp != {"32x32": "R2", "64x64": "R2", "128x128": "C1"}:
        problemas.append(f"configuraciones finales del MLP: {finales_mlp}")
    if problemas:
        raise SystemExit("Las lecturas de la presentación ya no describen los datos; revisa los textos del "
                         "script antes de generarla:\n- " + "\n- ".join(problemas))


def comprobar_detalle(d128):
    """Hechos de las diapositivas de mapas de activación y de Grad-CAM por personaje."""
    problemas = []
    if d128["gradcam_ejemplos"] != "{'original acertada': 64}":
        problemas.append(f"no todos los ejemplos por personaje son originales acertadas: {d128['gradcam_ejemplos']}")
    corr = d128["varias_correlacion"]
    if not corr["4"] < 0.1 or corr["4"] >= min(corr["1"], corr["2"], corr["3"]):
        problemas.append(f"la correlación entre los cuatro personajes ya no cae en el bloque 4: {corr}")
    if abs(d128["gradcam_clases_recall"]) >= 0.2:
        problemas.append(f"la concentración de Grad-CAM ya se correlaciona con el recall: {d128['gradcam_clases_recall']}")
    if len(d128["varias_confianza"]) != 4:
        problemas.append("el recorrido de cuatro imágenes no tiene cuatro predicciones")
    if problemas:
        raise SystemExit("Las lecturas de la presentación ya no describen los datos; revisa los textos del "
                         "script antes de generarla:\n- " + "\n- ".join(problemas))


# ---------------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------------

def figura(png, formato, alt, clase="lamina"):
    imagen = Image.open(io.BytesIO(png))
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        fondo = Image.new("RGB", imagen.size, "white")
        fondo.paste(imagen, mask=imagen.getchannel("A"))
        imagen = fondo
    else:
        imagen = imagen.convert("RGB")
    if imagen.width > ANCHO_MAXIMO:
        alto = round(imagen.height * ANCHO_MAXIMO / imagen.width)
        imagen = imagen.resize((ANCHO_MAXIMO, alto), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if formato == "jpeg":
        imagen.save(buffer, "JPEG", quality=CALIDAD_JPEG, optimize=True, progressive=True)
        tipo = "image/jpeg"
    else:
        imagen.save(buffer, "PNG", optimize=True)
        tipo = "image/png"
    datos = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<figure class="{clase}"><img src="data:{tipo};base64,{datos}" width="{imagen.width}" '
            f'height="{imagen.height}" alt="{esc(alt)}"></figure>')


# ---------------------------------------------------------------------------------------------------
# Piezas de los gráficos SVG, con las mismas clases y convenciones que la presentación del MLP
# ---------------------------------------------------------------------------------------------------

def texto_svg(x, y, contenido, clase, ancla="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{clase}" text-anchor="{ancla}">{esc(contenido)}</text>'


def linea_svg(x1, x2, y1, y2, clase):
    return f'<line class="{clase}" x1="{x1:.1f}" x2="{x2:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>'


def marca(tip, *formas):
    return f'<g class="marca" tabindex="0" data-tip="{esc(tip)}">' + "".join(formas) + "</g>"


def punto(cx, cy, r, clase):
    return (f'<circle class="golpe" cx="{cx:.1f}" cy="{cy:.1f}" r="11"/>'
            f'<circle class="visible {clase}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r}"/>')


def barra(x0, x1, y, alto, clase, redondeada=True):
    """Barra horizontal con el extremo de datos redondeado (4 px)."""
    if not redondeada or x1 - x0 < 5:
        return f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0.5):.1f}" height="{alto:.1f}" class="visible {clase}"/>'
    r = 4
    return (f'<path class="visible {clase}" d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
            f'V{y + alto - r:.1f} Q{x1:.1f},{y + alto:.1f} {x1 - r:.1f},{y + alto:.1f} H{x0:.1f} Z"/>')


def svg(alto, etiqueta, contenido):
    return (f'<svg class="graf" viewBox="0 0 620 {alto:g}" role="img" aria-label="{esc(etiqueta)}">'
            + "".join(contenido) + "</svg>")


def grafico_experimentos(cnn):
    """F1 macro de validación de cada configuración, a 128 x 128 y a 64 x 64."""
    e128, e64 = cnn["128x128"]["experimentos"], cnn["64x64"]["experimentos"]
    medidas = sorted((c for c in e128 if not e128[c]["no_cabe"]), key=lambda c: -e128[c]["f1"])
    sin_medir = sorted((c for c in e128 if e128[c]["no_cabe"]), key=lambda c: -e64[c]["f1"])
    filas = medidas + sin_medir
    x0, x1, minimo, maximo = 222, 600, 0.45, 0.76

    def x(v):
        return x0 + (v - minimo) / (maximo - minimo) * (x1 - x0)

    paso, y_inicial = 21.5, 12.0
    alto_base = y_inicial + paso * (len(filas) - 1) + 12
    partes = []
    y_base_fila = y_inicial + paso * filas.index("Base")
    partes.append(f'<rect class="banda" x="-6" y="{y_base_fila - 10.75:.1f}" width="632" height="21.5" rx="3"/>')
    for v in (0.5, 0.6, 0.7):
        partes.append(linea_svg(x(v), x(v), 0, alto_base, "rejilla"))
        partes.append(texto_svg(x(v), alto_base + 16, num(v, 1), "tick", "middle"))
    partes.append(linea_svg(x0, x1, alto_base, alto_base, "base"))
    for i, codigo in enumerate(filas):
        y = y_inicial + paso * i
        nombre = EXPERIMENTOS[codigo]
        partes.append(texto_svg(0, y + 4.2, codigo, "cfg"))
        partes.append(texto_svg(34, y + 4.5, nombre, "etq-fuerte" if codigo == "Base" else "etq"))
        v64 = e64[codigo]["f1"]
        tip64 = f"{codigo} · {nombre} — GPU 64 × 64: F1 macro {num(v64)} · accuracy {num(e64[codigo]['acc'])}"
        if e128[codigo]["no_cabe"]:
            partes.append(marca(tip64 + " · a 128 × 128 no cabe en la GPU", punto(x(v64), y, 5.5, "anillo m-r64")))
            partes.append(texto_svg(x(v64) - 9, y + 4, f"no cabe a 128 × 128 · {num(v64)} a 64", "valor-sutil", "end"))
            continue
        v128 = e128[codigo]["f1"]
        tip128 = f"{codigo} · {nombre} — GPU 128 × 128: F1 macro {num(v128)} · accuracy {num(e128[codigo]['acc'])}"
        partes.append(linea_svg(x(v128), x(v64), y, y, "conector"))
        partes.append(marca(tip64, punto(x(v64), y, 5.5, "anillo m-r64")))
        partes.append(marca(tip128, punto(x(v128), y, 5.5, "anillo m-r128")))
        if codigo == "Base":
            derecha, izquierda = max(v128, v64), min(v128, v64)
            partes.append(texto_svg(x(derecha) + 9, y + 4, num(derecha), "valor"))
            partes.append(texto_svg(x(izquierda) - 9, y + 4, num(izquierda), "valor", "end"))
    return svg(alto_base + 24, "F1 macro de validación de cada configuración en las corridas de GPU a 128 × 128 "
               "y a 64 × 64", partes)


def grafico_corridas(cnn, mlp):
    """Accuracy y accuracy estricta en test, por resolución, de la CNN y del MLP en GPU."""
    minimo, maximo, y_cero, y_tope = 0.45, 0.90, 280.0, 30.0

    def y(v):
        return y_cero - (v - minimo) / (maximo - minimo) * (y_cero - y_tope)

    partes = []
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        partes.append(linea_svg(58, 612, y(v), y(v), "rejilla"))
        partes.append(texto_svg(50, y(v) + 4, num(v, 1), "tick", "end"))
    partes.append(linea_svg(58, 612, 294, 294, "base"))
    for centro, resolucion in zip((150, 336, 522), RESOLUCIONES):
        partes.append(texto_svg(centro, 332, res(resolucion), "etq-fuerte", "middle"))
        for desplazamiento, modelo, datos, clase in ((-40, "CNN", cnn[resolucion]["json"], "r128"),
                                                     (40, "MLP", mlp["json"][resolucion], "mlp")):
            cx = centro + desplazamiento
            acc, estricta = datos["accuracy_test"], datos["accuracy_estricta_test"]
            codigo = datos["configuracion_final"].split(" - ")[0]
            tip = (f"{modelo} · {res(resolucion)} · {datos['configuracion_final']} — accuracy {num(acc)}, "
                   f"estricta {num(estricta)}, top-3 {num(datos['top3_test'])}, F1 macro {num(datos['f1_macro_test'])}")
            partes.append(linea_svg(cx, cx, y(acc), y(estricta), "conector"))
            partes.append(marca(tip, punto(cx, y(estricta), 4.5, f"hueco t-{clase}")))
            partes.append(marca(tip, punto(cx, y(acc), 6.0, f"anillo m-{clase}")))
            partes.append(texto_svg(cx, y(acc) - 12, num(acc), "valor", "middle"))
            partes.append(texto_svg(cx, y(acc) - 27, codigo, "cfg", "middle"))
            partes.append(texto_svg(cx, 312, modelo, "tick", "middle"))
    return svg(336, "Accuracy en test de la CNN y del MLP en GPU, por resolución", partes)


def grafico_personajes(cnn, mlp, n_train):
    """F1 en test por personaje: barras de la CNN a 128 x 128 y marca hueca del MLP a 64 x 64."""
    reporte, reporte_mlp = cnn["128x128"]["reporte"], mlp["reporte_64"]
    personajes = sorted(PERSONAJES_F1, key=lambda p: -reporte[p]["f1"])
    x0, ancho = 214.0, 366.0
    partes = []
    alto = 24 * len(personajes) + 12
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = x0 + ancho * v
        partes.append(linea_svg(x, x, 0, alto, "rejilla"))
        partes.append(texto_svg(x, alto + 16, num(v, 1), "tick", "middle"))
    partes.append(linea_svg(x0, x0, 0, alto, "base"))
    for i, personaje in enumerate(personajes):
        y = 6 + 24 * i
        f1, f1_mlp = reporte[personaje]["f1"], reporte_mlp[personaje]["f1"]
        nombre = NOMBRES[personaje]
        familia = personaje in FAMILIA
        partes.append(texto_svg(0, y + 15.5, nombre, "etq-fuerte" if familia else "etq"))
        partes.append(texto_svg(204, y + 15.5, entero(n_train[personaje]), "tick", "end"))
        x_cnn, x_mlp = x0 + ancho * f1, x0 + ancho * f1_mlp
        f1_64 = cnn["64x64"]["reporte"][personaje]["f1"]
        tip = (f"{nombre} — F1 de la CNN {num(f1)} a 128 × 128 ({num(f1_64)} a 64 × 64) · MLP {num(f1_mlp)} "
               f"a 64 × 64 · {entero(n_train[personaje])} imágenes de entrenamiento")
        partes.append(marca(tip, f'<rect class="golpe" x="{x0:.1f}" y="{y:.1f}" width="{x_cnn - x0 + 4:.1f}" height="24"/>',
                            barra(x0, x_cnn, y + 4, 15, "m-fam" if familia else "m-otro")))
        partes.append(marca(tip, punto(x_mlp, y + 11.5, 4.5, "hueco t-mlp")))
        partes.append(texto_svg(x_cnn + 7, y + 15.5, num(f1), "valor"))
    return svg(alto + 24, "F1 en test de doce personajes con la CNN a 128 × 128 y con el MLP a 64 × 64", partes)


def grafico_tipos(cnn, mlp):
    """Accuracy en test por tipo de imagen: rango de las tres corridas de cada modelo."""
    filas = (("Originales", "accuracy_originales_test", "3.101 en test"),
             ("Parecidas", "accuracy_parecidas_test", "912 en test"),
             ("Alteradas", "accuracy_alteradas_test", "1.050 en test"))
    minimo, maximo, x0, x1 = 0.45, 0.95, 218.0, 590.0

    def x(v):
        return x0 + (v - minimo) / (maximo - minimo) * (x1 - x0)

    partes = []
    alto = 3 * 66 + 2
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        partes.append(linea_svg(x(v), x(v), 0, alto, "rejilla"))
        partes.append(texto_svg(x(v), alto + 16, num(v, 1), "tick", "middle"))
    for i, (nombre, clave, nota) in enumerate(filas):
        y0 = 16 + 66 * i
        partes.append(texto_svg(0, y0 + 9, nombre, "etq-fuerte"))
        partes.append(texto_svg(0, y0 + 25, nota, "tick"))
        for desplazamiento, modelo, jsons, clase in ((0, "CNN", {r: cnn[r]["json"] for r in RESOLUCIONES}, "m-r128"),
                                                     (24, "MLP", mlp["json"], "m-mlp")):
            y = y0 + desplazamiento
            valores = {r: jsons[r][clave] for r in RESOLUCIONES}
            bajo, alto_v = min(valores.values()), max(valores.values())
            partes.append(f'<rect class="rango" x="{x(bajo) - 7:.1f}" y="{y - 7:.1f}" width="{x(alto_v) - x(bajo) + 14:.1f}" '
                          f'height="14" rx="7"/>')
            for r in sorted(valores, key=valores.get):
                partes.append(marca(f"{nombre} · {modelo} GPU {res(r)}: accuracy {num(valores[r])}",
                                    punto(x(valores[r]), y, 4.5, f"anillo {clase}")))
            partes.append(texto_svg(x(bajo) - 14, y + 4, f"{modelo} {num(bajo, 2)} – {num(alto_v, 2)}", "valor", "end"))
    return svg(alto + 24, "Accuracy en test de originales, parecidas y alteradas en las tres corridas de GPU de la CNN "
               "y del MLP", partes)


def minutos_corrida(datos):
    carga = (datos["segundos_carga_imagenes"] + datos.get("segundos_descriptores", 0.0)) / 60
    entrenamiento = datos["segundos_entrenamiento_total"] / 60
    total = datos["duracion_corrida_segundos"] / 60
    return carga, entrenamiento, total - carga - entrenamiento, total


def grafico_costo(cnn, mlp):
    """Duración de cada corrida en GPU, separada en carga, entrenamiento y resto."""
    x0, por_minuto = 122.0, 1.1
    partes = []
    alto = 3 * 70 + 4
    for minuto, etiqueta in ((0, "0"), (60, "1 h"), (120, "2 h"), (180, "3 h"), (240, "4 h"), (300, "5 h")):
        x = x0 + minuto * por_minuto
        partes.append(linea_svg(x, x, 4, alto, "rejilla"))
        partes.append(texto_svg(x, alto + 16, etiqueta, "tick", "middle"))
    for i, resolucion in enumerate(RESOLUCIONES):
        y_grupo = 13 + 70 * i
        partes.append(texto_svg(0, y_grupo + 29, res(resolucion), "etq-fuerte"))
        total_mlp = minutos_corrida(mlp["json"][resolucion])[3]
        for j, (modelo, datos, clase) in enumerate((("CNN", cnn[resolucion]["json"], "m-r128"),
                                                    ("MLP", mlp["json"][resolucion], "m-mlp"))):
            y = y_grupo + 25 * j
            carga, entrenamiento, resto, total = minutos_corrida(datos)
            partes.append(texto_svg(112, y + 14, modelo, "tick", "end"))
            x = x0
            tramos = (("carga de imágenes" + (" + descriptores" if modelo == "MLP" else ""), carga, f"{clase} lavado-1"),
                      ("entrenamiento", entrenamiento, clase),
                      ("resto (evaluación y análisis)", resto, f"{clase} lavado-2"))
            for k, (nombre, minutos, estilo) in enumerate(tramos):
                ancho = minutos * por_minuto
                tip = (f"{modelo} · GPU {res(resolucion)} — {nombre}: {num(minutos, 1)} min "
                       f"({entero(minutos / total * 100)} % de la corrida)")
                ultimo = k == len(tramos) - 1
                forma = barra(x, x + max(ancho - 2, 0.5), y + 3, 18, estilo, redondeada=ultimo)
                partes.append(marca(tip, f'<rect class="golpe" x="{x:.1f}" y="{y:.1f}" width="{max(ancho, 2):.1f}" height="24"/>',
                                    forma))
                x += ancho
            etiqueta = duracion_corta(total)
            if modelo == "CNN":
                etiqueta += f" · {veces(total / total_mlp)}"
            partes.append(texto_svg(x + 8, y + 16.5, etiqueta, "valor"))
    return svg(alto + 24, "Duración de las corridas en GPU de la CNN y del MLP, en minutos, separada en carga, "
               "entrenamiento y resto", partes)


# ---------------------------------------------------------------------------------------------------
# Diapositivas
# ---------------------------------------------------------------------------------------------------

def lista(items, clase="puntos chicos"):
    return f'<ul class="{clase}">' + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def cifra(etiqueta, valor_cifra, nota):
    return (f'<div class="cifra"><span class="etiqueta">{etiqueta}</span><span class="valor">{valor_cifra}</span>'
            f'<span class="nota">{nota}</span></div>')


def leyenda(*items):
    return '<div class="leyenda">' + "".join(f"<span>{item}</span>" for item in items) + "</div>"


def muestra(clase, texto):
    return f'<i class="muestra {clase}"></i>{texto}'


ESTADO_CUMPLE = ('<span class="estado"><svg viewBox="0 0 16 16" aria-hidden="true"><circle class="icono-fondo" '
                 'cx="8" cy="8" r="8"/><path class="icono-marca" d="M4.5 8.3l2.3 2.3 4.7-4.9"/></svg>Cumple</span>')


def tabla(cabecera, filas, clase="tabla", destacadas=()):
    cuerpo = []
    for i, fila in enumerate(filas):
        celdas = "".join(fila)
        cuerpo.append(f'<tr class="destacada">{celdas}</tr>' if i in destacadas else f"<tr>{celdas}</tr>")
    return (f'<table class="{clase}"><thead><tr>{"".join(cabecera)}</tr></thead><tbody>'
            + "".join(cuerpo) + "</tbody></table>")


def th(texto, numero=False):
    return f'<th class="num">{texto}</th>' if numero else f"<th>{texto}</th>"


def td(texto, numero=False, codigo=False):
    if codigo:
        return f'<td class="codigo">{texto}</td>'
    return f'<td class="num">{texto}</td>' if numero else f"<td>{texto}</td>"


def portada(fecha):
    return f"""<section class="diapo portada" id="d1" aria-label="Diapositiva 1: portada">
  <div class="lienzo">
    <p class="seccion">Técnicas Avanzadas de Machine Learning · Evaluación Parcial 2</p>
    <div class="titular">
      <h1>¿Quién sale en la imagen?</h1>
      <p class="sub">Una red neuronal convolucional que reconoce a 64 personajes de Los Simpson. Resultados de las tres corridas en GPU del <span style="white-space: nowrap">{fecha}</span>.</p>
    </div>
    <div class="equipo">
      <p><span class="rol">Equipo</span><br>Vincent Farenden Cerón · Rodrigo Martínez Becker<br>Iván Pavón Barría · Diego Peña y Lillo Luhrs</p>
      <p><span class="rol">Profesor</span><br>Francisco Jerez Salazar</p>
    </div>
  </div>
</section>
"""


def diapositiva(n, total, seccion, titulo, cuerpo, disposicion, estilo=""):
    atributo = f' style="{estilo}"' if estilo else ""
    return f"""<!-- {n} -->
<section class="diapo" id="d{n}" aria-label="Diapositiva {n}">
  <div class="lienzo">
    <header class="cabeza">
      <p class="seccion">{seccion}</p>
      <h2>{titulo}</h2>
    </header>
    <div class="cuerpo {disposicion}"{atributo}>
      {cuerpo}
    </div>
    <footer class="pie"><span>{PIE}</span><span>{n:02d} / {total}</span></footer>
  </div>
</section>
"""


def construir_diapositivas(cnn, mlp, d128, n_train):
    j = {r: cnn[r]["json"] for r in RESOLUCIONES}
    m = mlp["json"]
    c128, c64 = cnn["128x128"], cnn["64x64"]
    e128, e64 = c128["experimentos"], c64["experimentos"]
    base128, base64_ = e128["Base"]["f1"], e64["Base"]["f1"]
    mayoritaria = c128["mayoritaria"]
    figuras = {clave: c128["figuras"][clave] for clave in c128["figuras"]}
    registro = d128["registro"]
    primera = registro["reducciones"][0]
    ep = registro["epocas"]
    delta_acc = {r: j[r]["accuracy_test"] - m[r]["accuracy_test"] for r in RESOLUCIONES}
    delta_f1 = {r: j[r]["f1_macro_test"] - m[r]["f1_macro_test"] for r in RESOLUCIONES}
    razon_entreno = {r: j[r]["segundos_entrenamiento_total"] / m[r]["segundos_entrenamiento_total"] for r in RESOLUCIONES}
    razon_corrida = {r: j[r]["duracion_corrida_segundos"] / m[r]["duracion_corrida_segundos"] for r in RESOLUCIONES}
    costo_cnn = {"64x64": c64["costo_kpi"], "128x128": c128["costo_kpi"]}
    por_clase = c128["reporte"]
    fuertes = sum(1 for f in por_clase.values() if f["f1"] >= 0.60)
    debiles_mlp = sum(1 for f in mlp["reporte_64"].values() if f["f1"] < 0.30)
    familia = ("homer_simpson", "marge_simpson", "bart_simpson", "lisa_simpson")
    fam_mlp = [mlp["reporte_64"][p]["f1"] for p in familia]
    fam_cnn = [por_clase[p]["f1"] for p in familia]
    parte_entreno = [j[r]["segundos_entrenamiento_total"] / j[r]["duracion_corrida_segundos"] for r in RESOLUCIONES]
    escalado32 = j["32x32"]["escalado_batch_imagenes_por_segundo"]
    escalado64 = j["64x64"]["escalado_batch_imagenes_por_segundo"]
    uso_mlp = [m[r]["utilizacion_gpu_media"] for r in RESOLUCIONES]
    tipos = {r: (j[r]["accuracy_originales_test"], j[r]["accuracy_parecidas_test"], j[r]["accuracy_alteradas_test"])
             for r in RESOLUCIONES}
    ventaja_parecidas_mlp = [m[r]["accuracy_parecidas_test"] - m[r]["accuracy_originales_test"] for r in RESOLUCIONES]
    lisa_bart = (c128["confusiones"][("lisa_simpson", "bart_simpson")]["pct"],
                 c128["confusiones"][("bart_simpson", "lisa_simpson")]["pct"])
    lisa_bart_mlp = (mlp["confusiones_64"][("lisa_simpson", "bart_simpson")]["pct"],
                     mlp["confusiones_64"][("bart_simpson", "lisa_simpson")]["pct"])
    segunda = max((c for c in e128 if c != "Base" and not e128[c]["no_cabe"]), key=lambda c: e128[c]["f1"])
    caidas = sorted(((base128 - e128[c]["f1"], c) for c in e128 if c != "Base" and not e128[c]["no_cabe"]), reverse=True)
    mayores_caidas = [f"{EXPERIMENTOS[c][0].lower()}{EXPERIMENTOS[c][1:]} (−{puntos(caida)}{' puntos' if k == 0 else ''})"
                      for k, (caida, c) in enumerate(caidas[:3])]
    costo_exp = c128["costo_exp"]
    campo = d128["campo_receptivo"]
    prediccion = d128["prediccion"]

    rango = lambda valores, f=num: f"{f(min(valores))} – {f(max(valores))}"  # noqa: E731

    d = []

    d.append(("Contexto", "El mismo problema y los mismos KPIs que el MLP", "dos mitad", "", f"""<div class="columna">
        <p>Un archivo de 40.123 imágenes sin catalogar no se puede buscar ni reutilizar. La CNN propone a qué personaje corresponde cada imagen, y el catalogador <mark>revisa en vez de etiquetar</mark>.</p>
        {lista([
            "<strong>Clasificación multiclase de etiqueta única:</strong> imagen RGB de entrada, 64 probabilidades de salida. La etiqueta es siempre la carpeta, nunca el nombre del archivo.",
            "<strong>Mismo proceso que el MLP:</strong> la misma partición, las mismas imágenes alteradas y parecidas, los mismos pesos por clase, el mismo bucle de entrenamiento y el mismo criterio de acierto. Cambia el modelo.",
            "<strong>Criterio de acierto:</strong> nombrar al personaje de la carpeta o a otro que aparece en la imagen (misma escena a ±3 s, gemela en otra carpeta o misma toma verificada). La accuracy estricta, que solo acepta la carpeta, se informa como referencia.",
        ])}
      </div>
      <div class="columna">
        <div class="desliza">
          {tabla([th("KPI"), th("Métrica en test"), th("Umbral", True)], [
              [td("Cobertura de asistencia"), td("Top-3 accuracy"), td("≥ 0,70", True)],
              [td("Equidad entre personajes"), td("F1 macro"), td("≥ 0,45", True)],
              [td("Automatización sin revisión"), td("Accuracy"), td("≥ 0,50", True)],
              [td("Ventaja sobre lo trivial"), td("Accuracy / clase mayoritaria"), td("≥ 4×", True)],
              [td("Costo de catalogación"), td("Imágenes/s, entrada + red"), td("≥ 1.000", True)],
          ])}
        </div>
        <p class="leyenda-figura">Umbrales fijados antes de entrenar, los mismos del MLP. Referencias: azar {porcentaje(1 / 64)} y clase mayoritaria (Homero) {porcentaje(mayoritaria)}.</p>
      </div>"""))

    d.append(("Contexto · Datos", "64 personajes, 40.123 originales y la misma partición del MLP", "dos mitad", "", f"""<div class="columna">
        <div class="cifras" style="--n: 3">
          {cifra("Personajes", "64", "una carpeta cada uno")}
          {cifra("Originales", "40.123", "2,1 GB, mediana 640 × 480")}
          {cifra("Desbalance", "33:1", "Homero 3.118, Louie 94")}
        </div>
        {lista([
            'Cerca del 45 % son fotogramas de episodio. El nombre trae episodio e instante: <span class="nombre-archivo">homer_simpson__S02E15__t0675633.jpg</span>',
            "<strong>Alteradas:</strong> a un cuarto de cada personaje se le suma una variante con giro de 10 a 20°, cambio de color y filtro de textura. Cada una va al mismo conjunto que su original.",
            f"<strong>Parecidas:</strong> {entero(j['128x128']['imagenes_parecidas'])} imágenes casi idénticas a otra (miniatura 16 × 16, coseno ≥ 0,90) entran filtradas en lugar de la original.",
            f"<strong>Gemelas:</strong> {entero(j['128x128']['pares_gemelas'])} pares de la misma escena guardada en dos carpetas. Los dos personajes cuentan como presentes.",
        ])}
      </div>
      <div class="columna">
        <p class="bajada">Reparto estratificado de las originales por personaje, con semilla 42 y sin tope por personaje.</p>
        <div class="reparto" aria-hidden="true"><span style="width:70%"></span><span style="width:20%"></span><span style="width:10%"></span></div>
        <div class="conjuntos">
          <div class="conjunto"><span class="titulo-conjunto">Entrenamiento · 70 %</span><span class="cant">{entero(j['128x128']['imagenes_train'])}</span><span class="nota">imágenes, {entero(j['128x128']['imagenes_train_alteradas'])} alteradas</span></div>
          <div class="conjunto"><span class="titulo-conjunto">Validación · 20 %</span><span class="cant">{entero(j['128x128']['imagenes_val'])}</span><span class="nota">imágenes, {entero(j['128x128']['imagenes_val_alteradas'])} alteradas</span></div>
          <div class="conjunto"><span class="titulo-conjunto">Test · 10 %</span><span class="cant">{entero(j['128x128']['imagenes_test'])}</span><span class="nota">imágenes, {entero(j['128x128']['imagenes_test_alteradas'])} alteradas</span></div>
        </div>
        {lista([
            "El desbalance se compensa con <strong>pesos por clase</strong>: la raíz cuadrada de los balanceados, de 0,49 para Homero a 2,78 para Louie.",
            "Validación y test conservan la distribución real del dataset. Son exactamente las mismas imágenes que vio el MLP.",
        ])}
      </div>"""))

    d.append(("Contexto · Cambio de dataset", "Dejamos el dataset de la EP1 por uno más cercano al archivo real", "dos mitad", "", f"""<div class="columna">
        <div class="desliza">
          {tabla([th(""), th("EP1"), th("EP2")], [
              [td("Personajes"), td("25"), td("64")],
              [td("Imágenes originales"), td("20.074"), td("40.123")],
              [td("Partición"), td("train y test fijos de Kaggle, con criterio no documentado"), td("Propia: 70 / 20 / 10, estratificada y con semilla 42")],
              [td("Fotogramas con episodio e instante"), td("—"), td("Cerca del 45 %: permiten saber quién comparte la escena")],
              [td("Modelos entrenados"), td("MLP"), td("MLP y CNN, con los mismos datos y la misma partición")],
          ], "tabla densa")}
        </div>
        <p class="leyenda-figura">El MLP de la comparación es el que se volvió a entrenar sobre el dataset nuevo, no el de la EP1.</p>
      </div>
      <div class="columna">
        {lista([
            "<strong>Más cerca del negocio:</strong> un archivo audiovisual también tiene a los secundarios, no solo a los 25 personajes más conocidos.",
            "<strong>El doble de datos:</strong> una CNN aprende sus filtros de los píxeles y necesita más ejemplos que un MLP sobre descriptores.",
            "<strong>Partición bajo control:</strong> cada alterada en el conjunto de su original y las casi idénticas filtradas, para no acertar en test por haber visto la misma imagen.",
            "<strong>Comparación justa:</strong> las dos redes ven las mismas imágenes; la diferencia la pone el modelo.",
            "<strong>El costo:</strong> un desbalance de 33:1 y escenas con varios personajes, que se tratan con pesos por clase y con el criterio de acierto por escena.",
        ])}
      </div>"""))

    ultimo_campo = campo[-1]
    d.append(("Modelo · Fundamentos", "Filtros que se aprenden y recorren toda la imagen", "dos figura-ancha", "", f"""{figuras['recorrido']}
      <div class="columna">
        <div class="cifras" style="--n: 1; gap: 1.2cqw">
          {cifra("Pesos de la primera capa", entero(d128['pesos_conv_1']), f"frente a {entero(d128['pesos_mlp_pixeles'])} de una capa densa sobre los píxeles")}
          {cifra("Campo receptivo del último bloque", f"{ultimo_campo[1]} × {ultimo_campo[1]}", f"píxeles: el {ultimo_campo[2]} % del lado de la imagen")}
        </div>
        {lista([
            "<strong>Pesos compartidos:</strong> cada filtro de 3 × 3 se aplica en todas las posiciones de la imagen.",
            f"<strong>Del borde al personaje:</strong> en el cuarto bloque, los filtros más activos se encienden sobre la cabeza de {NOMBRES.get(prediccion[0], prediccion[0])}.",
        ])}
        <p class="leyenda-figura">Los 8 filtros más activos de cada bloque para una imagen de test, predicha con {num(prediccion[1], 1)} % de confianza.</p>
      </div>"""))

    corr, activos = d128["varias_correlacion"], d128["varias_activos"]
    corr_primeros = [corr[b] for b in ("1", "2", "3")]
    d.append(("Modelo · Mapas de activación", "Los mismos filtros con cuatro personajes: del color a la cara", "dos figura-ancha", "", f"""{figuras['recorrido_varias']}
      <div class="columna">
        <div class="cifras" style="--n: 1; gap: 1.2cqw">
          {cifra("Correlación entre los cuatro, bloque 4", num(corr['4'], 2), f"de {num(min(corr_primeros), 2)} a {num(max(corr_primeros), 2)} en los bloques 1 a 3")}
          {cifra("Valores activos", f"{entero(activos['1'])} % → {entero(activos['4'])} %", "del primer al cuarto bloque")}
        </div>
        {lista([
            "<strong>Bloques 1 a 3:</strong> zonas de color, contornos y combinaciones de bordes, del personaje y del fondo.",
            "<strong>Bloque 4:</strong> pocas manchas sobre el personaje, y cada uno activa sus propios filtros.",
        ])}
        <p class="leyenda-figura">Homero, Marge, Bart y Lisa, acertados con {num(min(d128['varias_confianza']), 1)} a {num(max(d128['varias_confianza']), 1)} % de confianza. Cada columna es el mismo filtro con la misma escala.</p>
      </div>"""))

    d.append(("Modelo", "La red recibe la imagen entera: 128 × 128 × 3 valores", "columna", "gap: 2.2cqw", f"""<ol class="flujo">
        <li><span class="nombre">Imagen RGB</span><span class="detalle">De 38 × 42 a 2.700 × 3.600 px</span></li>
        <li><span class="nombre">Redimensionado</span><span class="detalle">128 × 128; también se corrió a 32 y 64</span></li>
        <li><span class="nombre">Escalado</span><span class="detalle">De 0–255 a 0–1, dentro del modelo</span></li>
        <li><span class="nombre">4 bloques convolucionales</span><span class="detalle">Conv 3 × 3, BatchNorm, ReLU y MaxPool 2 × 2</span></li>
        <li><span class="nombre">Pooling global</span><span class="detalle">Un valor por filtro: 256</span></li>
        <li><span class="nombre">Cabeza densa</span><span class="detalle">256 neuronas y 64 probabilidades softmax</span></li>
      </ol>
      <div class="dos mitad">
        <div class="columna">
          <p class="seccion">Arquitectura base · {entero(j['128x128']['parametros_modelo_final'])} parámetros</p>
          <div class="capas">
            <div class="capa"><span class="nombre-capa">entrada</span><span>Imagen RGB en uint8, sin aplanar</span><span class="forma">128 × 128 × 3</span></div>
            <div class="capa"><span class="nombre-capa">conv 1–4</span><span>32, 64, 128 y 256 filtros de 3 × 3 + BatchNorm + ReLU + MaxPool</span><span class="forma">8 × 8 × 256</span></div>
            <div class="capa"><span class="nombre-capa">pooling</span><span>Promedio global de cada mapa</span><span class="forma">256</span></div>
            <div class="capa"><span class="nombre-capa">densa</span><span>ReLU + BatchNorm + Dropout 0,3</span><span class="forma">256</span></div>
            <div class="capa"><span class="nombre-capa">salida</span><span>Softmax, una neurona por personaje</span><span class="forma">64</span></div>
          </div>
          <p class="leyenda-figura">El último bloque concentra el 62 % de los parámetros. E2, con dos convoluciones por bloque, tiene {entero(j['64x64']['parametros_modelo_final'])}.</p>
        </div>
        <div class="columna">
          <p class="seccion">Entrenamiento</p>
          {lista([
              "Adam, learning rate 0,001, batch 128, hasta 60 épocas; las variantes, 35.",
              "Entropía cruzada categórica con suavizado 0,1; en escenas compartidas, 15 % del objetivo va a los acompañantes.",
              "Parada temprana con paciencia 12 que restaura la mejor época; el learning rate baja a la mitad tras 4 épocas sin mejora.",
              f"Keras 3 sobre PyTorch en la RTX 4060 Laptop. Las imágenes viven en la VRAM en uint8: {num(registro['vram_datos'], 2)} GB a 128 × 128.",
          ])}
        </div>
      </div>"""))

    d.append(("Experimentación", "Once variantes a 128 × 128: ninguna supera a la base", "dos", "", f"""<figure class="grafico">
        {leyenda(muestra("c-r128", "GPU · 128 × 128"), muestra("c-r64", "GPU · 64 × 64"), "F1 macro de validación. La fila sombreada es el modelo base.")}
        {grafico_experimentos(cnn)}
      </figure>
      <div class="columna">
        {lista([
            f"<strong>Gana la base.</strong> La más cercana, {segunda} ({EXPERIMENTOS[segunda].lower()}), queda a {puntos(base128 - e128[segunda]['f1'])} puntos, con 35 épocas frente a 60.",
            f"<strong>E2 y E5 no caben</strong> en los 8 GB de la GPU a 128 × 128. A 64 × 64, E2 era la ganadora: {num(e64['E2']['f1'])} frente a {num(base64_)}.",
            f"<strong>Lo que más cuesta:</strong> {mayores_caidas[0]}, {mayores_caidas[1]} y {mayores_caidas[2]}.",
            f"<strong>Sin color se pierde menos que a 64 × 64</strong> (−{puntos(base64_ - e64['R1']['f1'])}): con más resolución, la forma suple parte del color.",
            "<strong>C1 no se entrena</strong> en ninguna de las dos corridas: no hay dos factores que superen a la base.",
        ])}
        <p class="leyenda-figura">Tabla completa en el anexo A.</p>
      </div>"""))

    d.append(("Entrenamiento y validación", "La base generaliza mucho mejor que el MLP", "dos figura-ancha", "", f"""{figuras['curvas']}
      <div class="columna">
        {lista([
            f"Entrenamiento llega a <strong>{num(c128['diagnostico']['acc_train'], 2)}</strong> de accuracy y validación a <strong>{num(c128['diagnostico']['acc_val'], 2)}</strong>: una brecha de {num(c128['diagnostico']['brecha'], 2)}, frente a {num(c64['diagnostico']['brecha'], 2)} a 64 × 64 y <mark>{num(mlp['brecha_64'], 2)} en el MLP</mark>.",
            f"Usa las 60 épocas: la pérdida de validación sigue bajando y toca su mínimo en la época {c128['diagnostico']['mejor_epoca']} ({num(c128['diagnostico']['perdida_val'])}).",
            f"El learning rate baja a la mitad {en_palabras(len(registro['reducciones']))} veces. El primer escalón, tras la época {primera}, sube la validación de {num(ep[primera]['val_acc'])} a {num(ep[primera + 1]['val_acc'])} en una época.",
            "Desde la época 30 la validación casi no se mueve, mientras el entrenamiento sigue subiendo.",
        ])}
        <p class="leyenda-figura">Curvas del modelo base, corrida de GPU a 128 × 128.</p>
      </div>"""))

    veces_mayoritaria = {r: j[r]["accuracy_test"] / mayoritaria for r in RESOLUCIONES}
    d.append(("Evaluación en test", "Los cinco KPIs se cumplen en las tres corridas, con margen", "columna", "gap: 2cqw", f"""<div class="cifras">
        {cifra("Accuracy", num(j['128x128']['accuracy_test']), f"estricta {num(j['128x128']['accuracy_estricta_test'])}")}
        {cifra("Top-3 accuracy", num(j['128x128']['top3_test']), "umbral 0,70")}
        {cifra("F1 macro", num(j['128x128']['f1_macro_test']), f"estricto {num(j['128x128']['f1_macro_estricto_test'])}")}
        {cifra("Frente a la clase mayoritaria", veces(veces_mayoritaria['128x128']), f"{num(c128['azar_veces'], 1)} veces el azar")}
      </div>
      <p class="leyenda-figura">Arriba, la corrida de GPU a 128 × 128 (el modelo base), la de mejor accuracy. Abajo, el rango de las tres corridas.</p>
      <div class="desliza">
        {tabla([th("KPI"), th("Umbral", True), th("Tres corridas (mín. – máx.)", True), th("Estado")], [
            [td("Cobertura de asistencia · top-3"), td("≥ 0,70", True), td(rango([j[r]['top3_test'] for r in RESOLUCIONES]), True), td(ESTADO_CUMPLE)],
            [td("Equidad entre personajes · F1 macro"), td("≥ 0,45", True), td(rango([j[r]['f1_macro_test'] for r in RESOLUCIONES]), True), td(ESTADO_CUMPLE)],
            [td("Automatización sin revisión · accuracy"), td("≥ 0,50", True), td(rango([j[r]['accuracy_test'] for r in RESOLUCIONES]), True), td(ESTADO_CUMPLE)],
            [td("Ventaja sobre la clase mayoritaria"), td("≥ 4×", True), td(rango(list(veces_mayoritaria.values()), veces), True), td(ESTADO_CUMPLE)],
            [td("Costo de catalogación · imágenes/s"), td("≥ 1.000", True), td(rango(list(costo_cnn.values()), entero) + " (2 corridas medidas)", True), td(ESTADO_CUMPLE)],
        ])}
      </div>
      <p class="leyenda-figura">Con el criterio estricto la accuracy va de {num(min(j[r]['accuracy_estricta_test'] for r in RESOLUCIONES))} a {num(max(j[r]['accuracy_estricta_test'] for r in RESOLUCIONES))}: las tres corridas superan el umbral también con esa vara.</p>"""))

    d.append(("Evaluación · Tres corridas", "Más resolución sí ayuda a la CNN, cada vez menos", "dos", "", f"""<figure class="grafico">
        {leyenda(muestra("c-r128", "CNN"), muestra("c-mlp", "MLP, misma resolución"), muestra("c-entreno", "Accuracy con el criterio del proyecto"), muestra("vacia", "Accuracy estricta"))}
        {grafico_corridas(cnn, mlp)}
      </figure>
      <div class="columna">
        {lista([
            f"De 32 a 64 px la CNN gana <strong>{puntos(j['64x64']['accuracy_test'] - j['32x32']['accuracy_test'])} puntos</strong> de accuracy; de 64 a 128, {puntos(j['128x128']['accuracy_test'] - j['64x64']['accuracy_test'])} más, con el triple de duración.",
            f"El MLP se mueve menos: de {num(m['32x32']['accuracy_test'])} a {num(m['128x128']['accuracy_test'])}, y con R2 la resolución suma solo 2 a 3 puntos. Sus histogramas resumen el color, no la forma.",
            "A cada resolución gana una configuración distinta: E2 a 32 y 64 px, la base a 128 (E2 no cabe en la GPU).",
            f"A la misma resolución, la CNN supera al MLP por <mark>{puntos(min(delta_acc.values()))} a {puntos(max(delta_acc.values()))} puntos</mark>. La estricta queda unos 5 puntos bajo la oficial en las tres.",
        ])}
      </div>"""))

    d.append(("Evaluación · Por personaje", "Con la forma, la familia Simpson deja de confundirse", "dos", "", f"""<figure class="grafico">
        {leyenda(muestra("cuadro c-fam", "Familia Simpson"), muestra("cuadro c-otro", "Otros personajes"), muestra("vacia", "MLP, GPU a 64 × 64"), "F1 en test de la CNN a 128 × 128 · imágenes de entrenamiento a la izquierda")}
        {grafico_personajes(cnn, mlp, n_train)}
      </figure>
      <div class="columna">
        {lista([
            f"<strong>{fuertes} de 64</strong> personajes con F1 de 0,60 o más y ninguno bajo 0,30; el MLP tenía {en_palabras(debiles_mlp)} bajo 0,30.",
            f"Homero, Marge, Bart y Lisa pasan de {num(min(fam_mlp), 2)}–{num(max(fam_mlp), 2)} en el MLP a {num(min(fam_cnn), 2)}–{num(max(fam_cnn), 2)}: <mark>lo que separa a Bart de Lisa es el peinado</mark>, no el color.",
            f"Ahora sí pesa el tamaño: la correlación entre imágenes de entrenamiento y F1 es {num(c128['correlacion'], 2)} ({num(mlp['correlacion_64'], 2)} en el MLP).",
            "Abajo quedan personajes con alrededor de cien imágenes de entrenamiento: Disco Stu, Maggie, Rod Flanders y Helen Lovejoy.",
        ])}
      </div>"""))

    filas_confusion = []
    for real, predicho in PARES_CONFUSION:
        fila = [td(NOMBRES[real]), td(NOMBRES[predicho])]
        for pares in (c128["confusiones"], c64["confusiones"], mlp["confusiones_64"]):
            dato = pares.get((real, predicho))
            fila.append(td(porcentaje(dato["pct"]) if dato else "—", True))
        filas_confusion.append(fila)
    d.append(("Análisis de errores", "Quedan los personajes dibujados casi iguales", "dos mitad", "", f"""<div class="columna">
        <div class="desliza">
          {tabla([th("Personaje real"), th("Predicho"), th("CNN 128", True), th("CNN 64", True), th("MLP 64", True)], filas_confusion)}
        </div>
        <p class="leyenda-figura">Porcentaje de las imágenes de test del personaje real predichas como el otro, en GPU. «—»: fuera de los 15 pares más frecuentes de esa corrida.</p>
      </div>
      <div class="columna">
        {lista([
            "<strong>Gemelas y dibujos casi iguales:</strong> Patty y Selma, Carl y Lenny, el abuelo y Homero.",
            "<strong>Escenas compartidas:</strong> Helen sale casi siempre con el reverendo Lovejoy. Cuando la escena lo confirma, la predicción cuenta como acierto; las de la tabla son las que la escena no detectó.",
            f"<strong>La familia ya no:</strong> Lisa y Bart se confunden un {num(min(lisa_bart) * 100, 1)} a {porcentaje(max(lisa_bart))}, frente al {num(min(lisa_bart_mlp) * 100, 0)} a {porcentaje(max(lisa_bart_mlp), 0)} del MLP.",
            f"Homero sigue siendo el destino más frecuente de los errores: {c128['absorbe_homero']} imágenes entre los 15 pares más frecuentes.",
        ])}
      </div>"""))

    d.append(("Análisis · Grad-CAM", "La red mira la cabeza del personaje", "dos figura-ancha", "", f"""{figuras['gradcam']}
      <div class="columna">
        <div class="cifras" style="--n: 1; gap: 1.2cqw">
          {cifra("Calor en el 10 % de píxeles más cálidos", porcentaje(c128['gradcam_10']), f"{porcentaje(c64['gradcam_10'])} a 64 × 64 · 10 % si mirara toda la imagen")}
          {cifra("Calor en la mitad central", porcentaje(c128['gradcam_centro_aciertos']), f"{porcentaje(c128['gradcam_centro_errores'])} en los errores · 25 % si fuera uniforme")}
        </div>
        {lista([
            "En los aciertos, la zona cálida cae sobre la cabeza: el pelo de Martin Prince, Bart de espaldas, la cara de Burns.",
            "Los tres errores más confiados miran a un personaje que sí está: Homero en dos imágenes de la carpeta de Bart y Chalmers junto a Skinner.",
        ])}
        <p class="leyenda-figura">Tres aciertos y tres errores con la mayor confianza, corrida de GPU a 128 × 128.</p>
      </div>"""))

    d.append(("Análisis · Grad-CAM por personaje", "En los 64 personajes la red mira sobre todo la cabeza", "dos", "", f"""{figuras['gradcam_clases']}
      <div class="columna">
        <div class="cifras" style="--n: 1; gap: 1.2cqw">
          {cifra("Calor en el 10 % de píxeles más cálidos", porcentaje(d128['gradcam_clases_10']), "10 % si mirara toda la imagen")}
          {cifra("Calor en la mitad central", porcentaje(d128['gradcam_clases_centro']), "25 % si fuera uniforme")}
        </div>
        {lista([
            "Casi siempre en la cabeza: el peinado de Marge, el bigote de Ned, los lentes de Milhouse. En unos pocos, en la ropa.",
            f"No explica qué personajes cuestan más: correlación de {num(d128['gradcam_clases_recall'], 2)} con el recall.",
        ])}
        <p class="leyenda-figura">Grad-CAM de la clase de la carpeta sobre su imagen original de test más probable.</p>
      </div>"""))

    d.append(("Análisis de errores · Aciertos", "Aciertos seguros aunque el personaje sea pequeño", "dos figura-ancha", "", f"""{figuras['aciertos']}
      <div class="columna">
        {lista([
            "Los diez con confianza de 1,00, uno por personaje.",
            "Bart de espaldas, Homero pequeño en el living, Kirk Van Houten y Cletus con otros personajes: la red reconoce la forma, no solo la paleta.",
            f"En total acierta {entero(d128['aciertos_total'])} de {entero(j['128x128']['imagenes_test'])} imágenes ({porcentaje(d128['aciertos_total'] / j['128x128']['imagenes_test'])}), con una confianza media de {num(d128['confianza_aciertos'], 2)}.",
        ])}
        <p class="leyenda-figura">Un acierto por personaje, corrida de GPU a 128 × 128.</p>
      </div>"""))

    d.append(("Análisis de errores · Errores confiados", "Casi todos los errores confiados nombran a alguien que sí está", "dos figura-ancha", "", f"""{figuras['errores']}
      <div class="columna">
        {lista([
            "<strong>En 8 de los 10 el predicho está en el encuadre:</strong> Homero en dos imágenes de la carpeta de Bart, Chalmers junto a Skinner, Manjula, Bart junto a Lisa, Patty junto a Selma, el reverendo con Helen y Marge con Homero en el televisor.",
            f"Son pocos: {j['128x128']['errores_confiados_test']} con confianza mayor que 0,80, el {porcentaje(j['128x128']['errores_confiados_test'] / j['128x128']['imagenes_test'])} del test, frente a {m['128x128']['errores_confiados_test']} del MLP a la misma resolución.",
            f"La confianza separa: {num(d128['confianza_aciertos'], 2)} de media en los aciertos y {num(d128['confianza_errores'], 2)} en los errores; solo {d128['errores_090']} de {entero(d128['errores_total'])} errores pasan de 0,90.",
        ])}
        <p class="leyenda-figura">Corrida de GPU a 128 × 128. La accuracy oficial es una cota inferior: ni la escena ni las gemelas detectaron a esos acompañantes.</p>
      </div>"""))

    d.append(("Análisis de errores · Escenas", "Nombrar al acompañante también es acertar", "dos figura-ancha", "", f"""{figuras['escenas']}
      <div class="columna">
        <div class="cifras" style="--n: 1; gap: 1.2cqw">
          {cifra("Test con otro personaje presente", porcentaje(d128['escenas_presentes'] / j['128x128']['imagenes_test']), "por escena, gemela o misma toma")}
          {cifra("Accuracy en esas imágenes", num(d128['escenas_acc_oficial'], 2), f"estricta {num(d128['escenas_acc_estricta'], 2)} · GPU a 128 × 128")}
        </div>
        {lista([
            f"{c128['aciertos_escena']} aciertos por escena: suben la accuracy {puntos(j['128x128']['accuracy_test'] - j['128x128']['accuracy_estricta_test'])} puntos.",
            "Los pares son los que la serie pone juntos: Wiggum y Lou, Patty y Selma, Smithers y Burns, Skinner y Chalmers.",
        ])}
      </div>"""))

    o128, p128, a128 = tipos["128x128"]
    o64, _, a64 = tipos["64x64"]
    d.append(("Análisis de errores · Tipo de imagen", "Las alteradas casi no cuestan y las parecidas ayudan menos que en el MLP", "dos", "", f"""<div class="columna">
        <figure class="grafico">
          {leyenda(muestra("c-r128", "CNN"), muestra("c-mlp", "MLP"), "Accuracy en test por tipo de imagen. Cada punto es una corrida en GPU (32, 64 y 128 px); la barra va del mínimo al máximo.")}
          {grafico_tipos(cnn, mlp)}
        </figure>
        <div class="cifras" style="--n: 2">
          {cifra("Alteradas frente a originales", puntos(a128 - o128), f"puntos a 128 × 128 ({puntos(a64 - o64)} a 64 × 64)")}
          {cifra("Parecidas frente a originales", "+" + puntos(p128 - o128), f"puntos a 128 × 128 (MLP: +{puntos(min(ventaja_parecidas_mlp), 0)} a +{puntos(max(ventaja_parecidas_mlp), 0)})")}
        </div>
      </div>
      <div class="columna">
        {lista([
            "Las alteradas —giro, cambio de color y filtro de textura— cuestan menos de un punto a 128 × 128: el cambio de color desordena los histogramas del MLP, pero no la forma.",
            f"Las parecidas se aciertan {puntos(p128 - o128, 0)} puntos más que las originales: el filtro reduce la ventaja de las tomas casi idénticas y la CNN depende menos de ella.",
            f"Las originales son la medida más limpia: {num(o128)} a 128 × 128, {puntos(j['128x128']['accuracy_test'] - o128)} puntos bajo la global.",
            f"La CNN gana en los tres tipos y en las tres resoluciones: el MLP no pasa de {num(max(m[r]['accuracy_parecidas_test'] for r in RESOLUCIONES), 2)} ni en las parecidas.",
        ])}
      </div>"""))

    d.append(("Costo por resolución", f"La CNN sí aprovecha la GPU: {porcentaje(j['128x128']['utilizacion_gpu_media'] / 100, 0)} del tiempo a 128 × 128", "dos", "", f"""<figure class="grafico">
        {leyenda(muestra("cuadro c-r128", "CNN"), muestra("cuadro c-mlp", "MLP"), muestra("cuadro c-entreno", "entrenamiento"), muestra("cuadro c-pre", "carga de imágenes (y descriptores en el MLP)"), muestra("cuadro c-resto", "resto"), "×: veces la duración de la corrida del MLP")}
        {grafico_costo(cnn, mlp)}
      </figure>
      <div class="columna">
        {lista([
            f"Entrenar es entre el {porcentaje(min(parte_entreno), 0)} y el {porcentaje(max(parte_entreno), 0)} de cada corrida. La CNN no calcula descriptores: cargar las imágenes toma de {num(minutos_corrida(j['32x32'])[0], 1)} a {num(minutos_corrida(j['128x128'])[0], 1)} minutos.",
            f"La GPU trabaja el {num(j['32x32']['utilizacion_gpu_media'], 1)} % del tiempo a 32 × 32, el {num(j['64x64']['utilizacion_gpu_media'], 1)} % a 64 y el <strong>{num(j['128x128']['utilizacion_gpu_media'], 1)} %</strong> a 128, a {entero(j['128x128']['potencia_media_w'])} W. El MLP, entre el {num(min(uso_mlp), 1)} y el {num(max(uso_mlp), 1)} %.",
            f"Batch: a 32 × 32, pasar de 128 a 1.024 multiplica el rendimiento por {num(escalado32['1024'] / escalado32['128'], 1)}; a 64 × 64 batch 512 suma un {porcentaje(escalado64['512'] / escalado64['128'] - 1, 0)}; a 128 × 128 solo cabe batch 128.",
            f"De extremo a extremo la CNN clasifica {entero(costo_cnn['64x64'])} imágenes por segundo a 64 × 64 y {entero(costo_cnn['128x128'])} a 128 × 128, sobre el umbral de 1.000.",
        ])}
      </div>"""))

    filas_comparacion = [
        [td("Entrada"), td("370 histogramas de color (de 2.462 descriptores)"), td("La imagen de 128 × 128 × 3, sin aplanar")],
        [td("Características"), td("Elegidas a mano, calculadas en la CPU"), td("Filtros de 3 × 3 aprendidos")],
        [td("Configuración final (32 / 64 / 128)"), td(" / ".join(m[r]["configuracion_final"].split(" - ")[0] for r in RESOLUCIONES)),
         td(" / ".join(j[r]["configuracion_final"].split(" - ")[0] for r in RESOLUCIONES))],
        [td("Parámetros del modelo final a 128 × 128"), td(entero(m["128x128"]["parametros_modelo_final"]), True), td(entero(j["128x128"]["parametros_modelo_final"]), True)],
        [td("Accuracy (32 / 64 / 128)"), td(" / ".join(num(m[r]["accuracy_test"]) for r in RESOLUCIONES), True), td(" / ".join(num(j[r]["accuracy_test"]) for r in RESOLUCIONES), True)],
        [td("F1 macro (32 / 64 / 128)"), td(" / ".join(num(m[r]["f1_macro_test"]) for r in RESOLUCIONES), True), td(" / ".join(num(j[r]["f1_macro_test"]) for r in RESOLUCIONES), True)],
        [td("Entrenamiento en minutos (32 / 64 / 128)"), td(" / ".join(num(minutos_corrida(m[r])[1], 1) for r in RESOLUCIONES), True), td(" / ".join(num(minutos_corrida(j[r])[1], 1) for r in RESOLUCIONES), True)],
        [td("Imágenes/s de extremo a extremo"), td(rango(mlp["costos_kpi"], entero), True), td(rango(list(costo_cnn.values()), entero), True)],
        [td("Errores con confianza > 0,80 a 128 × 128"), td(str(m["128x128"]["errores_confiados_test"]), True), td(str(j["128x128"]["errores_confiados_test"]), True)],
        [td("Explicación de la decisión"), td("Importancia por permutación"), td("Grad-CAM")],
    ]
    d.append(("Comparación", f"El mismo proceso, otro modelo: de {entero(min(delta_acc.values()) * 100)} a {entero(max(delta_acc.values()) * 100)} puntos más", "dos", "", f"""<div class="desliza">
        {tabla([th(""), th("MLP (PMC/)"), th("CNN (CNN/)")], filas_comparacion, "tabla densa")}
      </div>
      <div class="columna">
        {lista([
            f"A la misma resolución, la CNN gana entre <strong>{puntos(min(delta_acc.values()))} y {puntos(max(delta_acc.values()))} puntos</strong> de accuracy y entre {puntos(min(delta_f1.values()))} y {puntos(max(delta_f1.values()))} de F1 macro.",
            f"Entrenar cuesta de {veces(min(razon_entreno.values()))} a {veces(max(razon_entreno.values()))} el tiempo del MLP; la corrida completa, de {veces(min(razon_corrida.values()))} a {veces(max(razon_corrida.values()))}.",
            "Pero clasifica más rápido de extremo a extremo: el MLP debe calcular sus descriptores en la CPU antes de usar la red.",
            "Los dos cumplen los cinco KPIs en todas sus corridas; la CNN, con margen en todos.",
        ])}
      </div>"""))

    d.append(("Cierre · Limitaciones y mejoras", "Lo que queda es de datos y de costo", "dos mitad", "", f"""<div class="columna">
        <p class="seccion">Limitaciones</p>
        {lista([
            "<strong>Resolución:</strong> incluso a 128 × 128 un personaje en un plano general ocupa pocas decenas de píxeles, y doblar el lado cuesta cuatro veces más.",
            "<strong>Memoria:</strong> a 128 × 128 E2 y E5 no caben en los 8 GB de la GPU, así que no se pudo medir la configuración que ganó a 32 y 64.",
            "<strong>Entrenada desde cero</strong> con unas 35.000 imágenes; algunos personajes tienen alrededor de cien.",
            "<strong>Nada la obliga a mirar al personaje:</strong> el fondo también puede separar clases. Grad-CAM lo muestra, pero no lo corrige.",
            "<strong>Una sola etiqueta por imagen:</strong> el criterio de acierto lo compensa en la evaluación, no en el modelo.",
        ])}
      </div>
      <div class="columna">
        <p class="seccion">Propuestas</p>
        {lista([
            "<strong>Transfer learning</strong> (MobileNet, ResNet, EfficientNet): filtros aprendidos sobre millones de imágenes.",
            "<strong>Aumento de datos en línea:</strong> un espejo, un recorte o un cambio de color distinto en cada época.",
            "<strong>Medir lo que no cupo:</strong> acumular gradientes o entrenar en bfloat16 para probar E2 y E5 a 128 × 128.",
            "<strong>Recortar al personaje</strong> con un detector antes de clasificar, para sacar el fondo.",
            "<strong>Salida multietiqueta</strong> para nombrar a todos los personajes de la escena.",
            "<strong>Varias semillas o validación cruzada</strong> para decidir entre configuraciones a menos de un punto.",
        ])}
      </div>"""))

    helen, homero = por_clase["helen_lovejoy"], por_clase["homer_simpson"]
    d.append(("Cierre · Impacto ético", "El modelo propone; una persona revisa y decide", "dos mitad", "", f"""<div class="columna">
        <p class="seccion">Derechos de autor</p>
        {lista([
            "Personajes y fotogramas son obras protegidas: los usamos solo con fines académicos, y ni el dataset ni los modelos entrenados se versionan o redistribuyen.",
            "Un uso comercial exige la licencia del titular, también sobre las imágenes de entrenamiento. Detectar a un personaje no prueba una infracción: la cita o la parodia pueden estar permitidas.",
        ])}
        <p class="seccion">Sesgo por el desbalance de 33:1</p>
        {lista([
            f"El modelo encuentra a Helen Lovejoy en el {porcentaje(helen['recall'], 0)} de sus imágenes y a Homero en el {porcentaje(homero['recall'], 0)}; Homero recibe {c128['absorbe_homero']} errores de otros personajes.",
            f"Contando apariciones, daría a Homero {num(homero['recall'] / homero['precision'], 2)} veces las reales y a Helen {num(helen['recall'] / helen['precision'], 2)}: un sesgo que llegaría a las licencias y al merchandising.",
        ])}
      </div>
      <div class="columna">
        <p class="seccion">Revisión humana</p>
        {lista([
            f"Se equivoca en el {porcentaje(1 - j['128x128']['accuracy_test'], 0)} del test y siempre responde algo, aunque la imagen no tenga a ningún personaje del catálogo.",
            f"{j['128x128']['errores_confiados_test']} errores superan una confianza de 0,80: una confianza alta no basta para automatizar.",
            f"Mostrar las tres sugerencias, que contienen la respuesta en el {porcentaje(j['128x128']['top3_test'], 0)} de los casos, y revisar siempre lo que afecte a terceros: reclamos, pagos de licencias o retiro de material.",
        ])}
        <p class="seccion">Fuera del dominio</p>
        {lista(["Con fotos de personas reales sería reconocimiento facial, con problemas de privacidad y consentimiento que este proyecto no aborda: el modelo no debe usarse así."])}
      </div>"""))

    d.append(("Cierre · Conclusiones", "Lo que nos llevamos de las tres corridas", "dos mitad", "", f"""{lista([
            f"<strong>La forma es lo que le faltaba al MLP.</strong> Con el mismo proceso, a 128 × 128 la accuracy pasa de {num(m['128x128']['accuracy_test'])} a {num(j['128x128']['accuracy_test'])} y el F1 macro de {num(m['128x128']['f1_macro_test'])} a {num(j['128x128']['f1_macro_test'])}.",
            "<strong>La representación decide más que el ajuste fino.</strong> Ninguna variante de hiperparámetros, en el MLP o en la CNN, ganó más de 3 puntos.",
            f"<strong>Más resolución ayuda, cada vez menos.</strong> +{puntos(j['64x64']['accuracy_test'] - j['32x32']['accuracy_test'])} puntos de 32 a 64 px y +{puntos(j['128x128']['accuracy_test'] - j['64x64']['accuracy_test'])} de 64 a 128, con el triple de tiempo.",
        ], "puntos")}
      {lista([
            f"<strong>El color pesa menos que en el MLP.</strong> Sin él la CNN pierde {entero((base128 - e128['R1']['f1']) * 100)} puntos de F1 ({entero((base64_ - e64['R1']['f1']) * 100)} a 64 × 64), y aun así supera al color solo del MLP.",
            "<strong>La accuracy es una cota inferior.</strong> Casi todos los errores confiados nombran a un personaje presente en la imagen.",
            f"<strong>El costo se mueve al entrenamiento.</strong> {entero(razon_entreno['128x128'])} veces el del MLP a 128 × 128, pero de extremo a extremo la CNN cataloga más rápido.",
        ], "puntos")}"""))

    filas_anexo_a = []
    for codigo in ("Base", "R1", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1", "P2", "S1", "S2", "A1", "C1"):
        if codigo == "C1":
            celdas = [td("no se entrenó", True)] * 4 + [td("—", True)] * 2
        else:
            a, b = e128[codigo], e64[codigo]
            if a["no_cabe"]:
                celdas = [td("no cabe", True), td("no cabe", True)]
            else:
                celdas = [td(num(a["f1"]), True), td(num(a["acc"]), True)]
            celdas += [td(num(b["f1"]), True), td(num(b["acc"]), True)]
            if a["no_cabe"]:
                celdas += [td("—", True), td("—", True)]
            else:
                celdas += [td(entero(costo_exp[codigo]["parametros"]), True), td(num(costo_exp[codigo]["s_epoca"], 1), True)]
        filas_anexo_a.append([td(codigo, codigo=True), td(EXPERIMENTOS[codigo])] + celdas)
    d.append(("Anexo A · Experimentos", "Métricas de validación de cada configuración", "desliza", "", tabla(
        [th(""), th("Configuración"), th("F1 macro · 128", True), th("Accuracy · 128", True), th("F1 macro · 64", True),
         th("Accuracy · 64", True), th("Parámetros · 128", True), th("s/época · 128", True)],
        filas_anexo_a, "tabla densa", destacadas=(0,))))

    def fila_b(nombre, funcion):
        return [td(nombre)] + [td(funcion(r), True) for r in RESOLUCIONES]

    d.append(("Anexo B · Tres corridas", "Resumen de cada corrida en test y en tiempo", "desliza", "", tabla(
        [th("")] + [th(f"GPU {res(r)}", True) for r in RESOLUCIONES], [
            fila_b("Configuración final", lambda r: j[r]["configuracion_final"].split(" - ")[0]),
            fila_b("Parámetros del modelo final", lambda r: entero(j[r]["parametros_modelo_final"])),
            fila_b("Accuracy", lambda r: num(j[r]["accuracy_test"])),
            fila_b("Accuracy estricta", lambda r: num(j[r]["accuracy_estricta_test"])),
            fila_b("Top-3", lambda r: num(j[r]["top3_test"])),
            fila_b("F1 macro", lambda r: num(j[r]["f1_macro_test"])),
            fila_b("F1 macro estricto", lambda r: num(j[r]["f1_macro_estricto_test"])),
            fila_b("Originales / parecidas / alteradas", lambda r: " / ".join(num(v, 2) for v in tipos[r])),
            fila_b("Errores con confianza > 0,80 (verificados)", lambda r: f"{j[r]['errores_confiados_test']} ({j[r]['errores_confiados_verificados_test']})"),
            fila_b("Carga de imágenes (min)", lambda r: num(minutos_corrida(j[r])[0], 1)),
            fila_b("Entrenamiento (min)", lambda r: num(minutos_corrida(j[r])[1], 1)),
            fila_b("Segundos por época (benchmark)", lambda r: num(j[r]["benchmark_segundos_por_epoca"], 2)),
            fila_b("Utilización media de la GPU", lambda r: f"{num(j[r]['utilizacion_gpu_media'], 1)} %"),
            fila_b("Inferencia de la red (imágenes/s)", lambda r: entero(j[r]["inferencia_imagenes_por_segundo"])),
            fila_b("Duración de la corrida", lambda r: duracion(j[r]["duracion_corrida_segundos"])),
        ], "tabla densa", destacadas=(2, 14))))
    return d


# ---------------------------------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------------------------------

CSS_PROPIO = """
/* ---------- series de la presentación de la CNN ---------- */
.c-r128 { background: var(--s1); }
.c-r64 { background: var(--s2); }
.c-mlp { background: var(--tinta-3); }
.graf .m-r128 { fill: var(--s1); }
.graf .m-r64 { fill: var(--s2); }
.graf .m-mlp { fill: var(--tinta-3); }
.graf .t-r128 { stroke: var(--s1); }
.graf .t-r64 { stroke: var(--s2); }
.graf .t-mlp { stroke: var(--tinta-3); }
/* ---------- figuras más altas que anchas: se limitan al alto de la diapositiva ---------- */
.lamina.alta { justify-self: start; }
.lamina.alta img { width: auto; height: 36cqw; }
@media (max-width: 760px) { .lamina.alta img { width: 100%; height: auto; } }
"""


def tramo(texto, inicio, fin):
    i = texto.index(inicio)
    return texto[i:texto.index(fin, i) + len(fin)]


def reemplazar(texto, viejo, nuevo):
    if texto.count(viejo) != 1:
        raise SystemExit(f"{PRESENTACION_MLP}: no se encontró una sola vez {viejo[:60]!r}; revisa el generador")
    return texto.replace(viejo, nuevo)


def ensamblar(diapositivas, fecha):
    mlp = (RAIZ / PRESENTACION_MLP).read_text(encoding="utf-8")
    cabecera = mlp[:mlp.index("<style>")]
    cabecera = reemplazar(cabecera, "<title>Resultados MLP Simpsons</title>", "<title>Resultados CNN Simpsons</title>")
    cabecera = re.sub(r'<meta name="description" content="[^"]*">',
                      '<meta name="description" content="Presentación de resultados de la red neuronal convolucional que '
                      'reconoce a 64 personajes de Los Simpson: experimentos, métricas, errores, Grad-CAM y comparación '
                      'de las tres corridas en GPU con el MLP.">', cabecera, count=1)
    estilos = tramo(mlp, "<style>", "</style>").replace("</style>", CSS_PROPIO + "</style>")
    barra = reemplazar(tramo(mlp, '<header class="barra">', "</header>"),
                       "<span>MLP sobre 64 personajes de Los Simpson</span>",
                       "<span>CNN sobre 64 personajes de Los Simpson</span>")
    total = len(diapositivas) + 1
    secciones = [portada(fecha)]
    for n, (seccion, titulo, disposicion, estilo, cuerpo) in enumerate(diapositivas, start=2):
        secciones.append(diapositiva(n, total, seccion, titulo, cuerpo, disposicion, estilo))
    cola = mlp[mlp.index('<div class="mando"'):]
    cola = reemplazar(cola, '<span class="contador" id="contador">1 / 19</span>',
                      f'<span class="contador" id="contador">1 / {total}</span>')
    final = ('<p class="final">Cifras de <span class="dato">resultados/newDataset/</span> y de las salidas de '
             '<span class="dato">notebooks/newDataset/LosSimpsons_GPU_4060M.ipynb</span>: las figuras son de la '
             'corrida de GPU a 128 × 128, y las cifras de la corrida a 64 × 64 que no están en su JSON se leen del '
             f'commit {REVISION_64}. Generada con <span class="dato">scripts/newDataset/generar_presentacion.py</span>.</p>')
    return (cabecera + estilos + "\n\n" + barra + '\n\n<main class="mazo" id="mazo">\n\n'
            + "\n".join(secciones) + "\n</main>\n\n" + final + "\n\n" + cola)


def main():
    global REVISION_64
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision-64", default=REVISION_64,
                        help="commit con las salidas de la corrida a 64 x 64 (por defecto %(default)s)")
    parser.add_argument("--salida", default=SALIDA, help="archivo HTML que se escribe (por defecto %(default)s)")
    argumentos = parser.parse_args()
    REVISION_64 = argumentos.revision_64

    jsons_cnn = {r: leer_json(f"{RESULTADOS_CNN}/rendimiento_gpu_{r}.json") for r in RESOLUCIONES}
    jsons_mlp = {r: leer_json(f"{RESULTADOS_MLP}/rendimiento_gpu_{r}.json") for r in RESOLUCIONES}
    nb128 = leer_notebook(NOTEBOOK_CNN)
    nb64 = leer_notebook(NOTEBOOK_CNN, REVISION_64)

    cnn = {"128x128": leer_cnn(nb128, jsons_cnn["128x128"]), "64x64": leer_cnn(nb64, jsons_cnn["64x64"]),
           "32x32": {"json": jsons_cnn["32x32"]}}
    mlp = leer_mlp(jsons_mlp)
    comprobar_lecturas(cnn, mlp)
    d128 = leer_detalle_128(nb128)
    comprobar_detalle(d128)
    n_train = imagenes_entrenamiento(nb128.texto("por_clase"), nb128.texto("confusiones"),
                                     nb64.texto("por_clase"), nb64.texto("confusiones"), *mlp["textos_n_train"])
    faltan = [p for p in PERSONAJES_F1 if p not in n_train]
    if faltan:
        raise SystemExit(f"no hay imágenes de entrenamiento de {faltan} en las salidas de los notebooks")

    prediccion = NOMBRES.get(d128["prediccion"][0], d128["prediccion"][0])
    cnn["128x128"]["figuras"] = {
        "recorrido": figura(nb128.imagenes("recorrido")[0], "jpeg",
                            f"Mapas de activación de los 8 filtros más activos de cada uno de los cuatro bloques para una "
                            f"imagen de test de {prediccion}: el primer bloque marca bordes y zonas de color, y en el "
                            f"cuarto los filtros se encienden sobre la cabeza del personaje."),
        "recorrido_varias": figura(nb128.imagenes("recorrido_varias")[0], "jpeg",
                                   "Mapas de activación de Homero, Marge, Bart y Lisa: la imagen de entrada y los dos "
                                   "filtros más activos de cada uno de los cuatro bloques, con la misma escala por columna. "
                                   "Los primeros bloques marcan zonas de color y contornos; en el cuarto quedan pocas "
                                   "manchas, distintas para cada personaje."),
        "gradcam_clases": figura(nb128.imagenes("gradcam_clases")[0], "jpeg",
                                 "Grad-CAM de la clase de cada uno de los 64 personajes sobre su imagen original de test "
                                 "acertada con mayor probabilidad: la zona cálida cae sobre la cabeza del personaje en "
                                 "casi todos.", "lamina alta"),
        "curvas": figura(nb128.imagenes("curvas")[0], "png",
                         "Curvas de pérdida y accuracy por época del modelo base a 128 × 128: la accuracy de entrenamiento "
                         "sube hasta 0,96 y la de validación se estabiliza cerca de 0,80; la mejor época es la 53."),
        "gradcam": figura(nb128.imagenes("gradcam")[0], "jpeg",
                          "Grad-CAM de tres aciertos (Martin Prince, Bart y Burns) y de los tres errores con mayor "
                          "confianza: dos imágenes de la carpeta de Bart en las que aparece Homero y una de Skinner junto "
                          "a Chalmers. La zona cálida cae sobre las cabezas."),
        "aciertos": figura(nb128.imagenes("aciertos")[0], "jpeg",
                           "Diez aciertos de test con confianza 1,00: Martin Prince, Bart de espaldas, Burns, Cletus, "
                           "Frink, Kirk Van Houten, Quimby, Homero en el sillón, Edna Krabappel y Otto."),
        "errores": figura(nb128.imagenes("errores")[0], "jpeg",
                          "Los diez errores de test con mayor confianza, entre 0,93 y 0,99, con el personaje real y el "
                          "predicho bajo cada imagen; en ocho de ellos el personaje predicho aparece en la imagen."),
        "escenas": figura(nb128.imagenes("escenas")[0], "jpeg",
                          "Diez aciertos por escena: el modelo nombró a otro personaje presente, como Burns junto a "
                          "Smithers, Selma junto a Patty o Lenny junto a Carl."),
    }
    fecha = "-".join(reversed(jsons_cnn["128x128"]["inicio_corrida"][:10].split("-")))
    documento = ensamblar(construir_diapositivas(cnn, mlp, d128, n_train), fecha)
    destino = RAIZ / argumentos.salida
    destino.write_bytes(documento.encode("utf-8"))
    print(f"Escrita {destino.relative_to(RAIZ)}: {documento.count('<section class=\"diapo')} diapositivas, "
          f"{len(documento.encode('utf-8')) / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
