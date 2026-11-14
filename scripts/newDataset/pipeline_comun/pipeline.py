"""El pipeline de los notebooks: sus nueve etapas, dibujadas y en una tabla.

Las etapas son las de la pauta de la EP3: carga de datos, análisis exploratorio, preprocesamiento, construcción del
modelo, entrenamiento, regularización, evaluación, auditoría y optimización.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def tabla_pipeline(etapas):
    """Tabla de las etapas: `etapas` es una lista de diccionarios con `etapa`, `funciones`, `seccion` y `por_que`."""
    return pd.DataFrame(etapas)[["etapa", "funciones", "seccion", "por_que"]].rename(
        columns={"etapa": "Etapa", "funciones": "Funciones", "seccion": "Sección del notebook",
                 "por_que": "Por qué es una etapa aparte"}
    )


def dibujar_pipeline(etapas, titulo, por_fila=5):
    """Diagrama de flujo de las etapas: una caja por etapa, con sus funciones principales, unidas por flechas en
    el orden en que se ejecutan. La optimización vuelve al entrenamiento: cada configuración se entrena de nuevo."""
    filas = (len(etapas) + por_fila - 1) // por_fila
    figura, eje = plt.subplots(figsize=(3.1 * por_fila, 2.6 * filas + 0.6))
    eje.set_xlim(0, por_fila)
    eje.set_ylim(0, filas)
    eje.axis("off")
    # Azul: los datos; naranja: el modelo y su entrenamiento; verde: la evaluación; morado: la auditoría
    colores = ["#dbe9f6", "#dbe9f6", "#dbe9f6", "#fde2c8", "#fde2c8", "#fde2c8", "#fde2c8", "#d9f0d3", "#f3d9ef"]
    centros = []
    for numero, etapa in enumerate(etapas):
        fila, columna = divmod(numero, por_fila)
        # Las filas pares van de izquierda a derecha y las impares al revés, para que el flujo no salte
        if fila % 2:
            columna = por_fila - 1 - columna
        x, y = columna + 0.5, filas - fila - 0.5
        centros.append((x, y))
        caja = FancyBboxPatch((x - 0.43, y - 0.36), 0.86, 0.72, boxstyle="round,pad=0.02",
                              facecolor=colores[numero % len(colores)], edgecolor="#555555", linewidth=1.2)
        eje.add_patch(caja)
        eje.text(x, y + 0.2, f"{numero + 1}. {etapa['etapa']}", ha="center", va="center", fontsize=9.5,
                 fontweight="bold")
        eje.text(x, y - 0.08, etapa["funciones_cortas"], ha="center", va="center", fontsize=7.2,
                 family="monospace", linespacing=1.4)
    for (x0, y0), (x1, y1) in zip(centros, centros[1:]):
        if y0 == y1:
            inicio, fin = (x0 + 0.44 * (1 if x1 > x0 else -1), y0), (x1 - 0.44 * (1 if x1 > x0 else -1), y1)
        else:
            inicio, fin = (x0, y0 - 0.37), (x1, y1 + 0.37)
        eje.add_patch(FancyArrowPatch(inicio, fin, arrowstyle="-|>", mutation_scale=14, color="#333333"))
    # La optimización vuelve a entrenar cada configuración
    nombres = [etapa["etapa"] for etapa in etapas]
    if "Optimización" in nombres and "Entrenamiento" in nombres:
        (xo, yo), (xe, ye) = centros[nombres.index("Optimización")], centros[nombres.index("Entrenamiento")]
        eje.add_patch(FancyArrowPatch((xo - 0.2, yo + 0.37), (xe + 0.2, ye - 0.37), arrowstyle="-|>",
                                      mutation_scale=12, color="#8e44ad", linestyle="--",
                                      connectionstyle="arc3,rad=0.25"))
        eje.text((xo + xe) / 2 - 0.1, (yo + ye) / 2 + 0.08, "cada configuración se entrena de nuevo", fontsize=7,
                 color="#8e44ad", ha="right", va="center")
    eje.set_title(titulo, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
