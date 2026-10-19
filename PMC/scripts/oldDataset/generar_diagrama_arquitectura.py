"""Genera PMC/images/oldDataset/arquitectura_mlp.png: representación visual del MLP (E2).

Uso:
    .venv-gpu/Scripts/python.exe PMC/scripts/oldDataset/generar_diagrama_arquitectura.py

Dibuja el recorrido completo de una imagen: redimensionado a 64 x 64, aplanado a
12.288 valores, las tres capas ocultas (1024 -> 512 -> 256) con sus neuronas y
conexiones, y la capa softmax de 25 salidas.
"""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

RAIZ = Path(__file__).resolve().parent.parent.parent
IMAGENES = RAIZ / "images" / "oldDataset"
SALIDA = IMAGENES / "arquitectura_mlp.png"

FONDO = "#FFFFFF"
TINTA = "#1F2933"
GRIS = "#6B7684"
AMARILLO = "#FFD90F"
BORDE_AMARILLO = "#C9A400"
AZUL = "#4A90D9"
BORDE_AZUL = "#2C5F91"
NARANJA = "#F2841E"
BORDE_NARANJA = "#B35C0A"
CONEXION = "#B7C2CC"

ANCHO, ALTO = 18.0, 8.6
CENTRO_Y = 4.55
RADIO = 0.17
PASO = 0.52


def posiciones(n_visibles: int, centro: float = CENTRO_Y) -> list[float]:
    """Posiciones verticales de las neuronas dibujadas, con hueco central."""
    idx = np.arange(n_visibles) - (n_visibles - 1) / 2
    ys = centro - idx * PASO
    mitad = n_visibles // 2
    ys[:mitad] += PASO * 0.45
    ys[mitad:] -= PASO * 0.45
    return list(ys)


def capa(ax, x, n_visibles, color, borde, centro=CENTRO_Y):
    ys = posiciones(n_visibles, centro)
    for y in ys:
        ax.add_patch(
            Circle(
                (x, y),
                RADIO,
                facecolor=color,
                edgecolor=borde,
                linewidth=1.1,
                zorder=3,
            )
        )
    ax.text(
        x,
        centro,
        "⋮",
        ha="center",
        va="center",
        fontsize=15,
        color=GRIS,
        zorder=4,
    )
    return ys


def conectar(ax, x1, ys1, x2, ys2, alpha=0.16):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot(
                [x1 + RADIO, x2 - RADIO],
                [y1, y2],
                color=CONEXION,
                linewidth=0.5,
                alpha=alpha,
                zorder=1,
            )


def flecha(ax, x1, x2, y=CENTRO_Y, texto=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=GRIS,
            zorder=2,
        )
    )
    if texto:
        ax.text(
            (x1 + x2) / 2,
            y + 0.22,
            texto,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=GRIS,
        )


def titulo_capa(ax, x, titulo, subtitulo=None, y=7.55):
    ax.text(
        x, y, titulo, ha="center", va="center", fontsize=11, color=TINTA, weight="bold"
    )
    if subtitulo:
        ax.text(x, y - 0.42, subtitulo, ha="center", va="center", fontsize=9, color=GRIS)


def pie_capa(ax, x, texto, y=1.5):
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.02, y - 0.3),
            2.04,
            0.6,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#F1F4F7",
            edgecolor="#D5DCE3",
            linewidth=0.9,
            zorder=2,
        )
    )
    ax.text(x, y, texto, ha="center", va="center", fontsize=8.5, color=GRIS, zorder=3)


def miniatura(ax, ruta, x, ancho, pixelar=None, etiqueta="", subetiqueta=""):
    img = Image.open(ruta).convert("RGB")
    if pixelar:
        img = img.resize((pixelar, pixelar), Image.Resampling.BILINEAR)
    alto = ancho
    extent = (x - ancho / 2, x + ancho / 2, CENTRO_Y - alto / 2, CENTRO_Y + alto / 2)
    ax.imshow(
        np.asarray(img),
        extent=extent,
        interpolation="nearest" if pixelar else "antialiased",
        zorder=3,
        aspect="auto",
    )
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            ancho,
            alto,
            facecolor="none",
            edgecolor=TINTA,
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.text(
        x,
        extent[3] + 0.28,
        etiqueta,
        ha="center",
        va="bottom",
        fontsize=10,
        color=TINTA,
        weight="bold",
    )
    if subetiqueta:
        ax.text(
            x,
            extent[2] - 0.3,
            subetiqueta,
            ha="center",
            va="top",
            fontsize=8.5,
            color=GRIS,
        )
    return extent


def vector_aplanado(ax, x, ruta):
    """Columna de píxeles que representa el vector de 12.288 valores."""
    img = Image.open(ruta).convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
    muestra = np.asarray(img).reshape(-1, 3)[20:34] / 255.0
    ancho, alto_celda = 0.34, 0.32
    y0 = CENTRO_Y + len(muestra) * alto_celda / 2
    for i, color in enumerate(muestra):
        y = y0 - i * alto_celda
        ax.add_patch(
            Rectangle(
                (x - ancho / 2, y - alto_celda),
                ancho,
                alto_celda,
                facecolor=color,
                edgecolor="#FFFFFF",
                linewidth=0.6,
                zorder=3,
            )
        )
    ax.text(
        x,
        y0 + 0.24,
        "Vector aplanado",
        ha="center",
        va="bottom",
        fontsize=10,
        color=TINTA,
        weight="bold",
    )
    ax.text(
        x,
        y0 - len(muestra) * alto_celda - 0.26,
        "12.288 valores\nnormalizados 0-1",
        ha="center",
        va="top",
        fontsize=8.5,
        color=GRIS,
    )
    return [CENTRO_Y + 0.9, CENTRO_Y, CENTRO_Y - 0.9]


def salidas(ax, x, ys):
    # Salida real de models/oldDataset/mlp_final.keras sobre images/oldDataset/ejemplo_homer.jpg
    # (test/homer_simpson/pic_0077.jpg), top-5 de las 25 probabilidades.
    clases = [
        ("homer_simpson", 0.8755),
        ("waylon_smithers", 0.0234),
        ("moe_szyslak", 0.0210),
        ("krusty_the_clown", 0.0132),
        ("abraham_grampa_simpson", 0.0132),
    ]
    x_barra = x + 0.75
    largo_max = 1.7
    for (nombre, prob), y in zip(clases, ys):
        destacado = prob == max(p for _, p in clases)
        ax.add_patch(
            Rectangle(
                (x_barra, y - 0.13),
                largo_max * prob,
                0.26,
                facecolor=AMARILLO if destacado else "#D7DEE5",
                edgecolor=BORDE_AMARILLO if destacado else "#BFC8D1",
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.text(
            x_barra + largo_max * prob + 0.12,
            y,
            f"{nombre}  {prob:.1%}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=TINTA if destacado else GRIS,
            weight="bold" if destacado else "normal",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(ANCHO, ALTO), dpi=170)
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.set_xlim(0, ANCHO)
    ax.set_ylim(0, ALTO)
    ax.axis("off")

    ax.text(
        0.35,
        8.15,
        "Perceptrón multicapa (E2): de la imagen al personaje",
        ha="left",
        va="center",
        fontsize=15,
        color=TINTA,
        weight="bold",
    )
    ax.text(
        0.35,
        7.72,
        "1024 → 512 → 256 neuronas ocultas · ~13,25 millones de parámetros · 25 clases",
        ha="left",
        va="center",
        fontsize=10,
        color=GRIS,
    )

    retrato = IMAGENES / "ejemplo_homer.jpg"

    ext1 = miniatura(
        ax,
        retrato,
        x=1.25,
        ancho=1.5,
        etiqueta="Imagen RGB",
        subetiqueta="tamaño variable",
    )
    flecha(ax, ext1[1] + 0.12, 3.0, texto="resize")

    ext2 = miniatura(
        ax,
        retrato,
        x=3.85,
        ancho=1.5,
        pixelar=64,
        etiqueta="64 × 64 × 3",
        subetiqueta="normalizada",
    )
    flecha(ax, ext2[1] + 0.12, 5.55, texto="flatten")

    x_vec = 6.1
    ys_vec = vector_aplanado(ax, x_vec, retrato)

    x1, x2, x3, x4 = 8.3, 10.6, 12.9, 15.2

    ys1 = capa(ax, x1, 8, AMARILLO, BORDE_AMARILLO)
    ys2 = capa(ax, x2, 8, NARANJA, BORDE_NARANJA)
    ys3 = capa(ax, x3, 8, AZUL, BORDE_AZUL)
    ys4 = capa(ax, x4, 6, "#7FB77E", "#3F7A3E")

    conectar(ax, x_vec + 0.17, ys_vec, x1, ys1, alpha=0.35)
    conectar(ax, x1, ys1, x2, ys2)
    conectar(ax, x2, ys2, x3, ys3)
    conectar(ax, x3, ys3, x4, ys4, alpha=0.2)

    titulo_capa(ax, x1, "Capa oculta 1", "Dense 1024 · ReLU")
    titulo_capa(ax, x2, "Capa oculta 2", "Dense 512 · ReLU")
    titulo_capa(ax, x3, "Capa oculta 3", "Dense 256 · ReLU")
    titulo_capa(ax, x4, "Salida", "Dense 25 · Softmax")

    for x in (x1, x2, x3):
        pie_capa(ax, x, "BatchNorm  ·  Dropout 0,3")
    pie_capa(ax, x4, "argmax → personaje")

    salidas(ax, x4, ys4)

    ax.text(
        0.35,
        0.36,
        "Cada neurona de una capa recibe la salida de todas las anteriores: por eso el MLP "
        "aplana la imagen y pierde su estructura espacial.\n"
        "El dibujo muestra solo unas pocas neuronas por capa; las reales son 1024, 512 y 256.\n"
        "Las cinco probabilidades son la salida real del modelo entrenado para esta imagen de prueba.",
        ha="left",
        va="center",
        fontsize=9,
        color=GRIS,
    )

    fig.savefig(SALIDA, facecolor=FONDO, bbox_inches="tight", pad_inches=0.25)
    print(f"Diagrama guardado en {SALIDA}")


if __name__ == "__main__":
    main()
