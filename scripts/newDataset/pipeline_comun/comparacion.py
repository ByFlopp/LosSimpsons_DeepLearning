"""Comparación entre el MLP y la CNN, y figuras del informe.

La comparación lee los dos resúmenes JSON que deja cada notebook (`resultados/newDataset/GPU/rendimiento_gpu_*.json`),
así que se puede volver a ejecutar sola, sin entrenar, cuando cambia la corrida de cualquiera de las dos redes. Las
figuras del informe se guardan desde el propio notebook, en el momento en que se dibujan.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (etiqueta, clave del JSON, "mayor" = más es mejor, "menor" = menos es mejor, None = no se compara)
FILAS_COMPARACION = [
    ("Configuración final", "configuracion_final", None),
    ("Parámetros del modelo final", "parametros_modelo_final", None),
    ("Accuracy en test", "accuracy_test", "mayor"),
    ("Precision macro en test", "precision_macro_test", "mayor"),
    ("Recall macro en test", "recall_macro_test", "mayor"),
    ("F1 macro en test", "f1_macro_test", "mayor"),
    ("ROC-AUC macro en test", "roc_auc_macro_test", "mayor"),
    ("Top-3 accuracy en test", "top3_test", "mayor"),
    ("Accuracy por escena en test (referencia)", "accuracy_por_escena_test", "mayor"),
    ("Accuracy en originales", "accuracy_originales_test", "mayor"),
    ("Accuracy en parecidas", "accuracy_parecidas_test", "mayor"),
    ("Accuracy en alteradas", "accuracy_alteradas_test", "mayor"),
    ("Error de calibración (ECE)", "ece_test", "menor"),
    ("Errores con confianza > 0,80", "errores_confiados_test", "menor"),
    ("Brecha train - val del modelo final", "brecha_train_val_final", "menor"),
    ("Segundos: entrenamiento total", "segundos_entrenamiento_total", "menor"),
    ("Segundos por época (benchmark)", "benchmark_segundos_por_epoca", "menor"),
    ("Inferencia: imágenes por segundo", "inferencia_imagenes_por_segundo", "mayor"),
    ("Duración de la corrida", "duracion_corrida", None),
]


def leer_resumen(ruta):
    """El resumen JSON de una corrida, o None si todavía no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def comparar_redes(json_mlp, json_cnn, filas=FILAS_COMPARACION):
    """Tabla comparativa cifra a cifra entre el MLP y la CNN, leída de sus dos resúmenes JSON.

    La columna «Diferencia» es CNN − MLP en las métricas en [0, 1] y una razón en los tiempos y velocidades. Una
    cifra que un resumen no guarda queda como «no medida»."""
    registros = []
    for etiqueta, clave, direccion in filas:
        valor_mlp, valor_cnn = json_mlp.get(clave), json_cnn.get(clave)
        diferencia = "-"
        comparables = isinstance(valor_mlp, (int, float)) and isinstance(valor_cnn, (int, float))
        if comparables and direccion is not None:
            if valor_mlp <= 1 and valor_cnn <= 1 and clave not in ("benchmark_segundos_por_epoca",):
                diferencia = f"{valor_cnn - valor_mlp:+.4f}"
            elif valor_mlp > 0:
                diferencia = f"CNN ×{valor_cnn / valor_mlp:.2f}"
        registros.append(
            {
                "Concepto": etiqueta,
                "MLP": valor_mlp if valor_mlp is not None else "no medida",
                "CNN": valor_cnn if valor_cnn is not None else "no medida",
                "Diferencia (CNN − MLP)": diferencia,
                "Mejor": _mejor(valor_mlp, valor_cnn, direccion) if comparables else "-",
            }
        )
    return pd.DataFrame(registros)


def _mejor(valor_mlp, valor_cnn, direccion):
    """Qué red gana en una fila de la comparación, según si más es mejor o menos es mejor."""
    if direccion is None or valor_mlp == valor_cnn:
        return "-"
    gana_cnn = valor_cnn > valor_mlp if direccion == "mayor" else valor_cnn < valor_mlp
    return "CNN" if gana_cnn else "MLP"


def indicadores_generalizacion(resumen):
    """Accuracy por tipo de imagen, caída en alteradas, ventaja de las parecidas, brecha entre entrenamiento y
    validación y errores confiados por cada 1.000 imágenes de test, leídos del resumen JSON de una red."""
    originales = resumen["accuracy_originales_test"]
    return {
        "Accuracy en originales": originales,
        "Accuracy en alteradas": resumen["accuracy_alteradas_test"],
        "Caída en alteradas frente a originales (puntos)": (originales - resumen["accuracy_alteradas_test"]) * 100,
        "Accuracy en parecidas": resumen["accuracy_parecidas_test"],
        "Ventaja de las parecidas sobre las originales (puntos)": (resumen["accuracy_parecidas_test"] - originales) * 100,
        "Brecha train - val del modelo base": resumen.get("brecha_train_val_base", np.nan),
        "Brecha train - val del modelo final": resumen.get("brecha_train_val_final", np.nan),
        "Accuracy en test": resumen["accuracy_test"],
        "Errores con confianza > 0,80 por cada 1.000 imágenes de test": (
            resumen["errores_confiados_test"] / resumen["imagenes_test"] * 1000
        ),
    }


def tabla_generalizacion(json_mlp, json_cnn):
    """Indicadores de generalización de las dos redes, lado a lado, con la diferencia CNN − MLP."""
    tabla = pd.DataFrame(
        {"MLP": indicadores_generalizacion(json_mlp), "CNN": indicadores_generalizacion(json_cnn)}
    ).round(4)
    tabla["Diferencia CNN − MLP"] = (tabla["CNN"] - tabla["MLP"]).round(4)
    return tabla


def graficar_barras_redes(json_mlp, json_cnn, metricas, titulo, etiqueta_eje="Valor en test"):
    """Barras agrupadas MLP frente a CNN para una lista de `(etiqueta, clave)` del resumen JSON."""
    metricas = [(etiqueta, clave) for etiqueta, clave in metricas
                if isinstance(json_mlp.get(clave), (int, float)) and isinstance(json_cnn.get(clave), (int, float))]
    posiciones = np.arange(len(metricas))
    figura, eje = plt.subplots(figsize=(max(8, 1.6 * len(metricas)), 4.2))
    barras_mlp = eje.bar(posiciones - 0.2, [json_mlp[clave] for _, clave in metricas], width=0.4, label="MLP",
                         color="#8fb8de")
    barras_cnn = eje.bar(posiciones + 0.2, [json_cnn[clave] for _, clave in metricas], width=0.4, label="CNN",
                         color="#d95f02")
    for barras in (barras_mlp, barras_cnn):
        eje.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
    eje.set_xticks(posiciones)
    eje.set_xticklabels([etiqueta for etiqueta, _ in metricas])
    eje.set_ylim(0, 1.08)
    eje.set_ylabel(etiqueta_eje)
    eje.set_title(titulo, fontsize=11)
    eje.legend()
    eje.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


# --- Figuras del informe -----------------------------------------------------------------------------------


def guardar_figura(nombre, carpeta, formato="png", dpi=110, figura=None):
    """Guarda la figura actual (o `figura`) en `carpeta/<nombre>.<formato>` para el informe. Se llama justo antes
    de `plt.show()`, que en Jupyter cierra la figura."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.{formato}"
    opciones = {"pil_kwargs": {"quality": 85}} if formato in ("jpg", "jpeg") else {}
    (figura or plt.gcf()).savefig(ruta, dpi=dpi, bbox_inches="tight", **opciones)
    return ruta


@contextmanager
def figura_para_informe(nombre, carpeta, formato="png", dpi=110):
    """Guarda en `carpeta/<nombre>.<formato>` cada figura que se muestre dentro del bloque `with`.

    Sustituye por un momento a `plt.show` para guardar la figura justo antes de mostrarla, así que también captura
    las figuras que dibujan las funciones del pipeline. Si el bloque muestra varias figuras, la segunda y las
    siguientes llevan un sufijo `_2`, `_3`... Devuelve la lista de rutas guardadas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    mostrar_original = plt.show
    guardadas = []

    def mostrar_y_guardar(*args, **kwargs):
        """Guarda la figura actual y luego la muestra como siempre."""
        sufijo = "" if not guardadas else f"_{len(guardadas) + 1}"
        ruta = carpeta / f"{nombre}{sufijo}.{formato}"
        opciones = {"pil_kwargs": {"quality": 85}} if formato in ("jpg", "jpeg") else {}
        plt.gcf().savefig(ruta, dpi=dpi, bbox_inches="tight", **opciones)
        guardadas.append(ruta)
        return mostrar_original(*args, **kwargs)

    plt.show = mostrar_y_guardar
    try:
        yield guardadas
    finally:
        plt.show = mostrar_original
