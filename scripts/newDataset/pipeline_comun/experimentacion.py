"""Experimentación: comparación de configuraciones, búsqueda de hiperparámetros e impacto de la regularización.

Etapa «Optimización» del pipeline. Las configuraciones se comparan con el mismo presupuesto (épocas máximas y
paciencia) y por el F1 macro de validación. El test no interviene en ninguna decisión de este módulo.
"""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .entrenamiento import PARADA_TOPE, _registro
from .evaluacion import evaluar_metricas

# --- Resultado de una configuración ------------------------------------------------------------------------


def resultado_de_entrenamiento(historial, modelo, X_valida, y_valida_idx, numero_clases, segundos=None):
    """Fila de la tabla de experimentos con lo que dejó un entrenamiento: épocas, mejor época, motivo de la parada,
    parámetros, segundos y las métricas de validación del modelo de su mejor época (accuracy, precision y recall
    macro, F1 macro y ROC-AUC macro)."""
    registro = _registro(historial)
    indice = int(np.argmin(registro["val_loss"]))
    metricas = evaluar_metricas(y_valida_idx, modelo.predict(X_valida, batch_size=256, verbose=0), numero_clases)
    return {
        "epocas": len(registro["loss"]),
        "mejor_epoca": indice + 1,
        "parada": getattr(historial, "motivo_parada", None) or "-",
        "parametros": int(modelo.count_params()),
        "val_loss": round(registro["val_loss"][indice], 4),
        "val_accuracy": round(metricas["accuracy"], 4),
        "val_precision_macro": round(metricas["precision_macro"], 4),
        "val_recall_macro": round(metricas["recall_macro"], 4),
        "val_f1_macro": round(metricas["f1_macro"], 4),
        "val_roc_auc": round(metricas["roc_auc_macro"], 4),
        "segundos": round(segundos, 1) if segundos is not None else np.nan,
    }


# --- Comparaciones de la experimentación de un factor ------------------------------------------------------


def graficar_comparacion(tabla, historiales, baseline_mayoritaria):
    """Accuracy y F1 macro de validación de cada configuración, y su pérdida de validación por época."""
    orden = tabla.sort_values("val_f1_macro", ascending=True)
    posiciones = np.arange(len(orden))
    figura, ejes = plt.subplots(1, 2, figsize=(14, max(6, 0.32 * len(orden))))

    ejes[0].barh(posiciones - 0.2, orden["val_accuracy"], height=0.4, label="accuracy val")
    ejes[0].barh(posiciones + 0.2, orden["val_f1_macro"], height=0.4, label="F1 macro val")
    ejes[0].set_yticks(posiciones)
    ejes[0].set_yticklabels(orden["experimento"], fontsize=8)
    ejes[0].axvline(baseline_mayoritaria / 100, color="red", linestyle="--", linewidth=1,
                    label="baseline clase mayoritaria")
    ejes[0].set_xlabel("Métrica de validación")
    ejes[0].set_title("Desempeño por configuración", fontsize=10)
    ejes[0].legend(fontsize=8)
    ejes[0].grid(alpha=0.3, axis="x")

    for nombre in orden["experimento"][::-1]:
        if nombre in historiales:
            registro = _registro(historiales[nombre])
            ejes[1].plot(range(1, len(registro["val_loss"]) + 1), registro["val_loss"], label=nombre, linewidth=1.5)
    ejes[1].set_xlabel("Epoca")
    ejes[1].set_ylabel("Pérdida de validación")
    ejes[1].set_title("Convergencia comparada", fontsize=10)
    ejes[1].grid(alpha=0.3)
    ejes[1].legend(fontsize=6)
    plt.tight_layout()
    plt.show()


def seleccionar(tabla, patron, columnas):
    """Filas de la tabla de experimentos cuyo nombre empieza por el patrón (una expresión regular, por ejemplo
    `r"Base|E3"`), con las columnas indicadas que existan."""
    filas = tabla[tabla["experimento"].str.match(patron)]
    return filas[[columna for columna in columnas if columna in filas.columns]].reset_index(drop=True)


def comparar_convergencia(tabla, historiales, patron, columnas, titulo):
    """Tabla de las configuraciones del patrón y su pérdida y accuracy de validación por época."""
    comparacion = seleccionar(tabla, patron, columnas)
    display(comparacion)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nombre in comparacion["experimento"]:
        registro = _registro(historiales[nombre])
        epocas = range(1, len(registro["loss"]) + 1)
        ejes[0].plot(epocas, registro["val_loss"], linewidth=2, label=nombre)
        ejes[1].plot(epocas, registro["val_accuracy"], linewidth=2, label=nombre)
    ejes[0].set_ylabel("Pérdida de validación")
    ejes[0].set_title(titulo, fontsize=10)
    ejes[1].set_ylabel("Accuracy de validación")
    ejes[1].set_title("Accuracy de validación por época", fontsize=10)
    for eje in ejes:
        eje.set_xlabel("Epoca")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return comparacion


def comparar_brecha(tabla, historiales, patron, columnas, titulo):
    """Tabla de las configuraciones del patrón, sus pérdidas de entrenamiento (punteada) y validación (sólida), y
    la brecha de generalización (accuracy de train menos accuracy de validación) por época."""
    comparacion = seleccionar(tabla, patron, columnas)
    display(comparacion)
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nombre in comparacion["experimento"]:
        registro = _registro(historiales[nombre])
        epocas = range(1, len(registro["loss"]) + 1)
        ejes[0].plot(epocas, registro["loss"], linewidth=1.6, linestyle="--")
        ejes[0].plot(epocas, registro["val_loss"], linewidth=2, label=nombre, color=ejes[0].lines[-1].get_color())
        ejes[1].plot(epocas, np.array(registro["accuracy"]) - np.array(registro["val_accuracy"]), linewidth=2,
                     label=nombre)
    ejes[0].set_ylabel("Perdida")
    ejes[0].set_title(titulo, fontsize=10)
    ejes[1].axhline(0, color="gray", linewidth=0.8)
    ejes[1].set_ylabel("accuracy train - accuracy val")
    ejes[1].set_title("Brecha de generalización por época", fontsize=10)
    for eje in ejes:
        eje.set_xlabel("Epoca")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return comparacion


def comparar_contra_base(tabla, patron, columnas, titulo):
    """Tabla de las configuraciones del patrón con su diferencia de accuracy y F1 macro de validación frente al
    modelo base, y un gráfico de esas diferencias."""
    comparacion = seleccionar(tabla, patron, columnas)
    fila_base = comparacion["experimento"].str.startswith("Base")
    for metrica, columna in (("val_accuracy", "delta_accuracy_vs_base"), ("val_f1_macro", "delta_f1_vs_base")):
        comparacion[columna] = (
            comparacion[metrica] - float(comparacion.loc[fila_base, metrica].iloc[0])
        ).round(4)
    display(comparacion)

    otras = comparacion[~fila_base].reset_index(drop=True)
    if otras.empty:
        return comparacion
    figura, eje = plt.subplots(figsize=(9, max(3, 0.45 * len(otras) + 1.5)))
    posiciones = np.arange(len(otras))
    eje.barh(posiciones - 0.2, otras["delta_accuracy_vs_base"], height=0.4, label="accuracy val")
    eje.barh(posiciones + 0.2, otras["delta_f1_vs_base"], height=0.4, label="F1 macro val")
    eje.set_yticks(posiciones)
    eje.set_yticklabels(otras["experimento"], fontsize=8)
    eje.invert_yaxis()
    eje.axvline(0, color="gray", linewidth=1)
    eje.set_xlabel("Diferencia frente al modelo base (validación)")
    eje.set_title(titulo, fontsize=10)
    eje.grid(alpha=0.3, axis="x")
    eje.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    return comparacion


def combinar_mejoras(experimentos, resultado_por_experimento, f1_macro_base):
    """C1: para cada factor, el valor del experimento que más superó a la base en F1 macro de validación.

    Devuelve `(factores, componentes)`: los factores de la combinación y los experimentos de los que salen. Si
    menos de dos factores superaron a la base no hay nada que combinar, y `factores` es None."""
    mejor_por_factor = {}
    for configuracion in experimentos:
        resultado = resultado_por_experimento.get(configuracion["nombre"])
        if resultado is None or resultado["val_f1_macro"] <= f1_macro_base:
            continue
        for factor, valor in configuracion.items():
            if factor == "nombre":
                continue
            if factor not in mejor_por_factor or resultado["val_f1_macro"] > mejor_por_factor[factor][1]:
                mejor_por_factor[factor] = (valor, resultado["val_f1_macro"], configuracion["nombre"])
    componentes = sorted({origen for _, _, origen in mejor_por_factor.values()})
    if len(mejor_por_factor) < 2:
        return None, componentes
    return {factor: valor for factor, (valor, _, _) in mejor_por_factor.items()}, componentes


# --- Búsqueda de hiperparámetros ---------------------------------------------------------------------------


def _formatear(valor):
    """Valor de un hiperparámetro como texto corto para el nombre de la configuración."""
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, float) and valor and (abs(valor) < 1e-2):
        return f"{valor:.0e}"
    return str(valor)


def generar_configuraciones(espacio, estrategia="grid", n_configuraciones=None, semilla=42):
    """Combinaciones de hiperparámetros a probar.

    - `grid`: todas las combinaciones del espacio (producto cartesiano).
    - `aleatoria`: `n_configuraciones` combinaciones distintas, sorteadas sin reemplazo con la semilla.

    `espacio` es `{factor: [valores]}`; cada combinación es un diccionario `{factor: valor}`."""
    factores = list(espacio)
    todas = [dict(zip(factores, valores)) for valores in itertools.product(*espacio.values())]
    if estrategia == "grid":
        return todas
    if estrategia != "aleatoria":
        raise ValueError(f"Estrategia desconocida: {estrategia} (usa 'grid' o 'aleatoria')")
    generador = np.random.default_rng(semilla)
    n = min(n_configuraciones or len(todas), len(todas))
    elegidas = generador.choice(len(todas), size=n, replace=False)
    return [todas[indice] for indice in sorted(elegidas)]


def optimizar_modelo(entrenar_configuracion, espacio, estrategia="grid", n_configuraciones=None, semilla=42,
                     prefijo="B", etiquetas=None, fijos=None):
    """Búsqueda de hiperparámetros sobre el pipeline: Grid Search o Random Search.

    1. Define el espacio de búsqueda con `generar_configuraciones` (todas las combinaciones, o una muestra
       aleatoria sin reemplazo).
    2. Entrena cada configuración con `entrenar_configuracion(nombre, **factores, **fijos)`, la misma función
       que usa la experimentación de un factor, así que comparten modelo, datos, pesos y presupuesto.
    3. Registra sus métricas de validación (accuracy, precision, recall, F1 macro y ROC-AUC), la mejor época,
       el motivo de la parada y los segundos.
    4. Devuelve `(tabla, modelos, historiales)`: la tabla comparativa ordenada por F1 macro de validación, con la
       columna `factores` para volver a entrenar cualquier fila, y los modelos e historiales por nombre.

    Una configuración que no cabe en la GPU queda fuera de la tabla con un aviso, sin detener la búsqueda."""
    etiquetas = etiquetas or {}
    fijos = fijos or {}
    configuraciones = generar_configuraciones(espacio, estrategia, n_configuraciones, semilla)
    print(f"{'Grid Search' if estrategia == 'grid' else 'Random Search'}: {len(configuraciones)} configuraciones"
          f" de un espacio de {int(np.prod([len(valores) for valores in espacio.values()]))}")

    resultados, modelos, historiales = [], {}, {}
    inicio = time.perf_counter()
    for numero, factores in enumerate(configuraciones, start=1):
        nombre = f"{prefijo}{numero:02d} - " + ", ".join(
            f"{etiquetas.get(factor, factor)} {_formatear(valor)}" for factor, valor in factores.items()
        )
        try:
            modelo, historial, resultado = entrenar_configuracion(nombre, **factores, **fijos)
        except Exception as error:  # noqa: BLE001 - sin memoria en la GPU u otro fallo de una sola configuración
            if "OutOfMemory" not in type(error).__name__:
                raise
            print(f"   {nombre}: sin memoria suficiente en la GPU, queda fuera de la búsqueda")
            continue
        modelos[nombre] = modelo
        historiales[nombre] = historial.history
        resultados.append({**resultado, "factores": dict(factores)})
        print(f"   {nombre}: F1 macro val {resultado['val_f1_macro']:.4f}   ROC-AUC {resultado['val_roc_auc']:.4f}"
              f"   épocas {resultado['epocas']} ({resultado['parada']})   {resultado['segundos']:.0f} s")

    tabla = pd.DataFrame(resultados).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    print(f"Búsqueda terminada en {time.perf_counter() - inicio:.0f} s")
    return tabla, modelos, historiales


def tabla_comparativa(tabla, referencia=None, n=None):
    """Tabla comparativa como la del material 3.5.2: accuracy, precision, recall, F1 y ROC-AUC de validación de
    cada configuración, con observaciones generadas desde las cifras (la ganadora, las que pararon por el tope y la
    diferencia frente a la referencia)."""
    columnas = ["experimento", "val_accuracy", "val_precision_macro", "val_recall_macro", "val_f1_macro",
                "val_roc_auc", "mejor_epoca", "epocas", "parada"]
    comparativa = tabla[[columna for columna in columnas if columna in tabla.columns]].copy()
    if n is not None:
        comparativa = comparativa.head(n)
    observaciones = []
    f1_referencia = None
    if referencia is not None:
        f1_referencia = float(tabla.loc[tabla["experimento"] == referencia, "val_f1_macro"].iloc[0])
    mejor = comparativa["val_f1_macro"].max()
    for fila in comparativa.itertuples():
        notas = []
        if fila.val_f1_macro == mejor:
            notas.append("★ mejor F1 macro")
        if getattr(fila, "parada", None) == PARADA_TOPE:
            notas.append("paró por el tope de épocas")
        if f1_referencia is not None and fila.experimento != referencia:
            notas.append(f"{(fila.val_f1_macro - f1_referencia) * 100:+.1f} pts de F1 frente a {referencia.split(' - ')[0]}")
        observaciones.append("; ".join(notas))
    comparativa["observaciones"] = observaciones
    return comparativa


# --- Impacto de cada técnica de regularización -----------------------------------------------------------


def brecha_en_mejor_epoca(historial):
    """Accuracy de entrenamiento menos accuracy de validación en la mejor época (la de menor val_loss)."""
    registro = _registro(historial)
    mejor = int(np.argmin(registro["val_loss"]))
    return registro["accuracy"][mejor] - registro["val_accuracy"][mejor]


def tabla_impacto(tabla, historiales, nombre_base, tecnicas, umbral_ruido=0.005):
    """Impacto de cada técnica de regularización, medido con un experimento de un solo factor frente a la base.

    `tecnicas` es una lista de diccionarios con:
    - `tecnica`: el nombre de la técnica (por ejemplo, "Dropout");
    - `experimento`: el nombre del experimento que la quita o la agrega;
    - `accion`: "quita" si el experimento la quita de la base, o "agrega" si la agrega;
    - `en_final`: si la técnica queda en el modelo final.

    El efecto de la técnica es el F1 macro de validación con ella menos el F1 sin ella. Por debajo de
    `umbral_ruido` (medio punto) se lee como ruido: es una sola semilla. La lectura de cada fila se genera desde
    las cifras, para que no quede desfasada si cambia la corrida."""
    fila_base = tabla.loc[tabla["experimento"] == nombre_base].iloc[0]
    brecha_base = brecha_en_mejor_epoca(historiales[nombre_base])
    filas = []
    for tecnica in tecnicas:
        seleccion = tabla.loc[tabla["experimento"] == tecnica["experimento"]]
        if seleccion.empty:
            continue
        fila = seleccion.iloc[0]
        brecha = brecha_en_mejor_epoca(historiales[tecnica["experimento"]])
        quita = tecnica["accion"] == "quita"
        efecto = (fila_base["val_f1_macro"] - fila["val_f1_macro"]) if quita else (
            fila["val_f1_macro"] - fila_base["val_f1_macro"])
        efecto_brecha = (brecha_base - brecha) if quita else (brecha - brecha_base)
        if efecto > umbral_ruido:
            veredicto = "ayuda"
        elif efecto < -umbral_ruido:
            veredicto = "perjudica"
        else:
            veredicto = "sin efecto apreciable (dentro de ±0,5 pts)"
        verbo = "Quitarla" if quita else "Agregarla"
        cambio_f1 = fila["val_f1_macro"] - fila_base["val_f1_macro"]
        lectura = (
            f"{verbo} {'sube' if cambio_f1 > 0 else 'baja'} el F1 macro {abs(cambio_f1) * 100:.1f} pts y "
            f"{'abre' if brecha > brecha_base else 'cierra'} la brecha ({brecha_base:.3f} → {brecha:.3f}): "
            f"la técnica {veredicto}"
            + (" y reduce el sobreajuste." if efecto_brecha < 0 else ".")
        )
        filas.append(
            {
                "técnica": tecnica["tecnica"],
                "experimento": tecnica["experimento"],
                "en el modelo final": "sí" if tecnica["en_final"] else "no",
                "F1 macro val": round(float(fila["val_f1_macro"]), 4),
                "Δ F1 vs base": round(float(cambio_f1), 4),
                "efecto de la técnica": round(float(efecto), 4),
                "brecha train-val": round(float(brecha), 4),
                "mejor época": int(fila["mejor_epoca"]),
                "parada": fila.get("parada", "-"),
                "lectura": lectura,
            }
        )
    impacto = pd.DataFrame(filas)
    print(f"Base: F1 macro val {fila_base['val_f1_macro']:.4f}, brecha train-val {brecha_base:.3f},"
          f" mejor época {int(fila_base['mejor_epoca'])} ({fila_base.get('parada', '-')})")
    return impacto
