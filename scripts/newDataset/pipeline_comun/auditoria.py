"""Auditoría del modelo: ROC-AUC, calibración, umbral de revisión, sesgos por subgrupo y SHAP.

Etapa «Auditoría» del pipeline (IL 3.2). Todo se calcula sin anotar nada a mano: los subgrupos salen del nombre del
archivo (Kaggle o fotograma de episodio), del tamaño de cada clase en entrenamiento y del nombre del personaje.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import f1_score, roc_curve

from .datos import FUENTE_EPISODIO, FUENTE_KAGGLE
from .evaluacion import evaluar_metricas

OBJETIVO_UMBRAL = 0.90
INTERVALOS_CALIBRACION = 15


# --- ROC ---------------------------------------------------------------------------------------------------


def curvas_roc(y_real, probabilidades, clases, f1_por_clase, n=5):
    """Curvas ROC uno contra el resto de los `n` personajes con mejor F1 y los `n` con peor, con su AUC.
    `f1_por_clase` va en el orden de `clases`. Devuelve la tabla de AUC de cada personaje."""
    y_real = np.asarray(y_real)
    auc_por_clase = []
    for indice, clase in enumerate(clases):
        positivos = y_real == indice
        if positivos.all() or not positivos.any():
            auc_por_clase.append(np.nan)
            continue
        falsos, verdaderos, _ = roc_curve(positivos, probabilidades[:, indice])
        auc_por_clase.append(float(np.trapezoid(verdaderos, falsos)))
    tabla = pd.DataFrame({"personaje": clases, "f1": np.round(f1_por_clase, 3), "roc_auc": np.round(auc_por_clase, 4)})

    orden = np.argsort(-np.asarray(f1_por_clase))
    grupos = (("Los 5 con mejor F1", orden[:n]), ("Los 5 con peor F1", orden[-n:][::-1]))
    figura, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
    for eje, (titulo, indices) in zip(ejes, grupos):
        for indice in indices:
            positivos = y_real == indice
            falsos, verdaderos, _ = roc_curve(positivos, probabilidades[:, indice])
            eje.plot(falsos, verdaderos, linewidth=2,
                     label=f"{clases[indice].replace('_', ' ')} (AUC {auc_por_clase[indice]:.3f}, F1 {f1_por_clase[indice]:.2f})")
        eje.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="azar (AUC 0,5)")
        eje.set_xlabel("Tasa de falsos positivos")
        eje.set_ylabel("Tasa de verdaderos positivos")
        eje.set_title(titulo, fontsize=10)
        eje.legend(fontsize=7, loc="lower right")
        eje.grid(alpha=0.3)
    figura.suptitle("Curvas ROC uno contra el resto (test)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return tabla


# --- Calibración y umbral de revisión ---------------------------------------------------------------------


def calibracion(y_real, probabilidades, intervalos=INTERVALOS_CALIBRACION, graficar=True):
    """Diagrama de confiabilidad: la confianza del modelo (su probabilidad más alta) frente a la proporción de
    aciertos, por intervalos de confianza. Un modelo calibrado acierta el 80 % de las veces que dice 0,80.

    El error de calibración esperado (ECE) es la distancia media entre las dos, pesada por las imágenes de cada
    intervalo. Devuelve `(tabla, ece)`."""
    y_real = np.asarray(y_real)
    confianza = probabilidades.max(axis=1)
    acierto = probabilidades.argmax(axis=1) == y_real
    bordes = np.linspace(0, 1, intervalos + 1)
    numero = np.clip(np.digitize(confianza, bordes[1:-1]), 0, intervalos - 1)
    filas = []
    for intervalo in range(intervalos):
        mascara = numero == intervalo
        if not mascara.any():
            continue
        filas.append(
            {
                "desde": round(bordes[intervalo], 3),
                "hasta": round(bordes[intervalo + 1], 3),
                "imagenes": int(mascara.sum()),
                "confianza_media": float(confianza[mascara].mean()),
                "accuracy": float(acierto[mascara].mean()),
            }
        )
    tabla = pd.DataFrame(filas)
    tabla["diferencia"] = tabla["accuracy"] - tabla["confianza_media"]
    ece = float((tabla["imagenes"] / len(y_real) * tabla["diferencia"].abs()).sum())

    if graficar:
        figura, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
        ejes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="calibración perfecta")
        ejes[0].plot(tabla["confianza_media"], tabla["accuracy"], marker="o", linewidth=2, color="#d95f02",
                     label=f"modelo (ECE = {ece:.3f})")
        ejes[0].set_xlabel("Confianza media del intervalo")
        ejes[0].set_ylabel("Accuracy del intervalo")
        ejes[0].set_title("Diagrama de confiabilidad", fontsize=10)
        ejes[0].legend(fontsize=8)
        ejes[0].grid(alpha=0.3)
        ejes[1].hist([confianza[acierto], confianza[~acierto]], bins=bordes, stacked=True,
                     color=["#1b7f3b", "#c0392b"], label=["aciertos", "errores"])
        ejes[1].set_xlabel("Confianza (probabilidad más alta)")
        ejes[1].set_ylabel("Imágenes")
        ejes[1].set_title("Confianza de aciertos y errores", fontsize=10)
        ejes[1].legend(fontsize=8)
        ejes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return tabla, ece


def elegir_umbral(confianza, acierto, objetivo=OBJETIVO_UMBRAL):
    """El umbral de confianza más bajo con el que la accuracy de las imágenes aceptadas (las de confianza mayor o
    igual) llega al objetivo. Las demás irían a revisión manual. Se elige en VALIDACIÓN y se aplica después en
    test, para que el test no intervenga en la decisión. Si ningún umbral llega, devuelve el más exigente."""
    orden = np.argsort(-confianza)
    acierto_ordenado = np.asarray(acierto)[orden].astype(float)
    accuracy_acumulada = np.cumsum(acierto_ordenado) / np.arange(1, len(orden) + 1)
    validos = np.flatnonzero(accuracy_acumulada >= objetivo)
    if not len(validos):
        return float(confianza.max())
    return float(confianza[orden][validos[-1]])


def aplicar_umbral(confianza, acierto, umbral):
    """Cobertura (fracción de imágenes que el sistema resuelve solo) y accuracy de las aceptadas y de las que van
    a revisión, con un umbral de confianza."""
    aceptadas = confianza >= umbral
    acierto = np.asarray(acierto)
    return {
        "umbral": float(umbral),
        "cobertura": float(aceptadas.mean()),
        "accuracy_aceptadas": float(acierto[aceptadas].mean()) if aceptadas.any() else float("nan"),
        "accuracy_a_revision": float(acierto[~aceptadas].mean()) if (~aceptadas).any() else float("nan"),
    }


# --- Sesgos por subgrupo ---------------------------------------------------------------------------------


def grupos_de_auditoria(datos_evaluados, clases, conteos_train):
    """Variables de subgrupo de cada imagen evaluada, todas automáticas:

    - **fuente**: Kaggle o fotograma de episodio (del nombre del archivo);
    - **tamaño de la clase**: tercil del personaje según sus imágenes de entrenamiento;
    - **familia Simpson**: los personajes cuya carpeta termina en `_simpson` frente al resto;
    - **tipo de imagen**: original, parecida (versión filtrada) o alterada."""
    conteos_train = np.asarray(conteos_train)
    corte_bajo, corte_alto = np.quantile(conteos_train, [1 / 3, 2 / 3])
    etiqueta_tercil = np.where(
        conteos_train <= corte_bajo,
        f"pequeñas (≤ {int(corte_bajo)} img de train)",
        np.where(conteos_train <= corte_alto, f"medianas ({int(corte_bajo) + 1}-{int(corte_alto)})",
                 f"grandes (> {int(corte_alto)})"),
    )
    etiquetas = datos_evaluados["etiqueta"].to_numpy()
    familia = np.array([clase.endswith("_simpson") for clase in clases])
    tipo = np.where(datos_evaluados["alterada"], "alterada",
                    np.where(datos_evaluados["parecida"], "parecida", "original"))
    return {
        "fuente": datos_evaluados["fuente"].to_numpy(),
        "tamaño de la clase": etiqueta_tercil[etiquetas],
        "familia Simpson": np.where(familia[etiquetas], "familia Simpson", "resto de personajes"),
        "tipo de imagen": tipo,
    }


def sesgos_por_subgrupo(y_real, predicciones, grupos):
    """Accuracy y F1 macro de cada subgrupo, y la brecha entre el mejor y el peor de cada variable.

    El F1 macro de un subgrupo promedia solo los personajes que tienen imágenes en él, para no castigarlo por
    clases que no le tocan."""
    y_real = np.asarray(y_real)
    predicciones = np.asarray(predicciones)
    filas = []
    for variable, etiquetas in grupos.items():
        for grupo in pd.unique(etiquetas):
            mascara = etiquetas == grupo
            y_grupo, p_grupo = y_real[mascara], predicciones[mascara]
            filas.append(
                {
                    "variable": variable,
                    "grupo": grupo,
                    "imagenes": int(mascara.sum()),
                    "personajes": int(len(np.unique(y_grupo))),
                    "accuracy": round(float((y_grupo == p_grupo).mean()), 4),
                    "f1_macro": round(float(f1_score(y_grupo, p_grupo, labels=np.unique(y_grupo), average="macro",
                                                     zero_division=0)), 4),
                }
            )
    tabla = pd.DataFrame(filas)
    brechas = (
        tabla.groupby("variable", sort=False)
        .agg(brecha_accuracy=("accuracy", lambda serie: serie.max() - serie.min()),
             brecha_f1_macro=("f1_macro", lambda serie: serie.max() - serie.min()))
        .round(4)
        .reset_index()
    )
    return tabla, brechas


# --- Auditoría completa ----------------------------------------------------------------------------------


def auditar_modelo(nombre, y_test, probabilidades_test, y_val, probabilidades_val, datos_test, clases,
                   conteos_train, f1_por_clase, objetivo_umbral=OBJETIVO_UMBRAL):
    """Auditoría del modelo final en cuatro partes, cada una con su tabla o su gráfico:

    1. **ROC-AUC** macro y ponderado (uno contra el resto) y las curvas ROC de los mejores y peores personajes;
    2. **Calibración**: diagrama de confiabilidad y ECE en test;
    3. **Umbral de revisión manual**: elegido en validación para que lo aceptado llegue a `objetivo_umbral` de
       accuracy, con la cobertura y la accuracy que deja en test;
    4. **Sesgos por subgrupo**: fuente, tamaño de la clase, familia Simpson y tipo de imagen.

    Devuelve un diccionario con las cifras para el resumen JSON de la corrida y las tablas."""
    y_test = np.asarray(y_test)
    predicciones_test = probabilidades_test.argmax(axis=1)
    metricas = evaluar_metricas(y_test, probabilidades_test, len(clases))

    print(f"=== 1. ROC-AUC de {nombre} en test ===")
    print(f"ROC-AUC macro: {metricas['roc_auc_macro']:.4f}   |   ponderado: {metricas['roc_auc_ponderado']:.4f}"
          "   (0,5 = azar; 1 = separa a cada personaje del resto)")
    auc_por_clase = curvas_roc(y_test, probabilidades_test, clases, f1_por_clase)
    peores_auc = auc_por_clase.sort_values("roc_auc").head(5)
    display(peores_auc.reset_index(drop=True))

    print(f"\n=== 2. Calibración de {nombre} en test ===")
    tabla_calibracion, ece = calibracion(y_test, probabilidades_test)
    confianza_test = probabilidades_test.max(axis=1)
    print(f"ECE: {ece:.4f}   |   confianza media {confianza_test.mean():.3f} frente a accuracy "
          f"{(predicciones_test == y_test).mean():.3f}"
          + ("  -> el modelo es sobreconfiado" if confianza_test.mean() > (predicciones_test == y_test).mean()
             else "  -> el modelo es subconfiado"))

    print(f"\n=== 3. Umbral de revisión manual (elegido en validación, objetivo {objetivo_umbral:.0%}) ===")
    confianza_val = probabilidades_val.max(axis=1)
    acierto_val = probabilidades_val.argmax(axis=1) == np.asarray(y_val)
    umbral = elegir_umbral(confianza_val, acierto_val, objetivo_umbral)
    en_validacion = aplicar_umbral(confianza_val, acierto_val, umbral)
    en_test = aplicar_umbral(confianza_test, predicciones_test == y_test, umbral)
    tabla_umbral = pd.DataFrame(
        [
            ("validación (donde se elige)", en_validacion["cobertura"], en_validacion["accuracy_aceptadas"],
             en_validacion["accuracy_a_revision"]),
            ("test (donde se comprueba)", en_test["cobertura"], en_test["accuracy_aceptadas"],
             en_test["accuracy_a_revision"]),
        ],
        columns=["conjunto", "cobertura (resuelve solo)", "accuracy de lo aceptado", "accuracy de lo que va a revisión"],
    ).round(4)
    display(tabla_umbral)
    print(f"Umbral de confianza: {umbral:.3f}. Con él, el sistema resuelve solo el {en_test['cobertura']:.1%} del test"
          f" con una accuracy de {en_test['accuracy_aceptadas']:.3f}; el resto va a revisión.")

    print(f"\n=== 4. Sesgos por subgrupo de {nombre} en test ===")
    grupos = grupos_de_auditoria(datos_test, clases, conteos_train)
    tabla_sesgos, brechas = sesgos_por_subgrupo(y_test, predicciones_test, grupos)
    display(tabla_sesgos)
    display(brechas)

    figura, ejes = plt.subplots(1, len(grupos), figsize=(4.2 * len(grupos), 4), sharey=True)
    for eje, variable in zip(ejes, grupos):
        filas = tabla_sesgos[tabla_sesgos["variable"] == variable]
        posiciones = np.arange(len(filas))
        eje.bar(posiciones - 0.2, filas["accuracy"], width=0.4, label="accuracy")
        eje.bar(posiciones + 0.2, filas["f1_macro"], width=0.4, label="F1 macro")
        eje.set_xticks(posiciones)
        eje.set_xticklabels([f"{grupo}\n(n={n})" for grupo, n in zip(filas["grupo"], filas["imagenes"])], fontsize=7)
        eje.set_title(variable, fontsize=10)
        eje.set_ylim(0, 1)
        eje.grid(alpha=0.3, axis="y")
    ejes[0].legend(fontsize=8)
    figura.suptitle(f"Desempeño de {nombre} por subgrupo (test)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    def valor(variable, grupo, metrica):
        """Métrica de un subgrupo, o None si no tiene imágenes."""
        fila = tabla_sesgos[(tabla_sesgos["variable"] == variable) & (tabla_sesgos["grupo"] == grupo)]
        return None if fila.empty else float(fila[metrica].iloc[0])

    resumen = {
        "roc_auc_macro_test": round(metricas["roc_auc_macro"], 4),
        "roc_auc_ponderado_test": round(metricas["roc_auc_ponderado"], 4),
        "precision_macro_test": round(metricas["precision_macro"], 4),
        "recall_macro_test": round(metricas["recall_macro"], 4),
        "ece_test": round(ece, 4),
        "umbral_revision": round(umbral, 4),
        "cobertura_umbral_test": round(en_test["cobertura"], 4),
        "accuracy_umbral_test": round(en_test["accuracy_aceptadas"], 4),
        "accuracy_kaggle_test": valor("fuente", FUENTE_KAGGLE, "accuracy"),
        "accuracy_episodio_test": valor("fuente", FUENTE_EPISODIO, "accuracy"),
        "f1_macro_kaggle_test": valor("fuente", FUENTE_KAGGLE, "f1_macro"),
        "f1_macro_episodio_test": valor("fuente", FUENTE_EPISODIO, "f1_macro"),
        "accuracy_familia_simpson_test": valor("familia Simpson", "familia Simpson", "accuracy"),
        "accuracy_resto_personajes_test": valor("familia Simpson", "resto de personajes", "accuracy"),
    }
    return {
        "resumen": resumen,
        "metricas": metricas,
        "auc_por_clase": auc_por_clase,
        "calibracion": tabla_calibracion,
        "umbral": tabla_umbral,
        "sesgos": tabla_sesgos,
        "brechas": brechas,
    }


# --- SHAP ------------------------------------------------------------------------------------------------


class _Envoltorio(torch.nn.Module):
    """Un modelo de Keras (backend PyTorch) como `torch.nn.Module` en modo de inferencia, para SHAP."""

    def __init__(self, modelo):
        """Guarda el modelo de Keras, que ya es un `nn.Module` con el backend de PyTorch."""
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        """Probabilidades del modelo con `training=False` (sin Dropout y con las medias de BatchNorm)."""
        return self.modelo(x, training=False)


def explicar_con_shap(modelo, X_fondo, X_explicar, grupos, dispositivo="cuda", muestras=200, semilla=42):
    """Valores SHAP de la clase predicha para cada imagen de `X_explicar`, agrupados por bloque de entrada.

    Usa `shap.GradientExplainer` (expected gradients): integra los gradientes del modelo entre cada imagen y las
    del fondo, y aproxima los valores de Shapley sin evaluar las 2^n coaliciones. Cada valor dice cuánto empuja
    esa entrada la probabilidad de la clase predicha, por encima o por debajo de la media del fondo.

    `grupos` es `{nombre_del_grupo: columnas}`: la suma de los valores SHAP de las columnas de un grupo es su
    aporte. Devuelve `(agrupados, entradas_agrupadas, clases_explicadas)`: un DataFrame `(imagen, grupo)` con los
    valores SHAP agrupados, otro con el valor medio de la entrada en cada grupo (para colorear el gráfico) y la
    clase predicha de cada imagen."""
    import shap

    torch.manual_seed(semilla)
    envoltorio = _Envoltorio(modelo).eval()
    fondo = torch.as_tensor(np.asarray(X_fondo), dtype=torch.float32, device=dispositivo)
    explicar = torch.as_tensor(np.asarray(X_explicar), dtype=torch.float32, device=dispositivo)
    explicador = shap.GradientExplainer(envoltorio, fondo)
    valores, indices = explicador.shap_values(explicar, nsamples=muestras, ranked_outputs=1,
                                              output_rank_order="max", rseed=semilla)
    valores = np.asarray(valores[0] if isinstance(valores, list) else valores, dtype=np.float64)
    if valores.ndim == 3:  # (imagen, entrada, salida explicada)
        valores = valores[..., 0]
    clases_explicadas = np.asarray(indices.detach().cpu() if hasattr(indices, "detach") else indices).reshape(-1)

    X_explicar = np.asarray(X_explicar, dtype=np.float64)
    agrupados = pd.DataFrame({nombre: valores[:, columnas].sum(axis=1) for nombre, columnas in grupos.items()})
    entradas = pd.DataFrame({nombre: X_explicar[:, columnas].mean(axis=1) for nombre, columnas in grupos.items()})
    return agrupados, entradas, clases_explicadas


def graficar_shap(agrupados, entradas, titulo="SHAP por bloque de entrada", maximo=20):
    """Importancia media (|SHAP| medio) de cada grupo y el gráfico de enjambre de SHAP por grupo."""
    import shap

    importancia = agrupados.abs().mean().sort_values(ascending=False)
    tabla = pd.DataFrame(
        {
            "grupo": importancia.index,
            "shap_medio_absoluto": importancia.values.round(5),
            "shap_medio": agrupados[importancia.index].mean().values.round(5),
            "% del total": (importancia / importancia.sum() * 100).round(1).values,
        }
    )
    display(tabla)

    figura, eje = plt.subplots(figsize=(8, 0.35 * min(len(importancia), maximo) + 1.5))
    seleccion = importancia.head(maximo)[::-1]
    eje.barh(seleccion.index, seleccion.values, color="#d95f02")
    eje.set_xlabel("|SHAP| medio sobre la probabilidad de la clase predicha")
    eje.set_title(titulo, fontsize=11)
    eje.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()

    columnas = list(importancia.head(maximo).index)
    shap.summary_plot(agrupados[columnas].to_numpy(), entradas[columnas].to_numpy(), feature_names=columnas,
                      max_display=maximo, show=False, plot_size=(9, 0.35 * len(columnas) + 1.5))
    plt.title(f"{titulo}: cada punto es una imagen", fontsize=11)
    plt.tight_layout()
    plt.show()
    return tabla
