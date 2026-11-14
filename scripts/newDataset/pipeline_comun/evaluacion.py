"""Evaluación: métricas de desempeño, matriz de confusión y análisis de aciertos y errores.

Etapa «Evaluación» del pipeline. `evaluar_metricas` calcula las métricas globales a partir de las probabilidades
(accuracy, precision, recall, F1, top-3 y ROC-AUC); la clase `Evaluacion` reúne lo que el análisis de resultados
necesita de un conjunto evaluado (probabilidades, imágenes, escena de cada imagen) y dibuja las galerías.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

COLOR_ACIERTO, COLOR_ESCENA, COLOR_ERROR = "#1b7f3b", "#d68910", "#c0392b"
TEXTO_RESULTADO = {
    COLOR_ACIERTO: "correcto",
    COLOR_ESCENA: "error: otro personaje de la escena",
    COLOR_ERROR: "incorrecto",
}


def evaluar_metricas(y_real, probabilidades, numero_clases=None):
    """Métricas globales de un conjunto a partir de las probabilidades del modelo.

    Devuelve un diccionario con la accuracy, el top-3, la precision, el recall y el F1 (promedio macro, que pesa
    igual a cada personaje, y ponderado por el tamaño de cada clase), el ROC-AUC uno contra el resto (macro y
    ponderado) y la pérdida logarítmica. El ROC-AUC mide si el modelo ordena bien las probabilidades: es la
    probabilidad de que una imagen del personaje reciba más puntaje para ese personaje que una de otro."""
    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades, dtype=np.float64)
    numero_clases = numero_clases or probabilidades.shape[1]
    etiquetas = np.arange(numero_clases)
    predicciones = probabilidades.argmax(axis=1)
    metricas = {
        "accuracy": float((predicciones == y_real).mean()),
        "top3": float((np.argsort(-probabilidades, axis=1)[:, :3] == y_real[:, None]).any(axis=1).mean()),
    }
    for promedio, sufijo in (("macro", "macro"), ("weighted", "ponderado")):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_real, predicciones, labels=etiquetas, average=promedio, zero_division=0
        )
        metricas[f"precision_{sufijo}"] = float(precision)
        metricas[f"recall_{sufijo}"] = float(recall)
        metricas[f"f1_{sufijo}"] = float(f1)
    # Las probabilidades se renormalizan por si el redondeo de float32 las deja sumando 0,9999
    normalizadas = probabilidades / probabilidades.sum(axis=1, keepdims=True)
    for promedio, sufijo in (("macro", "macro"), ("weighted", "ponderado")):
        try:
            metricas[f"roc_auc_{sufijo}"] = float(
                roc_auc_score(y_real, normalizadas, multi_class="ovr", average=promedio, labels=etiquetas)
            )
        except ValueError:
            # Alguna clase sin ejemplos en este conjunto: el ROC-AUC uno contra el resto no está definido
            metricas[f"roc_auc_{sufijo}"] = float("nan")
    metricas["log_loss"] = float(log_loss(y_real, np.clip(normalizadas, 1e-7, 1), labels=etiquetas))
    return metricas


def interpretar_metricas(accuracy, accuracy_por_escena, f1_ponderado, f1_macro, top3, metricas_por_clase,
                         roc_auc_macro=None):
    """Las lecturas globales lado a lado, la distancia entre la accuracy y el F1 macro, y cuántas clases quedan
    fuertes (F1 ≥ 0,60) o débiles (F1 < 0,30). Devuelve `(resumen, brecha, clases_debiles, clases_fuertes)`."""
    filas = [
        ("Accuracy (test)", accuracy, "Imágenes en que nombra al personaje de la carpeta; domina por las clases grandes"),
        ("Accuracy por escena (test)", accuracy_por_escena,
         "También acepta a otro personaje presente, como con la imagen completa; referencia"),
        ("F1 ponderado (test)", f1_ponderado, "Promedio de F1 pesado por el tamaño de cada clase"),
        ("F1 macro (test)", f1_macro, "Promedio simple de todas las clases; cada personaje pesa igual"),
        ("Top-3 accuracy (test)", top3, "El personaje de la carpeta está entre las 3 más probables"),
    ]
    if roc_auc_macro is not None:
        filas.append(("ROC-AUC macro (test)", roc_auc_macro,
                      "Si ordena bien las probabilidades: 0,5 es azar y 1 separa todo"))
    resumen = pd.DataFrame(
        [(nombre, round(float(valor), 4), que) for nombre, valor, que in filas],
        columns=["Metrica", "Valor", "Que mide"],
    )
    display(resumen)

    brecha = float(accuracy) - float(f1_macro)
    clases_debiles = metricas_por_clase[metricas_por_clase["f1"] < 0.30]
    clases_fuertes = metricas_por_clase[metricas_por_clase["f1"] >= 0.60]
    numero_clases = len(metricas_por_clase)
    print(f"Brecha accuracy - F1 macro: {brecha:.3f}")
    print(f"Clases con F1 >= 0.60 : {len(clases_fuertes)} de {numero_clases}")
    print(f"Clases con F1 <  0.30 : {len(clases_debiles)} de {numero_clases}")
    if len(clases_debiles):
        print("  " + ", ".join(clases_debiles["personaje"].tolist()))
    print(f"\nMediana de imagenes de entrenamiento en las clases fuertes: {clases_fuertes['n_train'].median():.0f}")
    if len(clases_debiles):
        print(f"Mediana de imagenes de entrenamiento en las clases debiles : {clases_debiles['n_train'].median():.0f}")
    return resumen, brecha, clases_debiles, clases_fuertes


def proporcion(valores):
    """Media de un arreglo booleano redondeada a cuatro decimales, o NaN si está vacío (un subgrupo sin
    imágenes)."""
    return round(float(valores.mean()), 4) if len(valores) else np.nan


def _marco(eje, color, grosor):
    """Quita las marcas de los ejes y pone un marco de color a una imagen."""
    eje.set_xticks([])
    eje.set_yticks([])
    for lado in eje.spines.values():
        lado.set_edgecolor(color)
        lado.set_linewidth(grosor)


@dataclass
class Evaluacion:
    """Un conjunto evaluado por el modelo final, con todo lo que necesitan las galerías y las tablas de errores.

    - `clases`: nombres de los personajes, en el orden de las salidas del modelo;
    - `y_real`: índice del personaje de la carpeta de cada imagen;
    - `probabilidades`: salida softmax del modelo, `(n, clases)`;
    - `imagenes`: las imágenes redimensionadas, para mostrarlas;
    - `escenas`: los otros personajes de la escena de cada imagen, como texto `"a, b"` (columna `escena`);
    - `conteos_train`: imágenes de entrenamiento de cada clase.

    El criterio de acierto es estricto: acierta si nombra al personaje de la carpeta. Nombrar a otro personaje de la
    escena es un error, que las galerías pintan en ámbar y la accuracy por escena cuenta solo como referencia."""

    clases: list
    y_real: np.ndarray
    probabilidades: np.ndarray
    imagenes: np.ndarray
    escenas: list
    conteos_train: np.ndarray
    predicciones: np.ndarray = field(init=False)
    confianza: np.ndarray = field(init=False)
    acompanantes: list = field(init=False)
    acierto: np.ndarray = field(init=False)
    acierto_por_escena: np.ndarray = field(init=False)
    top3_por_imagen: np.ndarray = field(init=False)

    def __post_init__(self):
        """Calcula las predicciones, la confianza y los aciertos de cada imagen."""
        self.y_real = np.asarray(self.y_real)
        self.predicciones = self.probabilidades.argmax(axis=1)
        self.confianza = self.probabilidades.max(axis=1)
        self.acompanantes = [set(texto.split(", ")) if texto else set() for texto in self.escenas]
        self.acierto = self.predicciones == self.y_real
        # Error que nombra a otro personaje presente según la escena (±3 s) o una gemela: referencia
        self.acierto_por_escena = np.array(
            [self.clases[predicho] in presentes for predicho, presentes in zip(self.predicciones, self.acompanantes)]
        )
        self.top3_por_imagen = (np.argsort(-self.probabilidades, axis=1)[:, :3] == self.y_real[:, None]).any(axis=1)

    @property
    def numero_clases(self):
        """Cantidad de personajes."""
        return len(self.clases)

    def color_resultado(self, indice):
        """Verde si el modelo nombró al personaje de la carpeta, ámbar si se equivocó nombrando a otro personaje de
        la escena y rojo en los demás errores."""
        if self.acierto[indice]:
            return COLOR_ACIERTO
        return COLOR_ESCENA if self.acierto_por_escena[indice] else COLOR_ERROR

    @staticmethod
    def rotulo_escena(indice):
        """Rótulo de un error que nombra a otro personaje de la escena."""
        return "error: otro personaje de la escena"

    def _nombre(self, indice_clase):
        """Nombre del personaje con espacios en lugar de guiones bajos."""
        return self.clases[indice_clase].replace("_", " ")

    # --- Métricas por clase y matriz de confusión -----------------------------------------------------

    def reporte_por_clase(self):
        """Precision, recall y F1 de cada personaje, de mayor a menor F1, y la correlación entre las imágenes de
        entrenamiento y el F1. Devuelve `(metricas_por_clase, correlacion)`."""
        print(classification_report(self.y_real, self.predicciones, labels=range(self.numero_clases),
                                    target_names=self.clases, digits=3, zero_division=0))
        precision, recall, f1, soporte = precision_recall_fscore_support(
            self.y_real, self.predicciones, labels=range(self.numero_clases), zero_division=0
        )
        metricas_por_clase = pd.DataFrame(
            {
                "personaje": self.clases,
                "n_test": soporte,
                "n_train": self.conteos_train,
                "precision": precision.round(3),
                "recall": recall.round(3),
                "f1": f1.round(3),
            }
        ).sort_values("f1", ascending=False, ignore_index=True)
        display(metricas_por_clase)
        correlacion = float(np.corrcoef(self.conteos_train, f1)[0, 1])
        print(f"Correlacion entre imagenes de entrenamiento y F1 por clase: {correlacion:.3f}")
        return metricas_por_clase, correlacion

    @staticmethod
    def graficar_por_clase(metricas_por_clase, correlacion):
        """Precision y recall de cada personaje, y el F1 frente a las imágenes de entrenamiento."""
        figura, ejes = plt.subplots(1, 2, figsize=(14, 12))
        orden_f1 = metricas_por_clase.sort_values("f1")
        posiciones = np.arange(len(orden_f1))
        ejes[0].barh(posiciones - 0.2, orden_f1["precision"], height=0.4, label="precision")
        ejes[0].barh(posiciones + 0.2, orden_f1["recall"], height=0.4, label="recall")
        ejes[0].set_yticks(posiciones)
        ejes[0].set_yticklabels(orden_f1["personaje"], fontsize=6)
        ejes[0].set_xlabel("Valor")
        ejes[0].set_title("Precision y recall por personaje", fontsize=10)
        ejes[0].legend(fontsize=8)
        ejes[0].grid(alpha=0.3, axis="x")
        ejes[1].scatter(metricas_por_clase["n_train"], metricas_por_clase["f1"], s=40)
        for _, fila in metricas_por_clase.iterrows():
            ejes[1].annotate(fila["personaje"], (fila["n_train"], fila["f1"]), textcoords="offset points",
                             xytext=(4, 3), fontsize=6)
        ejes[1].set_xscale("log")
        ejes[1].set_xlabel("Imágenes de entrenamiento (escala log)")
        ejes[1].set_ylabel("F1 en test")
        ejes[1].set_title(f"Mas ejemplos, mejor F1 (correlacion = {correlacion:.2f})", fontsize=10)
        ejes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def matriz_confusion(self):
        """Matriz de confusión normalizada por fila (cada fila suma 1, así el desbalance no distorsiona la lectura).
        Devuelve `(matriz, matriz_normalizada)`."""
        matriz = confusion_matrix(self.y_real, self.predicciones, labels=range(self.numero_clases))
        totales_fila = matriz.sum(axis=1, keepdims=True)
        matriz_normalizada = np.divide(matriz, totales_fila, out=np.zeros(matriz.shape), where=totales_fila > 0)

        figura, eje = plt.subplots(figsize=(18, 16))
        sns.heatmap(matriz_normalizada, annot=False, cmap="viridis", vmin=0, vmax=1, xticklabels=self.clases,
                    yticklabels=self.clases, cbar_kws={"label": "Proporción de la clase real"}, ax=eje)
        eje.set_xlabel("Prediccion")
        eje.set_ylabel("Clase real")
        eje.set_title("Matriz de confusión normalizada por fila (test)", fontsize=11)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()
        plt.show()

        diagonal = np.diag(matriz_normalizada)
        print(f"Recall medio (diagonal): {diagonal.mean():.3f}")
        print(f"Clase mejor reconocida : {self.clases[int(diagonal.argmax())]} ({diagonal.max():.3f})")
        print(f"Clase peor reconocida  : {self.clases[int(diagonal.argmin())]} ({diagonal.min():.3f})")
        return matriz, matriz_normalizada

    # --- Galerías ---------------------------------------------------------------------------------------

    def casos_top_k(self, casos, top_k=5):
        """Cada caso `(indice, descripcion)` con su imagen y las `top_k` clases más probables."""
        figura, ejes = plt.subplots(len(casos), 2, figsize=(11, 3.4 * len(casos)), layout="constrained",
                                    gridspec_kw={"width_ratios": [1, 2.6]})
        for fila, (indice, descripcion) in enumerate(casos):
            real = int(self.y_real[indice])
            predicho = int(self.predicciones[indice])
            color = self.color_resultado(indice)
            eje_imagen = ejes[fila, 0]
            eje_imagen.imshow(self.imagenes[indice])
            _marco(eje_imagen, color, 3)
            eje_imagen.set_xlabel(f"real: {self._nombre(real)}", fontsize=9, labelpad=6, color=color)
            eje_imagen.set_title(descripcion, fontsize=9, fontweight="bold")

            # Las top_k clases con mayor probabilidad, dibujadas de mayor a menor
            mejores = np.argsort(-self.probabilidades[indice])[:top_k][::-1]
            colores = [
                COLOR_ACIERTO if i == real else COLOR_ESCENA if self.clases[i] in self.acompanantes[indice]
                else "#b0c4de"
                for i in mejores
            ]
            eje_barras = ejes[fila, 1]
            barras = eje_barras.barh([self._nombre(i) for i in mejores], self.probabilidades[indice][mejores] * 100,
                                     color=colores)
            eje_barras.bar_label(barras, fmt="%.1f %%", fontsize=8, padding=3)
            eje_barras.set_xlim(0, 110)
            eje_barras.set_xlabel("probabilidad asignada (%)")
            eje_barras.tick_params(axis="y", labelsize=8)
            eje_barras.grid(alpha=0.3, axis="x")
            eje_barras.set_title(f"Top {top_k}: predice '{self._nombre(predicho)}' ({TEXTO_RESULTADO[color]})",
                                 fontsize=9, color=color)
        figura.suptitle("Distribucion de probabilidad en imagenes de test", fontsize=12, fontweight="bold")
        plt.show()

    def muestra_aleatoria(self, semilla, filas=4, columnas=6):
        """Muestra aleatoria sin seleccionar: marco verde si acertó, ámbar si se equivocó nombrando a otro
        personaje de la escena y rojo en los demás errores."""
        generador = np.random.default_rng(semilla)
        muestra = generador.choice(len(self.y_real), size=filas * columnas, replace=False)
        figura, ejes = plt.subplots(filas, columnas, figsize=(2.1 * columnas, 2.9 * filas), layout="constrained")
        for eje, indice in zip(np.ravel(ejes), muestra):
            color = self.color_resultado(indice)
            eje.imshow(self.imagenes[indice])
            _marco(eje, color, 2.5)
            eje.set_xlabel(
                f"real: {self._nombre(self.y_real[indice])}\n"
                f"pred: {self._nombre(self.predicciones[indice])} ({self.confianza[indice]:.2f})"
                + (f"\n{self.rotulo_escena(indice)}" if color == COLOR_ESCENA else ""),
                fontsize=7, color=color, labelpad=5, linespacing=1.5,
            )
        figura.suptitle(
            f"Muestra aleatoria de test: {int(self.acierto[muestra].sum())} aciertos de {len(muestra)}"
            f"  (accuracy global = {self.acierto.mean():.1%})",
            fontsize=12, fontweight="bold",
        )
        plt.show()

    def galeria_por_personaje(self, recall_por_clase, filas=8):
        """Una imagen por personaje (acertada si la hay), con el marco coloreado según el recall de su clase,
        de mejor a peor."""
        mapa_color = plt.get_cmap("RdYlGn")
        columnas = int(np.ceil(self.numero_clases / filas))
        figura, ejes = plt.subplots(filas, columnas, figsize=(2.0 * columnas, 2.7 * filas), layout="constrained")
        ejes = np.ravel(ejes)
        orden_clases = np.argsort(-recall_por_clase)
        for eje, clase in zip(ejes, orden_clases):
            indices_clase = np.flatnonzero(self.y_real == clase)
            correctas = indices_clase[self.predicciones[indices_clase] == clase]
            indice = int(correctas[0]) if len(correctas) else int(indices_clase[0])
            eje.imshow(self.imagenes[indice])
            _marco(eje, mapa_color(recall_por_clase[clase]), 3.5)
            eje.set_xlabel(f"{self._nombre(clase)}\nrecall {recall_por_clase[clase]:.0%}  ({len(indices_clase)} img)",
                           fontsize=7, labelpad=5, linespacing=1.5)
        for eje in ejes[self.numero_clases:]:
            eje.axis("off")
        figura.suptitle("Un personaje por celda, ordenados por recall en test  (verde = lo reconoce, rojo = se le escapa)",
                        fontsize=12, fontweight="bold")
        plt.show()
        print(f"Recall medio sobre las {self.numero_clases} clases: {recall_por_clase.mean():.1%}")
        print(f"Mejor: {self.clases[int(orden_clases[0])]} ({recall_por_clase[orden_clases[0]]:.0%})"
              f"   |   Peor: {self.clases[int(orden_clases[-1])]} ({recall_por_clase[orden_clases[-1]]:.0%})")

    def mostrar_ejemplos(self, indices, titulo, filas=2, columnas=5):
        """Rejilla de imágenes con su etiqueta real, la predicha y la confianza, con el marco del resultado."""
        indices = list(indices)
        figura, ejes = plt.subplots(filas, columnas, figsize=(2.2 * columnas, 3.0 * filas), layout="constrained")
        ejes = np.ravel(ejes)
        for eje, indice in zip(ejes, indices):
            color = self.color_resultado(indice)
            eje.imshow(self.imagenes[indice])
            _marco(eje, color, 2.5)
            # Las etiquetas van debajo de la imagen para que no invadan la fila superior
            eje.set_xlabel(
                f"real: {self._nombre(self.y_real[indice])}\n"
                f"pred: {self._nombre(self.predicciones[indice])}  ({self.confianza[indice]:.2f})"
                + (f"\n{self.rotulo_escena(indice)}" if color == COLOR_ESCENA else ""),
                fontsize=8, color=color, labelpad=6, linespacing=1.5,
            )
        for eje in ejes[len(indices):]:
            eje.axis("off")
        figura.suptitle(titulo, fontsize=13, fontweight="bold")
        plt.show()

    def aciertos_variados(self, n=10):
        """Los aciertos de mayor confianza sin repetir personaje, para ver variedad de clases."""
        aciertos = np.flatnonzero(self.acierto)
        vistos, seleccion = set(), []
        for indice in aciertos[np.argsort(-self.confianza[aciertos])]:
            if self.y_real[indice] not in vistos:
                vistos.add(self.y_real[indice])
                seleccion.append(indice)
            if len(seleccion) == n:
                break
        return seleccion

    def resumen_errores(self):
        """Errores totales, confianza en aciertos y errores, errores muy confiados y errores que nombran a otro
        personaje de la escena."""
        aciertos = np.flatnonzero(self.acierto)
        errores = np.flatnonzero(~self.acierto)
        return pd.DataFrame(
            [
                ("Errores totales", len(errores), f"{len(errores) / len(self.y_real):.1%} del test"),
                ("Confianza media en aciertos", round(float(self.confianza[aciertos].mean()), 3), ""),
                ("Confianza media en errores", round(float(self.confianza[errores].mean()), 3), ""),
                (
                    "Errores con confianza > 0.90",
                    int((self.confianza[errores] > 0.90).sum()),
                    f"{(self.confianza[errores] > 0.90).mean():.1%} de los errores",
                ),
                (
                    "Errores que nombran a otro personaje de la escena",
                    int(self.acierto_por_escena[errores].sum()),
                    f"{self.acierto_por_escena[errores].mean():.1%} de los errores (referencia)" if len(errores) else "",
                ),
            ],
            columns=["Indicador", "Valor", "Detalle"],
        )

    def confusiones_frecuentes(self, matriz, matriz_normalizada, n=15):
        """Pares (clase real, clase predicha) más frecuentes fuera de la diagonal, su gráfico y las clases que más
        absorben errores de otras."""
        pares = [
            {
                "real": self.clases[real],
                "predicho": self.clases[predicho],
                "casos": int(matriz[real, predicho]),
                "% de la clase real": round(float(matriz_normalizada[real, predicho]) * 100, 1),
                "n_train_real": int(self.conteos_train[real]),
                "n_train_predicho": int(self.conteos_train[predicho]),
            }
            for real in range(self.numero_clases)
            for predicho in range(self.numero_clases)
            if real != predicho and matriz[real, predicho] > 0
        ]
        confusiones = pd.DataFrame(pares).sort_values("casos", ascending=False, ignore_index=True).head(n)
        display(confusiones)

        figura, eje = plt.subplots(figsize=(9, 5))
        etiquetas = confusiones.head(10).apply(lambda fila: f"{fila['real']} -> {fila['predicho']}", axis=1)
        eje.barh(etiquetas[::-1], confusiones.head(10)["casos"][::-1], color="firebrick", alpha=0.8)
        eje.set_xlabel("Número de imágenes de test")
        eje.set_title("Las 10 confusiones más frecuentes", fontsize=10)
        eje.tick_params(axis="y", labelsize=8)
        eje.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()

        destino = confusiones.groupby("predicho")["casos"].sum().sort_values(ascending=False)
        print("Clases que más absorben errores de otras (el modelo las usa como respuesta por defecto):")
        for personaje, casos in destino.head(5).items():
            print(f"  {personaje}: {casos} imagenes de otras clases predichas como esta")
        return confusiones

    def errores_por_escena(self, semilla):
        """Separa los errores que nombran a otro personaje de la escena (referencia) de los demás: cuántos son, los
        pares más frecuentes y una muestra de ellos."""
        accuracy = float(self.acierto.mean())
        accuracy_escena = float((self.acierto | self.acierto_por_escena).mean())
        con_escena = np.array([bool(presentes) for presentes in self.acompanantes])
        errores_escena = np.flatnonzero(self.acierto_por_escena)
        distintos = int((~self.acierto).sum())
        display(
            pd.DataFrame(
                [
                    ("Imágenes de test", len(self.y_real), ""),
                    ("   con otro personaje presente (escena o gemela)", int(con_escena.sum()),
                     f"{con_escena.mean():.1%} del test"),
                    ("Errores: no nombra al personaje de la carpeta", distintos,
                     f"{distintos / len(self.y_real):.1%} del test"),
                    ("   que nombran a otro personaje presente", len(errores_escena),
                     f"{len(errores_escena) / max(distintos, 1):.1%} de los errores"),
                    ("Accuracy (personaje de la carpeta)", round(accuracy, 4), ""),
                    ("Accuracy por escena (referencia)", round(accuracy_escena, 4), f"{accuracy_escena - accuracy:+.4f}"),
                    ("Accuracy en imágenes con otro personaje presente", proporcion(self.acierto[con_escena]), ""),
                    ("Accuracy en el resto del test", proporcion(self.acierto[~con_escena]), ""),
                ],
                columns=["Indicador", "Valor", "Detalle"],
            )
        )
        display(
            pd.DataFrame(
                {
                    "carpeta": [self.clases[self.y_real[indice]] for indice in errores_escena],
                    "predicho": [self.clases[self.predicciones[indice]] for indice in errores_escena],
                }
            )
            .value_counts()
            .head(10)
            .reset_index(name="casos")
        )
        if not len(errores_escena):
            return
        generador = np.random.default_rng(semilla)
        muestra = generador.choice(errores_escena, size=min(10, len(errores_escena)), replace=False)
        figura, ejes = plt.subplots(2, 5, figsize=(11, 6.6), layout="constrained")
        ejes = np.ravel(ejes)
        for eje, indice in zip(ejes, muestra):
            eje.imshow(self.imagenes[indice])
            _marco(eje, COLOR_ESCENA, 2.5)
            escena = sorted(self.acompanantes[indice])
            resto = f" +{len(escena) - 2}" if len(escena) > 2 else ""
            eje.set_xlabel(
                f"carpeta: {self._nombre(self.y_real[indice])}\n"
                f"pred: {self._nombre(self.predicciones[indice])} ({self.confianza[indice]:.2f})\n"
                f"escena: {', '.join(n.replace('_', ' ') for n in escena[:2])}{resto}",
                fontsize=7, color="#9c640c", labelpad=6, linespacing=1.4,
            )
        for eje in ejes[len(muestra):]:
            eje.axis("off")
        figura.suptitle("Errores que nombran a otro personaje de la escena (referencia)", fontsize=13,
                        fontweight="bold")
        plt.show()

    def por_tipo_de_imagen(self, datos_evaluados):
        """Accuracy, accuracy por escena y top-3 en las originales, las parecidas (versión filtrada de una casi
        idéntica a otra) y las alteradas (variante de una original del mismo conjunto)."""
        es_alterada = datos_evaluados["alterada"].to_numpy()
        es_parecida = datos_evaluados["parecida"].to_numpy()
        es_original = ~es_alterada & ~es_parecida
        resumen = pd.DataFrame(
            [
                (
                    nombre,
                    int(mascara.sum()),
                    proporcion(self.acierto[mascara]),
                    proporcion((self.acierto | self.acierto_por_escena)[mascara]),
                    proporcion(self.top3_por_imagen[mascara]),
                )
                for nombre, mascara in (
                    ("Original, sin transformar", es_original),
                    ("Parecida: versión filtrada de una casi idéntica a otra", es_parecida),
                    ("Alterada: variante de una original de test", es_alterada),
                )
            ],
            columns=["Imágenes de test", "n", "accuracy", "accuracy por escena (referencia)", "top-3"],
        )
        display(resumen)
        return resumen
