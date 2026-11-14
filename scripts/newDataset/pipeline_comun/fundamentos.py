"""Gráficos de los fundamentos: funciones de activación, softmax, pérdida y tasa de aprendizaje.

Acompañan la sección «Fundamentos» de los dos notebooks. No dependen de los datos: solo del número de clases.
"""

import matplotlib.pyplot as plt
import numpy as np


def softmax(z, temperatura=1.0):
    """Softmax numéricamente estable: se resta el máximo antes de exponenciar."""
    escalado = np.asarray(z, dtype=float) / temperatura
    exponenciales = np.exp(escalado - escalado.max())
    return exponenciales / exponenciales.sum()


def graficar_activaciones():
    """Sigmoide, tanh y ReLU junto a sus derivadas: la saturación que frena la retropropagación."""
    z = np.linspace(-6, 6, 400)
    activaciones = {
        "Sigmoide": (1 / (1 + np.exp(-z)), lambda s: s * (1 - s)),
        "Tanh": (np.tanh(z), lambda s: 1 - s ** 2),
        "ReLU": (np.maximum(0, z), lambda s: (z > 0).astype(float)),
    }

    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for nombre, (valores, derivada) in activaciones.items():
        ejes[0].plot(z, valores, label=nombre, linewidth=2)
        ejes[1].plot(z, derivada(valores), label=nombre, linewidth=2)
    ejes[0].set_title("Funciones de activación", fontsize=10)
    ejes[1].set_title("Sus derivadas (lo que viaja en la retropropagación)", fontsize=10)
    for eje in ejes:
        eje.axhline(0, color="gray", linewidth=0.6)
        eje.axvline(0, color="gray", linewidth=0.6)
        eje.set_xlabel("z (preactivación)")
        eje.grid(alpha=0.3)
        eje.legend()
    plt.tight_layout()
    plt.show()

    print(
        "En el gráfico de la derecha se ve el problema de la saturación: las derivadas de sigmoide y tanh"
        "\ncaen a cero en los extremos (el gradiente se desvanece), mientras que la de ReLU vale 1 para"
        "\ntodo z > 0. Nótese además que la derivada de la sigmoide no supera 0.25."
    )


def graficar_softmax(numero_clases, semilla=7):
    """Logits, su softmax y la competencia entre clases. Softmax se grafica aparte: no es una activación elemento
    a elemento, sino una función vectorial que mira todas las salidas a la vez y las convierte en una
    distribución."""
    generador = np.random.default_rng(semilla)
    logits = generador.normal(0, 1.5, numero_clases)
    logits[min(9, numero_clases - 1)] += 4.0  # una clase claramente favorecida por la red
    probabilidades = softmax(logits)

    figura, ejes = plt.subplots(1, 3, figsize=(14, 4))
    # 1. Los logits: números sin escala definida, positivos y negativos
    ejes[0].bar(range(numero_clases), logits, color="#8fb8de")
    ejes[0].axhline(0, color="gray", linewidth=0.8)
    ejes[0].set_title(f"Entrada: {numero_clases} logits (sin escala)", fontsize=10)
    ejes[0].set_ylabel("z (preactivación de la capa de salida)")
    # 2. La misma información después de softmax: probabilidades que suman 1
    ejes[1].bar(range(numero_clases), probabilidades * 100, color="#1b7f3b")
    ejes[1].axhline(100 / numero_clases, color="red", linestyle="--", linewidth=1,
                    label=f"azar = {100 / numero_clases:.1f} %")
    ejes[1].set_title(f"Salida: softmax (suma = {probabilidades.sum():.2f})", fontsize=10)
    ejes[1].set_ylabel("probabilidad (%)")
    ejes[1].legend(fontsize=8)
    for eje in ejes[:2]:
        eje.set_xlabel("índice de clase")
        eje.grid(alpha=0.3, axis="y")

    # 3. La competencia entre clases: subir el logit de una hunde a todas las demás
    desplazamientos = np.linspace(-4, 6, 200)
    seguidas = sorted({0, min(9, numero_clases - 1), min(17, numero_clases - 1)})
    curvas = np.array([softmax(np.concatenate(([logits[0] + d], logits[1:]))) for d in desplazamientos])
    for indice in seguidas:
        ejes[2].plot(desplazamientos, curvas[:, indice] * 100, linewidth=2,
                     label=f"clase {indice}" + (" (la que sube)" if indice == 0 else ""))
    ejes[2].plot(desplazamientos, curvas.sum(axis=1) * 100, color="gray", linestyle="--", linewidth=1.2,
                 label=f"suma de las {numero_clases}")
    ejes[2].set_title("Competencia: las probabilidades se reparten", fontsize=10)
    ejes[2].set_xlabel("cuánto se suma al logit de la clase 0")
    ejes[2].set_ylabel("probabilidad (%)")
    ejes[2].grid(alpha=0.3)
    ejes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(
        "Softmax no se puede dibujar junto a ReLU o tanh porque no transforma un número en otro:"
        "\ntoma el vector completo de logits y devuelve una distribución. El denominador -- la suma"
        "\nde todas las exponenciales -- es lo que acopla las salidas entre si."
    )
    print(
        f"\nClase predicha: {int(probabilidades.argmax())} con {probabilidades.max() * 100:.1f} %"
        f" de probabilidad   |   suma de las {numero_clases} salidas = {probabilidades.sum():.6f}"
    )
    print(
        "En el gráfico de la derecha se ve la propiedad clave para etiqueta única: al crecer el logit"
        "\nde una clase, su probabilidad sube y la de las demas baja, mientras la suma se mantiene en 100 %."
    )


def graficar_perdida_y_tasa(numero_clases):
    """La entropía cruzada según la probabilidad de la clase correcta, y el efecto de la tasa de aprendizaje en
    el descenso del gradiente sobre una parábola."""
    probabilidades = np.linspace(0.001, 1.0, 500)
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))

    ejes[0].plot(probabilidades, -np.log(probabilidades), color="crimson", linewidth=2)
    ejes[0].set_xlabel("Probabilidad asignada a la clase correcta")
    ejes[0].set_ylabel("Pérdida  -log(p)")
    ejes[0].set_title("Entropía cruzada: castiga los errores confiados", fontsize=10)
    ejes[0].grid(alpha=0.3)
    for probabilidad in (0.01, 0.1, 0.5, 0.9):
        ejes[0].scatter([probabilidad], [-np.log(probabilidad)], zorder=3, s=35)
        ejes[0].annotate(f"p={probabilidad}  L={-np.log(probabilidad):.2f}", (probabilidad, -np.log(probabilidad)),
                         textcoords="offset points", xytext=(8, 6), fontsize=8)
    # Pérdida esperada al predecir al azar entre todas las clases
    perdida_azar = -np.log(1 / numero_clases)
    ejes[0].axhline(perdida_azar, color="gray", linestyle="--", linewidth=1,
                    label=f"prediccion al azar = {perdida_azar:.2f}")
    ejes[0].legend()

    # Efecto del learning rate sobre el descenso del gradiente en una parábola L(w) = w^2
    ejes[1].plot(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200) ** 2, color="gray", alpha=0.5)
    for tasa, color in ((0.05, "tab:blue"), (0.4, "tab:green"), (1.02, "tab:red")):
        w = 2.0
        trayectoria = [w]
        for _ in range(12):
            w = w - tasa * 2 * w  # derivada de w^2
            trayectoria.append(w)
        trayectoria = np.array(trayectoria)
        ejes[1].plot(trayectoria, trayectoria ** 2, "o-", color=color, markersize=4, linewidth=1, label=f"lr = {tasa}")
    ejes[1].set_xlabel("w (parámetro)")
    ejes[1].set_ylabel("L(w)")
    ejes[1].set_title("Efecto del learning rate en el descenso del gradiente", fontsize=10)
    ejes[1].grid(alpha=0.3)
    ejes[1].legend()
    plt.tight_layout()
    plt.show()

    print(f"Perdida esperada de un modelo sin entrenar (1/{numero_clases} a cada clase): {perdida_azar:.3f}")
    print("Es el valor de referencia con el que debería arrancar el entrenamiento en la primera época.")
    print(
        "\nEn el gráfico de la derecha: un lr pequeño (azul) converge pero lento, uno adecuado (verde)"
        "\nllega rápido al mínimo, y uno demasiado grande (rojo) se aleja y diverge."
    )
    return perdida_azar
