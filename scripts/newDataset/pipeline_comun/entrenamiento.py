"""Entrenamiento: bucle propio sobre PyTorch, regularización, parada temprana, historiales y curvas.

Etapas «Entrenamiento» y «Regularización» del pipeline. El modelo se define con Keras 3 (backend PyTorch), pero el
bucle se escribe a mano para que la GPU no espere: los datos viven enteros en la VRAM y los acumuladores son tensores
de la tarjeta. Reproduce los tres callbacks del curso: EarlyStopping con `restore_best_weights`, ReduceLROnPlateau y
el guardado de la mejor época.
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

PARADA_PACIENCIA = "paciencia"
PARADA_TOPE = "tope"


class Historial:
    """Misma interfaz que el objeto que devuelve `keras.Model.fit()`: un atributo `.history` con un diccionario
    `{metrica: [valor por epoca]}`. Además guarda la mejor época, el tope de épocas y por qué terminó."""

    def __init__(self, epocas_maximas=None):
        """Historial vacío de un entrenamiento que puede durar hasta `epocas_maximas` épocas."""
        self.history = {}
        self.epocas_maximas = epocas_maximas
        self.mejor_epoca = None
        self.motivo_parada = None
        self.segundos = None

    def anotar(self, **metricas):
        """Agrega el valor de cada métrica de una época al final de su lista."""
        for nombre, valor in metricas.items():
            self.history.setdefault(nombre, []).append(float(valor))


def a_gpu(arreglo, dispositivo="cuda"):
    """Copia un arreglo de NumPy a la VRAM como tensor de PyTorch, sin cambiar su tipo. Las imágenes viajan en
    uint8 (cuatro veces menos memoria que en float32) y el modelo las escala sobre cada lote."""
    return torch.as_tensor(np.ascontiguousarray(arreglo)).to(dispositivo)


GIRO_MAXIMO_GRADOS = 18.0
ZOOM_MAXIMO = 0.10
CONTRASTE_MAXIMO = 0.10


def aumentar_imagenes(lote, generador, giro_maximo=GIRO_MAXIMO_GRADOS, zoom_maximo=ZOOM_MAXIMO,
                      contraste_maximo=CONTRASTE_MAXIMO):
    """Aumento de datos en línea sobre un lote de imágenes `(n, alto, ancho, 3)` en [0, 255], ya en la GPU.

    Cada imagen recibe, con valores distintos en cada época: un volteo horizontal (con probabilidad 1/2), un giro
    de hasta ±`giro_maximo` grados, un zoom de hasta ±`zoom_maximo` y un cambio de contraste de hasta
    ±`contraste_maximo`. Los bordes que deja el giro se rellenan reflejando la imagen. Es el mismo aumento que las
    capas `RandomFlip`, `RandomRotation`, `RandomZoom` y `RandomContrast` de Keras (material 3.2.2), pero escrito
    con `affine_grid` y `grid_sample` de PyTorch: con el backend de PyTorch esas capas tardan unos 440 ms por lote
    de 128 imágenes y esto, unos pocos. Va en el bucle de entrenamiento y no dentro del modelo, así que solo actúa
    al entrenar y el modelo guardado no cambia.

    Devuelve el lote en float32, en la misma escala [0, 255]: la capa `Rescaling` del modelo lo lleva a [0, 1]."""
    n = lote.shape[0]
    dispositivo = lote.device
    x = lote.permute(0, 3, 1, 2).float()  # (n, 3, alto, ancho)

    def uniforme(minimo, maximo):
        return minimo + (maximo - minimo) * torch.rand(n, generator=generador, device=dispositivo)

    volteo = torch.where(torch.rand(n, generator=generador, device=dispositivo) < 0.5, -1.0, 1.0)
    angulo = torch.deg2rad(uniforme(-giro_maximo, giro_maximo))
    escala = uniforme(1 - zoom_maximo, 1 + zoom_maximo)
    coseno, seno = torch.cos(angulo) / escala, torch.sin(angulo) / escala
    # Matriz afín de cada imagen: giro y zoom, con el volteo horizontal en la primera columna
    theta = torch.zeros(n, 2, 3, device=dispositivo)
    theta[:, 0, 0] = coseno * volteo
    theta[:, 0, 1] = -seno
    theta[:, 1, 0] = seno * volteo
    theta[:, 1, 1] = coseno
    rejilla = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, rejilla, mode="bilinear", padding_mode="reflection", align_corners=False)

    # Contraste: se aleja o se acerca cada píxel a la media de su imagen y canal
    factor = uniforme(1 - contraste_maximo, 1 + contraste_maximo).view(n, 1, 1, 1)
    media = x.mean(dim=(2, 3), keepdim=True)
    x = ((x - media) * factor + media).clamp(0, 255)
    return x.permute(0, 2, 3, 1).contiguous()


def tiene_regularizacion(modelo):
    """Si alguna capa del modelo tiene un `kernel_regularizer` (L2)."""
    return any(getattr(variable, "regularizer", None) is not None for variable in modelo.trainable_weights)


def penalizacion(modelo):
    """Suma de las penalizaciones de los regularizadores del modelo (λ · Σw² de cada kernel con L2).

    Keras la suma sola dentro de `fit()`, pero este bucle calcula la entropía cruzada a mano: sin sumar
    `modelo.losses`, un `kernel_regularizer` no tendría ningún efecto."""
    perdidas = modelo.losses
    if not perdidas:
        return None
    return torch.stack([torch.as_tensor(perdida).reshape(()) for perdida in perdidas]).sum()


def _pasada(modelo, X, y, batch, entrenando, optimizador=None, pesos=None, generador=None, suavizado=0.0,
            aumento=None):
    """Recorre el conjunto una vez y devuelve `(perdida, accuracy, top_3)`.

    Al entrenar suma la penalización L2 a la pérdida que se optimiza y, si se indica `aumento` (una función
    `(lote, generador) -> lote`), transforma cada lote antes de pasarlo por la red. La pérdida que devuelve es solo
    la entropía cruzada, para que las curvas de configuraciones con y sin L2 se puedan comparar. Los acumuladores
    son tensores en la GPU: convertirlos a float en cada lote obligaría a sincronizar con la CPU en cada paso."""
    modelo.train(entrenando)
    n = len(X)
    if entrenando:
        orden = torch.randperm(n, device=X.device, generator=generador)
    else:
        orden = torch.arange(n, device=X.device)
    regularizar = entrenando and tiene_regularizacion(modelo)

    suma_perdida = torch.zeros((), device=X.device)
    suma_top1 = torch.zeros((), device=X.device)
    suma_top3 = torch.zeros((), device=X.device)

    for i in range(0, n, batch):
        indices = orden[i:i + batch]
        lote_x, lote_y = X[indices], y[indices]
        objetivo = lote_y.argmax(1)
        if entrenando and aumento is not None:
            lote_x = aumento(lote_x, generador)
        if entrenando and suavizado:
            # Suavizado de etiquetas: solo al entrenar; validación mide con la etiqueta estricta
            lote_y = lote_y * (1 - suavizado) + suavizado / lote_y.shape[1]

        with torch.set_grad_enabled(entrenando):
            probabilidades = modelo(lote_x, training=entrenando)
            # Entropía cruzada categórica, muestra a muestra
            perdida_muestra = -(lote_y * torch.log(probabilidades.clamp_min(1e-7))).sum(1)
            if pesos is not None:
                # class_weight: cada ejemplo pesa según su clase real
                perdida_muestra = perdida_muestra * pesos[objetivo]
            perdida = perdida_muestra.mean()
            if regularizar:
                perdida = perdida + penalizacion(modelo)

        if entrenando:
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()

        with torch.no_grad():
            suma_perdida += perdida_muestra.sum()
            suma_top1 += (probabilidades.argmax(1) == objetivo).sum()
            suma_top3 += (probabilidades.topk(3, dim=1).indices == objetivo.unsqueeze(1)).any(1).sum()

    return (float(suma_perdida) / n, float(suma_top1) / n, float(suma_top3) / n)


def entrenar(
    modelo,
    X_entrena,
    y_entrena,
    X_valida,
    y_valida,
    epocas=60,
    batch=128,
    tasa_aprendizaje=1e-3,
    pesos_por_clase=None,
    suavizado=0.0,
    paciencia=12,
    factor_lr=0.5,
    paciencia_lr=4,
    lr_minimo=1e-5,
    ruta_modelo=None,
    semilla=42,
    aumento=None,
    verbose=2,
):
    """Entrena el modelo y devuelve un `Historial` equivalente al de `model.fit()`.

    - **EarlyStopping** sobre `val_loss` con `restore_best_weights`: si la pérdida de validación no mejora en
      `paciencia` épocas, se detiene y vuelve a los pesos de la mejor época.
    - **ReduceLROnPlateau**: tras `paciencia_lr` épocas sin mejora, multiplica la tasa por `factor_lr`.
    - **ModelCheckpoint**: si se indica `ruta_modelo`, guarda ahí el modelo de la mejor época.
    - **Aumento en línea**: si se indica `aumento` (por ejemplo, `aumentar_imagenes`), cada lote de entrenamiento se
      transforma antes de pasar por la red; validación ve las imágenes intactas.

    El historial registra la mejor época y el motivo de la parada: `paciencia` si la cortó la parada temprana, o
    `tope` si llegó a `epocas` y tal vez seguía mejorando."""
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje, fused=X_entrena.device.type == "cuda")

    generador = torch.Generator(device=X_entrena.device)
    generador.manual_seed(semilla)

    pesos = None
    if pesos_por_clase is not None:
        pesos = torch.tensor(
            [pesos_por_clase[i] for i in range(len(pesos_por_clase))],
            dtype=torch.float32,
            device=X_entrena.device,
        )

    historial = Historial(epocas_maximas=epocas)
    mejor_perdida = float("inf")
    mejor_epoca = 0
    sin_mejora_parada = 0
    sin_mejora_lr = 0
    mejores_pesos = None
    historial.motivo_parada = PARADA_TOPE
    inicio_total = time.perf_counter()

    for epoca in range(1, epocas + 1):
        inicio = time.perf_counter()
        perdida, acierto, top3 = _pasada(
            modelo, X_entrena, y_entrena, batch, True, optimizador, pesos, generador, suavizado, aumento
        )
        val_perdida, val_acierto, val_top3 = _pasada(modelo, X_valida, y_valida, batch, False)
        duracion = time.perf_counter() - inicio

        historial.anotar(
            loss=perdida, accuracy=acierto, top_3=top3,
            val_loss=val_perdida, val_accuracy=val_acierto, val_top_3=val_top3,
            learning_rate=optimizador.param_groups[0]["lr"],
        )

        if val_perdida < mejor_perdida - 1e-6:
            mejor_perdida, mejor_epoca = val_perdida, epoca
            sin_mejora_parada = sin_mejora_lr = 0
            # Los mejores pesos se guardan en la RAM del equipo: una copia de los parámetros en la VRAM
            # competiría con el propio entrenamiento.
            mejores_pesos = {nombre: tensor.detach().cpu().clone() for nombre, tensor in modelo.state_dict().items()}
        else:
            sin_mejora_parada += 1
            sin_mejora_lr += 1

        if verbose >= 2:
            print(f"Epoca {epoca:3d}/{epocas}  {duracion:5.2f}s"
                  f"  loss={perdida:.4f}  acc={acierto:.4f}"
                  f"  val_loss={val_perdida:.4f}  val_acc={val_acierto:.4f}")

        # ReduceLROnPlateau
        if sin_mejora_lr >= paciencia_lr:
            lr_actual = optimizador.param_groups[0]["lr"]
            lr_nuevo = max(lr_actual * factor_lr, lr_minimo)
            if lr_nuevo < lr_actual:
                for grupo in optimizador.param_groups:
                    grupo["lr"] = lr_nuevo
                if verbose:
                    print(f"  ReduceLROnPlateau: learning rate {lr_actual:.2e} -> {lr_nuevo:.2e}")
            sin_mejora_lr = 0

        # EarlyStopping
        if sin_mejora_parada >= paciencia:
            historial.motivo_parada = PARADA_PACIENCIA
            if verbose:
                print(f"  EarlyStopping en la epoca {epoca}: val_loss no mejora desde la epoca {mejor_epoca}")
            break

    # restore_best_weights: se vuelve a la mejor época antes de devolver el modelo
    if mejores_pesos is not None:
        modelo.load_state_dict({nombre: tensor.to(X_entrena.device) for nombre, tensor in mejores_pesos.items()})
    historial.mejor_epoca = mejor_epoca
    historial.segundos = time.perf_counter() - inicio_total

    # Equivalente a ModelCheckpoint(save_best_only=True): el archivo que queda en disco es el de la mejor época. Se
    # escribe una sola vez, al final: el contenido es idéntico y no se paga la escritura en mitad de la medición.
    if ruta_modelo is not None:
        modelo.save(ruta_modelo)

    return historial


# --- Historiales en disco ----------------------------------------------------------------------------------


def _a_json(valor):
    """Convierte los tipos de NumPy a tipos de Python para guardarlos en JSON."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return None if np.isnan(valor) else float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, (tuple, np.ndarray)):
        return list(valor)
    if isinstance(valor, Path):
        return str(valor)
    raise TypeError(f"No se puede guardar en JSON: {type(valor)}")


def guardar_historial(directorio, identificador, historial, **metadatos):
    """Guarda el historial de una configuración en `<directorio>/<identificador>.json`, con su mejor época, su
    motivo de parada y los metadatos que se indiquen (por ejemplo, los factores y el resultado). Así las curvas se
    pueden volver a dibujar sin entrenar de nuevo."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    registro = historial.history if hasattr(historial, "history") else historial
    contenido = {
        "identificador": identificador,
        "mejor_epoca": getattr(historial, "mejor_epoca", None),
        "motivo_parada": getattr(historial, "motivo_parada", None),
        "epocas_maximas": getattr(historial, "epocas_maximas", None),
        "segundos": getattr(historial, "segundos", None),
        **metadatos,
        "history": registro,
    }
    ruta = directorio / f"{identificador}.json"
    ruta.write_text(json.dumps(contenido, indent=1, ensure_ascii=False, default=_a_json), encoding="utf-8")
    return ruta


def cargar_historiales(directorio):
    """Lee los historiales guardados con `guardar_historial`: `{identificador: contenido}`."""
    return {
        ruta.stem: json.loads(ruta.read_text(encoding="utf-8"))
        for ruta in sorted(Path(directorio).glob("*.json"))
    }


# --- Curvas de aprendizaje ---------------------------------------------------------------------------------


def _registro(historial):
    """El diccionario `{metrica: [valores]}` de un Historial, de un `fit()` o de un historial ya leído."""
    if hasattr(historial, "history"):
        return historial.history
    return historial.get("history", historial)


def graficar_curvas(historial, titulo="Modelo base"):
    """Dibuja pérdida y accuracy de entrenamiento y validación por época, con la mejor época marcada."""
    registro = _registro(historial)
    epocas = range(1, len(registro["loss"]) + 1)
    epoca_optima = int(np.argmin(registro["val_loss"])) + 1

    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].plot(epocas, registro["loss"], label="entrenamiento", linewidth=2)
    ejes[0].plot(epocas, registro["val_loss"], label="validacion", linewidth=2)
    ejes[0].set_ylabel("Pérdida (categorical crossentropy)")
    ejes[0].set_title("Pérdida por época", fontsize=10)
    ejes[1].plot(epocas, registro["accuracy"], label="entrenamiento", linewidth=2)
    ejes[1].plot(epocas, registro["val_accuracy"], label="validacion", linewidth=2)
    ejes[1].set_ylabel("Accuracy")
    ejes[1].set_title("Accuracy por época", fontsize=10)
    for eje in ejes:
        eje.axvline(epoca_optima, color="gray", linestyle="--", linewidth=1, label=f"mejor epoca = {epoca_optima}")
        eje.set_xlabel("Epoca")
        eje.grid(alpha=0.3)
        eje.legend()
    figura.suptitle(f"Curvas de aprendizaje - {titulo}")
    plt.tight_layout()
    plt.show()


def evaluar_aprendizaje(historiales, referencia, configuraciones, titulos=None, titulo=None):
    """Curvas de aprendizaje de varias configuraciones, cada una frente a la de referencia.

    Una fila por configuración: a la izquierda la pérdida y a la derecha la accuracy, de entrenamiento (línea
    punteada) y de validación (línea continua). La referencia (normalmente el modelo base) va en gris en todos los
    paneles, y un punto marca la mejor época de cada una. Así se ve, técnica por técnica, si cambia la velocidad
    de convergencia, el nivel que alcanza la validación o la brecha entre las dos curvas.

    `historiales` es `{nombre: historial}`; `titulos` permite poner un rótulo corto a cada configuración."""
    configuraciones = [nombre for nombre in configuraciones if nombre in historiales]
    if not configuraciones:
        print("No hay historiales para graficar.")
        return
    titulos = titulos or {}
    base = _registro(historiales[referencia])
    figura, ejes = plt.subplots(len(configuraciones), 2, figsize=(12, 3.0 * len(configuraciones)), squeeze=False)

    for fila, nombre in enumerate(configuraciones):
        registro = _registro(historiales[nombre])
        for columna, (metrica, etiqueta) in enumerate((("loss", "Pérdida"), ("accuracy", "Accuracy"))):
            eje = ejes[fila, columna]
            for datos_curva, color, rotulo in ((base, "#888888", "base"), (registro, "#d95f02", "variante")):
                epocas = np.arange(1, len(datos_curva[metrica]) + 1)
                eje.plot(epocas, datos_curva[metrica], linestyle="--", linewidth=1.2, color=color,
                         label=f"{rotulo}: train")
                eje.plot(epocas, datos_curva[f"val_{metrica}"], linewidth=2, color=color, label=f"{rotulo}: val")
                mejor = int(np.argmin(datos_curva["val_loss"]))
                eje.scatter([mejor + 1], [datos_curva[f"val_{metrica}"][mejor]], color=color, zorder=3, s=25)
            eje.set_ylabel(etiqueta)
            eje.grid(alpha=0.3)
            if fila == len(configuraciones) - 1:
                eje.set_xlabel("Época")
        ejes[fila, 0].set_title(titulos.get(nombre, nombre), fontsize=10, loc="left")
        if fila == 0:
            ejes[fila, 1].legend(fontsize=7, ncol=2)

    figura.suptitle(titulo or f"Curvas de aprendizaje frente a {titulos.get(referencia, referencia)}",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def diagnosticar_entrenamiento(historial, f1_macro_val, baseline_azar, baseline_mayoritaria):
    """Indicadores de la mejor época de un entrenamiento (pérdidas, accuracy, top-3, F1 macro y brecha entre
    entrenamiento y validación) y una lectura de la brecha. Los baselines van en porcentaje."""
    registro = _registro(historial)
    indice_optimo = int(np.argmin(registro["val_loss"]))
    brecha = registro["accuracy"][indice_optimo] - registro["val_accuracy"][indice_optimo]
    diagnostico = pd.DataFrame(
        [
            ("Épocas ejecutadas", len(registro["loss"])),
            ("Mejor época (val_loss)", indice_optimo + 1),
            ("Motivo de la parada", getattr(historial, "motivo_parada", None) or "-"),
            ("Pérdida entrenamiento", round(registro["loss"][indice_optimo], 4)),
            ("Pérdida validación", round(registro["val_loss"][indice_optimo], 4)),
            ("Accuracy entrenamiento", round(registro["accuracy"][indice_optimo], 4)),
            ("Accuracy validación", round(registro["val_accuracy"][indice_optimo], 4)),
            ("Top-3 accuracy validación", round(registro["val_top_3"][indice_optimo], 4)),
            ("F1 macro validación", round(float(f1_macro_val), 4)),
            ("Brecha train - val (accuracy)", round(brecha, 4)),
        ],
        columns=["Indicador", "Valor"],
    )
    display(diagnostico)

    if brecha > 0.15:
        lectura = "sobreajuste claro: la red memoriza el entrenamiento y generaliza bastante peor"
    elif brecha > 0.05:
        lectura = "sobreajuste moderado, contenido por el Dropout y la parada temprana"
    else:
        lectura = "sin sobreajuste apreciable; las dos curvas evolucionan juntas"
    print(f"Brecha train - val en accuracy: {brecha:.3f} -> {lectura}.")
    print(
        f"Accuracy de validacion ({registro['val_accuracy'][indice_optimo]:.3f}) frente a los baselines:"
        f" azar {baseline_azar / 100:.3f} y clase mayoritaria {baseline_mayoritaria / 100:.3f}."
    )
    return indice_optimo, diagnostico


def efecto_parada_temprana(historial):
    """Qué habría pasado sin parada temprana, sin volver a entrenar: la mejor época (la que restaura
    `restore_best_weights`) frente a la última que se ejecutó. Si la última tiene más pérdida de validación y más
    brecha, la parada temprana evitó usar un modelo sobreajustado."""
    registro = _registro(historial)
    mejor = int(np.argmin(registro["val_loss"]))
    ultima = len(registro["val_loss"]) - 1
    filas = []
    for etiqueta, indice in (("Mejor época (modelo que se usa)", mejor), ("Última época (sin parada temprana)", ultima)):
        filas.append(
            {
                "modelo": etiqueta,
                "epoca": indice + 1,
                "val_loss": round(registro["val_loss"][indice], 4),
                "val_accuracy": round(registro["val_accuracy"][indice], 4),
                "accuracy_train": round(registro["accuracy"][indice], 4),
                "brecha_train_val": round(registro["accuracy"][indice] - registro["val_accuracy"][indice], 4),
            }
        )
    return pd.DataFrame(filas)
