"""Entrena un MLP de píxeles robusto (R1 robusto) para que sus recreaciones se parezcan a los personajes.

El MLP de píxeles del notebook (R1) aprende «color en posición fija»: el gradiente de su salida respecto de los
píxeles apunta a cambios mínimos y repartidos por toda la imagen, que una CNN no percibe. Por eso lo que recrea
DeepInversion desde él la CNN lo trata como ruido. Un clasificador entrenado para resistir esos cambios mínimos
(entrenamiento adversarial, Madry et al., 2018) tiene gradientes que se alinean con lo que se percibe
(Santurkar et al., 2019; Engstrom et al., 2019): para subir su clase hay que cambiar la imagen de forma visible.

La red es la misma de R1 (Rescaling, 512 y 256 neuronas con ReLU, BatchNorm y dropout 0,3, salida softmax) y se
entrena igual que en el notebook (Adam 1e-3, suavizado de etiquetas 0,1, pesos por clase en raíz, parada temprana
con la val_loss y restauración de la mejor época), con dos cambios en cada lote de entrenamiento:
    1. Aumentos: desplazamiento de hasta `desplazamiento` píxeles (sin envolver los bordes), volteo horizontal y
       ruido gaussiano.
    2. Ataque PGD L2 con radio ε (en la escala [0, 1] de la imagen): la red aprende con la versión del lote que más
       la confunde dentro de ese radio. Con ε = 0 solo hay aumentos.

Datos. Las mismas imágenes y la misma partición que el notebook: las originales de `mlp_final_particion.csv`, cada
una con su versión __parecida si la tiene, más sus __alteradas en el mismo conjunto. Se cargan a 128 × 128 con
interpolación bilineal, como en el notebook. El notebook también puede llamar a `entrenar_robusto` con las matrices
que ya tiene en la VRAM.

Salidas, en PMC/models_gpu/newDataset/ (ignorada por Git):
    mlp_r1_robusto_e<ε>.keras            un modelo por cada ε probado
    mlp_r1_robusto.keras                 copia del elegido (--elegido), el que usa recrear_personajes.py
    mlp_r1_robusto_entrenamiento.json    por ε: épocas, accuracy de validación y test (limpias) y robusta, tiempo;
                                         y R1 medido igual, como referencia

Uso, desde la raíz del repositorio (entorno .venv-gpu):
    .venv-gpu/Scripts/python.exe scripts/newDataset/mlp_robusto.py --eps 0,0.5,1,2,4
    .venv-gpu/Scripts/python.exe scripts/newDataset/mlp_robusto.py --elegido 2

Pruebas (sin GPU): .venv-gpu/Scripts/python.exe -m pytest scripts/newDataset/tests/test_mlp_robusto.py -q
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[1]
MODELOS_MLP = RAIZ_REPO / "PMC" / "models_gpu" / "newDataset"
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARCA_ALTERADA = "__alterada"
MARCA_PARECIDA = "__parecida"
SEMILLA = 42
# Otra semilla que la de R1 (42): así la correlación de pesos entre los dos MLP mide si son independientes
SEMILLA_ROBUSTO = SEMILLA + 1


# --- Datos -----------------------------------------------------------------------------------------------

def indice_dataset(dataset: Path, particion: Path) -> list:
    """[(ruta, personaje, conjunto)] de las imágenes de la partición del notebook: cada original (o su versión
    __parecida) y sus __alteradas, en el conjunto de su original. Las imágenes que no están en la partición (agregadas
    después de la corrida) no entran."""
    import csv

    with open(particion, encoding="utf-8", newline="") as archivo:
        conjunto_de = {(f["personaje"], f["archivo"]): f["conjunto"] for f in csv.DictReader(archivo)}
    filas = []
    for carpeta in sorted(p for p in dataset.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        archivos = sorted(carpeta.iterdir())
        nombres = {ruta.name for ruta in archivos}
        for ruta in archivos:
            if ruta.suffix.lower() not in EXTENSIONES_VALIDAS or ruta.stem.endswith(MARCA_PARECIDA):
                continue
            alterada = ruta.stem.endswith(MARCA_ALTERADA)
            origen = ruta.stem.removesuffix(MARCA_ALTERADA) + ruta.suffix if alterada else ruta.name
            conjunto = conjunto_de.get((carpeta.name, origen))
            if conjunto is None:
                continue
            reemplazo = f"{ruta.stem}{MARCA_PARECIDA}{ruta.suffix}"
            if not alterada and reemplazo in nombres:
                ruta = ruta.with_name(reemplazo)
            filas.append((ruta, carpeta.name, conjunto))
    return filas


def cargar_pixeles(dataset: Path, particion: Path, clases: list, alto: int = 128, ancho: int = 128) -> dict:
    """{conjunto: (X uint8 (n, alto·ancho·3), y índices)} con la imagen completa redimensionada como en el notebook."""
    from joblib import Parallel, delayed
    from PIL import Image

    def cargar(ruta):
        with Image.open(ruta) as imagen:
            return np.asarray(imagen.convert("RGB").resize((ancho, alto), Image.Resampling.BILINEAR), dtype=np.uint8)

    filas = indice_dataset(dataset, particion)
    indice_de = {clase: i for i, clase in enumerate(clases)}
    imagenes = np.stack(Parallel(n_jobs=-1, prefer="threads")(delayed(cargar)(ruta) for ruta, _, _ in filas))
    etiquetas = np.array([indice_de[personaje] for _, personaje, _ in filas])
    conjuntos = np.array([conjunto for _, _, conjunto in filas])
    return {nombre: (imagenes[conjuntos == nombre].reshape(int((conjuntos == nombre).sum()), -1),
                     etiquetas[conjuntos == nombre]) for nombre in ("train", "val", "test")}


def pesos_raiz(y_train: np.ndarray, n_clases: int) -> np.ndarray:
    """Los pesos por clase del notebook: raíz del balanceado, reescalada para que el peso medio por imagen sea 1."""
    conteos = np.bincount(y_train, minlength=n_clases).astype(float)
    balanceados = len(y_train) / (n_clases * np.maximum(conteos, 1))
    raiz = np.sqrt(balanceados)
    return raiz * len(y_train) / (raiz * conteos).sum()


# --- Modelo ----------------------------------------------------------------------------------------------

def construir_mlp_pixeles(alto: int, ancho: int, n_clases: int, neuronas=(512, 256), dropout: float = 0.3,
                          nombre: str = "mlp_r1_robusto"):
    """La misma red que construir_mlp(representacion="pixeles") del notebook."""
    import keras
    from keras import layers

    modelo = keras.Sequential(name=nombre)
    modelo.add(layers.Input(shape=(alto * ancho * 3,), name="entrada"))
    modelo.add(layers.Rescaling(1.0 / 255, name="escalado"))
    for posicion, n in enumerate(neuronas, start=1):
        modelo.add(layers.Dense(n, activation="relu", name=f"oculta_{posicion}"))
        modelo.add(layers.BatchNormalization(name=f"norm_{posicion}"))
        modelo.add(layers.Dropout(dropout, name=f"dropout_{posicion}"))
    modelo.add(layers.Dense(n_clases, activation="softmax", name="salida"))
    return modelo


# --- Aumentos y ataque -----------------------------------------------------------------------------------

def aumentar(lote, alto: int, ancho: int, desplazamiento: int = 8, ruido: float = 0.0, generador=None):
    """Un lote aplanado (n, alto·ancho·3) en [0, 255] (uint8 o float) con, por imagen: un desplazamiento entero de
    hasta `desplazamiento` píxeles en cada eje (los bordes se repiten, no se envuelven), un volteo horizontal al azar
    y ruido gaussiano de desviación `ruido` (en la escala [0, 1]). Devuelve float32 aplanado en [0, 255]."""
    import torch
    import torch.nn.functional as F

    n = lote.shape[0]
    dispositivo = lote.device
    x = lote.float().reshape(n, alto, ancho, 3).permute(0, 3, 1, 2)
    voltear = torch.rand(n, generator=generador, device=dispositivo) < 0.5
    x = torch.where(voltear[:, None, None, None], x.flip(3), x)
    if desplazamiento:
        dx = torch.randint(-desplazamiento, desplazamiento + 1, (n,), generator=generador, device=dispositivo)
        dy = torch.randint(-desplazamiento, desplazamiento + 1, (n,), generator=generador, device=dispositivo)
        theta = torch.zeros(n, 2, 3, device=dispositivo)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        # En coordenadas normalizadas un píxel mide 2/ancho: con align_corners=False el muestreo cae justo en
        # píxeles enteros y la interpolación no mezcla vecinos
        theta[:, 0, 2] = 2.0 * dx.float() / ancho
        theta[:, 1, 2] = 2.0 * dy.float() / alto
        malla = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, malla, mode="bilinear", padding_mode="border", align_corners=False)
    if ruido:
        x = (x + ruido * 255.0 * torch.randn(x.shape, generator=generador, device=dispositivo)).clamp(0, 255)
    return x.permute(0, 2, 3, 1).reshape(n, -1)


def _log_probabilidades(modelo, x, training: bool):
    import torch

    return torch.log(modelo(x, training=training).clamp_min(1e-7))


def pgd_l2(modelo, x, objetivo, eps: float, pasos: int = 3, tamano_paso: float | None = None):
    """El lote x (aplanado, en [0, 255]) movido dentro de una bola L2 de radio eps (en la escala [0, 1]) hacia donde
    más sube la entropía cruzada de la clase objetivo. Con training=False: BatchNorm usa sus estadísticas guardadas y
    no las cambia, y el dropout no actúa. Los pesos no acumulan gradiente."""
    import torch

    if eps <= 0 or pasos <= 0:
        return x.detach()
    radio = eps * 255.0
    paso = tamano_paso * 255.0 if tamano_paso else 2.5 * radio / pasos
    x = x.detach()
    delta = torch.zeros_like(x)
    for _ in range(pasos):
        delta.requires_grad_(True)
        perdida = torch.nn.functional.nll_loss(_log_probabilidades(modelo, x + delta, False), objetivo)
        (gradiente,) = torch.autograd.grad(perdida, delta)
        with torch.no_grad():
            delta = delta + paso * gradiente / gradiente.norm(dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (radio / delta.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
            delta = (x + delta).clamp(0, 255) - x
    return (x + delta).detach()


# --- Entrenamiento ---------------------------------------------------------------------------------------

def _indices(y):
    return y.argmax(1) if y.ndim == 2 else y.long()


def evaluar(modelo, X, y, batch: int = 512, eps: float = 0.0, pasos: int = 10, maximo: int | None = None) -> dict:
    """Pérdida y accuracy de un conjunto, limpias o (con eps) bajo PGD de `pasos` pasos; `maximo` limita las filas."""
    import torch

    objetivo = _indices(y)
    n = len(X) if maximo is None else min(maximo, len(X))
    suma_perdida = torch.zeros((), device=X.device)
    aciertos = torch.zeros((), device=X.device)
    modelo.train(False)
    for inicio in range(0, n, batch):
        lote_x, lote_y = X[inicio:min(n, inicio + batch)].float(), objetivo[inicio:min(n, inicio + batch)]
        if eps > 0:
            lote_x = pgd_l2(modelo, lote_x, lote_y, eps, pasos)
        with torch.no_grad():
            log_p = _log_probabilidades(modelo, lote_x, False)
            suma_perdida += torch.nn.functional.nll_loss(log_p, lote_y, reduction="sum")
            aciertos += (log_p.argmax(1) == lote_y).sum()
    return {"perdida": float(suma_perdida) / n, "accuracy": float(aciertos) / n}


def entrenar_robusto(modelo, X_entrena, y_entrena, X_valida, y_valida, alto: int, ancho: int, eps: float,
                     pasos_pgd: int = 3, desplazamiento: int = 8, ruido: float = 0.02, epocas: int = 35,
                     batch: int = 128, tasa_aprendizaje: float = 1e-3, pesos_por_clase=None, suavizado: float = 0.1,
                     paciencia: int = 9, factor_lr: float = 0.5, paciencia_lr: int = 4, lr_minimo: float = 1e-5,
                     semilla: int = SEMILLA_ROBUSTO, ruta_modelo: Path | None = None, verbose: int = 1) -> dict:
    """Entrena con aumentos + PGD y devuelve el historial {métrica: [valor por época]}.

    El mismo esquema que entrenar() del notebook: Adam, suavizado de etiquetas solo al entrenar, pesos por clase,
    ReduceLROnPlateau y parada temprana con la val_loss limpia, y al final los pesos de la mejor época. Las y pueden
    ser one-hot (las del notebook) o índices.
    """
    import torch

    n_clases = modelo.output_shape[-1]
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    generador = torch.Generator(device=X_entrena.device)
    generador.manual_seed(int(semilla))
    pesos = None if pesos_por_clase is None else torch.as_tensor(
        np.asarray([pesos_por_clase[i] for i in range(n_clases)], dtype=np.float32), device=X_entrena.device)
    objetivo_entrena = _indices(y_entrena)
    historial = {}
    mejor, mejor_epoca, sin_mejora, sin_mejora_lr, mejores_pesos = float("inf"), 0, 0, 0, None

    for epoca in range(1, epocas + 1):
        inicio = time.perf_counter()
        orden = torch.randperm(len(X_entrena), device=X_entrena.device, generator=generador)
        suma, aciertos = torch.zeros((), device=X_entrena.device), torch.zeros((), device=X_entrena.device)
        for i in range(0, len(orden), batch):
            indices = orden[i:i + batch]
            objetivo = objetivo_entrena[indices]
            lote = aumentar(X_entrena[indices], alto, ancho, desplazamiento, ruido, generador)
            lote = pgd_l2(modelo, lote, objetivo, eps, pasos_pgd)
            modelo.train(True)
            log_p = _log_probabilidades(modelo, lote, True)
            blanco = torch.nn.functional.one_hot(objetivo, n_clases).float()
            blanco = blanco * (1 - suavizado) + suavizado / n_clases
            perdida_muestra = -(blanco * log_p).sum(1)
            if pesos is not None:
                perdida_muestra = perdida_muestra * pesos[objetivo]
            optimizador.zero_grad(set_to_none=True)
            perdida_muestra.mean().backward()
            optimizador.step()
            with torch.no_grad():
                suma += perdida_muestra.sum()
                aciertos += (log_p.argmax(1) == objetivo).sum()
        validacion = evaluar(modelo, X_valida, y_valida)
        for clave, valor in (("loss", float(suma) / len(orden)), ("accuracy", float(aciertos) / len(orden)),
                             ("val_loss", validacion["perdida"]), ("val_accuracy", validacion["accuracy"]),
                             ("learning_rate", optimizador.param_groups[0]["lr"]),
                             ("segundos", time.perf_counter() - inicio)):
            historial.setdefault(clave, []).append(valor)
        if verbose:
            print(f"  ε {eps:<4} época {epoca:>2}/{epocas}  {historial['segundos'][-1]:5.1f}s  "
                  f"loss {historial['loss'][-1]:.3f}  acc {historial['accuracy'][-1]:.3f}  "
                  f"val_loss {validacion['perdida']:.3f}  val_acc {validacion['accuracy']:.3f}")
        if validacion["perdida"] < mejor - 1e-6:
            mejor, mejor_epoca, sin_mejora, sin_mejora_lr = validacion["perdida"], epoca, 0, 0
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            sin_mejora_lr += 1
        if sin_mejora_lr >= paciencia_lr:
            for grupo in optimizador.param_groups:
                grupo["lr"] = max(grupo["lr"] * factor_lr, lr_minimo)
            sin_mejora_lr = 0
        if sin_mejora >= paciencia:
            if verbose:
                print(f"  parada temprana en la época {epoca}; la mejor fue la {mejor_epoca}")
            break
    if mejores_pesos is not None:
        modelo.load_state_dict({k: v.to(X_entrena.device) for k, v in mejores_pesos.items()})
    historial["mejor_epoca"] = mejor_epoca
    if ruta_modelo is not None:
        modelo.save(ruta_modelo)
    return historial


# --- Programa --------------------------------------------------------------------------------------------

RUTA_RESUMEN = MODELOS_MLP / "mlp_r1_robusto_entrenamiento.json"
# La robustez de todos los modelos se mide igual, para compararlos: PGD de 10 pasos, radio 1 y 2048 imágenes de
# validación
EPS_MEDICION, PASOS_MEDICION, MAXIMO_MEDICION = 1.0, 10, 2048


def _nombre_eps(eps: float) -> str:
    return f"{eps:g}".replace(".", "_")


def ruta_de(eps: float) -> Path:
    return MODELOS_MLP / f"mlp_r1_robusto_e{_nombre_eps(eps)}.keras"


def leer_resumen() -> dict:
    return json.loads(RUTA_RESUMEN.read_text(encoding="utf-8")) if RUTA_RESUMEN.is_file() else {}


def guardar_resumen(resumen: dict) -> None:
    RUTA_RESUMEN.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")


def medir(modelo, datos: dict) -> dict:
    """Accuracy limpia de validación y de test, y robusta de validación, con `datos` = {conjunto: (X, y)} en la GPU."""
    return {
        "val_accuracy": round(evaluar(modelo, *datos["val"])["accuracy"], 4),
        "test_accuracy": round(evaluar(modelo, *datos["test"])["accuracy"], 4),
        "val_accuracy_robusta": round(evaluar(modelo, *datos["val"], eps=EPS_MEDICION, pasos=PASOS_MEDICION,
                                              maximo=MAXIMO_MEDICION)["accuracy"], 4),
    }


def medir_r1(datos: dict, dispositivo: str) -> dict:
    """R1, el MLP de píxeles del notebook, medido con la misma vara que los robustos."""
    import keras

    ruta = MODELOS_MLP / "mlp_r1.keras"
    if not ruta.is_file():
        return {}
    modelo = keras.models.load_model(ruta, compile=False).to(dispositivo)
    return {"archivo": ruta.name, **medir(modelo, datos)}


def entrenar_radio(eps: float, construir, datos: dict, alto: int, ancho: int, pesos_por_clase=None,
                   **opciones) -> dict:
    """Entrena un MLP robusto con radio eps (construir() da la red sin entrenar), lo guarda en ruta_de(eps), lo mide
    y lo anota en mlp_r1_robusto_entrenamiento.json. `datos` = {"train"|"val"|"test": (X, y)} ya en la GPU; las y
    pueden ser one-hot o índices. Lo usan este script y el notebook."""
    import keras

    keras.utils.set_random_seed(SEMILLA_ROBUSTO)
    modelo = construir()
    inicio = time.perf_counter()
    historial = entrenar_robusto(modelo, *datos["train"], *datos["val"], alto, ancho, eps,
                                 pesos_por_clase=pesos_por_clase, ruta_modelo=ruta_de(eps), **opciones)
    segundos = time.perf_counter() - inicio
    registro = {
        "eps": eps, "archivo": ruta_de(eps).name, "semilla": SEMILLA_ROBUSTO,
        **{clave: opciones[clave] for clave in ("pasos_pgd", "desplazamiento", "ruido", "epocas") if clave in opciones},
        "epocas_ejecutadas": len(historial["loss"]), "mejor_epoca": historial["mejor_epoca"],
        "segundos": round(segundos, 1), **medir(modelo, datos),
        "historial": {k: [round(float(v), 5) for v in vs] for k, vs in historial.items() if isinstance(vs, list)},
    }
    resumen = leer_resumen()
    resumen.setdefault("medicion", {"eps": EPS_MEDICION, "pasos": PASOS_MEDICION, "imagenes": MAXIMO_MEDICION})
    resumen.setdefault("radios", {})[_nombre_eps(eps)] = registro
    guardar_resumen(resumen)
    print(f"ε {eps:g}: val {registro['val_accuracy']:.3f}, test {registro['test_accuracy']:.3f}, "
          f"val robusta {registro['val_accuracy_robusta']:.3f}, {segundos:.0f} s")
    return registro


def elegir(eps: float) -> Path:
    """Copia el modelo de radio eps a mlp_r1_robusto.keras, el que usa recrear_personajes.py, y lo anota."""
    origen = ruta_de(eps)
    if not origen.is_file():
        raise FileNotFoundError(f"No existe {origen}: entrénalo antes con --eps {eps:g}")
    destino = MODELOS_MLP / "mlp_r1_robusto.keras"
    shutil.copyfile(origen, destino)
    resumen = leer_resumen()
    resumen["elegido"] = _nombre_eps(eps)
    guardar_resumen(resumen)
    return destino


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eps", default="", help="radios a entrenar, separados por comas (0,0.5,1,2,4)")
    parser.add_argument("--elegido", type=float, help="ε cuyo modelo se copia a mlp_r1_robusto.keras")
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("SIMPSONS_DATASET", Path.home() / "Desktop" / "simpsons_dataset")))
    parser.add_argument("--epocas", type=int, default=35)
    parser.add_argument("--pasos-pgd", type=int, default=3)
    parser.add_argument("--desplazamiento", type=int, default=8)
    parser.add_argument("--ruido", type=float, default=0.02)
    args = parser.parse_args(argv)
    radios = [float(e) for e in args.eps.split(",") if e.strip()]

    if radios:
        import torch

        metadatos = json.loads((MODELOS_MLP / "mlp_final.json").read_text(encoding="utf-8"))
        clases, alto, ancho = metadatos["clases"], int(metadatos["alto"]), int(metadatos["ancho"])
        inicio = time.perf_counter()
        datos = cargar_pixeles(args.dataset, MODELOS_MLP / "mlp_final_particion.csv", clases, alto, ancho)
        print(f"Datos en {time.perf_counter() - inicio:.0f} s: "
              + ", ".join(f"{k} {len(v[1])}" for k, v in datos.items()))
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        en_gpu = {k: (torch.as_tensor(X).to(dispositivo), torch.as_tensor(y).to(dispositivo))
                  for k, (X, y) in datos.items()}
        pesos = pesos_raiz(datos["train"][1], len(clases))
        resumen = leer_resumen()
        resumen["r1"] = medir_r1(en_gpu, dispositivo)
        guardar_resumen(resumen)
        print(f"R1: {resumen['r1']}")
        for eps in radios:
            entrenar_radio(eps, lambda: construir_mlp_pixeles(alto, ancho, len(clases)).to(dispositivo), en_gpu,
                           alto, ancho, pesos, pasos_pgd=args.pasos_pgd, desplazamiento=args.desplazamiento,
                           ruido=args.ruido, epocas=args.epocas)
            torch.cuda.empty_cache()

    if args.elegido is not None:
        print(f"{elegir(args.elegido).name} <- {ruta_de(args.elegido).name}")
    return leer_resumen()


if __name__ == "__main__":
    main()
