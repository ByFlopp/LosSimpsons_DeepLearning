"""Separa con la CNN las imágenes de por_revisar/ en seguras, dudosas y ruido (`pnpm frinkiac clasificar`).

Corre en el entorno .venv-gpu, el de la CNN (Keras 3 sobre PyTorch). Usa el modelo final que guarda el
notebook (CNN/models_gpu/newDataset/cnn_final.keras) y lo carga igual que CNN/scripts/newDataset/
predecir_imagen.py.

Muchos fotogramas de Frinkiac no muestran al personaje: lo bajamos porque habla, pero puede estar fuera
de plano. Cada fotograma se mira en 4 vistas: completo y 3 recortes cuadrados (izquierda, centro y
derecha), porque en un plano ancho el personaje suele ocupar solo una parte. Se toma la probabilidad que
la CNN da al personaje de la carpeta en el fotograma completo y la mayor entre los recortes:

    segura   completo >= 0,80  o  algún recorte >= 0,90
    ruido    menos de 0,05 en las 4 vistas
    dudosa   el resto

Las imágenes quedan en clasificacion/<grupo>/<personaje>/ (ver clasificacion.py). Solo hay que revisar a
mano dudosas/. Cada ejecución clasifica únicamente lo nuevo, así que se puede repetir sin problema.

Uso, desde la raíz del repositorio:
    pnpm frinkiac clasificar
    pnpm frinkiac clasificar --personajes louie,lou
    .venv-gpu/Scripts/python.exe scripts/newDataset/frinkiac/clasificar_con_cnn.py --cpu
"""

import argparse
import importlib.util
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CARPETA = Path(__file__).resolve().parent
RAIZ_REPO = CARPETA.parents[2]
for ruta in (CARPETA, CARPETA.parent):  # clasificacion y generar_imagenes_alteradas, sin instalar paquetes
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import clasificacion as C  # noqa: E402
from generar_imagenes_alteradas import RUTA_POR_DEFECTO  # noqa: E402

DATASET = Path(os.environ.get("SIMPSONS_DATASET", str(RUTA_POR_DEFECTO)))
# El mismo lugar que usa ampliar_con_frinkiac.py: junto al dataset, nunca dentro
LOCAL_POR_DEFECTO = DATASET.parent / "frinkiac_ampliacion"
PREDECIR_IMAGEN = RAIZ_REPO / "CNN" / "scripts" / "newDataset" / "predecir_imagen.py"
LOTE = 256  # fotogramas por lote: 1024 vistas de 128x128, unos 50 MB
VISTAS = 4


def vistas(ruta: Path, ancho: int, alto: int) -> np.ndarray:
    """(4, alto, ancho, 3) uint8: fotograma completo y recortes cuadrados a la izquierda, al centro y a la
    derecha (arriba, al medio y abajo si la imagen es vertical). Mismo pre-procesamiento que el notebook:
    RGB y redimensionado bilineal; el escalado a [0, 1] lo hace la capa Rescaling del modelo."""
    with Image.open(ruta) as archivo:
        imagen = archivo.convert("RGB")
    w, h = imagen.size
    lado = min(w, h)
    cajas = [(x, 0, x + lado, lado) for x in (0, (w - lado) // 2, w - lado)] if w >= h else \
        [(0, y, lado, y + lado) for y in (0, (h - lado) // 2, h - lado)]
    recortes = [imagen] + [imagen.crop(caja) for caja in cajas]
    return np.stack([np.array(v.resize((ancho, alto), Image.Resampling.BILINEAR), dtype=np.uint8) for v in recortes])


def probabilidades(rutas: list[Path], predecir_lote, ancho: int, alto: int, hilos: int = 8) -> np.ndarray:
    """(n, 4, clases). predecir_lote recibe (m, alto, ancho, 3) uint8 y devuelve (m, clases)."""
    salida = []
    with ThreadPoolExecutor(hilos) as hilos_lectura:
        for inicio in range(0, len(rutas), LOTE):
            lote = np.concatenate(list(hilos_lectura.map(lambda r: vistas(r, ancho, alto), rutas[inicio:inicio + LOTE])))
            p = np.asarray(predecir_lote(lote), dtype=np.float32)
            salida.append(p.reshape(-1, VISTAS, p.shape[-1]))
    return np.concatenate(salida) if salida else np.zeros((0, VISTAS, 0), dtype=np.float32)


def evaluar(personaje: str, rutas: list[Path], p: np.ndarray, clases: list[str], umbrales: C.Umbrales) -> list[C.Resultado]:
    indice = clases.index(personaje)
    resultados = []
    for ruta, fila in zip(rutas, p):
        p_completa, p_recorte = float(fila[0, indice]), float(fila[1:, indice].max())
        mejor = int(fila[0].argmax())
        resultados.append(C.Resultado(ruta.name, p_completa, p_recorte, clases[mejor], float(fila[0, mejor]),
                                      C.grupo_de(p_completa, p_recorte, umbrales)))
    return resultados


def clasificar(por_revisar: Path, clases: list[str], predecir_lote, ancho: int, alto: int, umbrales: C.Umbrales,
               personajes: list[str] | None = None) -> dict[str, Counter]:
    """Clasifica las imágenes nuevas de cada personaje y las reparte. Cada personaje se registra al
    terminarlo: si se corta, lo hecho se conserva y lo demás se clasifica la próxima vez."""
    por_revisar = Path(por_revisar)
    registro = C.leer_registro(C.raiz_de(por_revisar))
    carpetas = sorted(p.name for p in por_revisar.iterdir() if p.is_dir() and not p.name.startswith("."))
    if personajes:
        carpetas = [p for p in carpetas if p in set(personajes)]
    hechos = {}
    for personaje in carpetas:
        rutas = C.pendientes(por_revisar, personaje, registro)
        if not rutas:
            continue
        if personaje not in clases:
            print(f"  {personaje}: la CNN no conoce este personaje; queda para revisión manual en por_revisar/")
            continue
        inicio = time.time()
        resultados = evaluar(personaje, rutas, probabilidades(rutas, predecir_lote, ancho, alto), clases, umbrales)
        cuenta = C.repartir(por_revisar, personaje, resultados)
        hechos[personaje] = cuenta
        print(f"  {personaje:<28} {len(rutas):5d}  seguras {cuenta[C.SEGURAS]:4d}  dudosas {cuenta[C.DUDOSAS]:4d}  "
              f"ruido {cuenta[C.RUIDO]:4d}  ({time.time() - inicio:.0f} s)", flush=True)
    return hechos


def cargar_cnn(ruta_modelo: Path | None, usar_cpu: bool):
    """Carga la CNN con el mismo código de predecir_imagen.py y devuelve (predecir_lote, metadatos, dispositivo)."""
    especificacion = importlib.util.spec_from_file_location("predecir_imagen", PREDECIR_IMAGEN)
    predecir_imagen = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(predecir_imagen)
    modelo, metadatos, dispositivo = predecir_imagen.cargar_modelo(ruta_modelo, usar_cpu)
    if metadatos.get("representacion", "rgb") != "rgb":
        raise SystemExit(f"El modelo usa la representación {metadatos['representacion']!r}; este script solo "
                         "prepara imágenes RGB.")
    import torch

    def predecir_lote(lote: np.ndarray) -> np.ndarray:
        return modelo.predict(torch.as_tensor(lote).to(dispositivo), verbose=0, batch_size=512)

    return predecir_lote, metadatos, dispositivo


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", type=Path, default=LOCAL_POR_DEFECTO,
                        help="carpeta de la ampliación (por defecto %(default)s)")
    parser.add_argument("--personajes", type=lambda t: [p.strip() for p in t.split(",") if p.strip()],
                        help="solo estos personajes, separados por comas")
    parser.add_argument("--modelo", type=Path, help="archivo .keras; a su lado debe estar el .json con las clases")
    parser.add_argument("--cpu", action="store_true", help="usa la CPU aunque haya GPU")
    por_defecto = C.Umbrales()
    parser.add_argument("--umbral-segura", type=float, default=por_defecto.segura,
                        help="probabilidad en el fotograma completo para ser segura (%(default)s)")
    parser.add_argument("--umbral-recorte", type=float, default=por_defecto.segura_recorte,
                        help="o probabilidad en algún recorte para ser segura (%(default)s)")
    parser.add_argument("--umbral-ruido", type=float, default=por_defecto.ruido,
                        help="bajo esta probabilidad en todas las vistas es ruido (%(default)s)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = leer_argumentos(argv)
    umbrales = C.Umbrales(args.umbral_segura, args.umbral_recorte, args.umbral_ruido)
    try:
        umbrales.validar()
    except ValueError as error:
        raise SystemExit(str(error)) from None
    por_revisar = args.local / "por_revisar"
    if not por_revisar.is_dir():
        raise SystemExit(f"No existe {por_revisar}: ejecuta antes 'pnpm frinkiac' para recolectar imágenes.")
    previas = C.resumen(por_revisar)
    if not any(c["sin_clasificar"] for c in previas.values()):
        print("No hay imágenes nuevas que clasificar en por_revisar/.")
    else:
        predecir_lote, metadatos, dispositivo = cargar_cnn(args.modelo, args.cpu)
        print(f"Modelo: {metadatos['configuracion']} ({metadatos['alto']}x{metadatos['ancho']}, {dispositivo})")
        print(f"Segura: completo >= {umbrales.segura:.2f} o recorte >= {umbrales.segura_recorte:.2f}. "
              f"Ruido: < {umbrales.ruido:.2f} en todas las vistas.\n")
        try:
            hechos = clasificar(por_revisar, metadatos["clases"], predecir_lote, metadatos["ancho"], metadatos["alto"],
                                umbrales, args.personajes)
        except KeyboardInterrupt:
            print("\nDetenido. Lo clasificado quedó guardado; vuelve a ejecutar para seguir.")
            return 3
        total = sum(hechos.values(), Counter())
        print(f"\nClasificadas {sum(total.values())}: {total[C.SEGURAS]} seguras, {total[C.DUDOSAS]} dudosas y "
              f"{total[C.RUIDO]} de ruido.")
    raiz = C.raiz_de(por_revisar)
    pendientes = sorted(p for p, c in C.resumen(por_revisar).items() if c["dudosas_pendientes"])
    print(f"\nResultado en {raiz}")
    if pendientes:
        print(f"Revisa {len(pendientes)} carpeta(s) de {raiz / C.DUDOSAS}: borra las imágenes en las que el "
              f"personaje no aparece y, al terminar cada carpeta, su {C.MARCA_PENDIENTE}.")
    print("Si alguna de ruido/ sí sirve, muévela a seguras/. Luego ejecuta 'pnpm frinkiac': aprueba seguras y "
          "dudosas, rechaza el resto y repone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
