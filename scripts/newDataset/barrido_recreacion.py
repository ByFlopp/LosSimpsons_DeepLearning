"""Barre los hiperparámetros de la recreación de un MLP sobre 16 personajes de ajuste, sin «entrenar contra el juez».

Cada configuración recrea con recrear_personajes.main a los 16 personajes de ajuste y la juzgan la red que genera y
las dos CNN. Los 16 se sortean con semilla fija entre los que no salen en las figuras (comparacion.png y
evolucion.png), así las figuras y los 48 restantes muestran personajes con los que no se eligió nada. Las dos CNN
comparten casi todos sus pesos (correlación 0,999): cuentan como un solo juez.

Puntaje de cada configuración, de más a menos importante:
    1. que la red que genera reconozca a los 16 (top-1 propio);
    2. la posición media del personaje para las CNN (al azar, la del medio: 32,5 de 64), promediada entre las dos;
    3. la probabilidad media que le dan las CNN.
La posición se mueve antes que el top-1: dice si las recreaciones se acercan a su personaje aunque aún no ganen.

Rejillas (--rejilla):
    mlp       el MLP de píxeles (R1): escalas, α_tv, α_bn (normalizada) y desplazamiento, sin volteo
    robusto   el MLP robusto: escalas, α_tv, α_bn (normalizada), desplazamiento y volteo
Pasos: 2000 (--pasos); los presets usan 3000, pero el MLP reconoce a todos antes de los 100.

Salidas, en --salida (por defecto PMC/models_gpu/newDataset/barridos/<generador>/, ignorada por Git):
    barrido.csv          una fila por configuración, en el orden en que se corrió; se escribe a medida que avanza
    <configuracion>/     la lámina, comparacion.png y evolucion.png de cada configuración, para revisarlas a ojo
No escribe en la caché de recreaciones.

Uso, desde la raíz del repositorio (entorno .venv-gpu):
    .venv-gpu/Scripts/python.exe scripts/newDataset/barrido_recreacion.py --generador mlp --rejilla mlp
    .venv-gpu/Scripts/python.exe scripts/newDataset/barrido_recreacion.py --generador mlp_robusto --rejilla robusto
"""

import argparse
import csv
import itertools
import random
import sys
from dataclasses import replace
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
if str(CARPETA) not in sys.path:
    sys.path.insert(0, str(CARPETA))

import recrear_personajes as R  # noqa: E402

N_AJUSTE = 16
SEMILLA_AJUSTE = 42
JUECES_CNN = ("cnn", "cnn_recortes")

REJILLAS = {
    "mlp": {
        "escalas": ["16,32,64,128", "16,32,64", "8,16,32"],
        "alfa-tv": [1e-3, 1e-2],
        "alfa-bn": [0.0, 1e-2, 1e-1],
        "desplazamiento": [0, 4],
        "volteo": [False],
    },
    "robusto": {
        "escalas": ["16,32,64,128", "16,32,64"],
        "alfa-tv": [1e-3, 1e-2],
        "alfa-bn": [0.0, 1e-2, 1e-1],
        "desplazamiento": [0, 4, 8],
        "volteo": [False, True],
    },
}


def personajes_de_ajuste(clases: list, n: int = N_AJUSTE, semilla: int = SEMILLA_AJUSTE) -> list:
    """n personajes sorteados con semilla fija entre los que no salen en las figuras del notebook."""
    en_figuras = set(R.PERSONAJES_F1) | set(R.PERSONAJES_EVOLUCION)
    candidatos = sorted(set(clases) - en_figuras)
    return sorted(random.Random(semilla).sample(candidatos, n))


def matriz_de(detalle: list, personajes) -> dict:
    """{juez: fila de la matriz} solo con `personajes`, desde el "detalle" de un recreacion.json. Sirve para informar
    aparte los personajes que no se usaron para ajustar."""
    elegidos = [d for d in detalle if d["personaje"] in set(personajes)]
    if not elegidos:
        return {}
    return {juez: R.fila_matriz({"posicion": [d["jueces"][juez]["posicion"] for d in elegidos],
                                 "p_objetivo": [d["jueces"][juez]["p_objetivo"] for d in elegidos]})
            for juez in elegidos[0]["jueces"]}


def configuraciones(rejilla: dict) -> list:
    """Cada combinación de la rejilla como {flag: valor}, en orden."""
    nombres = list(rejilla)
    return [dict(zip(nombres, valores)) for valores in itertools.product(*(rejilla[n] for n in nombres))]


def nombre_configuracion(configuracion: dict) -> str:
    partes = []
    for flag, valor in configuracion.items():
        if isinstance(valor, bool):
            partes.append(f"{flag}-{'si' if valor else 'no'}")
        else:
            partes.append(f"{flag}-{str(valor).replace(',', '_')}")
    return "__".join(partes)


def argumentos_main(generador: str, configuracion: dict, personajes: list, pasos: int, salida: Path) -> list:
    """Los argumentos de recrear_personajes.main para una configuración."""
    argv = ["--modelos", ",".join((*JUECES_CNN, generador)), "--generadores", generador,
            "--personajes", ",".join(personajes), "--semilla", "42", "--pasos", str(pasos),
            "--salida", str(salida), "--sin-cache", "--bn-normalizada"]
    for flag, valor in configuracion.items():
        if isinstance(valor, bool):
            argv.append(f"--{flag}" if valor else f"--no-{flag}")
        else:
            argv += [f"--{flag}", str(valor)]
    return argv


def fila(nombre: str, configuracion: dict, matriz: dict, generador: str) -> dict:
    """Una fila del CSV desde la matriz de recrear_personajes.main."""
    propia = matriz[generador]
    cnn = [matriz[j] for j in JUECES_CNN if j in matriz]
    salida = {"configuracion": nombre, **configuracion,
              "top1_propio": propia["top1"], "n": propia["n"], "confianza_propia": propia["confianza"]}
    for j in JUECES_CNN:
        if j in matriz:
            salida.update({f"top1_{j}": matriz[j]["top1"], f"top3_{j}": matriz[j]["top3"],
                           f"confianza_{j}": matriz[j]["confianza"], f"posicion_{j}": matriz[j]["posicion_media"]})
    salida["posicion_cnn"] = round(sum(m["posicion_media"] for m in cnn) / len(cnn), 2) if cnn else None
    salida["confianza_cnn"] = round(sum(m["confianza"] for m in cnn) / len(cnn), 4) if cnn else None
    return salida


def orden(f: dict) -> tuple:
    """Mejor primero: más top-1 propio, menor posición para las CNN y más probabilidad para las CNN."""
    return (-f["top1_propio"], f["posicion_cnn"] if f["posicion_cnn"] is not None else 99,
            -(f["confianza_cnn"] or 0))


def main(argv=None) -> list:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generador", choices=["mlp", "mlp_robusto"], default="mlp")
    parser.add_argument("--rejilla", choices=list(REJILLAS), default="mlp")
    parser.add_argument("--pasos", type=int, default=2000)
    parser.add_argument("--salida", type=Path)
    parser.add_argument("--desde", type=int, default=0, help="saltar las primeras N configuraciones (reanudar)")
    parser.add_argument("--modelo", type=Path, help="otro .keras para el generador (p. ej. mlp_r1_robusto_e2.keras)")
    parser.add_argument("--solo", type=int, nargs="*", help="solo estas configuraciones (números desde 1)")
    args = parser.parse_args(argv)

    if args.modelo is not None:
        R.ESPECIFICACIONES[args.generador] = replace(R.ESPECIFICACIONES[args.generador], candidatos=(args.modelo,))
    carpeta = args.generador if args.modelo is None else f"{args.generador}_{args.modelo.stem}"
    salida = args.salida or (R.MODELOS_MLP / "barridos" / carpeta)
    salida.mkdir(parents=True, exist_ok=True)
    red, avisos = R.resolver(R.ESPECIFICACIONES[args.generador], usar_cpu=False)
    if red is None:
        raise SystemExit("\n".join(avisos))
    ajuste = personajes_de_ajuste(red.clases)
    print(f"Personajes de ajuste ({len(ajuste)}): {', '.join(ajuste)}")
    todas = configuraciones(REJILLAS[args.rejilla])
    print(f"{len(todas)} configuraciones, {args.pasos} pasos cada una")

    ruta_csv = salida / "barrido.csv"
    filas = []
    if args.desde and ruta_csv.is_file():
        with open(ruta_csv, encoding="utf-8", newline="") as archivo:
            filas = [{k: _numero(v) for k, v in f.items()} for f in csv.DictReader(archivo)]
    for numero, configuracion in enumerate(todas):
        if numero < args.desde or (args.solo and numero + 1 not in args.solo):
            continue
        nombre = nombre_configuracion(configuracion)
        print(f"\n[{numero + 1}/{len(todas)}] {nombre}")
        resultado = R.main(argumentos_main(args.generador, configuracion, ajuste, args.pasos, salida / nombre))
        filas.append(fila(nombre, configuracion, resultado["matriz"][args.generador], args.generador))
        with open(ruta_csv, "w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
            escritor.writeheader()
            escritor.writerows(filas)

    print("\nLas 10 mejores (top-1 propio, posición media para las CNN, probabilidad para las CNN):")
    for f in sorted(filas, key=orden)[:10]:
        print(f"  {f['top1_propio']}/{f['n']}  posición CNN {f['posicion_cnn']:>5}  prob. CNN {f['confianza_cnn']:.3f}"
              f"  {f['configuracion']}")
    print(f"\nCSV: {ruta_csv}")
    return filas


def _numero(valor: str):
    for tipo in (int, float):
        try:
            return tipo(valor)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(valor, valor)


if __name__ == "__main__":
    main()
