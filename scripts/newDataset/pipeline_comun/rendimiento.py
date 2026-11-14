"""Rendimiento de la corrida: entorno de ejecución, tiempo por época y uso de la GPU.

Lo usa la sección «Resumen de rendimiento de la corrida en la GPU» de los dos notebooks, que guarda el resumen en
`resultados/newDataset/GPU/rendimiento_gpu_<resolución>.json`.
"""

import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .entrenamiento import _pasada

# Ventana mínima de muestreo. Una época puede durar 0,2 s: medir la utilización sobre una sola es leer ruido, así
# que se repiten épocas hasta cubrir este tiempo.
SEGUNDOS_MINIMOS_MUESTREO = 4.0
INTERVALO_MUESTREO = 0.05


def describir_cpu():
    """Nombre del procesador, en Windows y en Linux."""
    if platform.system() == "Windows":
        try:
            nombre = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10
            ).stdout.strip().splitlines()
            if len(nombre) > 1 and nombre[1].strip():
                return nombre[1].strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return platform.processor() or platform.machine()
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def describir_gpu():
    """Nombre comercial y capacidad de cómputo de la GPU que ve PyTorch."""
    mayor, menor = torch.cuda.get_device_capability(0)
    return f"{torch.cuda.get_device_name(0)} (CC {mayor}.{menor})"


class MuestreadorGPU(threading.Thread):
    """Lee nvidia-smi en paralelo para saber cuánto de la tarjeta se está usando.

    La utilización responde a «¿la GPU está ocupada o esperando?». No mide potencia entregada: una tarjeta limitada
    por consumo puede marcar 100 % y rendir la mitad."""

    def __init__(self, intervalo=INTERVALO_MUESTREO):
        """Prepara el hilo: lee nvidia-smi cada `intervalo` segundos hasta que se active `parar`."""
        super().__init__(daemon=True)
        self.parar = threading.Event()
        self.intervalo = intervalo
        self.muestras = []

    def run(self):
        """Bucle del hilo: guarda (utilización %, potencia W, reloj MHz) en cada lectura."""
        while not self.parar.is_set():
            try:
                salida = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                self.muestras.append(tuple(float(v) for v in salida.split(",")))
            except Exception:  # noqa: BLE001 - una lectura fallida no detiene la medición
                pass
            time.sleep(self.intervalo)

    def resumen(self):
        """Media sobre la ventana muestreada, con el número de muestras. No se descartan lecturas bajas: filtrar
        sesgaría la media justo en los casos que interesan, los de tarjeta infrautilizada."""
        if len(self.muestras) < 5:
            return {"muestras_utilizacion": len(self.muestras)}
        util, potencia, reloj = (np.array(v) for v in zip(*self.muestras))
        return {
            "utilizacion_gpu_media": round(float(util.mean()), 1),
            "potencia_media_w": round(float(potencia.mean()), 1),
            "reloj_medio_mhz": round(float(reloj.mean())),
            "muestras_utilizacion": len(util),
        }


def cronometrar_epocas(modelo, X, y, batch, epocas, tasa_aprendizaje, semilla, muestrear=False):
    """Entrena `epocas` épocas seguidas y devuelve `(segundos_por_epoca, estado_gpu)`.

    Se descarta la primera época, que incluye la reserva de memoria y el calentamiento de los kernels. Si
    `muestrear`, mide después la utilización de la GPU en una ventana aparte, lo bastante larga para que el muestreo
    tenga sentido."""
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje, fused=(X.device.type == "cuda"))
    generador = torch.Generator(device=X.device)
    generador.manual_seed(semilla)

    duraciones = []
    for _ in range(epocas):
        inicio = time.perf_counter()
        _pasada(modelo, X, y, batch, True, optimizador, None, generador)
        if X.device.type == "cuda":
            torch.cuda.synchronize()
        duraciones.append(time.perf_counter() - inicio)
    segundos_epoca = float(np.mean(duraciones[1:] or duraciones))

    estado = {}
    if muestrear:
        repeticiones = max(2, int(np.ceil(SEGUNDOS_MINIMOS_MUESTREO / max(segundos_epoca, 1e-6))))
        muestreador = MuestreadorGPU()
        muestreador.start()
        for _ in range(repeticiones):
            _pasada(modelo, X, y, batch, True, optimizador, None, generador)
        if X.device.type == "cuda":
            torch.cuda.synchronize()
        muestreador.parar.set()
        muestreador.join(timeout=2)
        estado = muestreador.resumen()
    return segundos_epoca, estado


def cuatro_decimales(valor):
    """Métrica redondeada; la de un subconjunto vacío queda como None (null en el JSON)."""
    return None if valor is None or np.isnan(valor) else round(float(valor), 4)
