"""Clasificar una imagen cualquiera con el modelo final, con el mismo preprocesamiento que en el entrenamiento.

Lo usa la sección «Probar el modelo con una imagen cualquiera» de los dos notebooks. Los rótulos siguen el formato de
`scripts/newDataset/detectar_cajas.py`: el personaje y la confianza del modelo, por ejemplo `Jasper Beardly · 92 %`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Nombres que no salen bien de la carpeta con title()
NOMBRES_ESPECIALES = {
    "kang_kodos": "Kang y Kodos",
    "krusty_the_clown": "Krusty the Clown",
    "captain_horatio_mccallister": "Captain Horatio McCallister",
    "dr_julius_hibbert": "Dr. Julius Hibbert",
    "dr_nick_riviera": "Dr. Nick Riviera",
    "rev_timothy_lovejoy": "Rev. Timothy Lovejoy",
}


def nombre_legible(personaje):
    """'jasper_beardly' -> 'Jasper Beardly'."""
    return NOMBRES_ESPECIALES.get(personaje, personaje.replace("_", " ").title())


def rotulo(personaje, confianza):
    """El personaje y la confianza del modelo: 'Jasper Beardly · 92 %'."""
    return f"{nombre_legible(personaje)} · {round(100 * float(confianza))} %"


def predecir_imagen(ruta_imagen, modelo, clases, alto, ancho, preparar_entrada, top_k=3, mostrar=True):
    """Clasifica una imagen completa con el modelo.

    - `ruta_imagen`: ruta a un archivo de imagen (jpg, png, ...);
    - `clases`: los personajes en el orden de las salidas del modelo;
    - `alto`, `ancho`: la resolución con que se entrenó;
    - `preparar_entrada`: la función que lleva un lote de imágenes RGB redimensionadas a la entrada del modelo
      (la escribe «Selección del modelo final» en `<modelo>_entrada.py`);
    - `top_k`: cuántas candidatas devolver, de mayor a menor probabilidad.

    Devuelve un DataFrame con una fila por candidata: el personaje, su confianza y su rótulo. Si `mostrar`, dibuja
    la imagen con su rótulo: el marco es verde si la carpeta de la imagen es un personaje y el modelo lo acierta, y
    rojo si se equivoca."""
    ruta_imagen = Path(str(ruta_imagen).strip().strip('"').strip("'"))
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontro la imagen: {ruta_imagen}")
    with Image.open(ruta_imagen) as archivo:
        imagen = archivo.convert("RGB")

    # Mismo preprocesamiento que en el entrenamiento: la imagen completa, redimensionada con bilineal
    reducida = np.array(imagen.resize((ancho, alto), Image.Resampling.BILINEAR), dtype=np.uint8)
    entrada = np.ascontiguousarray(preparar_entrada(reducida[np.newaxis]))
    dispositivo = next(modelo.parameters()).device
    probabilidades = modelo.predict(torch.as_tensor(entrada).to(dispositivo), verbose=0)[0]
    mejores = np.argsort(-probabilidades)[:top_k]
    resultado = pd.DataFrame(
        [
            {
                "puesto": puesto,
                "personaje": nombre_legible(clases[indice]),
                "confianza": round(float(probabilidades[indice]), 4),
                "rotulo": rotulo(clases[indice], probabilidades[indice]),
            }
            for puesto, indice in enumerate(mejores, start=1)
        ]
    )

    if mostrar:
        carpeta = ruta_imagen.parent.name if ruta_imagen.parent.name in clases else None
        prediccion = clases[int(mejores[0])]
        color = "#1f77b4" if carpeta is None else ("#1b7f3b" if prediccion == carpeta else "#c0392b")
        figura, eje = plt.subplots(figsize=(9, 6.5))
        eje.imshow(imagen)
        eje.set_xticks([])
        eje.set_yticks([])
        for lado in eje.spines.values():
            lado.set_edgecolor(color)
            lado.set_linewidth(4)
        titulo = f"{resultado.loc[0, 'rotulo']}   |   {ruta_imagen.name}"
        if carpeta is not None:
            titulo += f"   |   carpeta: {nombre_legible(carpeta)} ({'acierto' if prediccion == carpeta else 'error'})"
        eje.set_title(titulo, fontsize=11, fontweight="bold")
        eje.set_xlabel("Otras candidatas: " + ", ".join(resultado["rotulo"].iloc[1:]), fontsize=10)
        plt.show()

    return resultado
