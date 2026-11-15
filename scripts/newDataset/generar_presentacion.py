"""Genera la presentación de resultados de una de las dos redes a partir de su corrida.

    .venv-gpu\\Scripts\\python.exe scripts\\newDataset\\generar_presentacion.py --modelo cnn
    .venv-gpu\\Scripts\\python.exe scripts\\newDataset\\generar_presentacion.py --modelo mlp

Lee tres cosas, todas escritas por el notebook de la red al ejecutarse:

- el resumen de la corrida, `<RED>/resultados/newDataset/GPU/rendimiento_gpu_128x128.json`;
- el resumen de la otra red, para las diapositivas de comparación;
- las figuras del informe, `<RED>/docs/newDataset/figuras/`.

El texto de cada diapositiva vive en `presentacion/<modelo>.py`, con las cifras tomadas del JSON. El estilo y la
navegación salen de `presentacion/plantilla.html`, la misma para las dos redes. Escribe
`CNN/presentacion_resultados_CNN.html` o `PMC/presentacion_resultados_PMC.html`, un solo archivo con las imágenes
embebidas.
"""

import argparse
import base64
import html
import importlib
import io
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = Path(__file__).resolve().parent / "presentacion"
ANCHO_MAXIMO = 1200
CALIDAD_JPEG = 85

REDES = {
    "cnn": {"carpeta": "CNN", "salida": "CNN/presentacion_resultados_CNN.html", "otra": "mlp"},
    "mlp": {"carpeta": "PMC", "salida": "PMC/presentacion_resultados_PMC.html", "otra": "cnn"},
}
CLAVES_OBLIGATORIAS = (
    "inicio_corrida", "configuracion_final", "accuracy_test", "f1_macro_test", "top3_test", "roc_auc_macro_test",
    "precision_macro_test", "recall_macro_test", "ece_test", "umbral_revision", "cobertura_umbral_test",
    "accuracy_umbral_test", "accuracy_kaggle_test", "accuracy_episodio_test", "epocas_maximas", "paciencia",
)


# --- Formato de cifras ---------------------------------------------------------------------------------------


def num(valor, decimales=3):
    """Número con coma decimal y punto de miles, redondeado hacia arriba en el medio: 0,7465 -> '0,747'."""
    redondeado = Decimal(str(valor)).quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)
    texto = f"{redondeado:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def entero(valor):
    """Entero con punto de miles: 7058 -> '7.058'."""
    return f"{int(valor):,}".replace(",", ".")


def pct(valor, decimales=1):
    """Proporción como porcentaje: 0,253 -> '25,3 %'."""
    return f"{num(valor * 100, decimales)} %"


def puntos_de(diferencia, decimales=1):
    """Diferencia entre proporciones en puntos, con signo: 0,019 -> '+1,9'."""
    signo = "+" if diferencia > 0 else "−" if diferencia < 0 else "±"
    return f"{signo}{num(abs(diferencia) * 100, decimales)}"


def duracion(texto):
    """'9:11:40' -> '9 h 11 min'."""
    horas, minutos, _ = (int(parte) for parte in texto.split(":"))
    return f"{horas} h {minutos:02d} min" if horas else f"{minutos} min"


def fecha(inicio):
    """'2026-11-13 21:40:12' -> '13-11-2026'."""
    dia = inicio.split(" ")[0].split("-")
    return f"{dia[2]}-{dia[1]}-{dia[0]}"


def esc(texto):
    """Texto escapado para HTML."""
    return html.escape(str(texto), quote=True)


# --- Piezas de las diapositivas ------------------------------------------------------------------------------


def figura(ruta, alt, clase="lamina"):
    """Una figura del informe embebida como imagen, reducida a un ancho máximo."""
    ruta = Path(ruta)
    imagen = Image.open(ruta)
    if imagen.mode in ("RGBA", "LA", "P"):
        imagen = imagen.convert("RGBA")
        fondo = Image.new("RGB", imagen.size, "white")
        fondo.paste(imagen, mask=imagen.getchannel("A"))
        imagen = fondo
    else:
        imagen = imagen.convert("RGB")
    if imagen.width > ANCHO_MAXIMO:
        alto = round(imagen.height * ANCHO_MAXIMO / imagen.width)
        imagen = imagen.resize((ANCHO_MAXIMO, alto), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if ruta.suffix.lower() in (".jpg", ".jpeg"):
        imagen.save(buffer, "JPEG", quality=CALIDAD_JPEG, optimize=True, progressive=True)
        tipo = "image/jpeg"
    else:
        imagen.save(buffer, "PNG", optimize=True)
        tipo = "image/png"
    datos = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (f'<figure class="{clase}"><img src="data:{tipo};base64,{datos}" width="{imagen.width}" '
            f'height="{imagen.height}" alt="{esc(alt)}"></figure>')


def cifras(elementos, columnas=None):
    """Fila de cifras destacadas: `elementos` es una lista de (etiqueta, valor, nota)."""
    estilo = f' style="--n: {columnas}"' if columnas else ""
    celdas = "".join(
        f'<div class="cifra"><span class="etiqueta">{etiqueta}</span><span class="valor">{valor}</span>'
        f'<span class="nota">{nota}</span></div>'
        for etiqueta, valor, nota in elementos
    )
    return f'<div class="cifras"{estilo}>{celdas}</div>'


def tabla(cabecera, filas, numericas=(), densa=False):
    """Tabla HTML: `numericas` son los índices de las columnas que se alinean a la derecha."""
    def celda(texto, k, etiqueta):
        """Una celda `etiqueta` (th o td), alineada a la derecha si su columna es numérica."""
        clase = ' class="num"' if k in numericas else ""
        return f"<{etiqueta}{clase}>{texto}</{etiqueta}>"

    encabezado = "".join(celda(texto, k, "th") for k, texto in enumerate(cabecera))
    cuerpo = "".join("<tr>" + "".join(celda(texto, k, "td") for k, texto in enumerate(fila)) + "</tr>"
                     for fila in filas)
    clase = "tabla densa" if densa else "tabla"
    return (f'<div class="desliza"><table class="{clase}"><thead><tr>{encabezado}</tr></thead>'
            f"<tbody>{cuerpo}</tbody></table></div>")


def lista(puntos, chicos=False):
    """Lista de viñetas; cada punto puede llevar HTML (por ejemplo, <strong>)."""
    clase = "puntos chicos" if chicos else "puntos"
    return f'<ul class="{clase}">' + "".join(f"<li>{punto}</li>" for punto in puntos) + "</ul>"


def columna(*piezas):
    """Columna de piezas apiladas."""
    return '<div class="columna">' + "".join(piezas) + "</div>"


def subtitulo(texto):
    """Rótulo pequeño dentro del cuerpo de una diapositiva."""
    return f'<p class="seccion">{texto}</p>'


def leyenda(texto):
    """Pie de figura o de tabla."""
    return f'<p class="leyenda-figura">{texto}</p>'


def estado(cumple):
    """Marca de cumplimiento de un KPI."""
    if cumple:
        return ('<span class="estado"><svg viewBox="0 0 16 16" aria-hidden="true"><circle class="icono-fondo" '
                'cx="8" cy="8" r="8"/><path class="icono-marca" d="M4.5 8.3l2.3 2.3 4.7-4.9"/></svg>Cumple</span>')
    return "No cumple"


def portada(red, subtitulo_portada, fecha_portada):
    """Primera diapositiva: título, red, equipo y profesor."""
    return f"""<section class="diapo portada" id="d1" aria-label="Diapositiva 1: portada">
  <div class="lienzo">
    <p class="seccion">Técnicas Avanzadas de Machine Learning · Evaluación Parcial 3</p>
    <div class="titular">
      <h1>¿Quién sale en la imagen?</h1>
      <p class="sub">{subtitulo_portada} <span style="white-space: nowrap">{fecha_portada}</span>.</p>
    </div>
    <div class="equipo">
      <p><span class="rol">Equipo</span><br>Vincent Farenden Cerón · Rodrigo Martínez Becker<br>Iván Pavón Barría · Diego Peña y Lillo Luhrs</p>
      <p><span class="rol">Profesor</span><br>Francisco Jerez Salazar</p>
    </div>
  </div>
</section>
"""


def diapositiva(n, total, pie, seccion, titulo, cuerpo, disposicion="dos mitad", estilo=""):
    """Una diapositiva con su encabezado, su cuerpo y su pie numerado."""
    atributo = f' style="{estilo}"' if estilo else ""
    return f"""<!-- {n} -->
<section class="diapo" id="d{n}" aria-label="Diapositiva {n}">
  <div class="lienzo">
    <header class="cabeza">
      <p class="seccion">{seccion}</p>
      <h2>{titulo}</h2>
    </header>
    <div class="cuerpo {disposicion}"{atributo}>
      {cuerpo}
    </div>
    <footer class="pie"><span>{pie}</span><span>{n:02d} / {total}</span></footer>
  </div>
</section>
"""


# --- Ensamblado ------------------------------------------------------------------------------------------------


class Contexto:
    """Lo que necesita el contenido de una presentación: los dos resúmenes JSON y las figuras de la red."""

    def __init__(self, modelo):
        """Carga el resumen de `modelo` y el de la otra red, y comprueba que el primero tenga las claves de la EP3."""
        configuracion = REDES[modelo]
        self.modelo = modelo
        self.j = self._resumen(configuracion["carpeta"])
        self.otra = self._resumen(REDES[configuracion["otra"]]["carpeta"], obligatorio=False)
        self.figuras_dir = RAIZ / configuracion["carpeta"] / "docs" / "newDataset" / "figuras"
        faltan = [clave for clave in CLAVES_OBLIGATORIAS if clave not in self.j]
        if faltan:
            sys.exit(f"El resumen de la corrida no tiene {', '.join(faltan)}: ejecuta antes el notebook completo.")

    @staticmethod
    def _resumen(carpeta, obligatorio=True):
        """El resumen JSON de la corrida de `carpeta`; sin él, termina si es obligatorio o devuelve None."""
        ruta = RAIZ / carpeta / "resultados" / "newDataset" / "GPU" / "rendimiento_gpu_128x128.json"
        if not ruta.exists():
            if obligatorio:
                sys.exit(f"No existe {ruta}: ejecuta antes el notebook.")
            return None
        return json.loads(ruta.read_text(encoding="utf-8"))

    def fig(self, nombre, alt, clase="lamina"):
        """Figura `nombre` (con su extensión) de la carpeta de figuras de la red."""
        ruta = self.figuras_dir / nombre
        if not ruta.exists():
            sys.exit(f"Falta la figura {ruta}: ejecuta antes el notebook completo.")
        return figura(ruta, alt, clase)


def generar(modelo):
    """Arma la presentación de `modelo` y la escribe en su carpeta. Devuelve la ruta y el número de diapositivas."""
    contexto = Contexto(modelo)
    contenido = importlib.import_module(f"presentacion.{modelo}")
    datos = contenido.diapositivas(contexto)
    total = len(datos) + 1
    partes = [portada(modelo, contenido.SUBTITULO_PORTADA, contenido.fecha_portada(contexto))]
    for numero, (seccion, titulo, cuerpo, disposicion) in enumerate(datos, start=2):
        partes.append(diapositiva(numero, total, contenido.PIE, seccion, titulo, cuerpo, disposicion))
    plantilla = (CARPETA / "plantilla.html").read_text(encoding="utf-8")
    pagina = (
        plantilla.replace("{{TITULO}}", esc(contenido.TITULO))
        .replace("{{DESCRIPCION}}", esc(contenido.DESCRIPCION))
        .replace("{{BARRA}}", contenido.BARRA)
        .replace("{{DIAPOSITIVAS}}", "\n".join(partes))
        .replace("{{FINAL}}", contenido.final(contexto))
        .replace("{{TOTAL}}", str(total))
    )
    salida = RAIZ / REDES[modelo]["salida"]
    salida.write_text(pagina, encoding="utf-8")
    return salida, total


def main(argumentos=None):
    """Punto de entrada de la línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modelo", choices=sorted(REDES), required=True)
    opciones = parser.parse_args(argumentos)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    salida, total = generar(opciones.modelo)
    print(f"{salida.relative_to(RAIZ)}: {total} diapositivas, {salida.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
