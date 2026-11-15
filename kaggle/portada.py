"""Portada 2:1 (1200x600) del dataset: mosaico de 32 personajes con un panel central."""
import csv, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DATASET = Path.home() / "Desktop" / "simpsons_dataset"
SALIDA = Path(sys.argv[1])
renombradas = {f["nuevo"] for f in csv.DictReader(open(DATASET / "_renombres.csv", encoding="utf-8"))}

personajes = sorted((d for d in DATASET.iterdir() if d.is_dir()),
                    key=lambda d: -sum(1 for f in d.iterdir() if "__" not in f.name))[:32]
LADO, COLS, FILAS = 150, 8, 4
lienzo = Image.new("RGB", (LADO * COLS, LADO * FILAS))
for i, carpeta in enumerate(personajes):
    # una imagen curada (no fotograma) cercana al cuadrado y de buen tamaño
    candidatas = [f for f in sorted(carpeta.iterdir())
                  if "__" not in f.name and f.name not in renombradas and f.suffix.lower() == ".jpg"]
    mejor = None
    for f in candidatas[:40]:
        with Image.open(f) as im:
            w, h = im.size
        if min(w, h) >= 200 and 0.6 <= w / h <= 1.4:
            mejor = f
            break
    mejor = mejor or (candidatas or [f for f in sorted(carpeta.iterdir()) if "__" not in f.name and f.suffix.lower() in (".jpg", ".png")])[0]
    im = Image.open(mejor).convert("RGB")
    w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((LADO, LADO), Image.Resampling.LANCZOS)
    lienzo.paste(im, ((i % COLS) * LADO, (i // COLS) * LADO))

# Panel central legible también en la miniatura cuadrada (el centro de 600 x 600)
capa = Image.new("RGBA", lienzo.size, (0, 0, 0, 0))
d = ImageDraw.Draw(capa)
x0, y0, x1, y1 = 330, 175, 870, 425
d.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=(20, 24, 38, 225), outline=(255, 217, 15, 255), width=4)
lienzo = Image.alpha_composite(lienzo.convert("RGBA"), capa)
d = ImageDraw.Draw(lienzo)
F = "C:/Windows/Fonts/"
titulo = ImageFont.truetype(F + "seguibl.ttf", 58)
sub = ImageFont.truetype(F + "segoeuib.ttf", 30)
peq = ImageFont.truetype(F + "segoeuib.ttf", 22)
def centrado(y, texto, fuente, color):
    w = d.textlength(texto, font=fuente)
    d.text(((x0 + x1 - w) / 2, y), texto, font=fuente, fill=color)
centrado(195, "THE SIMPSONS", titulo, (255, 217, 15))
centrado(272, "64 characters · 56,268 images", sub, (255, 255, 255))
centrado(318, "34,106 episode frames · 557 episodes", peq, (210, 214, 225))
centrado(360, "CNN 77.5% · MLP 55.6% accuracy", peq, (150, 220, 160))
lienzo.convert("RGB").save(SALIDA, quality=92)
print(SALIDA, lienzo.size)
