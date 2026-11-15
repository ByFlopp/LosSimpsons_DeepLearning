"""Escribe dataset-metadata.json en ASCII puro: la CLI de Kaggle lo abre sin encoding (cp1252 en Windows)."""
import json, sys
from pathlib import Path

carpeta = Path(sys.argv[1])
P = ""  # Kaggle descomprime el zip en la raíz del dataset
texto = lambda n, d: {"name": n, "description": d, "type": "string"}
entero = lambda n, d: {"name": n, "description": d, "type": "integer"}
decimal = lambda n, d: {"name": n, "description": d, "type": "decimal"}
personaje = texto("personaje", "Character folder (the label), e.g. homer_simpson")
transformacion = [
    decimal("rotacion", "Rotation in degrees, between 10 and 20 in either direction; the borders are cropped and the image is resized back to its original size"),
    texto("color", "Color change: brillo (brightness), contraste (contrast), saturacion (saturation) or tono (hue)"),
    decimal("parametro_color", "Factor for brillo and contraste (0.7-1.3) and saturacion (0.6-1.4); for tono, hue shift as a fraction of the color wheel (-0.04 to 0.04)"),
    texto("textura", "Texture filter: desenfoque (Gaussian blur), nitidez (unsharp mask), ruido (Gaussian noise) or bordes (edge enhance)"),
    decimal("parametro_textura", "Blur radius in px (0.5-1.5), unsharp percent (100-200), noise sigma (5-15) or 1.0 for bordes"),
]
datos = [
    {"name": P + "_renombres.csv",
     "description": "Original name of every renamed image. For the 34,106 episode frames it keeps the season, episode and timestamp.",
     "columns": [personaje,
                 texto("antiguo", "Name before renaming; episode frames use {character}__SxxEyy__t{milliseconds}.jpg (season, episode, time in the episode)"),
                 texto("nuevo", "Current file name inside the character folder: {character}_{number}.{ext}")]},
    {"name": P + "_alteraciones.csv",
     "description": "One row per __alterada image: the augmented copy generated for 25% of each character's originals (seed 42).",
     "columns": [personaje,
                 texto("original", "Original image the variant comes from"),
                 texto("alterada", "File name of the augmented copy ({original}__alterada.{ext})"),
                 *transformacion]},
    {"name": P + "_parecidas.csv",
     "description": "Near-duplicate groups inside one character folder (cosine similarity >= 0.90 between 16x16 color thumbnails) and the __parecida image that replaces each duplicate.",
     "columns": [personaje,
                 entero("grupo", "Id of the near-duplicate group within the character"),
                 texto("representante", "Image of the group that stays intact"),
                 texto("original", "Near-duplicate original that is replaced"),
                 texto("parecida", "File that replaces it ({original}__parecida.{ext}); use it instead of the original"),
                 decimal("similitud_original", "Highest cosine similarity between the original and the rest of its group"),
                 decimal("similitud_final", "Highest similarity between the replacement and the final images of the group (always below 0.90)"),
                 entero("intentos", "Random transformations drawn until the replacement stopped looking like the group"),
                 *transformacion]},
    {"name": P + "_gemelas.csv",
     "description": "Near-identical originals stored in the folders of two different characters (similarity >= 0.90): the same scene, where both characters appear.",
     "columns": [personaje,
                 texto("original", "Image in this character's folder"),
                 texto("otro_personaje", "The other character folder"),
                 texto("otro_original", "The near-identical image in the other folder"),
                 decimal("similitud", "Cosine similarity between both 16x16 color thumbnails"),
                 texto("copia_original", "Path of the copy proposed for manual review outside the dataset; empty if not copied. Can be ignored"),
                 texto("copia_otro", "Same as copia_original, for otro_original. Can be ignored")]},
]
metadata = {
    "id": "diegopyll/los-simpsons-dataset",
    "title": "The Simpsons Characters: 64 Classes, 56K Images",
    "subtitle": "56,268 images of 64 characters: curated images + 34,106 episode frames",
    "description": (carpeta / "descripcion.md").read_text(encoding="utf-8"),
    "isPrivate": False,
    "licenses": [{"name": "other"}],
    # Kaggle admite como mucho 5 etiquetas: con mas, "You have exceeded the max category limit"
    "keywords": ["computer vision", "deep learning", "multiclass classification", "cnn", "movies and tv shows"],
    "expectedUpdateFrequency": "annually",
    "userSpecifiedSources": (
        "Curated images: Los Simpson by alfaro96 (https://www.kaggle.com/datasets/alfaro96/los-simpson). "
        "Episode frames: Frinkiac (https://frinkiac.com), located with the script lines of The Simpsons by the Data "
        "(https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset): each frame is taken when the character "
        "speaks, filtered by a CNN, and the doubtful ones were checked by hand. Variants and logs generated with "
        "generar_imagenes_alteradas.py (seed 42) from https://github.com/ByFlopp/LosSimpsons_DeepLearning"),
    "data": datos,
}
assert 6 <= len(metadata["title"]) <= 50 and 20 <= len(metadata["subtitle"]) <= 80, (len(metadata["title"]), len(metadata["subtitle"]))
(carpeta / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="ascii")
print("titulo", len(metadata["title"]), "subtitulo", len(metadata["subtitle"]), "descripcion", len(metadata["description"]))
