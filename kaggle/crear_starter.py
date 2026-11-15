"""Crea el notebook de exploración (starter) que acompaña al dataset en Kaggle."""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
celdas = [
    md("""# The Simpsons Characters: Starter EDA

A short tour of the [**The Simpsons Characters** dataset](https://www.kaggle.com/datasets/diegopyll/los-simpsons-dataset):

- 64 characters and 56,268 original images: curated character images and 34,106 episode frames;
- how the `__alterada` and `__parecida` variants work;
- a **leakage-free split** you can copy.

The full study compares a CNN (0.775 accuracy, 0.755 F1 macro) with an MLP on hand-crafted color features (0.556 accuracy) on this data. Code and reports are on [GitHub](https://github.com/ByFlopp/LosSimpsons_DeepLearning)."""),
    code("""import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


def find_dataset() -> Path:
    \"\"\"The folder that holds one subfolder per character (and _renombres.csv).\"\"\"
    if os.environ.get("SIMPSONS_DATASET"):
        return Path(os.environ["SIMPSONS_DATASET"])
    for depth in range(1, 5):
        for log in Path("/kaggle/input").glob("/".join(["*"] * depth) + "/_renombres.csv"):
            return log.parent
    return Path.home() / "Desktop" / "simpsons_dataset"


ROOT = find_dataset()
print(ROOT)
print(sorted(p.name for p in ROOT.iterdir() if p.is_file()))"""),
    md("""## 1. Index every image

The label is always the folder name. Each original is `<character>_<number>.<ext>`, and each variant keeps the name of its original. `_renombres.csv` keeps the original name of every episode frame (`<character>__SxxEyy__t<ms>.jpg`), so the source, season and episode of every image can be recovered."""),
    code("""EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FRAME = re.compile(r"__S(\\d+)E(\\d+)__t(\\d+)")

renames = pd.read_csv(ROOT / "_renombres.csv")
old_name = dict(zip(zip(renames.personaje, renames.nuevo), renames.antiguo))

rows = []
for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for f in folder.iterdir():
        if f.suffix.lower() not in EXTENSIONS:
            continue
        kind = "alterada" if "__alterada" in f.stem else "parecida" if "__parecida" in f.stem else "original"
        original = f.stem.split("__")[0] + f.suffix
        match = FRAME.search(old_name.get((folder.name, original), ""))
        rows.append({
            "character": folder.name,
            "file": f.name,
            "kind": kind,
            "original": original,
            "source": "episode frame" if match else "curated image",
            "season": int(match.group(1)) if match else np.nan,
            "episode": int(match.group(2)) if match else np.nan,
        })
df = pd.DataFrame(rows)
originals = df[df.kind == "original"]

print(f"{df.character.nunique()} characters, {len(df):,} images")
display(df.kind.value_counts().rename("images").to_frame())
display(originals.source.value_counts().rename("originals").to_frame())
df.sample(5, random_state=0)"""),
    md("""## 2. Class balance

The classes are imbalanced by about 20:1, from Homer down to Maggie. Episode frames dominate most of the secondary characters, so the source of an image is almost as important as its class: models do much better on curated images than on frames."""),
    code("""def pretty(name: str) -> str:
    return name.replace("_", " ").title()


counts = originals.groupby(["character", "source"]).size().unstack(fill_value=0)
counts = counts.loc[counts.sum(axis=1).sort_values().index]
ax = counts.plot.barh(stacked=True, figsize=(10, 16), width=0.8, color=["#f2c300", "#3b6fb6"])
ax.set_yticklabels([pretty(c) for c in counts.index], fontsize=9)
ax.set_xlabel("original images")
ax.set_ylabel("")
ax.set_title(f"Original images per character ({len(originals):,} in total)")
ax.legend(title="")
plt.tight_layout()
plt.show()
largest, smallest = counts.sum(axis=1).iloc[-1], counts.sum(axis=1).iloc[0]
print(f"imbalance: {largest} / {smallest} = {largest / smallest:.1f}:1")"""),
    md("## 3. What the characters look like"),
    code("""top = counts.sum(axis=1).sort_values(ascending=False).index[:24]
fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
for ax, character in zip(axes.flat, top):
    pool = originals[(originals.character == character) & (originals.source == "curated image")]
    pool = pool if len(pool) else originals[originals.character == character]
    row = pool.sample(1, random_state=3).iloc[0]
    ax.imshow(Image.open(ROOT / character / row.file).convert("RGB"))
    ax.set_title(f"{pretty(character)}\\n{int(counts.loc[character].sum()):,} images", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()"""),
    md("""## 4. Episode frames by season

The frames were taken from [Frinkiac](https://frinkiac.com) at the moments when each character speaks, according to the script lines of [The Simpsons by the Data](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset)."""),
    code("""frames = originals[originals.source == "episode frame"]
per_season = frames.season.astype(int).value_counts().sort_index()
ax = per_season.plot.bar(figsize=(11, 3.5), color="#3b6fb6", width=0.8)
ax.set_xlabel("season")
ax.set_ylabel("frames")
n_episodes = frames[["season", "episode"]].drop_duplicates().shape[0]
ax.set_title(f"{len(frames):,} episode frames from {n_episodes} episodes")
plt.tight_layout()
plt.show()"""),
    md("""## 5. The variants

- **`__alterada`** is an augmented copy of 25% of each character's originals: a 10-20° rotation, one color change and one texture filter. `_alteraciones.csv` logs every parameter.
- **`__parecida`** replaces a near-duplicate. Frames from the same scene look almost identical. In each group of near-identical images (cosine similarity >= 0.90 between 16x16 color thumbnails), one image stays and each of the others is replaced by a transformed version that no longer looks like the group. `_parecidas.csv` logs every group."""),
    code("""altered = pd.read_csv(ROOT / "_alteraciones.csv").sample(3, random_state=7)
similar = pd.read_csv(ROOT / "_parecidas.csv").sample(2, random_state=7)

fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
for col, row in enumerate(altered.itertuples()):
    for ax, name, title in ((axes[0, col], row.original, "original"),
                            (axes[1, col], row.alterada,
                             f"rotation {row.rotacion:+.0f}°, {row.color} {row.parametro_color:.2f},\\n{row.textura} {row.parametro_textura:.2f}")):
        ax.imshow(Image.open(ROOT / row.personaje / name).convert("RGB"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
for col, row in enumerate(similar.itertuples(), start=3):
    for ax, name, title in ((axes[0, col], row.representante, "kept (group representative)"),
                            (axes[1, col], row.parecida, f"replaces {row.original}\\nsimilarity {row.similitud_original:.2f} -> {row.similitud_final:.2f}")):
        ax.imshow(Image.open(ROOT / row.personaje / name).convert("RGB"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
fig.suptitle("Left: augmented copies (__alterada).   Right: near-duplicates and their replacement (__parecida)")
plt.tight_layout()
plt.show()"""),
    md("""## 6. A leakage-free split

1. Split the **originals**, stratified by character (here 70 / 20 / 10, seed 42).
2. Send every variant to the split of its original.
3. Use each `__parecida` **instead of** its original.

With these rules no split shares an image, an augmented copy or a near-duplicate with another split. The project's split gives 49,148 / 14,130 / 7,067 images."""),
    code("""train, rest = train_test_split(originals, test_size=0.30, stratify=originals.character, random_state=42)
val, test = train_test_split(rest, test_size=1 / 3, stratify=rest.character, random_state=42)
split_of = {(r.character, r.file): name
            for name, part in (("train", train), ("val", val), ("test", test))
            for r in part.itertuples()}
df["split"] = [split_of[key] for key in zip(df.character, df.original)]

replaced = set(zip(df.character[df.kind == "parecida"], df.original[df.kind == "parecida"]))
is_replaced = [(c, f) in replaced for c, f in zip(df.character, df.file)]
usable = df[~((df.kind == "original") & pd.Series(is_replaced, index=df.index))]

table = usable.pivot_table(index="split", columns="kind", values="file", aggfunc="count", fill_value=0)
table["total"] = table.sum(axis=1)
display(table.loc[["train", "val", "test"]])
usable.to_csv("split.csv", index=False)  # character, file, kind, original, source, split"""),
    md("""## 7. One image, two characters

Many frames show more than one character. `_gemelas.csv` lists 6,101 pairs of near-identical images stored under two different characters. When you evaluate, treat a prediction that names the other character in the pair as plausible, not as a plain error."""),
    code("""twins = pd.read_csv(ROOT / "_gemelas.csv").sample(3, random_state=11)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for col, row in enumerate(twins.itertuples()):
    for ax, character, name in ((axes[0, col], row.personaje, row.original),
                                (axes[1, col], row.otro_personaje, row.otro_original)):
        ax.imshow(Image.open(ROOT / character / name).convert("RGB"))
        ax.set_title(f"folder: {pretty(character)}", fontsize=10)
        ax.axis("off")
fig.suptitle("Near-identical images filed under two characters")
plt.tight_layout()
plt.show()"""),
    md("""## Next steps

- `split.csv` is ready to feed a Keras or PyTorch data loader.
- **Baselines.** On 128x128 images, a 4-block CNN with 473k parameters reaches **0.775 accuracy, 0.908 top-3 and 0.755 F1 macro**. An MLP on hand-crafted color histograms stops at **0.556**. The training loop, the Grad-CAM and SHAP analyses and the bias audit are on [GitHub](https://github.com/ByFlopp/LosSimpsons_DeepLearning).
- Report accuracy **by source**: curated images and episode frames behave very differently.

If this helped you, an upvote on the notebook or the dataset is appreciated."""),
]
nb = nbf.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, sys.argv[1])
print("ok", sys.argv[1])
