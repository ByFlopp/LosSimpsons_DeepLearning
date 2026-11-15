## The Simpsons Characters: 64 classes, 56,268 images

An image classification dataset with **64 characters** from *The Simpsons*, one folder per character. The label of every image is the name of its folder.

- **56,268 original RGB images**
  - **22,162 curated character images**, mostly from [Los Simpson (alfaro96)](https://www.kaggle.com/datasets/alfaro96/los-simpson), plus a few screenshots added by the team.
  - **34,106 episode frames** from **557 episodes** (seasons 1 to 28).
- **23,117 synthetic variants**, used by the training pipeline:
  - 14,077 augmented copies (`__alterada`);
  - 9,040 de-duplicated replacements (`__parecida`).
- **Class imbalance of about 20:1**: from 3,118 images of `homer_simpson` down to 151 of `maggie_simpson`.

### Folder structure

```
/kaggle/input/los-simpsons-dataset/
├── homer_simpson/
│   ├── homer_simpson_0001.jpg             original
│   ├── homer_simpson_0003__alterada.jpg   augmented copy of homer_simpson_0003.jpg
│   └── homer_simpson_0017__parecida.jpg   replaces the near-duplicate homer_simpson_0017.jpg
├── ... 63 more character folders
├── _renombres.csv      original file name of every renamed image (episode and timestamp of each frame)
├── _alteraciones.csv   transformation applied to every __alterada image
├── _parecidas.csv      near-duplicate groups and the __parecida image that replaces each duplicate
└── _gemelas.csv        near-identical images found in the folders of two different characters
```

Every original is named `{character}_{number}.{ext}`. Most files are `.jpg`; there are 40 `.png` and 2 `.webp`.

### How to use the variants

| Suffix | What it is | How to use it |
|---|---|---|
| none | Original image | Split these first: stratified by character |
| `__alterada` | Copy of 25% of each character's originals with a 10-20° rotation, one color change and one texture filter (seed 42) | Put it in the **same split as its original**, or drop it if you use your own augmentation |
| `__parecida` | Inside one folder, near-identical originals (cosine similarity >= 0.90 between 16x16 color thumbnails) form a group. One image per group stays intact; each of the others gets a transformed version that no longer looks like the group | Use it **instead of** its original, so a split never holds two near-identical images |

To work only with the originals, keep the files whose name has no `__`.

### Where the images come from

1. **Curated images.** These come from [Los Simpson](https://www.kaggle.com/datasets/alfaro96/los-simpson) by Juan Carlos Alfaro Jiménez. The character usually fills the frame.
2. **Episode frames.** The team took frames from [Frinkiac](https://frinkiac.com) at the moments when a character speaks.
   - **Finding the moments.** The script lines of [The Simpsons by the Data](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset) say who speaks, in which episode and when.
   - **Filtering.** A CNN sorted the frames into safe, doubtful and noise; only the doubtful ones were checked by hand.
   - **Tracing a frame.** `_renombres.csv` keeps the original name `{character}__SxxEyy__t{milliseconds}.jpg`, so every frame can be traced back to its episode and timestamp.
3. **Variants.** The script `generar_imagenes_alteradas.py` generated the `__alterada` and `__parecida` images and the three `_*.csv` logs. It is deterministic: with seed 42 it gives the same output every time.

### Baselines

Both models were trained on the same stratified 70/20/10 split (seed 42) at 128x128, on the same GPU (RTX 4060 Laptop). A prediction is correct when it names the character of the folder.

| Model | Input | Accuracy | Top-3 | F1 macro |
|---|---|---:|---:|---:|
| CNN, 4 convolutional blocks, 473k parameters | The 128x128 image | **0.775** | **0.908** | **0.755** |
| MLP, 1.06M parameters | 370 hand-crafted color descriptors (HSV and Lab histograms) | 0.556 | 0.732 | 0.557 |

The code, both notebooks and the technical reports (in Spanish) are on [GitHub](https://github.com/ByFlopp/LosSimpsons_DeepLearning).

### Known limitations

- **Source bias.** Episode frames are much harder than curated images: the CNN reaches 0.67 accuracy on frames and 0.93 on curated images. Report both numbers.
- **Several characters in one image.** Many frames show more than one character. `_gemelas.csv` lists 6,101 pairs of near-identical images stored under two different characters.
- **Label noise.** The label of a frame is the character who speaks at that moment, so a small share of frames may show that character only partially, or not at all.

### License and copyright

All images are frames or artwork from *The Simpsons* © 20th Television / Disney. They are shared **only for non-commercial research and education**. No ownership of the images is claimed. The CSV metadata and the curation work by the team are released under CC BY-NC-SA 4.0.

### Authors

Diego Peña y Lillo Lührs, Vincent Farenden Cerón, Rodrigo Martínez Becker and Iván Pavón Barría. Duoc UC, course *Técnicas Avanzadas para Machine Learning*, 2026.
