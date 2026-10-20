# Dataset

345 real cat/dog photos, already split:

| | cat | dog | total |
|---|---|---|---|
| train | 95 | 180 | 275 |
| val | 24 | 46 | 70 |

```
data/
├── train/cat/*.jpg
├── train/dog/*.jpg
├── val/cat/*.jpg
└── val/dog/*.jpg
```

This exact folder shape is what `torchvision.datasets.ImageFolder` expects — the folder name is the label (`cat` = class 0, `dog` = class 1, alphabetical order).

Note: each of `train/` and `val/` also contains a small `classname.txt` file from the original dataset export. `ImageFolder` only looks at sub-*folders* for class names, so this stray file is ignored automatically and causes no issue.
