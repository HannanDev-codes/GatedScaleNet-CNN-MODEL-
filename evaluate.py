"""Check a trained model on the validation images: accuracy, per-class report, confusion matrix.

    python evaluate.py
    python evaluate.py --checkpoint models/best_model.pt --data-dir data
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from catsdogs.data import get_transforms
from catsdogs.inference import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="val", help="folder name to evaluate on (val, or a test folder)")
    p.add_argument("--out", default="outputs/confusion_matrix.png")
    args = p.parse_args()

    pred = Predictor(args.checkpoint)
    _, eval_tf = get_transforms(pred.img_size)
    ds = datasets.ImageFolder(Path(args.data_dir) / args.split, transform=eval_tf)
    if ds.classes != pred.classes:
        raise ValueError(f"Dataset classes {ds.classes} differ from model classes {pred.classes}")
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = pred.model(images.to(pred.device))
            y_pred += logits.argmax(1).cpu().tolist()
            y_true += labels.tolist()

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"Accuracy on '{args.split}': {acc:.1%}  ({len(y_true)} images)\n")
    print(classification_report(y_true, y_pred, target_names=pred.classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(pred.classes)), yticks=range(len(pred.classes)),
           xticklabels=pred.classes, yticklabels=pred.classes,
           xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Confusion matrix saved to {args.out}")


if __name__ == "__main__":
    main()
