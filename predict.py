"""Predict one image from the command line.

    python predict.py path/to/photo.jpg
    python predict.py path/to/photo.jpg --heatmap outputs/heat.png
"""
import argparse

from PIL import Image

from catsdogs.inference import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--checkpoint", default="models/best_model.pt")
    p.add_argument("--heatmap", default=None, help="optional path to save the 'where it looked' picture")
    args = p.parse_args()

    predictor = Predictor(args.checkpoint)
    scores, overlay = predictor.predict(Image.open(args.image), heatmap=bool(args.heatmap))
    for name, prob in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{name:>4}: {prob:.1%}")
    if args.heatmap and overlay is not None:
        overlay.save(args.heatmap)
        print(f"Heatmap saved to {args.heatmap}")


if __name__ == "__main__":
    main()
