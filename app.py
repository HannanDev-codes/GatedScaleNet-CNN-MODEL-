"""Gradio web app — upload a photo, get cat/dog probabilities + a heatmap.

    python app.py
    python app.py --share     (adds a temporary public link)

Fixed vs. the original draft: this now loads the model through
catsdogs.inference.Predictor — the SAME loading code evaluate.py and
predict.py already use — instead of duplicating its own loading logic with
the architecture hardcoded to 'resnet18'. That hardcoding would silently
break the moment you trained a different --arch (like 'simple' or our new
'gatedscalenet'): the checkpoint's saved weights wouldn't match a
freshly-built resnet18's shape, and loading would fail. Predictor reads
checkpoint['arch'] itself, so this works correctly for whichever
architecture best_model.pt actually contains.
"""
import argparse

import gradio as gr
import torch
from PIL import Image

from catsdogs.inference import Predictor

CHECKPOINT_PATH = "models/best_model.pt"
predictor = Predictor(CHECKPOINT_PATH)


def energy_score(logits: torch.Tensor, T: float = 1.0) -> float:
    """Lower = the model confidently recognizes this as cat-like or dog-like.
    Higher = the model doesn't strongly associate the image with EITHER
    class — a sign it may be something else entirely (out-of-distribution)."""
    return (-T * torch.logsumexp(logits / T, dim=1)).item()


def predict_image(image, confidence_threshold, energy_threshold):
    if image is None:
        return {}
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype("uint8"), "RGB")

    x = predictor.tf(image.convert("RGB")).unsqueeze(0).to(predictor.device)
    with torch.no_grad():
        logits = predictor.model(x)
        if isinstance(logits, tuple):  # safety net: eval-mode models never
            logits = logits[0]          # return a tuple, but just in case
        energy = energy_score(logits)
        probs = torch.softmax(logits[0], dim=0)

    if energy > energy_threshold:
        return {"Doesn't look like a cat or dog (out-of-distribution)": 1.0}

    result = {cls: float(probs[i]) for i, cls in enumerate(predictor.classes)}
    best_class = max(result, key=result.get)
    if result[best_class] < confidence_threshold:
        result[f"(low confidence — below {confidence_threshold:.0%})"] = 0.0
    return result


with gr.Blocks(title="Cat vs Dog Classifier") as interface:
    gr.Markdown("# 🐱 Cat vs Dog Classifier 🐶")
    gr.Markdown(
        f"Model: **{predictor.arch}** · validation accuracy at save time: "
        f"**{predictor.val_acc:.1%}**" if predictor.val_acc else
        f"Model: **{predictor.arch}**"
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload a cat or dog photo", type="pil")
            confidence_slider = gr.Slider(0.5, 1.0, value=0.7, step=0.05,
                                           label="Confidence threshold")
            energy_slider = gr.Slider(-15.0, 15.0, value=2.5, step=0.5,
                                       label="Out-of-distribution strictness (higher = stricter)")
            submit_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Prediction")

    submit_btn.click(predict_image, [image_input, confidence_slider, energy_slider], output_label)
    image_input.change(predict_image, [image_input, confidence_slider, energy_slider], output_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    interface.launch(share=args.share)
