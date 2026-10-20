"""Load a trained model and use it on a single picture (used by app.py and predict.py)."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .data import get_transforms
from .model import build_model, gradcam_layer


def gradcam(model, arch: str, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM: shows WHICH parts of the picture pushed the model towards its answer.

    Returns a (H, W) array in [0, 1]; bright = important. For a good model the bright
    area should sit on the cat/dog's face and body, not on the background.
    """
    store = {}

    def forward_hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = gradcam_layer(model, arch).register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act = store["act"]
    weights = act.grad.mean(dim=(2, 3), keepdim=True)          # how important is each feature map
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))     # weighted sum, keep positive only
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    from matplotlib import colormaps

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(image.convert("RGB")) / 255.0
    mixed = (1 - alpha) * base + alpha * heat
    return Image.fromarray((mixed * 255).astype(np.uint8))


class Predictor:
    def __init__(self, checkpoint: str = "models/best_model.pt", device: str = None):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file '{checkpoint}' not found. Train first with:  python train.py"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.arch = ckpt["arch"]
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.val_acc = ckpt.get("val_acc")
        self.model = build_model(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        _, self.tf = get_transforms(self.img_size)

    def predict(self, image: Image.Image, heatmap: bool = True):
        """Returns ({'cat': 0.97, 'dog': 0.03}, heatmap_image_or_None)."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0].cpu().numpy()
        scores = {c: float(p) for c, p in zip(self.classes, probs)}

        overlay = None
        if heatmap:
            try:  # the heatmap is a bonus; never let it break the prediction
                cam = gradcam(self.model, self.arch, x, int(probs.argmax()))
                overlay = overlay_heatmap(image, cam)
            except Exception as err:  # noqa: BLE001
                print(f"[warn] heatmap failed: {err}")
        return scores, overlay
