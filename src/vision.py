"""Inference wrapper for the trained panel-condition ResNet."""
import io
from functools import lru_cache

import torch
from PIL import Image

from config import IMAGE_MODEL_PATH
from train_image import build_model, eval_tf


@lru_cache(maxsize=1)
def _load():
    ckpt = torch.load(IMAGE_MODEL_PATH, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], device


def model_ready() -> bool:
    return IMAGE_MODEL_PATH.exists()


@torch.no_grad()
def classify(image_bytes: bytes) -> dict:
    model, classes, device = _load()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    probs = torch.softmax(model(eval_tf(img).unsqueeze(0).to(device)), dim=1)[0].cpu()
    top = int(probs.argmax())
    return {
        "label": classes[top],
        "confidence": float(probs[top]),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }
