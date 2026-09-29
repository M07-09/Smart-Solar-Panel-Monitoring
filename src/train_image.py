"""Train the image model: ResNet-18 (ImageNet pre-trained) fine-tuned to classify
solar panel condition into 6 classes. Runs on the GPU (RTX 5060) when available.

Two-phase transfer learning:
  phase 1 — backbone frozen, only the new classifier head is trained
  phase 2 — whole network fine-tuned with a small learning rate
"""
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import CLASSES, IMAGE_METRICS_PATH, IMAGE_MODEL_PATH, IMAGE_SIZE, KAGGLE_IMAGES, RAW_DIR, RESULTS_DIR, ROOT

SEED = 42
IMG_DIR = RAW_DIR / "panel_images"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.25, 0.25, 0.2, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def ensure_images():
    """Copy the 6 class folders into data/raw/panel_images.

    Source: data/raw/solar_panel_images.zip if present (downloaded manually), otherwise kagglehub.
    """
    if IMG_DIR.exists() and any(IMG_DIR.iterdir()):
        return
    archive = RAW_DIR / "solar_panel_images.zip"
    if archive.exists():
        import zipfile

        src = RAW_DIR / "_images_extracted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(src)
    else:
        import kagglehub

        src = Path(kagglehub.dataset_download(KAGGLE_IMAGES))
    lower = {c.lower(): c for c in CLASSES}
    for d in src.rglob("*"):
        if d.is_dir() and d.name.lower() in lower:
            shutil.copytree(d, IMG_DIR / lower[d.name.lower()], dirs_exist_ok=True)
    if archive.exists():
        shutil.rmtree(src, ignore_errors=True)  # temporary extraction, the images now live in IMG_DIR


def collect() -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for p in sorted((IMG_DIR / cls).rglob("*")):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                continue  # skip corrupted files
            paths.append(p); labels.append(idx)
    return paths, labels


class PanelDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def build_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


def run_epoch(model, loader, device, crit, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    seed_all()
    ensure_images()
    paths, labels = collect()
    counts = {c: labels.count(i) for i, c in enumerate(CLASSES)}
    print("images per class:", counts)

    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    print(f"train {len(p_tr)} | val {len(p_va)} | test {len(p_te)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    kw = dict(num_workers=4, pin_memory=device.type == "cuda", persistent_workers=True)
    dl_tr = DataLoader(PanelDataset(p_tr, y_tr, train_tf), batch_size=32, shuffle=True, **kw)
    dl_va = DataLoader(PanelDataset(p_va, y_va, eval_tf), batch_size=64, **kw)
    dl_te = DataLoader(PanelDataset(p_te, y_te, eval_tf), batch_size=64, **kw)

    # Class weights counter the imbalance between classes
    freq = np.bincount(y_tr, minlength=len(CLASSES))
    weights = torch.tensor(len(y_tr) / (len(CLASSES) * np.maximum(freq, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = build_model(len(CLASSES)).to(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history, best_acc, best_state = [], 0.0, None
    t0 = time.time()

    phases = [("head", 4, 1e-3), ("finetune", 26, 3e-4)]
    for phase, epochs, lr in phases:
        for name, p in model.named_parameters():
            p.requires_grad = phase == "finetune" or name.startswith("fc")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, dl_tr, device, crit, opt, scaler)
            va_loss, va_acc = run_epoch(model, dl_va, device, crit)
            sched.step()
            history.append({"phase": phase, "epoch": len(history) + 1, "train_loss": tr_loss,
                            "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
            flag = ""
            if va_acc >= best_acc:
                best_acc, best_state, flag = va_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, " *"
            print(f"[{phase:8s}] ep {len(history):2d}  train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}{flag}")

    train_sec = time.time() - t0
    model.load_state_dict(best_state)
    y_true, y_pred = predict_all(model, dl_te, device)
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    out = {
        "model": "ResNet-18 (ImageNet pre-trained, fine-tuned)",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "classes": CLASSES,
        "images_per_class": counts,
        "split": {"train": len(p_tr), "val": len(p_va), "test": len(p_te)},
        "train_seconds": round(train_sec, 1),
        "best_val_acc": best_acc,
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
        "history": history,
    }
    IMAGE_METRICS_PATH.write_text(json.dumps(out, indent=2))
    torch.save({"state_dict": best_state, "classes": CLASSES, "image_size": IMAGE_SIZE}, IMAGE_MODEL_PATH)
    # Keep the test file list so the dashboard can offer unseen demo images
    (RESULTS_DIR / "image_test_files.json").write_text(json.dumps(   # paths relative to the project root
        [{"path": p.relative_to(ROOT).as_posix(), "label": CLASSES[y]} for p, y in zip(p_te, y_te)], indent=1))
    print(f"test accuracy {out['test_accuracy']:.4f} | macro F1 {out['test_f1_macro']:.4f} | {train_sec:.0f}s")


if __name__ == "__main__":
    main()
