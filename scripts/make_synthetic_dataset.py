from __future__ import annotations
from pathlib import Path
import os
import math
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw

def _resolve_project_root() -> Path:
    try:
        import IPython
        ip = IPython.get_ipython()
        if ip is not None:
            try:
                import ipynbname
                nb_path = Path(str(ipynbname.path()))
                notebook_dir = nb_path.parent.resolve()
                return notebook_dir.parent.resolve()
            except Exception:
                pass
    except Exception:
        pass

    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        return script_dir.parent.resolve()
    return Path.cwd().resolve().parent

PROJECT_ROOT: Path = _resolve_project_root()
FOLDER_ROOT: Path = PROJECT_ROOT / "scripts"
SYNTH_ROOT: Path = PROJECT_ROOT / "synthetic_dataset"
SYNTH_ZIP: Path = PROJECT_ROOT / "synthetic_dataset.zip"

assert (SYNTH_ROOT / "train").exists(), "Falta synthetic_dataset/train. Genera o descomprime el dataset sintético."
assert (SYNTH_ROOT / "val").exists(),   "Falta synthetic_dataset/val."
assert (SYNTH_ROOT / "test").exists(),  "Falta synthetic_dataset/test."

CLASS_CODES = ['0DS', '1DS', '2DS', '3DS']

IMG_SIZE = 255
BG_COLOR = (245, 245, 245)
FG_COLOR = (40, 40, 40)

def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def _shape_params_by_code(code):
    base_h = 150
    mapping = {
        '0DS': 120,
        '1DS': 100,
        '2DS':  85,
        '3DS':  70
    }
    return mapping[code], base_h

def _draw_torso(draw: ImageDraw.Draw, cx, cy, base_w, base_h, jitter=0.10, color=FG_COLOR):
    jw = 1.0 + np.random.uniform(-jitter, jitter)
    jh = 1.0 + np.random.uniform(-jitter, jitter)
    w = int(base_w * jw); h = int(base_h * jh)

    bbox = [cx - w//2, cy - h//2, cx + w//2, cy + h//2]
    draw.ellipse(bbox, fill=color)

    shoulder_w = int(w * (1.30 + np.random.uniform(-0.05, 0.05)))
    shoulder_h = int(h * 0.18)
    x0, y0 = cx - shoulder_w//2, cy - h//2 - int(0.15*h)
    poly = [(x0, y0 + shoulder_h), (x0 + shoulder_w, y0 + shoulder_h),
            (cx + int(0.45*shoulder_w), y0), (cx - int(0.45*shoulder_w), y0)]
    draw.polygon(poly, fill=color)

def _make_image(code, seed=None):
    if seed is not None:
        _set_seed(seed)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)

    cx = IMG_SIZE//2 + np.random.randint(-8, 8)
    cy = IMG_SIZE//2 + np.random.randint(-6, 6)

    base_w, base_h = _shape_params_by_code(code)
    _draw_torso(draw, cx, cy, base_w, base_h, jitter=0.10)

    for _ in range(np.random.randint(15, 30)):
        r = np.random.randint(4, 10)
        x = np.random.randint(0, IMG_SIZE)
        y = np.random.randint(0, IMG_SIZE)
        alpha = np.random.randint(10, 25)
        shade = (alpha,)*3
        draw.ellipse([x-r, y-r, x+r, y+r], fill=shade)

    return img

def _make_archive(root: Path, archive_format: str = "zip", archive_name: str | None = None) -> str:
    if archive_format not in ("zip", "gztar"):
        raise ValueError("archive_format debe ser 'zip' o 'gztar'.")

    base_name = str(root if archive_name is None else root.parent / archive_name)
    archive_path = shutil.make_archive(
        base_name=base_name,
        format=archive_format,
        root_dir=str(root.parent),
        base_dir=root.name
    )
    return archive_path

def generate_synthetic_ds_0DS_format(
    root="synthetic_ds",
    per_class=120,
    seed=42,
    make_archive=True,
    archive_format="zip",
    archive_name=None,
    try_colab_download=False
):
    _set_seed(seed)
    root = Path(root)
    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    n_train = math.floor(per_class * 0.70)
    n_val   = math.floor(per_class * 0.15)
    n_test  = per_class - n_train - n_val
    counts = {"train": n_train, "val": n_val, "test": n_test}
    for code in CLASS_CODES:
        for split, k in counts.items():
            for i in range(k):
                img_seed = random.randint(0, 10**9)
                img = _make_image(code, seed=img_seed)
                fname = f"{code}_{i:05d}.png"
                img.save(root / split / fname, format="PNG")

    total = len(CLASS_CODES) * per_class
    print(f"Dataset sintético creado en: {root.resolve()}")
    print(f"Por clase -> train={n_train}, val={n_val}, test={n_test} | total/clase={per_class} | total={total}")

    archive_path = None
    if make_archive:
        archive_path = _make_archive(root, archive_format=archive_format, archive_name=archive_name)
        print(f"Archivo generado: {archive_path}")

        if try_colab_download:
            try:
                from google.colab import files
                files.download(archive_path)
            except Exception as e:
                print(f"[Aviso] Descarga directa no disponible: {e}")

    return {
        "root": str(root.resolve()),
        "per_class": per_class,
        "splits_per_class": counts,
        "total_images": total,
        "archive_path": archive_path
    }

info = generate_synthetic_ds_0DS_format(
  root=SYNTH_ROOT,
  per_class=120,
  seed=42,
  make_archive=True,          
  archive_format="zip",       
  archive_name=None,          
  try_colab_download=True     
)