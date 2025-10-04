import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import shutil

def parse_custom_label(txt_path: str):
    """
    Your format: last (3rd) line contains: x1 y1 x2 y2 (pixel coords)
    Returns (x1, y1, x2, y2) or None
    """
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        return None
    parts = lines[-1].split()
    if len(parts) != 4:
        return None
    x1, y1, x2, y2 = map(float, parts)
    return int(x1), int(y1), int(x2), int(y2)

def export_crops_from_custom(
    images_root: str,
    labels_root: str,
    out_root: str,
    split: str = "train",
    # how to get class label:
    # 1) From parent folder of the image (e.g., .../Toyota/images/img.jpg)
    use_parent_dir_as_class: bool = True,
    # 2) OR from a mapping dict { "<relative/path/without_ext>": "ClassName" }
    relpath_to_class: Optional[Dict[str, str]] = None,
    # 3) OR from a global class name (all crops same class) — not common, but handy
    constant_class: Optional[str] = None,
    labels_suffix: str = ".txt",       # your custom txt suffix
    images_subdir_token: str = "/images/",  # used to build rel path for label lookup
    resize: Optional[Tuple[int,int]] = (224, 224),
    pad: int = 4,
    min_side: int = 10,
    overwrite_split: bool = True,
):
    """
    Walks images_root recursively, for each image finds a matching custom label in labels_root
    and writes a single crop per image to out_root/<split>/<class_name>/...
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    out_split = Path(out_root) / split
    if overwrite_split and out_split.exists():
        shutil.rmtree(out_split)
    out_split.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg"}
    n_written = 0

    for img_path in images_root.rglob("*"):
        if img_path.suffix.lower() not in img_exts:
            continue

        # Derive label path using the same logic you showed
        # Example: /.../<class>/images/foo/bar/img.jpg  -> rel = foo/bar/img
        p_str = str(img_path)
        if images_subdir_token in p_str:
            rel = p_str.split(images_subdir_token, 1)[-1]
        else:
            # fallback: relative to images_root
            rel = str(img_path.relative_to(images_root))
        rel_wo_ext = os.path.splitext(rel)[0]
        txt_custom = labels_root / f"{rel_wo_ext}{labels_suffix}"
        if not txt_custom.exists():
            continue

        bbox = parse_custom_label(str(txt_custom))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # clamp, pad, and skip tiny
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            continue

        crop = img[y1:y2, x1:x2]
        if resize:
            crop = cv2.resize(crop, resize, interpolation=cv2.INTER_LINEAR)

        # Determine class name
        if constant_class:
            cls_name = constant_class
        elif relpath_to_class is not None:
            # lookup by rel path without extension
            cls_name = relpath_to_class.get(rel_wo_ext)
            if cls_name is None:
                # try fallback to parent dir if mapping missing
                cls_name = img_path.parent.parent.name if use_parent_dir_as_class else "unknown"
        elif use_parent_dir_as_class:
            # If images live under .../<Class>/images/...
            # parent of "images" dir is the class folder
            # e.g., .../Toyota/images/xxx.jpg -> parent.parent.name == "Toyota"
            try:
                if img_path.parent.name.lower() == "images":
                    cls_name = img_path.parent.parent.name
                else:
                    cls_name = img_path.parent.name
            except Exception:
                cls_name = "unknown"
        else:
            cls_name = "unknown"

        out_dir = out_split / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # save; allow multiple crops/image if needed; here one bbox per file
        stem = img_path.stem
        out_file = out_dir / f"{stem}.jpg"
        i = 0
        while out_file.exists():  # avoid collisions
            i += 1
            out_file = out_dir / f"{stem}_{i}.jpg"
        cv2.imwrite(str(out_file), crop)
        n_written += 1

    return n_written
