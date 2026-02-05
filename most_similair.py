#!/usr/bin/env python3
from pathlib import Path
import contextlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import cv2
from scipy.ndimage import gaussian_filter
import kornia as K
from torchvision.transforms.functional import pil_to_tensor

from metauas import MetaUAS, normalize, safely_load_state_dict

QUERY_IMAGE = "/home/awais/Datasets/analyze/query.jpg"
# Root folder containing per-part subfolders (e.g., .../134/*.jpg).
GOOD_IMAGES_ROOT = "/home/awais/Datasets/gm_good_images_sorted"
DEFECT_FOLDER = "/home/awais/Datasets/gm_defects"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

BATCH_SIZE = 256
NUM_WORKERS = 6
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Save cache to: <part_folder>/cache/{paths.json, embeds.pt}
CACHE_SUBDIR = "cache"
CACHE_PATHS = "paths.json"
CACHE_EMBEDS = "embeds.pt"

# ---- MetaUAS one-shot config ----
METAUAS_CHECKPOINT = "weights/metauas-512.ckpt"
METAUAS_IMAGE_SIZE = 512
METAUAS_ENCODER = "efficientnet-b4"
METAUAS_DECODER = "unet"
METAUAS_ENCODER_DEPTH = 5
METAUAS_DECODER_DEPTH = 5
METAUAS_NUM_ALIGNMENT_LAYERS = 3
METAUAS_ALIGNMENT_TYPE = "sa"
METAUAS_FUSION_POLICY = "cat"

# ---- Output ----
OUTPUT_DIR = "/home/awais/Datasets/gm_anomaly_test_results"
HEATMAP_ALPHA = 0.6
MIN_REGION_PCT = 0.002
MASK_OVERLAY_ALPHA = 0.45
MASK_DISPLAY_BLUR = 2.0

# ---- Segmentation mask thresholds (mean + k*std on byte-scale map) ----
DYNAMIC_THRESHOLDING = True
MASK_STD_K = 2.5
MASK_STD_K_SECOND = 2.0
SECONDARY_BLUR = 1.0

# ---- flip/orientation-sensitive rerank (no timing) ----
TOPK_RERANK = 50     # CLIP shortlist size
PIX_SIZE = 64        # 48/64/96; higher = stricter about layout/orientation
ALPHA = 0.35         # how much pixel-sim matters vs CLIP (0.2–0.6)
FLIP_PENALTY = 0.20  # penalize if candidate matches better when horizontally flipped


def list_images(folder: str):
    p = Path(folder)
    paths = []
    for ext in EXTS:
        paths += list(p.rglob(f"*{ext}"))
        paths += list(p.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def extract_leading_number(path: Path) -> int:
    digits = ""
    for ch in path.name:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f"No leading digits found in filename: {path.name}")
    return int(digits)


def select_good_dir_for_path(good_root: Path, image_path: Path) -> Path:
    part_num = extract_leading_number(image_path)
    candidate = good_root / str(part_num)
    if candidate.exists() and candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"No folder for file number {part_num} under: {good_root}")


class ImgDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), str(path)


@torch.inference_mode()
def build_cache(img_paths, cache_paths_file: Path, cache_embeds_file: Path, model, preprocess, device: str):
    valid_paths = []
    for p in img_paths:
        try:
            Image.open(p).verify()
        except Exception:
            print(f"skip_bad_image: {p}")
            continue
        valid_paths.append(p)
    if not valid_paths:
        raise FileNotFoundError(f"No valid images found in: {cache_paths_file.parent}")

    ds = ImgDS(valid_paths, preprocess)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    embeds = []
    out_paths = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    for xb, pb in dl:
        xb = xb.to(device, non_blocking=True)
        with autocast_ctx:
            feat = model.encode_image(xb)
        feat = F.normalize(feat, dim=-1)
        if device == "cuda":
            feat = feat.half()
        embeds.append(feat)
        out_paths += list(pb)

    embeds = torch.cat(embeds, dim=0).contiguous()

    cache_paths_file.write_text(json.dumps(out_paths), encoding="utf-8")
    torch.save(embeds.cpu(), cache_embeds_file)
    return out_paths, embeds  # embeds on GPU


@torch.inference_mode()
def load_cache(cache_paths_file: Path, cache_embeds_file: Path):
    paths = json.loads(cache_paths_file.read_text(encoding="utf-8"))
    embeds = torch.load(cache_embeds_file, map_location="cpu", weights_only=True)  # no warning
    return paths, embeds


@torch.inference_mode()
def pixel_embed(img: Image.Image, device: str) -> torch.Tensor:
    # orientation/layout-sensitive descriptor: low-res raw pixels
    im = img.resize((PIX_SIZE, PIX_SIZE), Image.BILINEAR).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    v = torch.from_numpy(arr).to(device).reshape(-1)  # (H*W*3,)
    v = v / (v.norm() + 1e-12)
    return v


def load_metauas(device: str):
    model = MetaUAS(
        METAUAS_ENCODER,
        METAUAS_DECODER,
        METAUAS_ENCODER_DEPTH,
        METAUAS_DECODER_DEPTH,
        METAUAS_NUM_ALIGNMENT_LAYERS,
        METAUAS_ALIGNMENT_TYPE,
        METAUAS_FUSION_POLICY,
    )
    model = safely_load_state_dict(model, METAUAS_CHECKPOINT)
    model.to(device).eval()
    return model, METAUAS_IMAGE_SIZE


@torch.inference_mode()
def run_one_shot(query_path: str, prompt_path: str, device: str):
    model, image_size = load_metauas(device)
    query_img = Image.open(query_path).convert("RGB")
    prompt_img = Image.open(prompt_path).convert("RGB")

    query_tensor = pil_to_tensor(query_img).float().unsqueeze(0) / 255.0
    prompt_tensor = pil_to_tensor(prompt_img).float().unsqueeze(0) / 255.0

    if query_tensor.shape[-2] != image_size or query_tensor.shape[-1] != image_size:
        resize_trans = K.augmentation.Resize([image_size, image_size], return_transform=False)
        query_tensor = resize_trans(query_tensor)
        prompt_tensor = resize_trans(prompt_tensor)

    test_data = {
        "query_image": query_tensor.to(device),
        "prompt_image": prompt_tensor.to(device),
    }

    predicted_masks = model(test_data)
    pixel_anomaly_map = predicted_masks[0, 0].detach()
    image_anomaly_pred = float(predicted_masks.max().item())

    return query_img, pixel_anomaly_map, image_anomaly_pred


def save_outputs(query_img: Image.Image, anomaly_map: torch.Tensor, output_dir: Path, rel_stem: str, suffix: str):
    raw_dir = output_dir / "raw"
    heat_dir = output_dir / "heat map"
    infer_dir = output_dir / "inference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    anomaly_map = normalize(anomaly_map).cpu().numpy()
    heat_gray = (anomaly_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    height, width = anomaly_map.shape
    resized_img = query_img.resize((width, height), Image.BICUBIC)
    resized_np = np.asarray(resized_img, dtype=np.uint8)
    heatmap = (HEATMAP_ALPHA * resized_np + (1.0 - HEATMAP_ALPHA) * heat).astype(np.uint8)
    Image.fromarray(heatmap).save(heat_dir / f"{rel_stem}.png")

    # Mean + k*std thresholding on byte-scale anomaly map.
    heat_f = heat_gray.astype(np.float32)
    thresh = float(heat_f.mean() + MASK_STD_K * heat_f.std())
    mask = (heat_f >= thresh).astype(np.uint8) * 255
    if DYNAMIC_THRESHOLDING:
        secondary = gaussian_filter(heat_f, sigma=SECONDARY_BLUR)
        thresh2 = float(secondary.mean() + MASK_STD_K_SECOND * secondary.std())
        mask2 = (secondary >= thresh2).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, mask2)
    mask_rgb = np.zeros_like(resized_np)
    mask_rgb[..., 0] = 255
    mask_alpha = (mask[..., None] / 255.0) * MASK_OVERLAY_ALPHA
    mask_overlay = (
        resized_np * (1.0 - mask_alpha) + mask_rgb * mask_alpha
    ).astype(np.uint8)
    Image.fromarray(mask_overlay).save(infer_dir / f"{rel_stem}.png")
    query_img.save(raw_dir / f"{rel_stem}{suffix}")


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(DEFECT_FOLDER).exists():
        raise FileNotFoundError(f"DEFECT_FOLDER not found: {DEFECT_FOLDER}")
    if not Path(METAUAS_CHECKPOINT).exists():
        raise FileNotFoundError(f"METAUAS_CHECKPOINT not found: {METAUAS_CHECKPOINT}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    defect_paths = list_images(DEFECT_FOLDER)
    if not defect_paths:
        raise FileNotFoundError(f"No images found in DEFECT_FOLDER: {DEFECT_FOLDER}")

    output_dir = Path(OUTPUT_DIR)
    good_root = Path(GOOD_IMAGES_ROOT)
    if not good_root.exists():
        raise FileNotFoundError(f"GOOD_IMAGES_ROOT not found: {good_root}")
    clip_cache: dict[str, tuple[list[str], torch.Tensor]] = {}

    for defect_path in defect_paths:
        good_dir = select_good_dir_for_path(good_root, defect_path)
        cache_key = str(good_dir.resolve())
        cached = clip_cache.get(cache_key)
        if cached is None:
            cache_dir = good_dir / CACHE_SUBDIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_paths_file = cache_dir / CACHE_PATHS
            cache_embeds_file = cache_dir / CACHE_EMBEDS
            if cache_paths_file.exists():
                cache_paths_file.unlink()
            if cache_embeds_file.exists():
                cache_embeds_file.unlink()
            img_paths = list_images(str(good_dir))
            if not img_paths:
                raise FileNotFoundError(f"No images found in folder: {good_dir}")
            paths, embeds = build_cache(img_paths, cache_paths_file, cache_embeds_file, model, preprocess, device)
            clip_cache[cache_key] = (paths, embeds)
        else:
            paths, embeds = cached

        qimg = Image.open(defect_path).convert("RGB")
        q = preprocess(qimg).unsqueeze(0).to(device)
        with autocast_ctx:
            qfeat = model.encode_image(q)
        qfeat = F.normalize(qfeat, dim=-1).squeeze(0)
        if device == "cuda":
            qfeat = qfeat.half()

        clip_scores = embeds @ qfeat  # (N,)
        k = min(TOPK_RERANK, clip_scores.numel())
        topk_idx = torch.topk(clip_scores, k=k).indices.tolist()

        q_pix = pixel_embed(qimg, device)
        best_i = topk_idx[0]
        best_final = -1e9

        for idx in topk_idx:
            cimg = Image.open(paths[idx]).convert("RGB")
            pix_sim = float((pixel_embed(cimg, device) @ q_pix).item())
            cimg_flip = cimg.transpose(Image.FLIP_LEFT_RIGHT)
            flip_sim = float((pixel_embed(cimg_flip, device) @ q_pix).item())

            penalty = FLIP_PENALTY if flip_sim > pix_sim else 0.0
            final = float(clip_scores[idx].item()) + ALPHA * pix_sim - penalty

            if final > best_final:
                best_final = final
                best_i = idx

        query_img, anomaly_map, image_score = run_one_shot(str(defect_path), paths[best_i], device)
        rel_stem = defect_path.relative_to(DEFECT_FOLDER).with_suffix("")
        rel_stem = str(rel_stem).replace("/", "_")
        save_outputs(query_img, anomaly_map, output_dir, rel_stem, defect_path.suffix or ".png")
        print(f"{defect_path} -> {paths[best_i]} | score={image_score:.4f}")


if __name__ == "__main__":
    main()
