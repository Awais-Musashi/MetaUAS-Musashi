#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip

QUERY_IMAGE = "/home/awais/Datasets/analyze/query.jpg"
FOLDER = "/home/awais/Datasets/gm_good_images"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

BATCH_SIZE = 256
NUM_WORKERS = 6
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Save cache to: FOLDER/cache/{paths.json, embeds.pt}
CACHE_SUBDIR = "cache"
CACHE_PATHS = "paths.json"
CACHE_EMBEDS = "embeds.pt"

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
    ds = ImgDS(img_paths, preprocess)
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

    for xb, pb in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feat = model.encode_image(xb)
        feat = F.normalize(feat, dim=-1).half()
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


@torch.inference_mode()
def main():
    device = "cuda"

    cache_dir = Path(FOLDER) / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths_file = cache_dir / CACHE_PATHS
    cache_embeds_file = cache_dir / CACHE_EMBEDS

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()

    if cache_paths_file.exists() and cache_embeds_file.exists():
        paths, embeds = load_cache(cache_paths_file, cache_embeds_file)
        embeds = embeds.to(device, non_blocking=True)
    else:
        img_paths = list_images(FOLDER)
        paths, embeds = build_cache(img_paths, cache_paths_file, cache_embeds_file, model, preprocess, device)

    # ---- query CLIP embedding ----
    qimg = Image.open(QUERY_IMAGE).convert("RGB")
    q = preprocess(qimg).unsqueeze(0).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        qfeat = model.encode_image(q)
    qfeat = F.normalize(qfeat, dim=-1).squeeze(0).half()

    # ---- CLIP shortlist ----
    clip_scores = embeds @ qfeat  # (N,)
    k = min(TOPK_RERANK, clip_scores.numel())
    topk_idx = torch.topk(clip_scores, k=k).indices.tolist()

    # ---- orientation-sensitive rerank with flip penalty ----
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

    print("Most similar:", paths[best_i])
    print("Cosine similarity:", float(clip_scores[best_i].item()))


if __name__ == "__main__":
    main()