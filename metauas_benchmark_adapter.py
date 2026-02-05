# python run_metauas_benchmark.py \
#   --async-preprocess --prefetch-size 64 \
#   --async-save --save-queue-size 300

#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from benchmark_anomaly_models import BaseModelAdapter, list_images
from metauas import MetaUAS, safely_load_state_dict


class MetaUASAdapter(BaseModelAdapter):
    name = "metauas"
    is_one_shot = True

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cfg = self.config
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(cfg.get("image_size", 512))
        self.checkpoint = cfg.get("checkpoint", "weights/metauas-512.ckpt")

        self.encoder = cfg.get("encoder", "efficientnet-b4")
        self.decoder = cfg.get("decoder", "unet")
        self.encoder_depth = int(cfg.get("encoder_depth", 5))
        self.decoder_depth = int(cfg.get("decoder_depth", 5))
        self.num_alignment_layers = int(cfg.get("num_alignment_layers", 3))
        self.alignment_type = cfg.get("alignment_type", "sa")
        self.fusion_policy = cfg.get("fusion_policy", "cat")

        self.prompt_path = cfg.get("prompt_path")
        self.reference_dir = cfg.get("reference_dir")
        self.prompt_mode = cfg.get("prompt_mode", "first")

        self.good_images_dir = cfg.get("good_images_dir")
        self.clip_model_name = cfg.get("clip_model_name", "ViT-B-32")
        self.clip_pretrained = cfg.get("clip_pretrained", "laion2b_s34b_b79k")
        self.cache_subdir = cfg.get("cache_subdir", "cache")
        self.cache_paths_name = cfg.get("cache_paths", "paths.json")
        self.cache_embeds_name = cfg.get("cache_embeds", "embeds.pt")
        self.cache_batch_size = int(cfg.get("cache_batch_size", 256))
        self.cache_num_workers = int(cfg.get("cache_num_workers", 4))
        self.rebuild_clip_cache = bool(cfg.get("rebuild_clip_cache", True))
        self.use_clip_prompt = bool(self.good_images_dir)

        self.model = None
        self.prompt_tensor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_cache: dict[str, tuple[list[str], torch.Tensor]] = {}
        self._nn_mismatches: list[tuple[Path, Path]] = []
        self._nn_checks = 0

    def _select_prompt_path(self) -> Path:
        if self.use_clip_prompt:
            raise RuntimeError("Dynamic prompt selection requires per-query inference")
        if self.prompt_path:
            path = Path(self.prompt_path)
            if not path.exists():
                raise FileNotFoundError(f"prompt_path not found: {path}")
            return path

        if self.reference_dir:
            ref_dir = Path(self.reference_dir)
            if not ref_dir.exists():
                raise FileNotFoundError(f"reference_dir not found: {ref_dir}")
            ref_paths = list_images(ref_dir)
            if not ref_paths:
                raise RuntimeError(f"No reference images found in: {ref_dir}")
            if self.prompt_mode == "random":
                return ref_paths[np.random.randint(0, len(ref_paths))]
            return ref_paths[0]

        raise ValueError("Must provide prompt_path or reference_dir in model config")

    def _prepare_tensor(self, img: Image.Image) -> torch.Tensor:
        t = pil_to_tensor(img).float().unsqueeze(0) / 255.0
        if t.shape[-2] != self.image_size or t.shape[-1] != self.image_size:
            t = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return t

    def _ensure_clip_model(self) -> None:
        if self._clip_model is not None and self._clip_preprocess is not None:
            return
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        self._clip_model = model.to(self.device).eval()
        self._clip_preprocess = preprocess

    @staticmethod
    def _extract_leading_number(path: Path) -> int:
        digits = ""
        for ch in path.name:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            raise ValueError(f"No leading digits found in filename: {path.name}")
        return int(digits)

    def _select_good_dir_for_path(self, image_path: Path | None) -> Path:
        good_root = Path(self.good_images_dir)
        if not good_root.exists():
            raise FileNotFoundError(f"good_images_dir not found: {good_root}")

        if image_path is None:
            return good_root

        part_num = self._extract_leading_number(image_path)
        candidate = good_root / str(part_num)
        if candidate.exists() and candidate.is_dir():
            return candidate

        # Fallback: if the root contains images directly (non-sorted layout), use it.
        has_direct_images = any(
            f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            for f in good_root.iterdir()
        )
        if has_direct_images:
            return good_root

        raise FileNotFoundError(f"No folder for file number {part_num} under: {good_root}")

    def _is_general_mode(self, good_dir: Path | None) -> bool:
        if good_dir is None:
            return False
        return good_dir.resolve() == Path(self.good_images_dir).resolve()

    class _ClipImgDS(Dataset):
        def __init__(self, paths: list[Path], preprocess):
            self.paths = paths
            self.preprocess = preprocess

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            path = self.paths[idx]
            img = Image.open(path).convert("RGB")
            return self.preprocess(img), str(path)

    def _build_clip_cache(
        self,
        img_paths: list[Path],
        cache_paths_file: Path,
        cache_embeds_file: Path,
    ) -> tuple[list[str], torch.Tensor]:
        valid_paths: list[Path] = []
        for p in img_paths:
            try:
                Image.open(p).verify()
            except Exception:
                print(f"skip_bad_image: {p}")
                continue
            valid_paths.append(p)
        if not valid_paths:
            raise FileNotFoundError(f"No valid images found in: {cache_paths_file.parent}")

        ds = self._ClipImgDS(valid_paths, self._clip_preprocess)
        dl = DataLoader(
            ds,
            batch_size=self.cache_batch_size,
            shuffle=False,
            num_workers=self.cache_num_workers,
            pin_memory=True,
            persistent_workers=(self.cache_num_workers > 0),
        )

        embeds = []
        out_paths: list[str] = []

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )

        for xb, pb in dl:
            xb = xb.to(self.device, non_blocking=True)
            with autocast_ctx:
                feat = self._clip_model.encode_image(xb)
            feat = F.normalize(feat, dim=-1)
            if self.device == "cuda":
                feat = feat.half()
            embeds.append(feat)
            out_paths += list(pb)

        embeds = torch.cat(embeds, dim=0).contiguous()
        cache_paths_file.write_text(json.dumps(out_paths), encoding="utf-8")
        torch.save(embeds.cpu(), cache_embeds_file)
        return out_paths, embeds

    def _load_clip_cache_for_dir(self, good_dir: Path) -> tuple[list[str], torch.Tensor]:
        key = str(good_dir.resolve())
        cached = self._clip_cache.get(key)
        if cached is not None:
            return cached

        cache_dir = good_dir / self.cache_subdir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_paths_file = cache_dir / self.cache_paths_name
        cache_embeds_file = cache_dir / self.cache_embeds_name

        if self.rebuild_clip_cache:
            if cache_paths_file.exists():
                cache_paths_file.unlink()
            if cache_embeds_file.exists():
                cache_embeds_file.unlink()
            paths = []
            embeds = torch.empty(0)
        else:
            paths = []
            embeds = torch.empty(0)

        if not paths or embeds.numel() == 0:
            img_paths = list_images(good_dir)
            if not img_paths:
                raise FileNotFoundError(f"No images found in: {good_dir}")
            paths, embeds = self._build_clip_cache(img_paths, cache_paths_file, cache_embeds_file)

        self._clip_cache[key] = (paths, embeds)
        return paths, embeds

    def load(self) -> None:
        model = MetaUAS(
            self.encoder,
            self.decoder,
            self.encoder_depth,
            self.decoder_depth,
            self.num_alignment_layers,
            self.alignment_type,
            self.fusion_policy,
        )
        model = safely_load_state_dict(model, self.checkpoint)
        model.to(self.device).eval()
        self.model = model

        if self.use_clip_prompt:
            self._ensure_clip_model()
        else:
            prompt_path = self._select_prompt_path()
            prompt_img = Image.open(prompt_path).convert("RGB")
            prompt_tensor = self._prepare_tensor(prompt_img).to(self.device)
            self.prompt_tensor = prompt_tensor

    def warmup(self, sample_input: Any | None = None) -> None:
        if self.model is None or (self.prompt_tensor is None and not self.use_clip_prompt):
            return
        if sample_input is None and self.prompt_tensor is not None:
            sample_input = self.prompt_tensor
        if sample_input is None:
            return
        if isinstance(sample_input, dict):
            test_data = dict(sample_input)
            if "prompt_image" not in test_data and self.prompt_tensor is not None:
                test_data["prompt_image"] = self.prompt_tensor
            if "query_image" not in test_data:
                return
            # Ensure tensors are on the right device for warmup
            test_data["query_image"] = test_data["query_image"].to(self.device)
            test_data["prompt_image"] = test_data["prompt_image"].to(self.device)
        else:
            test_data = {
                "query_image": sample_input.to(self.device),
                "prompt_image": self.prompt_tensor.to(self.device),
            }
        with torch.inference_mode():
            _ = self.model(test_data)

    def _preprocess_image(self, img: Image.Image, image_path: Path | None) -> dict[str, Any]:
        image_arr = np.asarray(img)
        query_tensor = self._prepare_tensor(img)
        prompt_tensor = self.prompt_tensor
        prompt_path = None
        good_dir = None
        part_num = None

        if self.use_clip_prompt:
            if self._clip_model is None or self._clip_preprocess is None:
                raise RuntimeError("CLIP model not initialized for prompt selection")
            good_dir = self._select_good_dir_for_path(image_path)
            if image_path is not None:
                part_num = self._extract_leading_number(image_path)
            try:
                paths, embeds = self._load_clip_cache_for_dir(good_dir)
            except FileNotFoundError as exc:
                print(f"warn_no_valid_images: {exc} | using self-prompt for {image_path}")
                prompt_tensor = query_tensor.to(self.device)
                paths, embeds = [], None
            else:
                q = self._clip_preprocess(img).unsqueeze(0).to(self.device)
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.device == "cuda"
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), autocast_ctx:
                    qfeat = self._clip_model.encode_image(q)
                qfeat = F.normalize(qfeat, dim=-1).squeeze(0)
                if embeds.dtype == torch.float16:
                    qfeat = qfeat.half()
                sims = embeds @ qfeat
                best_i = int(torch.argmax(sims).item())
                prompt_path = Path(paths[best_i])
                prompt_img = Image.open(prompt_path).convert("RGB")
                prompt_tensor = self._prepare_tensor(prompt_img).to(self.device)
                if image_path is not None:
                    print(f"nearest_neighbor: {image_path} -> {prompt_path}")
                    if self._is_general_mode(good_dir):
                        try:
                            qn = self._extract_leading_number(image_path)
                            pn = self._extract_leading_number(prompt_path)
                            self._nn_checks += 1
                            if qn != pn:
                                self._nn_mismatches.append((image_path, prompt_path))
                        except Exception:
                            pass
        elif self.prompt_path:
            prompt_path = Path(self.prompt_path)
        elif self.reference_dir:
            prompt_path = self._select_prompt_path()

        meta = {
            "prompt_mode": self.prompt_mode,
            "prompt_path": str(prompt_path) if prompt_path else None,
        }
        if good_dir is not None:
            meta["good_images_dir"] = str(good_dir)
        if part_num is not None:
            meta["file_number"] = part_num

        return {
            "input": {"query_image": query_tensor, "prompt_image": prompt_tensor},
            "image": image_arr,
            "meta": meta,
        }

    def report_nn_mismatches(self) -> None:
        if self._nn_checks == 0:
            print("nn_mismatch_summary: no comparable nearest-neighbor checks")
            return
        total = self._nn_checks
        failed = len(self._nn_mismatches)
        passed = total - failed
        if failed:
            print("nn_mismatch_details:")
            for query_path, prompt_path in self._nn_mismatches:
                print(f"  mismatch: {query_path} -> {prompt_path}")
        pct = (passed / total) * 100.0
        print(f"nn_mismatch_summary: passed={passed} failed={failed} total={total} pass_pct={pct:.2f}%")

    def preprocess(self, image_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.use_clip_prompt:
            # Warmup path uses bytes only; avoid building a full cache.
            good_root = Path(self.good_images_dir)
            if not good_root.exists():
                raise FileNotFoundError(f"good_images_dir not found: {good_root}")
            prompt_candidates = list_images(good_root)
            if not prompt_candidates:
                raise FileNotFoundError(f"No images found in: {good_root}")
            prompt_img = Image.open(prompt_candidates[0]).convert("RGB")
            prompt_tensor = self._prepare_tensor(prompt_img).to(self.device)
            image_arr = np.asarray(img)
            query_tensor = self._prepare_tensor(img)
            return {
                "input": {"query_image": query_tensor, "prompt_image": prompt_tensor},
                "image": image_arr,
                "meta": {
                    "prompt_mode": "fallback",
                    "prompt_path": str(prompt_candidates[0]),
                },
            }
        return self._preprocess_image(img, None)

    def preprocess_path(self, image_path: Path) -> dict[str, Any]:
        img = Image.open(image_path).convert("RGB")
        return self._preprocess_image(img, image_path)

    def infer(self, model_input: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        query = model_input["query_image"].to(self.device)
        prompt = model_input["prompt_image"].to(self.device)
        with torch.inference_mode():
            pred = self.model({"query_image": query, "prompt_image": prompt})
        if isinstance(pred, torch.Tensor):
            anomaly_map = pred[0, 0].detach().cpu().numpy()
            score = float(pred.max().item())
            return {"anomaly_map": anomaly_map, "score": score, "label": None}
        return {"anomaly_map": None, "score": None, "label": None}
