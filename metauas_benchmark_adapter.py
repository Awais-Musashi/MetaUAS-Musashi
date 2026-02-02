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
        self.use_clip_prompt = bool(self.good_images_dir)

        self.model = None
        self.prompt_tensor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_embeds = None
        self._clip_paths = None

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

    def _load_clip_cache(self) -> None:
        good_dir = Path(self.good_images_dir)
        if not good_dir.exists():
            raise FileNotFoundError(f"good_images_dir not found: {good_dir}")

        cache_dir = good_dir / self.cache_subdir
        cache_paths_file = cache_dir / self.cache_paths_name
        cache_embeds_file = cache_dir / self.cache_embeds_name
        if not cache_paths_file.exists() or not cache_embeds_file.exists():
            raise RuntimeError(
                f"Missing cache in {cache_dir}. Build it first with m1.py or most_similair.py."
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        model = model.to(self.device).eval()

        paths = json.loads(cache_paths_file.read_text(encoding="utf-8"))
        embeds = torch.load(cache_embeds_file, map_location="cpu", weights_only=True)
        embeds = embeds.to(self.device, non_blocking=True)

        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_paths = paths
        self._clip_embeds = embeds

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
            self._load_clip_cache()
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
        else:
            test_data = {"query_image": sample_input, "prompt_image": self.prompt_tensor}
        with torch.inference_mode():
            _ = self.model(test_data)

    def preprocess(self, image_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_arr = np.asarray(img)
        query_tensor = self._prepare_tensor(img)
        prompt_tensor = self.prompt_tensor
        prompt_path = None

        if self.use_clip_prompt:
            if self._clip_model is None or self._clip_preprocess is None:
                raise RuntimeError("CLIP model not initialized for prompt selection")
            q = self._clip_preprocess(img).unsqueeze(0).to(self.device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device == "cuda"
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                qfeat = self._clip_model.encode_image(q)
            qfeat = F.normalize(qfeat, dim=-1).squeeze(0)
            if self._clip_embeds.dtype == torch.float16:
                qfeat = qfeat.half()
            sims = self._clip_embeds @ qfeat
            best_i = int(torch.argmax(sims).item())
            prompt_path = Path(self._clip_paths[best_i])
            prompt_img = Image.open(prompt_path).convert("RGB")
            prompt_tensor = self._prepare_tensor(prompt_img).to(self.device)
        elif self.prompt_path:
            prompt_path = Path(self.prompt_path)
        elif self.reference_dir:
            prompt_path = self._select_prompt_path()

        return {
            "input": {"query_image": query_tensor, "prompt_image": prompt_tensor},
            "image": image_arr,
            "meta": {"prompt_mode": self.prompt_mode, "prompt_path": str(prompt_path) if prompt_path else None},
        }

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
