from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import io

import numpy as np

from benchmark_anomaly_models import BaseModelAdapter, PostprocessResult


class DummyAnomalyAdapter(BaseModelAdapter):
    """
    Minimal example adapter. Replace internals with your model code.
    """

    name = "dummy_anomaly"
    is_one_shot = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model = None

    def load(self) -> None:
        # Load weights, allocate device, etc.
        self.model = object()

    def preprocess(self, image_bytes: bytes) -> dict[str, Any]:
        # Return dict with at least "input"; optionally include "image" (HWC uint8 RGB).
        # Here we decode with PIL only if needed for heatmaps.
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)
        model_input = arr.astype(np.float32) / 255.0
        return {"input": model_input, "image": arr}

    def infer(self, model_input: Any) -> dict[str, Any]:
        # Return dict with anomaly_map (H,W), score, label.
        h, w = model_input.shape[:2]
        anomaly_map = np.random.rand(h, w).astype(np.float32)
        return {"anomaly_map": anomaly_map, "score": float(anomaly_map.mean()), "label": 0}

    def postprocess(
        self, raw_output: Any, image: np.ndarray | None = None, meta: dict[str, Any] | None = None
    ) -> PostprocessResult:
        # Use the default heatmap/mask generation.
        return super().postprocess(raw_output, image, meta)

    # Optional embedding/index hooks.
    def compute_embedding(self, model_input: Any) -> np.ndarray:
        return np.random.rand(512).astype(np.float32)

    def build_index(self, embeddings: list[Any]) -> dict[str, Any]:
        # Return dict with optional index_path to compute size on disk.
        return {"index_path": None}

    def query_index(self, embedding: Any) -> Any:
        return None

    def update_index(self, embedding: Any) -> Any:
        return None


class EfficientADAdapter(BaseModelAdapter):
    """
    EfficientAD adapter using anomalib Engine.predict on single images.
    Config keys:
      - ckpt_path: path to model checkpoint
      - imagenet_dir: path to imagenette/imagenet directory (required by EfficientAd)
      - image_size: [H, W] or [H] (default 256)
      - accelerator: "auto" | "cpu" | "gpu"
      - devices: int or None
    """

    name = "efficientad"
    is_one_shot = False

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.model = None
        self.engine = None
        self.ckpt_path = self.config.get("ckpt_path")
        self.imagenet_dir = self.config.get("imagenet_dir")
        self.output_base = self.config.get("output_base")
        self.image_size = self.config.get("image_size", 256)
        self.accelerator = self.config.get("accelerator", "auto")
        self.devices = self.config.get("devices", 1)

        if isinstance(self.image_size, (list, tuple)) and len(self.image_size) == 1:
            self.image_size = int(self.image_size[0])

    def load(self) -> None:
        if self.imagenet_dir is None:
            # Defaults from infer_ad.py/train_efficientad.py
            self.imagenet_dir = Path("/home/awais/Datasets/imagenette/imagenette2/train")
        if self.output_base is None:
            self.output_base = Path("/home/awais/Datasets/gm_anomaly_test_results_efficientad")
        if self.ckpt_path is None:
            self.ckpt_path = self.find_newest_ckpt(Path(self.output_base))
        print(f"[EfficientADAdapter] Using checkpoint: {self.ckpt_path}")

        from anomalib.engine import Engine
        from anomalib.models import EfficientAd
        import torch

        if self.ckpt_path:
            try:
                self.model = EfficientAd.load_from_checkpoint(
                    str(self.ckpt_path),
                    imagenet_dir=str(self.imagenet_dir),
                )
            except Exception:
                self.model = EfficientAd(imagenet_dir=str(self.imagenet_dir))
                state = torch.load(str(self.ckpt_path), map_location="cpu")
                state_dict = state.get("state_dict", state)
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = EfficientAd(imagenet_dir=str(self.imagenet_dir))
        self.engine = Engine(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=1,
        )

    @staticmethod
    def find_newest_ckpt(output_base: Path) -> Path:
        runs_glob = "efficientad_*"
        ckpts: list[Path] = []
        runs = sorted(output_base.glob(runs_glob))
        for run in runs:
            ckpts.extend(run.rglob("anomalib_run/EfficientAd/gm_folder/**/weights/lightning/model.ckpt"))
        if not ckpts:
            raise FileNotFoundError(
                f"No EfficientAD checkpoints found under: {output_base}/{runs_glob}\\n"
                "Expected: .../anomalib_run/EfficientAd/gm_folder/.../weights/lightning/model.ckpt"
            )
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0].resolve()

    def preprocess(self, image_bytes: bytes) -> dict[str, Any]:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(image_bytes)
        tmp.flush()
        tmp.close()

        return {"input": tmp.name, "image": arr, "meta": {"temp_path": tmp.name}}

    def preprocess_path(self, image_path: Path) -> dict[str, Any]:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        return {"input": str(image_path), "image": arr, "meta": {"path": str(image_path)}}

    def infer(self, model_input: Any) -> dict[str, Any]:
        if self.model is None or self.engine is None:
            raise RuntimeError("Model not loaded; call load() first")

        from anomalib.data import PredictDataset

        img_path = str(model_input)
        dataset = PredictDataset(path=img_path, image_size=self.image_size)
        preds = self.engine.predict(model=self.model, dataset=dataset, ckpt_path=None)
        if preds is None:
            raise RuntimeError("Engine.predict returned None")

        # Extract first prediction
        pred = preds[0]
        anomaly_maps = getattr(pred, "anomaly_map")
        pred_scores = getattr(pred, "pred_score")
        pred_labels = getattr(pred, "pred_label")

        if hasattr(anomaly_maps, "detach"):
            anomaly_maps = anomaly_maps.detach().cpu()
        if hasattr(pred_scores, "detach"):
            pred_scores = pred_scores.detach().cpu()
        if hasattr(pred_labels, "detach"):
            pred_labels = pred_labels.detach().cpu()

        amap = anomaly_maps[0].squeeze().numpy() if hasattr(anomaly_maps, "__len__") else np.array(anomaly_maps).squeeze()
        score = float(pred_scores[0]) if hasattr(pred_scores, "__len__") else float(pred_scores)
        label = int(pred_labels[0]) if hasattr(pred_labels, "__len__") else int(pred_labels)

        return {"anomaly_map": amap, "score": score, "label": label}
