#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import importlib
import json
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import time

import numpy as np
from PIL import Image
import cv2
import psutil

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None


# -------------------------------
# Utilities
# -------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for ext in IMG_EXTS:
        out.extend(folder.rglob(f"*{ext}"))
        out.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(out))


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_rss_bytes(proc: psutil.Process) -> int:
    return int(proc.memory_info().rss)


def get_vram_bytes() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.memory_allocated())


def reset_vram_peak() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def get_vram_peak() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    dur = time.perf_counter() - start
    return out, dur


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Postprocessing helpers
# -------------------------------
@dataclass
class PostprocessResult:
    anomaly_map: np.ndarray | None
    mask: np.ndarray | None
    heatmap: np.ndarray | None
    score: float | None
    label: int | None


def normalize_robust_01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if b - a < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, a, b)
    return (x - a) / (b - a)


def generate_heatmap_mask(anomaly_map_hw: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    amap = np.asarray(anomaly_map_hw, dtype=np.float32).squeeze()
    if amap.ndim != 2:
        raise ValueError(f"Expected anomaly map 2D (H,W). Got {amap.shape}")

    m01 = normalize_robust_01(amap, lo=2, hi=98)
    heat_gray = (m01 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    h, w = image_hw
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    hf = heat_gray.astype(np.float32)
    thresh = float(hf.mean() + 2.5 * hf.std())
    mask = (hf >= thresh).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return heat, mask


# -------------------------------
# Adapter interface
# -------------------------------
class BaseModelAdapter:
    name = "base"
    is_one_shot = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def load(self) -> None:
        pass

    def warmup(self, sample_input: Any | None = None) -> None:
        pass

    def preprocess(self, image_bytes: bytes) -> dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)
        return {"input": arr, "image": arr}

    def infer(self, model_input: Any) -> Any:
        return {"anomaly_map": None, "score": None, "label": None}

    def postprocess(self, raw_output: Any, image: np.ndarray | None = None, meta: dict[str, Any] | None = None) -> PostprocessResult:
        if isinstance(raw_output, dict):
            anomaly_map = raw_output.get("anomaly_map")
            score = raw_output.get("score")
            label = raw_output.get("label")
        else:
            anomaly_map = raw_output
            score = None
            label = None

        heatmap = None
        mask = None
        if anomaly_map is not None and image is not None:
            heatmap, mask = generate_heatmap_mask(anomaly_map, (image.shape[0], image.shape[1]))

        return PostprocessResult(anomaly_map=anomaly_map, mask=mask, heatmap=heatmap, score=score, label=label)

    # Optional embedding/index hooks
    def compute_embedding(self, model_input: Any) -> Any:
        raise NotImplementedError

    def build_index(self, embeddings: list[Any]) -> Any:
        raise NotImplementedError

    def query_index(self, embedding: Any) -> Any:
        raise NotImplementedError

    def update_index(self, embedding: Any) -> Any:
        raise NotImplementedError

    def infer_batch(self, batch_inputs: list[Any]) -> list[Any]:
        return [self.infer(x) for x in batch_inputs]


def load_adapter(module_path: str, class_name: str, config: dict[str, Any]) -> BaseModelAdapter:
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    adapter = cls(config)
    if not isinstance(adapter, BaseModelAdapter):
        raise TypeError(f"Adapter {class_name} must inherit from BaseModelAdapter")
    return adapter


def is_overridden(obj: Any, method_name: str) -> bool:
    base_impl = getattr(BaseModelAdapter, method_name, None)
    impl = getattr(obj.__class__, method_name, None)
    return impl is not None and impl is not base_impl


# -------------------------------
# Output saving
# -------------------------------
@dataclass
class SavePaths:
    raw_dir: Path
    heat_dir: Path
    mask_dir: Path
    raw_output_dir: Path


def rel_stem(path: Path, base: Path) -> str:
    return str(path.relative_to(base).with_suffix("")).replace("/", "_")


def save_outputs(
    image_path: Path,
    base_dir: Path,
    image_arr: np.ndarray | None,
    post: PostprocessResult,
    raw_output: Any,
    data_root: Path,
) -> dict[str, str]:
    raw_dir = base_dir / "raw"
    heat_overlay_dir = base_dir / "heatmap_overlay"
    mask_overlay_dir = base_dir / "mask_overlay"
    ensure_dirs(raw_dir, heat_overlay_dir, mask_overlay_dir)

    stem = rel_stem(image_path, data_root)
    outputs: dict[str, str] = {}

    if image_arr is not None:
        if hasattr(image_arr, "detach"):
            image_arr = image_arr.detach().cpu().numpy()
        if image_arr.ndim == 3 and image_arr.shape[0] in (1, 3) and image_arr.shape[-1] not in (1, 3):
            image_arr = np.transpose(image_arr, (1, 2, 0))
        if image_arr.dtype.kind in {"f", "c"}:
            vmax = float(np.max(image_arr)) if image_arr.size else 0.0
            if vmax <= 1.0:
                image_arr = (image_arr * 255.0).clip(0, 255)
        if image_arr.dtype != np.uint8:
            image_arr = image_arr.astype(np.uint8)
        raw_path = raw_dir / f"{stem}{image_path.suffix or '.png'}"
        Image.fromarray(image_arr).save(raw_path)
        outputs["raw"] = str(raw_path)

    if post.heatmap is not None and image_arr is not None:
        heat_overlay = (0.6 * image_arr + 0.4 * post.heatmap).astype(np.uint8)
        heat_path = heat_overlay_dir / f"{stem}.png"
        Image.fromarray(heat_overlay).save(heat_path)
        outputs["heatmap_overlay"] = str(heat_path)

    if post.mask is not None and image_arr is not None:
        mask_rgb = np.zeros_like(image_arr, dtype=np.uint8)
        mask_rgb[..., 0] = 255
        mask_alpha = (post.mask[..., None] / 255.0) * 0.45
        mask_overlay = (
            image_arr.astype(np.float32) * (1.0 - mask_alpha) + mask_rgb.astype(np.float32) * mask_alpha
        ).astype(np.uint8)
        mask_path = mask_overlay_dir / f"{stem}.png"
        Image.fromarray(mask_overlay).save(mask_path)
        outputs["mask_overlay"] = str(mask_path)

    return outputs


# -------------------------------
# Benchmarking
# -------------------------------
@dataclass
class PerImageMetrics:
    path: str
    io_s: float
    preprocess_s: float
    embedding_s: float | None
    query_s: float | None
    update_s: float | None
    inference_s: float
    postprocess_s: float
    e2e_s: float
    score: float | None
    label: int | None
    outputs: dict[str, str]


@dataclass
class SaveJob:
    image_path: Path
    base_dir: Path
    image_arr: np.ndarray | None
    post: PostprocessResult
    raw_output: Any
    data_root: Path


def save_worker(job_queue: "queue.Queue[SaveJob | None]", results: list[dict[str, str]]) -> None:
    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        outputs = save_outputs(
            image_path=job.image_path,
            base_dir=job.base_dir,
            image_arr=job.image_arr,
            post=job.post,
            raw_output=job.raw_output,
            data_root=job.data_root,
        )
        results.append(outputs)
        job_queue.task_done()


def prefetch_worker(
    adapter: BaseModelAdapter,
    image_paths: list[Path],
    job_queue: "queue.Queue[tuple[Path, float, dict[str, Any]] | None]",
) -> None:
    for path in image_paths:
        if hasattr(adapter, "preprocess_path"):
            pre, pre_t = timed(adapter.preprocess_path, path)
            io_t = 0.0
        else:
            img_bytes, io_t = timed(read_bytes, path)
            pre, pre_t = timed(adapter.preprocess, img_bytes)
        pre["__timing__"] = {"io_s": io_t, "preprocess_s": pre_t}
        job_queue.put((path, io_t, pre))
    job_queue.put(None)


def coerce_postprocess(result: Any) -> PostprocessResult:
    if isinstance(result, PostprocessResult):
        return result
    if isinstance(result, dict):
        return PostprocessResult(
            anomaly_map=result.get("anomaly_map"),
            mask=result.get("mask"),
            heatmap=result.get("heatmap"),
            score=result.get("score"),
            label=result.get("label"),
        )
    return PostprocessResult(anomaly_map=None, mask=None, heatmap=None, score=None, label=None)


def benchmark_images(
    adapter: BaseModelAdapter,
    image_paths: list[Path],
    out_dir: Path,
    data_root: Path,
    warmup_count: int,
    enable_embeddings: bool,
    update_count: int,
    proc: psutil.Process,
    async_preprocess: bool,
    prefetch_size: int,
    async_save: bool,
    save_queue_size: int,
) -> tuple[list[PerImageMetrics], dict[str, Any]]:
    per_image: list[PerImageMetrics] = []
    mem_peak_pre = get_rss_bytes(proc)
    mem_peak_inf = get_rss_bytes(proc)

    reset_vram_peak()

    warmup_time = None
    if warmup_count > 0 and image_paths:
        if hasattr(adapter, "preprocess_path"):
            pre, _ = timed(adapter.preprocess_path, image_paths[0])
        else:
            sample_bytes = read_bytes(image_paths[0])
            pre, _ = timed(adapter.preprocess, sample_bytes)
        warm_start = time.perf_counter()
        for _ in range(warmup_count):
            adapter.warmup(pre.get("input"))
        warmup_time = time.perf_counter() - warm_start

    save_results: list[dict[str, str]] = []
    save_queue: "queue.Queue[SaveJob | None] | None" = None
    save_thread: threading.Thread | None = None
    if async_save:
        save_queue = queue.Queue(maxsize=max(1, save_queue_size))
        save_thread = threading.Thread(target=save_worker, args=(save_queue, save_results), daemon=True)
        save_thread.start()

    prefetch_queue: "queue.Queue[tuple[Path, float, dict[str, Any]] | None] | None" = None
    prefetch_thread: threading.Thread | None = None
    if async_preprocess:
        prefetch_queue = queue.Queue(maxsize=max(1, prefetch_size))
        prefetch_thread = threading.Thread(
            target=prefetch_worker, args=(adapter, image_paths, prefetch_queue), daemon=True
        )
        prefetch_thread.start()

    def next_item(idx: int) -> tuple[Path, float, dict[str, Any]]:
        if async_preprocess and prefetch_queue is not None:
            item = prefetch_queue.get()
            if item is None:
                raise StopIteration
            return item
        path = image_paths[idx]
        if hasattr(adapter, "preprocess_path"):
            pre, pre_t = timed(adapter.preprocess_path, path)
            io_t = 0.0
        else:
            img_bytes, io_t = timed(read_bytes, path)
            mem_peak_pre = max(mem_peak_pre, get_rss_bytes(proc))
            pre, pre_t = timed(adapter.preprocess, img_bytes)
        pre["__timing__"] = {"io_s": io_t, "preprocess_s": pre_t}
        return path, io_t, pre

    idx = 0
    total = len(image_paths)
    while idx < len(image_paths):
        rss_before = get_rss_bytes(proc)
        try:
            path, io_t, pre = next_item(idx)
        except StopIteration:
            break
        timing = pre.pop("__timing__", {})
        pre_t = float(timing.get("preprocess_s", 0.0))
        io_t = float(timing.get("io_s", 0.0))
        mem_peak_pre = max(mem_peak_pre, get_rss_bytes(proc))

        model_input = pre.get("input")
        image_arr = pre.get("image")
        meta = pre.get("meta", {})

        embedding = None
        emb_t = None
        if enable_embeddings and is_overridden(adapter, "compute_embedding"):
            embedding, emb_t = timed(adapter.compute_embedding, model_input)

        query_t = None
        if embedding is not None and is_overridden(adapter, "query_index"):
            _, query_t = timed(adapter.query_index, embedding)

        update_t = None
        if embedding is not None and update_count > 0 and is_overridden(adapter, "update_index") and idx < update_count:
            _, update_t = timed(adapter.update_index, embedding)

        infer_start = time.perf_counter()
        raw_output = adapter.infer(model_input)
        infer_t = time.perf_counter() - infer_start
        mem_peak_inf = max(mem_peak_inf, get_rss_bytes(proc))

        post, post_t = timed(adapter.postprocess, raw_output, image_arr, meta)
        post = coerce_postprocess(post)

        if async_save and save_queue is not None:
            save_queue.put(
                SaveJob(
                    image_path=path,
                    base_dir=out_dir / "outputs",
                    image_arr=image_arr,
                    post=post,
                    raw_output=raw_output,
                    data_root=data_root,
                )
            )
            outputs: dict[str, str] = {}
        else:
            outputs = save_outputs(path, out_dir / "outputs", image_arr, post, raw_output, data_root)

        e2e = io_t + pre_t + (emb_t or 0.0) + (query_t or 0.0) + (update_t or 0.0) + infer_t + post_t

        per_image.append(
            PerImageMetrics(
                path=str(path),
                io_s=io_t,
                preprocess_s=pre_t,
                embedding_s=emb_t,
                query_s=query_t,
                update_s=update_t,
                inference_s=infer_t,
                postprocess_s=post_t,
                e2e_s=e2e,
                score=post.score,
                label=post.label,
                outputs=outputs,
            )
        )
        print(f"Inference {idx + 1}/{total} done")
        idx += 1

    vram_peak = get_vram_peak()
    if async_save and save_queue is not None:
        print("Waiting for async save to finish...")
        save_queue.put(None)
        save_queue.join()
    if async_save and save_results:
        for i, out in enumerate(save_results[: len(per_image)]):
            per_image[i].outputs = out
    return per_image, {
        "warmup_time_s": warmup_time,
        "mem_peak_preprocess_rss_bytes": mem_peak_pre,
        "mem_peak_inference_rss_bytes": mem_peak_inf,
        "mem_rss_end_bytes": get_rss_bytes(proc),
        "vram_peak_bytes": vram_peak,
    }


def benchmark_batch_throughput(
    adapter: BaseModelAdapter,
    image_paths: list[Path],
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 1 or not image_paths:
        return {"batch_size": batch_size, "throughput_records_per_s": None, "batch_latency_s": None}

    batch_inputs: list[Any] = []
    for path in image_paths[:batch_size]:
        img_bytes = read_bytes(path)
        pre = adapter.preprocess(img_bytes)
        batch_inputs.append(pre.get("input"))

    start = time.perf_counter()
    _ = adapter.infer_batch(batch_inputs)
    batch_latency = time.perf_counter() - start
    throughput = batch_size / batch_latency if batch_latency > 0 else None
    return {"batch_size": batch_size, "throughput_records_per_s": throughput, "batch_latency_s": batch_latency}


def compute_summary(per_image: list[PerImageMetrics]) -> dict[str, Any]:
    io_times = [m.io_s for m in per_image]
    pre_times = [m.preprocess_s for m in per_image]
    emb_times = [m.embedding_s for m in per_image if m.embedding_s is not None]
    query_times = [m.query_s for m in per_image if m.query_s is not None]
    update_times = [m.update_s for m in per_image if m.update_s is not None]
    infer_times = [m.inference_s for m in per_image]
    post_times = [m.postprocess_s for m in per_image]
    e2e_times = [m.e2e_s for m in per_image]

    return {
        "counts": {"images": len(per_image)},
        "io_time_s": {"avg": mean(io_times), "p50": pct(io_times, 50), "p95": pct(io_times, 95), "p99": pct(io_times, 99)},
        "preprocess_time_s": {
            "avg": mean(pre_times),
            "p50": pct(pre_times, 50),
            "p95": pct(pre_times, 95),
            "p99": pct(pre_times, 99),
        },
        "embedding_time_s": {
            "avg": mean(emb_times),
            "p50": pct(emb_times, 50),
            "p95": pct(emb_times, 95),
            "p99": pct(emb_times, 99),
        },
        "query_time_s": {
            "avg": mean(query_times),
            "p50": pct(query_times, 50),
            "p95": pct(query_times, 95),
            "p99": pct(query_times, 99),
        },
        "update_time_s": {
            "avg": mean(update_times),
            "p50": pct(update_times, 50),
            "p95": pct(update_times, 95),
            "p99": pct(update_times, 99),
        },
        "inference_time_s": {
            "avg": mean(infer_times),
            "p50": pct(infer_times, 50),
            "p95": pct(infer_times, 95),
            "p99": pct(infer_times, 99),
        },
        "postprocess_time_s": {
            "avg": mean(post_times),
            "p50": pct(post_times, 50),
            "p95": pct(post_times, 95),
            "p99": pct(post_times, 99),
        },
        "e2e_time_s": {"avg": mean(e2e_times), "p50": pct(e2e_times, 50), "p95": pct(e2e_times, 95), "p99": pct(e2e_times, 99)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anomaly models")
    parser.add_argument("--model-module", required=True, help="Python module path, e.g. my_models.adapters")
    parser.add_argument("--model-class", required=True, help="Adapter class name, must extend BaseModelAdapter")
    parser.add_argument("--model-config", default=None, help="Path to JSON config for the adapter")
    parser.add_argument("--model-kwargs", default=None, help="Inline JSON string with adapter kwargs")
    parser.add_argument("--data-dir", required=True, help="Directory of images to run inference on")
    parser.add_argument("--reference-dir", default=None, help="Reference images for one-shot models")
    parser.add_argument("--one-shot", action="store_true", help="Force one-shot mode")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (count)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for throughput benchmark")
    parser.add_argument("--enable-embeddings", action="store_true", help="Enable embedding/index benchmarks")
    parser.add_argument("--update-count", type=int, default=0, help="Number of queries to measure index update cost")
    parser.add_argument("--async-preprocess", action="store_true", help="Preprocess in a background worker")
    parser.add_argument("--prefetch-size", type=int, default=32, help="Async preprocess queue size")
    parser.add_argument("--async-save", action="store_true", help="Save outputs in a background worker")
    parser.add_argument("--save-queue-size", type=int, default=300, help="Async save queue size")
    parser.add_argument("--results-dir", default="results", help="Results root")
    return parser.parse_args()


def load_config(path: str | None, inline_json: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    if inline_json:
        cfg.update(json.loads(inline_json))
    return cfg


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data-dir not found: {data_root}")

    image_paths = list_images(data_root)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError("No images found for benchmarking")

    config = load_config(args.model_config, args.model_kwargs)
    adapter = load_adapter(args.model_module, args.model_class, config)

    results_root = Path(args.results_dir)
    run_dir = results_root / f"{now_ts()}_{adapter.name}"
    ensure_dirs(run_dir)

    proc = psutil.Process(os.getpid())

    if torch is not None:
        torch.set_grad_enabled(False)

    load_start = time.perf_counter()
    adapter.load()
    model_load_time = time.perf_counter() - load_start

    is_one_shot = args.one_shot or getattr(adapter, "is_one_shot", False)
    reference_metrics: dict[str, Any] = {}
    if is_one_shot and args.reference_dir:
        ref_paths = list_images(Path(args.reference_dir))
        if not ref_paths:
            raise RuntimeError("Reference directory has no images")

        if args.enable_embeddings and is_overridden(adapter, "compute_embedding"):
            emb_times: list[float] = []
            embeddings: list[Any] = []
            for ref_path in ref_paths:
                img_bytes = read_bytes(ref_path)
                pre = adapter.preprocess(img_bytes)
                model_input = pre.get("input")
                emb, emb_t = timed(adapter.compute_embedding, model_input)
                if emb_t is not None:
                    emb_times.append(emb_t)
                embeddings.append(emb)

            reference_metrics["embedding_time_s"] = {
                "avg": mean(emb_times),
                "p50": pct(emb_times, 50),
                "p95": pct(emb_times, 95),
                "p99": pct(emb_times, 99),
            }

            if is_overridden(adapter, "build_index"):
                index_obj, build_t = timed(adapter.build_index, embeddings)
                reference_metrics["index_build_time_s"] = build_t
                if isinstance(index_obj, dict):
                    reference_metrics["index_info"] = index_obj
                    idx_path = index_obj.get("index_path")
                    if idx_path:
                        idx_path = Path(idx_path)
                        if idx_path.exists():
                            reference_metrics["index_size_bytes"] = idx_path.stat().st_size
                elif isinstance(index_obj, (str, Path)):
                    idx_path = Path(index_obj)
                    if idx_path.exists():
                        reference_metrics["index_size_bytes"] = idx_path.stat().st_size

    per_image, extra_metrics = benchmark_images(
        adapter=adapter,
        image_paths=image_paths,
        out_dir=run_dir,
        data_root=data_root,
        warmup_count=args.warmup,
        enable_embeddings=args.enable_embeddings,
        update_count=args.update_count,
        proc=proc,
        async_preprocess=args.async_preprocess,
        prefetch_size=args.prefetch_size,
        async_save=args.async_save,
        save_queue_size=args.save_queue_size,
    )

    if hasattr(adapter, "report_nn_mismatches"):
        adapter.report_nn_mismatches()

    batch_metrics = benchmark_batch_throughput(adapter, image_paths, args.batch_size)

    summary = compute_summary(per_image)

    metrics = {
        "run": {
            "timestamp": run_dir.name,
            "model": {"module": args.model_module, "class": args.model_class, "name": adapter.name},
            "one_shot": is_one_shot,
            "data_dir": str(data_root),
            "reference_dir": str(args.reference_dir) if args.reference_dir else None,
            "limit": args.limit,
            "batch_size": args.batch_size,
        },
        "model_load_time_s": model_load_time,
        "warmup_time_s": extra_metrics.get("warmup_time_s"),
        "memory": {
            "preprocess_peak_rss_bytes": extra_metrics.get("mem_peak_preprocess_rss_bytes"),
            "inference_peak_rss_bytes": extra_metrics.get("mem_peak_inference_rss_bytes"),
            "rss_end_bytes": extra_metrics.get("mem_rss_end_bytes"),
            "vram_peak_bytes": extra_metrics.get("vram_peak_bytes"),
        },
        "summary": summary,
        "batch": batch_metrics,
        "reference": reference_metrics,
    }

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = run_dir / "per_image.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "io_s",
                "preprocess_s",
                "embedding_s",
                "query_s",
                "update_s",
                "inference_s",
                "postprocess_s",
                "e2e_s",
                "score",
                "label",
                "outputs",
            ],
        )
        writer.writeheader()
        for row in per_image:
            writer.writerow(
                {
                    "path": row.path,
                    "io_s": row.io_s,
                    "preprocess_s": row.preprocess_s,
                    "embedding_s": row.embedding_s,
                    "query_s": row.query_s,
                    "update_s": row.update_s,
                    "inference_s": row.inference_s,
                    "postprocess_s": row.postprocess_s,
                    "e2e_s": row.e2e_s,
                    "score": row.score,
                    "label": row.label,
                    "outputs": json.dumps(row.outputs),
                }
            )

    config_out = run_dir / "run_config.json"
    with config_out.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "adapter_config": config}, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved per-image to: {csv_path}")
    print(f"Outputs under: {run_dir / 'outputs'}")


if __name__ == "__main__":
    import io

    main()
