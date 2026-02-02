# WinClip Benchmarking

This repo includes a general-purpose anomaly model benchmarking script:

- `benchmark_anomaly_models.py`
- Results are written under `./results/<timestamp>_<model_name>/`

## Quick start

```bash
python benchmark_anomaly_models.py \
  --model-module my_models.adapters \
  --model-class MyAdapter \
  --data-dir /path/to/images \
  --batch-size 8 \
  --warmup 1
```

Optional flags:

- `--model-config /path/to/config.json`
- `--model-kwargs '{"checkpoint":"/home/awais/Datasets/path/to/model.ckpt"}'`
- `--reference-dir /path/to/reference` (one-shot mode)
- `--one-shot` (force one-shot behavior)
- `--enable-embeddings` (embedding + index benchmarks)
- `--update-count 10` (measure index update cost for first N queries)
- `--limit 100` (limit images)
- `--results-dir results`

## Output structure

Each run creates:

- `metrics.json`: summary metrics, load time, warmup, memory peaks
- `per_image.csv`: per-image timings and outputs
- `run_config.json`: resolved CLI + adapter config
- `outputs/`:
  - `raw/` (copied input images)
  - `heatmaps/` (color heatmaps)
  - `masks/` (binary segmentation masks)
  - `raw_output/` (numpy arrays or JSON fallbacks)

## Adapter API (how to add a new model)

Create a module (e.g., `my_models/adapters.py`) and implement a class that inherits `BaseModelAdapter`:

```python
from benchmark_anomaly_models import BaseModelAdapter, PostprocessResult
import numpy as np

class MyAdapter(BaseModelAdapter):
    name = "my_model"
    is_one_shot = False

    def __init__(self, config):
        super().__init__(config)
        self.model = None

    def load(self):
        # load weights, allocate GPU, etc.
        pass

    def preprocess(self, image_bytes):
        # return a dict with at least "input" and optionally "image"
        # image should be an HWC uint8 RGB array if you want built-in heatmaps/masks
        image = ...
        model_input = ...
        return {"input": model_input, "image": image}

    def infer(self, model_input):
        # return a dict that may include "anomaly_map", "score", "label"
        return {"anomaly_map": np.random.rand(256, 256), "score": 0.1, "label": 0}

    def postprocess(self, raw_output, image=None, meta=None):
        # optional: if you already have a mask/heatmap, return them here
        return super().postprocess(raw_output, image, meta)

    # Optional: enable embedding benchmarks
    def compute_embedding(self, model_input):
        return np.random.rand(512)

    def build_index(self, embeddings):
        # build FAISS/HNSW/etc
        return None

    def query_index(self, embedding):
        # return nearest neighbor info (optional)
        return None

    def update_index(self, embedding):
        # add a point to the index (optional)
        return None
```

Template available at `my_models/adapters.py` with a `DummyAnomalyAdapter` example.

### One-shot models

If your model needs a reference set:

- Set `is_one_shot = True`, or pass `--one-shot`.
- Use `--reference-dir` to supply reference images.
- If you implement `compute_embedding` and `build_index`, the script will time index build and embedding costs.

## Notes

- The script saves heatmaps/masks when an `anomaly_map` is provided (H×W float array).
- GPU memory stats are reported when CUDA is available.
- Batch throughput uses `infer_batch` if implemented, otherwise falls back to per-image inference.
