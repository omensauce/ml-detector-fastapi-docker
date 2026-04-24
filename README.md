# Fruit Object Detector

Faster R-CNN (ResNet-50 FPN) fine-tuned to detect **apples, bananas, and oranges**.  
Served via a three-container stack: FastAPI inference API · SvelteKit frontend · MLflow tracking server.

---

## Architecture

```
Browser → frontend :3000 (nginx + SvelteKit SPA)
                ↓ /predict proxy
         inference :8000 (FastAPI + TorchScript model)

MLflow UI  :5000 (experiment tracking & model registry)
```

---

## Cold-start instructions

### Prerequisites

| Tool | Purpose |
|------|---------|
| [Podman](https://podman.io/docs/installation) + [podman-compose](https://github.com/containers/podman-compose) | build and run containers |
| Python 3.10+ | run the training notebook |
| Kaggle dataset [`fruit-images-for-object-detection`](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection) | training data |

> **podman-compose** ships with Podman Desktop on Windows (`podman compose`).  
> Alternatively: `pip install podman-compose` and call `podman-compose`.

---

### Step 1 — Prepare the dataset

Extract the Kaggle dataset so the layout matches:

```
prac 6/
└── fruit_detection/
    ├── train/   ← .jpg + .xml pairs
    └── test/    ← .jpg + .xml pairs
```

---

### Step 2 — Generate the model artifact

Open `practice_7.ipynb` in Jupyter and **run all cells top to bottom**.

The notebook trains two phases (frozen backbone → partial unfreeze) and saves checkpoints automatically:

| File | Created by | Notes |
|------|-----------|-------|
| `baseline_fasterrcnn.pth` | Part 5 | skipped on re-run if file exists |
| `finetuned_fasterrcnn.pth` | Part 6 | skipped on re-run if file exists |
| `fruits_model.pt` | Part 15 (last cell) | TorchScript export — **required by the inference container** |

The checkpoint skip logic means re-running the notebook is fast if the `.pth` files are already present.

---

### Step 3 — Track experiments in the MLflow container

Install MLflow in your notebook environment if it is not already present:

```powershell
pip install mlflow
```

Start the tracking server container first:

```powershell
podman compose up -d mlflow
```

Set tracking URI to the container server before launching Jupyter:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
jupyter lab
```

Then use one of these notebooks:

| Notebook | Use case |
|----------|----------|
| `practice_7.ipynb` | full training pipeline + Part 16 MLflow logging |
| `practice_7_light.ipynb` | quicker experiments with alternate hyperparameters and lighter logging loop |

Expected target:

```
Tracking URI : http://localhost:5000
```

If the URI prints `mlruns`, runs are being logged locally instead of the container.

---

### Step 4 — Start the container stack

```powershell
podman compose up -d --build
```

If inference loads checkpoints from MLflow (`runs:/...`), MLflow must allow internal container host headers.
This repo config already sets:
`--allowed-hosts mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000`.

Inference model loading is now environment-driven:

1. `MLFLOW_MODEL_URI` (highest priority), example: `runs:/<run_id>/torchscript/fruits_model_light.pt`
2. `MLFLOW_RUN_ID` + `MLFLOW_MODEL_ARTIFACT` (default artifact path: `torchscript/fruits_model_light.pt`)
3. `LOCAL_MODEL_PATH` fallback (default: `fruits_model.pt`)

If no MLflow variables are set, inference loads the local TorchScript file.

What happens on the **first build** (cold pull):

| Layer | Size | Cached after |
|-------|------|--------------|
| `python:3.10-slim` | ~45 MB | 1st inference + mlflow build |
| `torch==2.4.1+cpu` + `torchvision` | ~520 MB | stays cached unless `Containerfile` changes |
| `node:20-alpine` | ~40 MB | 1st frontend build |
| `nginx:alpine` | ~8 MB | stays cached |

Subsequent `--build` calls only rebuild layers whose inputs changed.  
Changing `main.py` or `+page.svelte` does **not** invalidate the torch layer.

---

### Step 5 — Open the services

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Inference API (Swagger UI) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

Upload a fruit photo on the frontend — the annotated result appears alongside a detection table.

---

## Switch checkpoint in inference

### Option A: point inference to a specific MLflow run artifact

```powershell
$env:MLFLOW_RUN_ID = "<run_id_here>"
$env:MLFLOW_MODEL_ARTIFACT = "torchscript/fruits_model_light.pt"
podman compose up -d --build inference
```

### Option B: use a full MLflow model URI (recommended when copying from MLflow UI)

```powershell
$env:MLFLOW_MODEL_URI = "runs:/<run_id_here>/torchscript/fruits_model_light.pt"
podman compose up -d --build inference
```

### Verify which checkpoint is active

Open `http://localhost:8000/` and check `model_source` in the response.

---

### Stopping and cleaning up

```powershell
# Stop containers (keeps volumes)
podman compose down

# Stop and remove the mlruns volume too
podman compose down -v
```

---

## Running without containers

```powershell
# Create and activate venv (uv recommended)
uv venv .venv
.venv\Scripts\activate

# Install inference deps + torch CPU
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu torchvision==0.19.1+cpu

# Start the API
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`.

---

## API reference

### `GET /`
Health check.

```json
{ "message": "Fruit Detector API is running", "classes": ["apple", "banana", "orange"] }
```

### `POST /predict`

**Request:** `multipart/form-data` with field `file` — any image format PIL can open.

**Response:**

```json
{
  "detections": [
    {
      "box": [1940.8, 137.1, 3719.7, 1904.5],
      "label": "apple",
      "score": 0.9914
    }
  ],
  "image": "<base64-encoded JPEG with boxes drawn>"
}
```

Box format: `[x_min, y_min, x_max, y_max]` in absolute pixels.  
The `image` field is only present when at least one detection exceeds the 0.6 confidence threshold.

**curl example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "accept: application/json" \
  -F "file=@apple.jpg;type=image/jpeg"
```

To preview the annotated image paste the `image` value into a browser address bar:

```
data:image/jpeg;base64,<paste value here>
```

---

## Model details

| Property | Value |
|----------|-------|
| Architecture | Faster R-CNN, ResNet-50 FPN backbone |
| Classes | background · apple · banana · orange |
| Input | RGB image, any resolution (model handles it internally) |
| Export format | TorchScript (`fruits_model.pt`) |
| Baseline mAP@0.5 | 0.717 |
| Fine-tuned mAP@0.5 | 0.828 |
| Confidence threshold | 0.6 |

The model is exported with `torch.jit.script` so the inference container does not need `model.py`; the architecture is embedded in the `.pt` file.  Note that `torchvision` must still be installed because the TorchScript graph calls the `torchvision::nms` C++ operator.
