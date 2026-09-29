# CV Playground

**English** | [日本語](README.ja.md)

A web page for trying out and comparing computer vision models right in the browser.
On the same screen you can choose **"run in the browser" (WebGPU / WASM on your device)** or **"run on the server"**, and see inference time, fps, and a per-stage breakdown. That makes it easy to check how far a model actually goes on a phone or a laptop.

Open it at **https://yoshiri.github.io/cv-playground/** (no server; browser-only models; works on phones).

> The UI text is currently in Japanese (an English UI is on the [TODO list](docs/TODO.md)). The developer docs under `docs/` are also in Japanese.

![Screenshot](docs/images/hero.jpg)

## Demo

| | | |
| --- | --- | --- |
| ![Pose and tracking](docs/images/pose-track.jpg)<br>**Human pose + tracking** (YOLO26n-pose + ByteTrack, with IDs and trails) | ![Object detection](docs/images/detect.jpg)<br>**Object detection** (YOLO26n) | ![Hands and eyes](docs/images/wholebody.jpg)<br>**Hands and eyes** (PINTO's DEIMv2 + OCEC eye open/closed) |
| ![Semantic](docs/images/semantic.jpg)<br>**Semantic segmentation** (EoMT DINOv3, ADE20K) | ![Panoptic](docs/images/panoptic.jpg)<br>**Panoptic segmentation** (EoMT DINOv3, COCO; each chair and desk separately) | ![Prompt](docs/images/segment.jpg)<br>**Promptable segmentation** (SAM 2.1, clicked person) |
| ![Automatic mask generation](docs/images/segment-auto.jpg)<br>**Segment everything** (EdgeTAM, from a 144-point grid) | ![Depth estimation](docs/images/depth.jpg)<br>**Depth estimation** (Depth Anything 3, also estimates the field of view) | ![Text-prompted detection](docs/images/zsdetect.jpg)<br>**Text-prompted object detection** (Grounding DINO, "orange, lemon") |
| ![Zero-shot classification](docs/images/classify.jpg)<br>**Zero-shot classification** (SigLIP2) | ![Background removal](docs/images/matting.jpg)<br>**Background removal** (BiRefNet lite) | |

The demo frames come from [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0).

## Features

- **Tasks** (grouped into four categories)
  - Detection & tracking: object detection / human pose / hands & eyes (PINTO's ultra-light models) / gestures (pose + PINTO run together on each frame: hand raised, face direction, eyes closed, pointing) / text-prompted object detection
  - Segmentation: promptable (SAM; click points or segment everything) / semantic / panoptic / background removal
  - Depth & 3D: depth estimation
  - Vision & language: zero-shot classification / image captioning and VQA (VLM)
- **Applications** (add-ons in any tab that returns boxes — detection, pose, hands & eyes, text-prompted detection): counting per class (current count and, with tracking, a cumulative count of track IDs). Add `?apps=count` to the URL to turn it on from the start
- **Input**: images, video files, and live camera. Video and camera run continuously with results overlaid, showing fps and a breakdown (grab, preprocess, model, postprocess)
- **Tracking**: ByteTrack / BoT-SORT / BoT-SORT + ReID (ported from Ultralytics; verified to give identical results on the same detection sequence)
- **Runtime options**: by default, generic ONNX models run with fp16 (when the model has an fp16 file) and WebGPU graph capture, which is dropped automatically for models that cannot use it. You can switch to fp32, no graph capture, or CPU (WASM), and pick the input long side (320–960) for models with dynamic input. Fixed-size models also get their symbolic input dimensions pinned
- **Compare**: the same model in the browser and on the server (generic ONNX models build pre/post-processing from shared blocks, so both sides follow the same steps). Timings are kept in the run history
- **Benchmark**: on a fixed sample image, 3 warm-up runs followed by 5 / 20 / 50 / 100 measured runs, recording median, p90, p95, and fps. Compare devices under identical conditions. Add `?profile=1` to the URL to also get per-operator GPU time for generic ONNX models (see [docs/NOTES.md](docs/NOTES.md))
- **Use local ONNX files**: if you already have a published model's file, pick it and use it without downloading (only files whose SHA-256 matches exactly; generic ONNX models only)
- **Export**: the displayed image (with an optional caption strip; on phones, the share sheet lets you save to Photos), result data (JSON), and run records (CSV / JSON / Markdown table). Records include device, browser, and GPU info, so CSVs from different devices can be concatenated and compared as is

## Data, network usage, and licenses

- **Models that run in the browser are downloaded to your device on first use** (from a few MB up to about 1.4 GB; the size is shown next to each model name). **Watch your data usage on mobile networks.** Models over 100 MB ask for confirmation before downloading, and later runs load from the browser cache. "ダウンロード済みのモデルを消す" (clear downloaded models) at the bottom of the page removes them
- **Images, videos, and camera frames never leave your device as long as you run in the browser** (the only network traffic is fetching models and libraries). Only when you choose "run on the server" is the image sent to the server
- **Each model has its own license**, and some do not allow commercial use (e.g., YOLO26 is AGPL-3.0; Depth Anything V2 Large and SegFormer are non-commercial). The license is shown in each model's description on screen
- Supported browsers: Chrome / Edge / Safari (iOS 26 or later) with WebGPU are recommended. Without WebGPU it falls back to WASM, which is slower

## Usage

### Three ways to serve it (all from the same code in `web/`)

| Form | What it is | Differences |
| --- | --- | --- |
| Server | `server.py` serves `web/` plus the API | Server-side models (on a Mac, etc.) are also available. COOP/COEP headers enable multi-threaded WASM |
| Static | GitHub Pages or any static host serves `web/` as is | The API is unreachable, so only browser models. Multi-threaded WASM is enabled by the bundled coi-serviceworker |
| Single file | `dist/cv-playground.html`, built by `python3 build.py` | Just open the file (works from `file://`). Browser models only; single-threaded WASM |

The page detects at runtime whether a server is present (whether `api/models` responds).

### Running the server

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # per-environment settings (optional; .env is not committed)
scripts/serve.sh                # http://127.0.0.1:8010 (--bg to run in the background)
```

- Server-side models are downloaded from Hugging Face on first use (several GB in total). Up to 3 are kept in memory; beyond that the least recently used is unloaded
- To open it from another device (e.g., a phone), **HTTPS is required** (for WebGPU and the camera). With Tailscale, for example: `tailscale serve --bg --https=8443 http://127.0.0.1:8010`
- On Apple Silicon it uses MPS / CoreML; elsewhere, CPU (CUDA is not supported or tested)
- To use YOLO26 with dynamic input size, export it with `tools/export_yolo26_dynamic.py` (the weights are AGPL, so they are not in the repo and only the server serves them)

Environment variables (`.env`)

| Variable | Meaning | Default |
| --- | --- | --- |
| `CVPG_PORT` | Port to listen on | 8010 |
| `CVPG_MAX_LOADED` | Number of server models kept loaded at once | 3 |
| `CVPG_OLLAMA_MODEL` | Model used by the "Ollama VLM" entry | `qwen3-vl:8b` |
| `OLLAMA_HOST_URL` | Ollama URL | `http://127.0.0.1:11434` |
| `CVPG_GPU_LOCK` | If this file exists, the page shows "GPU is in use by another process" (optional) | none |

## Forking and extending

| What you want to do | Docs (Japanese) |
| --- | --- |
| Add a model (browser / server) | [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) |
| Add tabs, settings fields, or result views | [docs/ADDING_UI.md](docs/ADDING_UI.md) |
| How it works, measurements, findings | [docs/NOTES.md](docs/NOTES.md) |
| Future work | [docs/TODO.md](docs/TODO.md) |

In most cases, adding one entry to `web/models.json` is enough (for plain ONNX models, even pre/post-processing is composed in JSON).

## Layout

| File | Role |
| --- | --- |
| `web/models.json` | **Definitions of tabs (tasks) and models, shared by browser and server** |
| `web/app.js` | UI, continuous video/camera runs, wiring for tracking and cascades, run history, benchmark |
| `web/renderers.js` | How each result kind is shown (boxes, masks, depth, segment maps, labels, text) |
| `web/apps.js` | Applications added on top of a tab's results (stateful aggregation across frames and its overlay, e.g. counting) |
| `web/export.js` | Image saving, result data, run-record export (CSV / JSON / Markdown), device info, statistics |
| `web/worker.js` | Browser-side adapters (Web Workers; separate workers for onnxruntime-web and transformers.js) |
| `web/onnx_generic.js` | Browser-side generic ONNX (pre/post-processing blocks) |
| `web/tracker.js` | ByteTrack / BoT-SORT (+ ReID) |
| `adapters.py` | Server-side adapters (the generic ONNX blocks mirror onnx_generic.js) |
| `server.py` | FastAPI. Serves `web/`, `/api/run`, `/api/status`, `/api/unload`, and `/local-models/` |
| `build.py` | Builds the single-file version (a stale build is caught by `.github/workflows/check-dist.yml`) |
| `web/pinto/` | Bundled models from PINTO_model_zoo (MIT) |
| `tools/` | Tracker comparison against Ultralytics, YOLO26 export, fetching model file SHA-256 hashes |

## License

The code is under the MIT License (LICENSE). **Model weights are generally not bundled; they are fetched from each distributor at runtime. Check each model's license with its distributor** (e.g., YOLO26 is AGPL-3.0 from Ultralytics).
The exception is `web/pinto/`, which bundles models from PINTO_model_zoo (MIT) with attribution. Sample images are loaded at runtime from [Xenova/transformers.js-docs](https://huggingface.co/datasets/Xenova/transformers.js-docs) on Hugging Face. The README demo images are frames from [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0) with results overlaid.
