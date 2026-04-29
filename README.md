# ReID Model Conversion Kit
## .etlt → .onnx → .engine (TensorRT)

---

## Folder Structure

```
reid_conversion/
├── pretrained_model/        ← PUT YOUR .etlt FILE HERE
│   └── resnet50_market1501_aicity156.etlt
├── onnx_model/              ← outputs saved here automatically
│   ├── resnet50_reid.onnx   (after setup.sh)
│   └── resnet50_reid.engine (after convert_to_tensorrt.sh)
├── setup.sh                 ← Step 1: .etlt → .onnx (needs Docker, one time)
├── convert_to_tensorrt.sh   ← Step 2: .onnx → .engine (no Docker)
├── export.yaml              ← TAO config (do not edit)
└── README.md
```

---

## Step 1: Copy your .etlt file

```bash
cp resnet50_market1501_aicity156.etlt pretrained_model/
```

---

## Step 2: Run setup.sh (.etlt → .onnx)

> Requires: Docker + NVIDIA Container Toolkit

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Docker and NVIDIA runtime
- Pull TAO Toolkit image (~5GB, first time only)
- Decrypt and export .etlt → .onnx
- Save to onnx_model/resnet50_reid.onnx

**After this step, Docker is never needed again.**

---

## Step 3: Run convert_to_tensorrt.sh (.onnx → .engine)

> No Docker needed — runs on your desktop GPU

```bash
chmod +x convert_to_tensorrt.sh
./convert_to_tensorrt.sh
```

This will:
- Install TensorRT if not present
- Build optimized FP16 engine for your GPU
- Save to onnx_model/resnet50_reid.engine

---

## Step 4: Use in your project

Copy the engine to your project:
```bash
cp onnx_model/resnet50_reid.engine \
   /your/project/src/local_models/people_gpu/code/pretrained/
```

Then update `OSNetFeatureExtractor` in `osnet_deepsort_reid.py`
to load the TensorRT engine instead of OSNet weights.

---

## Requirements

| Tool | When needed |
|------|-------------|
| Docker | Step 2 only (one time) |
| NVIDIA Container Toolkit | Step 2 only (one time) |
| TensorRT (pip) | Step 3 onwards |
| CUDA 12.x | Step 3 onwards |

---

## Model Key

NVIDIA TAO decrypt key: `nvidia_tao`
