# Deepfake-GPU-Trainer

We have prepared two solutions:

1. **Single-GPU mode**: achieves an ACER of **3.6**
2. **Two-GPU mode**: achieves an EER of **2.10**

---

## Prepare Docker

A Docker image ensures reproducible training for both pipelines. We have saved the image separately, but you can build it yourself:

```bash
cd docker
sudo docker build -t deepfake-gpu-trainer:latest .
```

---

## Data unpacking

Place all raw data in the `data/` folder. Txt files should be placed in `dataset/` folder 

## Train One-GPU Model

```bash
cd train_one_gpu

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ../dataset:/dataset \
  -v ../data:/data \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-trainer:latest \
  tools/train.py --config_path configs/config_one_gpu.py
```

---

## Evaluate One-GPU Model

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  --shm-size=16gb \
  deepfake-gpu-trainer:latest \
  eval_one_gpu/run_one_gpu.py
```

---

## Train Two-GPU Model

```bash
cd train_two_gpus

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ../dataset:/dataset \
  -v ../data:/data \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-trainer:latest \
  tools/train.py --config_path configs/config_two_gpus.py
```

---

## Evaluate Two-GPU Model

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  --shm-size=16gb \
  deepfake-gpu-trainer:latest \
  eval_two_gpus/run_two_gpu.py
```

---

## Data Preparation

We use a face model to extract faces from images. For training, we prepare pairs of "live-attack" samples with the same user identity to stabilize training. 

### Prepare Docker for Face Matcher

```bash
cd docker_face
sudo docker build -t deepfake-gpu-face_matcher:latest .
```

### Extract Face Embeddings (Optional)

> *Could be skipped because CSVs are already prepared.*

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/extract_embeddings.py \
    --src data/Data-train/ \
    --dst dataset/embeddings/train
```

### Extract Boxes and Face Crops (Optional)

> *Could be skipped because CSVs are already prepared.*

```bash
# Train set
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/crop_faces_v2.py \
    --src /workspace/data/Data-train/ \
    --dst /data/face_crops_v2_224/Data-train \
    --csv-name /workspace/dataset/train_bboxes_rerun.csv

# Validation set
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/crop_faces_v2.py \
    --src /workspace/data/Data-val/ \
    --dst /data/face_crops_v2_224/Data-val \
    --csv-name /workspace/dataset/val_bboxes_rerun.csv

# Test set
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/crop_faces_v2.py \
    --src /workspace/data/Data-test/ \
    --dst /data/face_crops_v2_224/Data-test \
    --csv-name /workspace/dataset/test_bboxes_rerun.csv
```

### Filter Training Data for Pairing (Optional)

> *Could be skipped because CSVs are already prepared.*

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/filter_train.py \
    --data-dir /workspace/data \
    --embeddings-dir /workspace/dataset/embeddings/train \
    --clean-df-path /workspace/dataset/train_no_misslabel.csv \
    --sim-thr 0.90 \
    --output-csv /workspace/dataset/train_matched_09_th_rerun.csv

# for threshold 0.92
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/filter_train.py \
    --data-dir /workspace/data \
    --embeddings-dir /workspace/dataset/embeddings/train \
    --clean-df-path /workspace/dataset/train_no_misslabel.csv \
    --sim-thr 0.92 \
    --output-csv /workspace/dataset/train_matched_092_th_rerun.csv
```

### Add Face Crop Boxes to Protocols (Optional)

> *Could be skipped if CSVs are already prepared.*

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16gb \
  deepfake-gpu-face_matcher:latest \
  prepare_data/bbox_to_data.py \
    --train-df /workspace/dataset/train_matched_09_th_rerun.csv \
    --val-df /workspace/dataset/Protocol-val-labels.txt \
    --test-df /workspace/dataset/Protocol-test.txt \
    --train-bboxes /workspace/dataset/train_bboxes_rerun.csv \
    --val-bboxes /workspace/dataset/val_bboxes_rerun.csv \
    --test-bboxes /workspace/dataset/test_bboxes_rerun.csv \
    --output-train-csv /workspace/dataset/train_matched_092_bboxes_rerun.csv \
    --output-val-csv /workspace/dataset/val_bboxes_rerun.csv \
    --output-test-csv /workspace/dataset/test_bboxes_rerun.csv
