import os, sys, random, torch, timm, json, math, pandas as pd, numpy as np, cv2
import timm, torch.nn as nn, torch.nn.functional as F
from scipy.special import expit as sigmoid
import zipfile
import matplotlib.pyplot as plt
from mmengine.config import Config

import os
import random
import csv
import cv2
import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Dict, Tuple, List, Optional

import pandas as pd, numpy as np, cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def calculate_eer(df, score_name):
    fpr, tpr, thresholds = roc_curve(df['label'], df[score_name])
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    eer = (fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] + (1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])) / 2
    return eer, eer_threshold

def get_threshold_for_bpcer(df, score_name, target_bpcer):
    bona_fide = df[df['label'] == 1]

    thresholds = np.sort(bona_fide[score_name].unique())
    
    best_diff = float('inf')
    best_threshold = None
    best_bpcer = None

    for threshold in thresholds:
        false_rejections = (bona_fide[score_name] < threshold).sum()
        bpcer = false_rejections / len(bona_fide) if len(bona_fide) > 0 else np.nan
        diff = abs(bpcer - target_bpcer)

        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            best_bpcer = bpcer

    return best_threshold, best_bpcer

def binarize_scores(df, score_name, threshold, output_col='binary_score'):
    df = df.copy()
    df[output_col] = (df[score_name] >= threshold).astype(int)
    return df


def calculate_acers(df, thresholds, score_col,
                    attack_label=1, bona_label=0):

    y_true   = df["label"].to_numpy()
    scores   = df[score_col].to_numpy()

    attack_mask = y_true == attack_label
    bona_mask   = y_true == bona_label

    acers = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        apcer = (y_pred[attack_mask] == bona_label).mean()
        bpcer = (y_pred[bona_mask] == attack_label).mean()
        acer = (apcer + bpcer) / 2.0
        acers.append(acer)

    return acers
    
# ------------------------------------------------------------
# 📦  Utility helpers
# ------------------------------------------------------------

def _load_bbox_csv(csv_path: str, root_dir: str = "") -> Dict[str, Tuple[float, float, float, float]]:
    """
    Build a lookup table:  absolute_path → (x1, y1, x2, y2)

    Rows with missing coords OR any coordinate < 0  (e.g. -1 -1 -1 -1 from
    a failed detector) are skipped, so those images will fall back to
    full-frame processing downstream.
    """
    bbox_by_path: Dict[str, Tuple[float, float, float, float]] = {}

    if not csv_path:
        return bbox_by_path

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = (row.get("path") or "").strip()
            if not rel_path:
                continue

            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except (KeyError, TypeError, ValueError):
                continue  # malformed numbers → ignore row

            # detector failure encodes all -1 or any negative → skip
            if min(x1, y1, x2, y2) < 0:
                continue
            # degenerate box (zero or negative size) → skip
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            full_path = os.path.join(root_dir, rel_path)
            bbox_by_path[full_path] = (x1, y1, x2, y2)

    return bbox_by_path



def _crop_with_margin(img, bbox: Tuple[float, float, float, float], margin: float):
    """Crop *img* to *bbox* expanded by *margin* (e.g. 1.2 enlarges 20 %).

    Falls back to returning *img* if bbox is invalid or crops outside bounds.
    """
    if bbox is None:
        return img

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return img

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    new_bw = bw * margin
    new_bh = bh * margin

    nx1 = max(0, int(cx - new_bw / 2.0))
    ny1 = max(0, int(cy - new_bh / 2.0))
    nx2 = min(w, int(cx + new_bw / 2.0))
    ny2 = min(h, int(cy + new_bh / 2.0))

    if nx2 <= nx1 or ny2 <= ny1:
        return img

    return img[ny1:ny2, nx1:nx2]


def read_protocol(protocol_path, root_dir="") -> List[dict]:
    """
    Accepts either 2‑column (path + label) **or** 1‑column (path only) files.
    Returns list[dict] with keys: *filepath*, *code*, *class*.
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            rel_path = parts[0]
            full_path = os.path.join(root_dir, rel_path)

            if len(parts) == 2:
                label_code = parts[1]
                binary_class = "live" if label_code == "0_0_0" else "attack"
            elif len(parts) == 1:
                label_code = None
                binary_class = None
            else:
                raise ValueError(f"Bad line in protocol file: {line}")

            entries.append({"filepath": full_path, "code": label_code, "class": binary_class})
    return entries


def preprocess_image(img, size):
    """BGR → RGB, then resize to `size` *(H, W)*."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    return img_resized



class ValNoLabelDataset(Dataset):
    """Protocol with *no* labels. Returns *(img_tensor, path)* with bbox crop."""

    def __init__(self, protocol_file: str, model_input_size: Tuple[int, int], root_dir: str = "", bbox_csv: str = None):
        entries = read_protocol(protocol_file, root_dir=root_dir)
        self.files = [e["filepath"] for e in entries]
        self.model_input_size = model_input_size
        self.bbox_by_path = _load_bbox_csv(bbox_csv, root_dir) if bbox_csv else {}
        self.margin = 1.2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        bbox = self.bbox_by_path.get(path)
        if bbox is not None:
            img = _crop_with_margin(img, bbox, self.margin)
        img_resized = preprocess_image(img, self.model_input_size)
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return img_tensor, path

class FeatHead(nn.Module):
    """
    Wrap any timm backbone so `forward(..., return_feats=True)` returns:
        logits  – shape [B, 1]
        feats   – L2-normalised projection, shape [B, 128]
    """
    def __init__(self, backbone_name: str, pretrained=True, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg'
        )
        in_f = self.backbone.num_features
        self.cls = nn.Linear(in_f, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_f, in_f // 2), nn.ReLU(inplace=True),
            nn.Linear(in_f // 2, proj_dim)
        )

    def forward(self, x, return_feats=False):
        f = self.backbone(x)
        logits = self.cls(f)
        if return_feats:
            feats = F.normalize(self.proj(f), dim=1)   # for SupCon
            return logits, feats
        return logits

def tensor_to_img_uint8(t: torch.Tensor) -> np.ndarray:
    t = torch.clamp(t, 0., 1.).permute(1, 2, 0).cpu().numpy() * 255
    return t.astype(np.uint8)

def show_predictions(imgs, scores, files, n=16):
    n  = min(n, len(imgs))
    ids = random.sample(range(len(imgs)), n)
    cols, rows = 4, math.ceil(n / 4)

    plt.figure(figsize=(4*cols, 4*rows))
    for k, j in enumerate(ids):
        plt.subplot(rows, cols, k+1)
        plt.imshow(tensor_to_img_uint8(imgs[j]))
        plt.axis("off")
        plt.title(f"{os.path.basename(files[j])}\nscore={scores[j]:.3f}", fontsize=9)
    plt.tight_layout();  plt.show()


CHECKPOINT = "/workspace/train_two_gpus/weights/pt/two_gpus_val0.0000.pth"

VAL_PROTOCOL  = "/workspace/dataset/Protocol-val-test.txt"
ROOT_DIR      = "/workspace/data/"
CSV_OUT       = "/submission_two_gpus.csv"
val_bbox_csv = "/workspace/dataset/val_bbox.csv",
test_bbox_csv= "/workspace/dataset/test_bbox.csv",

BATCH_SIZE    = 32
NUM_WORKERS   = 8
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42

PROTOCOL_PATH  = VAL_PROTOCOL
CSV_PATH       = None                           # or "val_predictions.csv"
USE_RAW        = True                          # True → all_raw_scores → apply sigmoid
OUTPUT_TXT     = "submission_two_gpus.txt"
OUTPUT_ZIP     = "submission_two_gpus.zip"


#cfg = Config.fromfile(CONFIG_PATH)
MODEL_INPUT_SIZE = (224, 224)

print("✓ Config loaded")
print("✓ Device:", DEVICE)

val_dataset = ValNoLabelDataset(
    protocol_file    = VAL_PROTOCOL,
    model_input_size = MODEL_INPUT_SIZE,
    root_dir         = ROOT_DIR,
    bbox_csv         = "/workspace/dataset/val_test_bbox.csv",
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size   = BATCH_SIZE,
    shuffle      = False,
    num_workers  = NUM_WORKERS,
    pin_memory   = torch.cuda.is_available(),
)

print(f"Validation images: {len(val_dataset)}")

model = FeatHead(
        backbone_name="convnextv2_tiny",
        proj_dim=128
    ).to(DEVICE)

state = torch.load(CHECKPOINT, map_location=DEVICE)
if next(iter(state)).startswith("module."):        # DP → strip prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=True)
model.eval()

print("✓ Weights restored")

all_scores, all_files, all_raw_scores = [], [], []
sigmoid = torch.nn.Sigmoid()

with torch.no_grad():
    for images, paths in val_loader:               # no labels
        images = images.to(DEVICE)
        logits = model(images).squeeze(1)
        probs  = sigmoid(logits)

        all_scores.extend(probs.cpu().tolist())
        all_files.extend(paths)
        all_raw_scores.extend(logits.cpu().tolist())

print("✓ Inference complete –", len(all_scores), "samples")

if CSV_PATH is not None:
    df_pred  = pd.read_csv(CSV_PATH)
    scores   = df_pred["score"].to_numpy()
    files_in_pred = df_pred["filename"].tolist()
elif "all_scores" in globals():
    scores = np.asarray(all_scores)
    files_in_pred = None
elif "all_raw_scores" in globals():
    scores = sigmoid(np.asarray(all_raw_scores)) if USE_RAW else np.asarray(all_raw_scores)
    files_in_pred = None
else:
    raise RuntimeError("No scores found: provide CSV, all_scores or all_raw_scores.")

with open(PROTOCOL_PATH, "r") as f:
    protocol_lines = [ln.strip() for ln in f if ln.strip()]

sub_lines = [
    f"{path} {score:.5f}"
    for path, score in zip(protocol_lines, scores[: len(protocol_lines)])
]
with open(OUTPUT_TXT, "w") as f:
    f.write("\n".join(sub_lines))



val_labels_path = "/workspace/dataset/Protocol-val-labels.txt"
val_labels = pd.read_csv(val_labels_path, sep=" ", names=["path", "label"])
val_labels['label'] = val_labels['label'].apply(lambda x: 0 if x == "0_0_0" else 1)
df_scores_main = pd.read_csv(OUTPUT_TXT, sep=" ", header=None, names=["path", "score"])
#df_scores_main = pd.read_csv("/mnt/usernfshdd/aganiev/projects/cvpr_2025/notebooks/submission_phase2_trial_19_f78408f6_epoch011.txt", sep=" ", header=None, names=["path", "score"])
df = pd.merge(df_scores_main, val_labels, on="path", how="left")
df_val = df[df['path'].str.contains("val")]
eer_val_1, eer_val_thr_1 = calculate_eer(df_val, "score")
print(eer_val_1, eer_val_thr_1)
threshold_1_5, achieved_bpcer_1_5 = get_threshold_for_bpcer(df_val, 'score', target_bpcer=0.015)
#threshold_2, achieved_bpcer_2 = get_threshold_for_bpcer(df_val, 'score', target_bpcer=0.02)
df_binarized_1_5 = binarize_scores(df, 'score', threshold_1_5)
#df_binarized_2 = binarize_scores(df, 'score', threshold_2)
print(threshold_1_5, achieved_bpcer_1_5)
print(df_binarized_1_5['binary_score'].value_counts())
df_binarized_1_5[['path', 'binary_score']].to_csv(OUTPUT_TXT, index=False, header=False, sep=' ')

with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(OUTPUT_TXT, arcname=os.path.basename(OUTPUT_TXT))

print(f"✓ Submission text  ➜ {OUTPUT_TXT}")
print(f"✓ Zipped for upload ➜ {OUTPUT_ZIP}  (size {os.path.getsize(OUTPUT_ZIP)/1024:.1f} KB)")