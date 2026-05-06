from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import torch


class StatKeeper:
    def __init__(self, config=None) -> None:
        self.config = config
        self.binary_preds: list[float] = []
        self.binary_gts:   list[int]   = []


    def step(self, preds: np.ndarray | torch.Tensor, gts: np.ndarray | torch.Tensor):
        """Append one batch of predictions / ground-truth labels."""
        # detach / to CPU if coming from torch
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(gts):
            gts = gts.detach().cpu().numpy()

        self.binary_preds.extend(preds.ravel())
        self.binary_gts.extend(gts.ravel())

    def get_stat(self) -> dict[str, float]:
        preds = np.asarray(self.binary_preds, dtype=float)
        gts   = np.asarray(self.binary_gts,   dtype=float)

        if np.any((gts != 0) & (gts != 1)):
            gts = (gts > 0.5).astype(int)

        unique_labels = np.unique(gts)
        if len(unique_labels) == 1:
            print("StatKeeper warning: only one label in gts – metrics skipped")
            return {}

        # ── ROC & EER ---------------------------------------------------
        fpr, tpr, thr = roc_curve(gts, preds, pos_label=1)
        fnr = 1.0 - tpr
        idx_eer = int(np.nanargmin(np.abs(fnr - fpr)))
        eer     = float(fpr[idx_eer])
        eer_thr = float(thr[idx_eer])

        # ── APCER at fixed BPCER targets (ISO/IEC 30107-3) -------------
        target_bpcers = (0.10, 0.20)          # 10 % & 20 %
        apcer_bpcer_dict: dict[float, dict[str, float]] = {}
        for tgt in target_bpcers:
            idx = int(np.nanargmin(np.abs(fnr - tgt)))
            apcer_bpcer_dict[tgt] = {
                "apcer":  float(fpr[idx]),
                "bpcer":  float(fnr[idx]),
                "thr":    float(thr[idx]),
            }

        return {
            "eer": eer,
            "eer_thr": eer_thr,
            "apcer_at_bpcer10": apcer_bpcer_dict[0.10]["apcer"],
            "apcer_at_bpcer20": apcer_bpcer_dict[0.20]["apcer"],
        }


    def rates_at_thr(self, thr: float) -> tuple[float, float, float]:
        preds = np.asarray(self.binary_preds, dtype=float)
        gts   = np.asarray(self.binary_gts,   dtype=float)
        if np.any((gts != 0) & (gts != 1)):
            gts = (gts > 0.5).astype(int)

        bin_preds = (preds >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(gts, bin_preds, labels=[0, 1]).ravel()

        apcer = fp / (fp + tn) if (fp + tn) else np.nan  # FPR
        bpcer = fn / (fn + tp) if (fn + tp) else np.nan  # FNR
        acer  = (apcer + bpcer) / 2.0

        return float(apcer), float(bpcer), float(acer)
