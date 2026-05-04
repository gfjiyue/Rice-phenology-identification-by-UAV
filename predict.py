import os, re, csv, json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score
)

# ==========================================================
# Dataset & model
# ==========================================================
from dataset import PhaseFolderSequenceDualDataset20mAnchor, collate_fn_dual_mask
from model_mv3 import get_model_dual


# ==========================================================
# Path configuration: modify these paths if needed
# ==========================================================
TEST_ROOT_20M = r"your dataset folder"#20m input path
TEST_ROOT_4M  = r"your dataset folder"#4m input path

SAVE_ROOT     = r"Output root directory"   # Output root directory, automatically created
weight=r"..\1_code\weight"# weight
CKPT_ROOT     = os.path.join(weight, "checkpoints")

os.makedirs(SAVE_ROOT, exist_ok=True)#save path


# ==========================================================
# Configuration: must be consistent with the training settings
# ==========================================================
BACKBONE_20M   = "mobilenetv3"
BACKBONE_4M    = "mobilenetv3"
SEQUENCE_LEN   = 5
MAX_DATE_DIFF  = 0
PROJ_DIM       = 512
HIDDEN_SIZE    = 128
NUM_LAYERS     = 1
BIDIRECTIONAL  = False
DROPOUT        = 0.0

BATCH_SIZE     = 64
WORKERS        = 4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DELTAS         = [1]  # 1,2,3,4,5,6,7,10,14


# ==========================================================
# Utility functions: parse date and plot ID
# ==========================================================
def _parse_date_any_from_path(path: str) -> str:
    base = os.path.basename(path)
    m8  = re.search(r"(\d{8})", base)
    m14 = re.search(r"(\d{14})", base)
    m6  = re.search(r"(\d{6})", base)
    try:
        if m8:  return datetime.strptime(m8.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        if m14: return datetime.strptime(m14.group(1)[:8], "%Y%m%d").strftime("%Y-%m-%d")
        if m6:
            yy = int(m6.group(1)[:2]); year = 2000 + yy
            rest = m6.group(1)[2:]
            return datetime.strptime(f"{year}{rest}", "%Y%m%d").strftime("%Y-%m-%d")
    except:
        pass
    return ""

def _parse_plot_any_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    toks = base.split("_")
    for t in toks:
        if re.match(r"[A-Za-z]+_?\d+", t):
            return t
    for t in toks:
        if re.search(r"[A-Za-z]", t):
            return t
    return base


# ==========================================================
# Built-in confusion matrix plotting function
# Does not depend on utils.py
# ==========================================================
def plot_confusion_matrix_v2(y_true, y_pred, classes, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================================================
# Complete prediction pipeline for a single delta
# ==========================================================
def run_one_delta(delta_days: int):
    print("\n" + "="*90)
    print(f"###  Start prediction  Δ = {delta_days}")
    print("="*90)

    OUT_DIR = os.path.join(SAVE_ROOT, f"pred_outputs_delta{delta_days}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Best checkpoint corresponding to the training stage
    exp_name = f"delta{delta_days}_{BACKBONE_20M}_{BACKBONE_4M}_T{SEQUENCE_LEN}_proj{PROJ_DIM}"
    ckpt = os.path.join(CKPT_ROOT, exp_name, f"{exp_name}_best.pt")

    if not os.path.isfile(ckpt):
        print(f"[WARN] Weight file not found: {ckpt} (skipped)")
        return None

    # Preprocessing pipeline: must be consistent with training
    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Build prediction dataset
    ds = PhaseFolderSequenceDualDataset20mAnchor(
        root_20m=TEST_ROOT_20M,
        root_4m=TEST_ROOT_4M,
        sequence_length=SEQUENCE_LEN,
        transform_20m=t,
        transform_4m=t,
        stride=1,
        drop_short=True,
        max_date_diff_days=MAX_DATE_DIFF,
        delta_days=delta_days
    )

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True,
        collate_fn=collate_fn_dual_mask
    )

    num_classes = len(ds.classes)
    print(f"[Δ={delta_days}] Windows={len(ds)} | Classes={ds.classes}")

    # Build model
    model = get_model_dual(
        backbone_20m=BACKBONE_20M,
        backbone_4m=BACKBONE_4M,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    ).to(DEVICE)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # ------------------------
    # Inference
    # ------------------------
    all_probs = []
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for x20, x4, y, lengths, m4 in loader:
            x20 = x20.to(DEVICE)
            x4  = x4.to(DEVICE)
            m4  = m4.to(DEVICE)

            out = model(x20, x4, lengths, mask4=m4)
            prob = F.softmax(out, dim=-1)

            all_probs.append(prob.cpu())
            all_preds.append(prob.argmax(dim=-1).cpu())
            all_masks.append(m4.cpu())

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    masks = torch.cat(all_masks)

    N, T, C = probs.shape

    # ==========================================================
    # Save frame-level prediction results to CSV
    # ==========================================================
    frame_csv = os.path.join(OUT_DIR, "frame_preds.csv")
    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window","t","plot","date","phase_gt","pred","prob","has_4m"])

        for i in range(N):
            seq = ds.samples[i]
            L = ds.lengths[i]

            for t_idx in range(L):
                p20, p4, ci, m = seq[t_idx]
                pred_c = int(preds[i, t_idx])
                prob_c = float(probs[i, t_idx, pred_c])

                w.writerow([
                    i,
                    t_idx,
                    _parse_plot_any_from_path(p20),
                    _parse_date_any_from_path(p20),
                    ds.classes[ci],
                    ds.classes[pred_c],
                    f"{prob_c:.6f}",
                    int(masks[i, t_idx] > 0.5)
                ])

    # ==========================================================
    # Save window-center prediction results to CSV
    # ==========================================================
    center_csv = os.path.join(OUT_DIR, "window_center_preds.csv")
    with open(center_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window","center_t","plot","date","phase_gt","pred","prob","has_4m"])

        for i in range(N):
            L = ds.lengths[i]
            c = L // 2
            p20, p4, ci, m = ds.samples[i][c]
            pred_c = int(preds[i, c])
            prob_c = float(probs[i, c, pred_c])

            w.writerow([
                i, c,
                _parse_plot_any_from_path(p20),
                _parse_date_any_from_path(p20),
                ds.classes[ci],
                ds.classes[pred_c],
                f"{prob_c:.6f}",
                int(masks[i, c] > 0.5)
            ])

    # ==========================================================
    # Compute confusion matrix at the frame level
    # ==========================================================
    y_true, y_pred = [], []
    for i in range(N):
        seq = ds.samples[i]
        L = ds.lengths[i]
        for t_idx in range(L):
            _, _, ci, _ = seq[t_idx]
            y_true.append(ci)
            y_pred.append(int(preds[i, t_idx]))

    cm_path       = os.path.join(OUT_DIR, "confusion_matrix.png")
    cm_norm_path  = os.path.join(OUT_DIR, "confusion_matrix_norm.png")

    plot_confusion_matrix_v2(y_true, y_pred, ds.classes, cm_path)
    plot_confusion_matrix_v2(y_true, y_pred, ds.classes, cm_norm_path, normalize=True)

    # ==========================================================
    # Prediction metrics at the frame level
    # ==========================================================
    OA       = accuracy_score(y_true, y_pred)
    P        = precision_score(y_true, y_pred, average="macro", zero_division=0)
    R        = recall_score(y_true, y_pred, average="macro", zero_division=0)
    F1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    Kappa    = cohen_kappa_score(y_true, y_pred)

    print(f"[Δ={delta_days}] OA={OA:.4f} P={P:.4f} R={R:.4f} F1={F1:.4f} Kappa={Kappa:.4f}")

    # Return summary information
    return {
        "delta": delta_days,
        "windows": len(ds),
        "OA": OA,
        "precision": P,
        "recall": R,
        "F1": F1,
        "kappa": Kappa
    }


# ==========================================================
# Main program: loop through all delta values
# ==========================================================
def main():
    print("Device:", DEVICE)

    summary_path = os.path.join(SAVE_ROOT, "summary_predict.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["delta","num_windows","OA","Precision","Recall","F1","Kappa"]
            )

    for d in DELTAS:
        stat = run_one_delta(d)
        if stat:
            with open(summary_path, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([
                    stat["delta"],
                    stat["windows"],
                    f"{stat['OA']:.6f}",
                    f"{stat['precision']:.6f}",
                    f"{stat['recall']:.6f}",
                    f"{stat['F1']:.6f}",
                    f"{stat['kappa']:.6f}",
                ])

    print("\nAll Delta predictions are complete. Summary:")
    print(summary_path)


if __name__ == "__main__":
    main()