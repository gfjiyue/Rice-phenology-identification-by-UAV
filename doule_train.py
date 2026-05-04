# ======================================================

# ======================================================
import time
import os, csv, json, re, argparse, shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, classification_report)
import matplotlib.pyplot as plt

# ====== Project-specific modules ======
from dataset import PhaseFolderSequenceDualDataset20mAnchor, collate_fn_dual_mask
from model_mv3 import get_model_dual
from utils import save_log, plot_confusion_matrix


# =========================
# Path configuration (modify as needed)
# =========================
datasetfolder = r"your dataset folder"# your dataset folder
resultfolder= r"your result folder"# your result folder
TRAIN_ROOT_20M = datasetfolder+r"/20m_split_stage_new/train"
VAL_ROOT_20M   = datasetfolder+r"/20m_split_stage_new/val"
TEST_ROOT_20M  = datasetfolder+r"/20m_split_stage_new/test"

TRAIN_ROOT_4M  = datasetfolder+r"/4m_split_stage_new/train"
VAL_ROOT_4M    = datasetfolder+r"/4m_split_stage_new/val"
TEST_ROOT_4M   = datasetfolder+r"/4m_split_stage_new/test"

SAVE_ROOT      = resultfolder+r"/20_4m_full_version"
os.makedirs(SAVE_ROOT, exist_ok=True)


# =========================
# Training configuration (adjust as needed)
# =========================
BACKBONE_20M   = "mobilenetv3"   # mobilenetv3 / resnet, etc.
BACKBONE_4M    = "mobilenetv3"
SEQUENCE_LEN   = 5
MAX_DATE_DIFF  = 1               # Same-day 4m-20m matching; can be relaxed to 1/2/...
PROJ_DIM       = 512
BIDIRECTIONAL  = False
DROPOUT        = 0.0

EPOCHS         =100
BATCH_SIZE     = 64
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
USE_AMP        = True
SEED           = 42
# Settings for training with different Δ values
DELTAS         = [1,2,3,4,5,6,7,10,14]  # Add or remove values as needed: 3,4,5,6,7,10,/dev/shm/
# Quick debugging: use only 10% of the data
FAST_DEBUG     = False           # Set to True for debugging; set back to False for formal runs

# Early stopping patience (stop if Macro-F1 does not improve for N consecutive epochs)
EARLY_STOP_PATIENCE = 15

# Default number of DataLoader workers (overridden by the /dev/shm check)
WORKERS_DEFAULT = 4


# =========================
# Utility functions
# =========================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # Prioritize performance; for full reproducibility, set deterministic=True and disable benchmark


def _parse_date_any_from_path(path: str) -> str:
    base = os.path.basename(path)
    m8  = re.search(r"(\d{8})", base)
    m14 = re.search(r"(\d{14})", base)
    m6  = re.search(r"(\d{6})", base)
    try:
        if m8:
            return datetime.strptime(m8.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        if m14:
            return datetime.strptime(m14.group(1)[:8], "%Y%m%d").strftime("%Y-%m-%d")
        if m6:
            yy = int(m6.group(1)[:2]); rest = m6.group(1)[2:]
            return datetime.strptime(f"{2000+yy}{rest}", "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        pass
    return ""


def _parse_plot_any_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    toks = base.split("_")
    if toks and re.match(r"[A-Za-z]+_?\d+", toks[0]): return toks[0]
    cand = [t for t in toks if re.match(r"[A-Za-z]+_?\d+", t)]
    if cand: return cand[-1]
    for t in toks:
        if re.search(r"[A-Za-z]", t): return t
    return ""


def safe_save_log(file_path, epoch, train_loss, val_loss, acc, **extra):
    exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['Epoch','TrainLoss','ValLoss','Accuracy','MacroF1','LR'])
        w.writerow([epoch, train_loss, val_loss, acc, extra.get("macro_f1",0), extra.get("lr",0)])


def auto_dataloader_config(default_workers=WORKERS_DEFAULT):
    """
    Automatically check /dev/shm space and adjust DataLoader parameters to prevent “Bus error: out of shared memory”.
    """
    shm_path = "/dev/shm"
    num_workers = default_workers
    persistent_workers = True
    prefetch_factor = 2
    try:
        usage = shutil.disk_usage(shm_path)
        free_gb = usage.free / (1024**3)
        print(f"[INFO] Available /dev/shm space: {free_gb:.2f} GB")
        if free_gb < 2.0:
            print("[WARN] /dev/shm is low -> num_workers=0, persistent_workers=False, prefetch_factor=1")
            num_workers = 0
            persistent_workers = False
            prefetch_factor = 1
        elif free_gb < 4.0:
            print("[INFO] /dev/shm is moderate -> num_workers=2, persistent_workers=False, prefetch_factor=1")
            num_workers = 2
            persistent_workers = False
            prefetch_factor = 1
    except Exception as e:
        print(f"[WARN] Unable to check /dev/shm: {e}; using the default DataLoader configuration")
    return num_workers, persistent_workers, prefetch_factor


def setup_multi_gpu(model):
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"[INFO] Detected {n_gpu} GPU(s)")
        if n_gpu > 1:
            model = nn.DataParallel(model)
            print("[INFO] ✅ DataParallel multi-GPU training enabled")
    return model


def export_used_to_csv(train_set, val_set, test_set, out_dir: str):
    """
    Export the sample list used in this run (supports Subset).
    """
    os.makedirs(out_dir, exist_ok=True)

    def extract_samples(ds):
        if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
            base, idxs = ds.dataset, ds.indices
            return [base.samples[i] for i in idxs if hasattr(base, "samples")]
        elif hasattr(ds, "samples"):
            return ds.samples
        else:
            return []

    def dump(ds, split):
        samples = extract_samples(ds)
        if not samples: return
        rec20, rec4 = {}, {}
        for seq in samples:
            for p20, p4, ci, _ in seq:
                phase = ds.dataset.classes[ci] if hasattr(ds, 'dataset') else ds.classes[ci]
                rec20[p20] = (phase, 1)
                if p4 is not None:
                    rec4[p4]  = (phase, 1)

        for scale, rec in [("20m", rec20), ("4m", rec4)]:
            with open(os.path.join(out_dir, f"{split}_{scale}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["scale","split","phase","filepath","filename","plot_id","date","ext","available"])
                for p,(ph,avail) in sorted(rec.items()):
                    w.writerow([scale, split, ph, os.path.abspath(p), os.path.basename(p),
                                _parse_plot_any_from_path(p), _parse_date_any_from_path(p),
                                os.path.splitext(p)[1], avail])

    dump(train_set, "train"); dump(val_set, "val"); dump(test_set, "test")
    print(f"[INFO] CSV file list has been written to: {out_dir}")


# =========================
# Evaluation and training (with masks)
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, save_dir=None, tag="eval"):
    """
    Returns:
      val_loss, acc, macro_f1, gts, preds, metrics(dict: OA/Precision/Recall/F1/Kappa)
    If save_dir is provided, also save the classification report and confusion matrices (including the normalized one).
    Note: only valid frames are counted (according to the lengths mask).
    """
    model.eval()
    val_loss, steps = 0.0, 0
    preds, gts = [], []
    for x20, x4, y, lengths, m4 in loader:
        x20 = x20.to(device, non_blocking=True)
        x4  = x4 .to(device, non_blocking=True)
        y   = y  .to(device, non_blocking=True)
        m4  = m4 .to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type=="cuda"), dtype=torch.float16):
            out = model(x20, x4, lengths, mask4=m4)  # [B, T, C]
            B, T = y.size()
            valid_mask = torch.arange(T, device=y.device)[None, :] < lengths.to(y.device)[:, None]
            y_masked = y.clone()
            y_masked[~valid_mask] = -100  # Ignore padding
            loss = criterion(out.view(-1, out.size(-1)), y_masked.view(-1))

        val_loss += float(loss.item()); steps += 1

        # Collect valid frames only
        pred_flat = out.argmax(dim=2)[valid_mask].detach().cpu().tolist()
        gt_flat   = y[valid_mask].detach().cpu().tolist()
        preds += pred_flat
        gts   += gt_flat

    # Metrics (all calculated on valid frames)
    acc       = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
    recall    = recall_score(gts, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
    macro_f1  = f1_score(gts, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
    kappa     = cohen_kappa_score(gts, preds)
    metrics = {"OA": acc, "Precision": precision, "Recall": recall, "F1": macro_f1, "Kappa": kappa}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        rep = classification_report(gts, preds, labels=list(range(num_classes)), output_dict=True, digits=4)
        with open(os.path.join(save_dir, f"{tag}_cls_report.json"), "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "report": rep}, f, ensure_ascii=False, indent=2)
        plot_confusion_matrix(gts, preds, labels=list(range(num_classes)),
                              save_path=os.path.join(save_dir, f"{tag}_cm.png"))
        plot_confusion_matrix(gts, preds, labels=list(range(num_classes)),
                              save_path=os.path.join(save_dir, f"{tag}_cm_norm.png"), normalize=True)

    return val_loss / max(steps, 1), acc, macro_f1, gts, preds, metrics


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    total_loss, steps = 0.0, 0
    for x20, x4, y, lengths, m4 in loader:
        x20 = x20.to(device, non_blocking=True)
        x4  = x4 .to(device, non_blocking=True)
        y   = y  .to(device, non_blocking=True)
        m4  = m4 .to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                out = model(x20, x4, lengths, mask4=m4)
                B, T = y.size()
                valid_mask = torch.arange(T, device=y.device)[None, :] < lengths.to(y.device)[:, None]
                y_masked = y.clone()
                y_masked[~valid_mask] = -100
                loss = criterion(out.view(-1, out.size(-1)), y_masked.view(-1))
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(x20, x4, lengths, mask4=m4)
            B, T = y.size()
            valid_mask = torch.arange(T, device=y.device)[None, :] < lengths.to(y.device)[:, None]
            y_masked = y.clone()
            y_masked[~valid_mask] = -100
            loss = criterion(out.view(-1, out.size(-1)), y_masked.view(-1))
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()); steps += 1
    return total_loss / max(steps, 1)


# =========================
# Main training + testing workflow (with early stopping)
# =========================
def run_train_and_test():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Automatically adapt the DataLoader
    num_workers, persistent_workers, prefetch_factor = auto_dataloader_config(default_workers=WORKERS_DEFAULT)

    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Initialize summary.csv (including OA/Precision/Recall/F1/Kappa)
    summary_path = os.path.join(SAVE_ROOT, "summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                "delta_days", "best_macro_f1", "best_acc", "best_epoch",
                "last_macro_f1", "last_acc",
                "OA", "Precision", "Recall", "F1", "Kappa", "time_minutes"
            ])

    all_results = []

    for DELTA in DELTAS:
        start_time = time.time()
        print(f"\n{'=' * 70}\n### Δ = {DELTA} days -- training started\n{'=' * 70}")

        # Datasets
        train_set = PhaseFolderSequenceDualDataset20mAnchor(
            root_20m=TRAIN_ROOT_20M, root_4m=TRAIN_ROOT_4M,
            sequence_length=SEQUENCE_LEN, transform_20m=t, transform_4m=t,
            stride=1, drop_short=True, max_date_diff_days=MAX_DATE_DIFF, delta_days=DELTA
        )
        val_set = PhaseFolderSequenceDualDataset20mAnchor(
            root_20m=VAL_ROOT_20M, root_4m=VAL_ROOT_4M,
            sequence_length=SEQUENCE_LEN, transform_20m=t, transform_4m=t,
            stride=SEQUENCE_LEN, drop_short=True, max_date_diff_days=MAX_DATE_DIFF, delta_days=DELTA
        )
        test_set = PhaseFolderSequenceDualDataset20mAnchor(
            root_20m=TEST_ROOT_20M, root_4m=TEST_ROOT_4M,
            sequence_length=SEQUENCE_LEN, transform_20m=t, transform_4m=t,
            stride=SEQUENCE_LEN, drop_short=True, max_date_diff_days=MAX_DATE_DIFF, delta_days=DELTA
        )

        num_classes = len(train_set.classes)
        print(f"Classes({num_classes}): {train_set.classes}")
        print(f"[Δ={DELTA}] Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

        # FAST_DEBUG: only 10% of the data
        if FAST_DEBUG:
            def _subset(ds):
                n = max(1, int(len(ds) * 0.1))
                return Subset(ds, range(n))
            train_set = _subset(train_set)
            val_set   = _subset(val_set)
            test_set  = _subset(test_set)
            print("⚡ FAST_DEBUG: using only 10% of the data")

        # Directories
        exp_name = f"delta{DELTA}_{BACKBONE_20M}_{BACKBONE_4M}_T{SEQUENCE_LEN}_proj{PROJ_DIM}"
        logs_dir = os.path.join(SAVE_ROOT, "logs", exp_name)
        ckpt_dir = os.path.join(SAVE_ROOT, "checkpoints", exp_name)
        os.makedirs(logs_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

        # Export the sample list
        export_used_to_csv(train_set, val_set, test_set, os.path.join(logs_dir, "dataset_csv"))

        # Loader
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn_dual_mask
        )
        val_loader   = DataLoader(
            val_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn_dual_mask
        )
        test_loader  = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn_dual_mask
        )

        # Model & optimizer
        model = get_model_dual(
            backbone_20m=BACKBONE_20M, backbone_4m=BACKBONE_4M,
            hidden_size=128, num_layers=1, num_classes=num_classes,
            proj_dim=PROJ_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL
        ).to(device)
        model = setup_multi_gpu(model)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda") if (USE_AMP and device.type=="cuda") else None

        log_csv = os.path.join(logs_dir, "train_log.csv")
        if not os.path.exists(log_csv):
            with open(log_csv, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss,acc,macro_f1,lr\n")

        best_metric = -1.0
        best_acc_for_print = 0.0
        best_epoch = 0
        last_acc, last_macro = 0.0, 0.0
        patience = EARLY_STOP_PATIENCE
        no_improve_epochs = 0

        # ===== Training loop =====
        for epoch in range(1, EPOCHS+1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                         scaler=scaler, grad_clip=GRAD_CLIP)
            val_loss, acc, macro_f1, gts, preds, _ = evaluate(
                model, val_loader, criterion, device, num_classes
            )
            cur_lr = next(iter(optimizer.param_groups))["lr"]
            safe_save_log(log_csv, epoch, train_loss, val_loss, acc, macro_f1=macro_f1, lr=cur_lr)

            print(f"[Δ={DELTA}] Epoch {epoch:03d} | "
                  f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                  f"Acc={acc*100:.2f}% | MacroF1={macro_f1:.4f} | LR={cur_lr:.2e}")

            # Save last.pt (every epoch)
            last_path = os.path.join(ckpt_dir, f"{exp_name}_last.pt")
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, last_path)
            last_acc, last_macro = acc, macro_f1

            # Save best.pt + validation-set confusion matrices
            if macro_f1 > best_metric:
                best_metric = macro_f1
                best_acc_for_print = acc
                best_epoch = epoch
                no_improve_epochs = 0
                best_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
                torch.save(state, best_path)

                plot_confusion_matrix(gts, preds, labels=list(range(num_classes)),
                                      save_path=os.path.join(logs_dir, "conf_best.png"))
                plot_confusion_matrix(gts, preds, labels=list(range(num_classes)),
                                      save_path=os.path.join(logs_dir, "conf_best_norm.png"), normalize=True)
                print(f"✅ [Δ={DELTA}] Saved the best model: {best_path} | MacroF1={macro_f1:.4f}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"[Δ={DELTA}] Early stopping triggered: validation Macro-F1 did not improve for {patience} consecutive epochs")
                    break

            scheduler.step()

        # ===== Testing stage (using best.pt) =====
        print(f"\n[Δ={DELTA}] Test evaluation (best.pt)...")
        # Rebuild the model and load weights here to ensure consistency between DP and non-DP modes
        test_model = get_model_dual(
            backbone_20m=BACKBONE_20M, backbone_4m=BACKBONE_4M,
            hidden_size=128, num_layers=1, num_classes=num_classes,
            proj_dim=PROJ_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL
        ).to(device)
        best_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
        test_model.load_state_dict(torch.load(best_path, map_location=device))
        # Note: do not wrap with DP during testing to avoid DP wrapping affecting saved paths

        test_loss, test_acc, test_macro, _, _, test_metrics = evaluate(
            test_model, test_loader, criterion, device, num_classes,
            save_dir=os.path.join(logs_dir, "test_results"), tag="test"
        )

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60.0

        # Write to summary.csv (including OA / Precision / Recall / F1 / Kappa)
        with open(summary_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                DELTA,
                f"{best_metric:.6f}",
                f"{best_acc_for_print:.6f}",
                best_epoch,
                f"{last_macro:.6f}",
                f"{last_acc:.6f}",
                f"{test_metrics['OA']:.6f}",
                f"{test_metrics['Precision']:.6f}",
                f"{test_metrics['Recall']:.6f}",
                f"{test_metrics['F1']:.6f}",
                f"{test_metrics['Kappa']:.6f}",
                f"{runtime_minutes:.2f} min"
            ])

        print(f"[Δ={DELTA}] Completed: best Macro-F1={best_metric:.4f} (epoch {best_epoch}), "
              f"last Macro-F1={last_macro:.4f}, last Acc={last_acc*100:.2f}%")
        print(f"[Δ={DELTA}] ✅ Test → OA={test_metrics['OA']*100:.2f}% | "
              f"P={test_metrics['Precision']:.4f} | R={test_metrics['Recall']:.4f} | "
              f"F1={test_metrics['F1']:.4f} | Kappa={test_metrics['Kappa']:.4f}")
        print(f"⏱️ [Δ={DELTA}] Total runtime: {runtime_minutes:.2f} minutes")
        print("-"*80)

        all_results.append((DELTA, best_metric, best_acc_for_print, test_metrics["F1"], test_metrics["OA"]))

    # Trend plot (Val vs Test)
    deltas   = [r[0] for r in all_results]
    val_f1   = [r[1] for r in all_results]
    val_acc  = [r[2] for r in all_results]
    test_f1  = [r[3] for r in all_results]
    test_oa  = [r[4] for r in all_results]

    plt.figure(figsize=(8,6))
    plt.plot(deltas, val_f1, "o-", label="Val Macro-F1")
    plt.plot(deltas, test_f1, "s--", label="Test Macro-F1")
    plt.plot(deltas, val_acc, "o-", label="Val Acc")
    plt.plot(deltas, test_oa, "s--", label="Test OA")
    plt.xlabel("Δ Days (Temporal Gap)")
    plt.ylabel("Score")
    plt.title("Performance vs Δ (20m Anchor)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "delta_performance.png"), dpi=300)
    print(f"\nPerformance trend plot has been saved to: {SAVE_ROOT}/delta_performance.png")


# =========================
# Test-only mode (reuse best.pt)
# =========================
def run_test_only():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Adaptive DataLoader (also prevents test_only from crashing)
    num_workers, persistent_workers, prefetch_factor = auto_dataloader_config(default_workers=WORKERS_DEFAULT)

    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    for DELTA in DELTAS:
        exp_name = f"delta{DELTA}_{BACKBONE_20M}_{BACKBONE_4M}_T{SEQUENCE_LEN}_proj{PROJ_DIM}"
        ckpt_dir = os.path.join(SAVE_ROOT, "checkpoints", exp_name)
        logs_dir = os.path.join(SAVE_ROOT, "logs", exp_name)
        best_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
        if not os.path.exists(best_path):
            print(f"⚠️ [Δ={DELTA}] Model file not found: {best_path}")
            continue

        test_set = PhaseFolderSequenceDualDataset20mAnchor(
            root_20m=TEST_ROOT_20M, root_4m=TEST_ROOT_4M,
            sequence_length=SEQUENCE_LEN, transform_20m=t, transform_4m=t,
            stride=SEQUENCE_LEN, drop_short=True, max_date_diff_days=MAX_DATE_DIFF, delta_days=DELTA
        )
        num_classes = len(test_set.classes)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=persistent_workers,
                                 prefetch_factor=prefetch_factor,
                                 collate_fn=collate_fn_dual_mask)

        model = get_model_dual(
            backbone_20m=BACKBONE_20M, backbone_4m=BACKBONE_4M,
            hidden_size=128, num_layers=1, num_classes=num_classes,
            proj_dim=PROJ_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL
        ).to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        test_dir = os.path.join(logs_dir, "test_only_results")
        test_loss, test_acc, test_macro, _, _, test_metrics = evaluate(
            model, test_loader, criterion, device, num_classes,
            save_dir=test_dir, tag="test_only"
        )
        print(f"[Δ={DELTA}] TestOnly → OA={test_metrics['OA']*100:.2f}% | "
              f"P={test_metrics['Precision']:.4f} | R={test_metrics['Recall']:.4f} | "
              f"F1={test_metrics['F1']:.4f} | Kappa={test_metrics['Kappa']:.4f}")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action="store_true", help="Evaluate only on the test set (skip the training stage)")
    args = parser.parse_args()

    if args.test_only:
        print("\nEntering TEST-ONLY mode\n")
        run_test_only()
    else:
        print("\nEntering full training + validation + testing pipeline mode\n")
        run_train_and_test()
