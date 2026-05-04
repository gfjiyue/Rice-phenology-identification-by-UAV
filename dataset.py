import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp",
            ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP"}


# ---------- Common parsing and utility functions ----------

def _find_classes(root: str) -> Tuple[List[str], Dict[str, int]]:
    """Identify classes: first-level subfolder names under root are used as class names."""
    classes = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if len(classes) == 0:
        raise RuntimeError(f"No class folders found under: {root}")
    return classes, class_to_idx


def _parse_date_any(s: str) -> datetime:
    """
    Parse the date from the filename whenever possible:
      - Prefer 8-digit YYYYMMDD
      - Then 14-digit YYYYMMDDhhmmss -> use the first 8 digits
      - Then 6-digit YYMMDD -> convert to 20YYMMDD
    """
    base = os.path.basename(s)
    m8 = re.search(r"(\d{8})", base)
    if m8:
        return datetime.strptime(m8.group(1), "%Y%m%d")
    m14 = re.search(r"(\d{14})", base)
    if m14:
        return datetime.strptime(m14.group(1)[:8], "%Y%m%d")
    m6 = re.search(r"(\d{6})", base)
    if m6:
        yy = int(m6.group(1)[:2])
        year = 2000 + yy  # Assume 20xx
        rest = m6.group(1)[2:]
        return datetime.strptime(f"{year}{rest}", "%Y%m%d")
    raise ValueError(f"Cannot parse date from filename: {s}")


def _parse_plot_any(s: str) -> str:
    """

    """
    base = os.path.splitext(os.path.basename(s))[0]
    toks = base.split("_")
    # Option 1: the first token looks like JC239
    if toks and re.match(r"[A-Za-z]+[_]?\d+", toks[0]):
        return toks[0]
    # Option 2: find a token containing letters and numbers
    cand = [t for t in toks if re.match(r"[A-Za-z]+[_]?\d+", t)]
    if cand:
        return cand[-1]
    # Fallback: return the first token containing letters
    for t in toks:
        if re.search(r"[A-Za-z]", t):
            return t
    raise ValueError(f"Cannot parse PlotID from filename: {s}")


def _open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")



class PhaseFolderSequenceDualDataset20mAnchor(Dataset):

    def __init__(self,
                 root_20m: str,
                 root_4m: str,
                 sequence_length: int = 5,
                 transform_20m=None,
                 transform_4m=None,
                 transform=None,                 # Use the same transform for both streams if needed
                 stride: int = 1,
                 drop_short: bool = True,
                 max_date_diff_days: int = 0,
                 delta_days: int = 0):
        super().__init__()
        from collections import defaultdict
        self.root_20m = root_20m
        self.root_4m  = root_4m
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.drop_short = bool(drop_short)
        self.max_diff = int(max_date_diff_days)
        self.delta = int(delta_days)

        if transform_20m is None and transform is not None: transform_20m = transform
        if transform_4m  is None and transform is not None: transform_4m  = transform
        self.t20, self.t4 = transform_20m, transform_4m

        # Classes: use identical classes on both sides or their intersection
        classes_20, _ = _find_classes(root_20m)
        classes_4 , _ = _find_classes(root_4m)
        if classes_20 != classes_4:
            inter = [c for c in classes_20 if c in set(classes_4)]
            if not inter: raise RuntimeError("No overlapping class folders between 20m and 4m")
            self.classes = inter
        else:
            self.classes = classes_20
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        # Scan indices on both sides: {plot: [(date, path, cls_idx), ...]}
        idx20, idx4 = defaultdict(list), defaultdict(list)

        def scan_side(root, target):
            for cls in self.classes:
                cdir = os.path.join(root, cls)
                if not os.path.isdir(cdir): continue
                ci = self.class_to_idx[cls]
                for fn in os.listdir(cdir):
                    if os.path.splitext(fn)[1] not in IMG_EXTS: continue
                    fp = os.path.join(cdir, fn)
                    try:
                        dt = _parse_date_any(fn)
                        pid = _parse_plot_any(fn)
                    except Exception:
                        continue
                    target[pid].append((dt, fp, ci))

        scan_side(root_20m, idx20)
        scan_side(root_4m,  idx4)

        def downsample_4m(items4: List[Tuple[datetime,str,int]]):
            # Downsample 4m by Δ; fall back if there are not enough samples
            if self.delta <= 1 or len(items4) <= 1: return items4
            items4 = sorted(items4, key=lambda z: z[0])
            base = items4[0][0].date()
            picked = [(d,p,c) for (d,p,c) in items4 if ((d.date()-base).days % self.delta) == 0]
            if len(picked) < self.sequence_length:
                picked, last = [], None
                for d,p,c in items4:
                    if last is None or (d.date()-last).days >= self.delta:
                        picked.append((d,p,c)); last = d.date()
                if len(picked) < self.sequence_length:
                    return items4
            return picked

        self.samples = []   # [(p20, p4_or_None, cls_idx, mask4)]*T
        self.lengths = []

        for pid in set(idx20.keys()) | set(idx4.keys()):
            items20 = sorted(idx20.get(pid, []), key=lambda z: z[0])
            items4  = sorted(idx4 .get(pid, []), key=lambda z: z[0])
            if not items20: continue
            items4_ds = downsample_4m(items4)

            # Use daily 20m images as anchors and find the nearest 4m image (within ±max_diff and with the same class); missing values are allowed
            j = 0
            pairs = []  # (d20, p20, p4_or_None, ci, m4)
            for d20, p20, c20 in items20:
                while j < len(items4_ds) and items4_ds[j][0] < d20:
                    j += 1
                cand = []
                if j > 0:              cand.append(items4_ds[j-1])
                if j < len(items4_ds): cand.append(items4_ds[j])
                best = None
                for d4, p4, c4 in cand:
                    if c4 != c20: continue
                    gap = abs((d4 - d20).days)
                    if gap <= self.max_diff:
                        if best is None or abs((best[0]-d20).days) > gap:
                            best = (d4, p4, c4)
                if best is None:
                    pairs.append((d20, p20, None, c20, 0))
                else:
                    pairs.append((d20, p20, best[1], c20, 1))

            n = len(pairs)
            if n == 0: continue
            if self.drop_short and n < self.sequence_length: continue

            i = 0
            while i < n:
                end = i + self.sequence_length
                if end <= n:
                    win = pairs[i:end]
                    seq = [(p20, p4, ci, m) for (_,p20,p4,ci,m) in win]
                    self.samples.append(seq); self.lengths.append(self.sequence_length)
                else:
                    if not self.drop_short:
                        win = pairs[i:n]
                        seq = [(p20, p4, ci, m) for (_,p20,p4,ci,m) in win]
                        while len(seq) < self.sequence_length:
                            seq.append(seq[-1])
                        self.samples.append(seq); self.lengths.append(n-i)
                i += self.stride

        if len(self.samples) == 0:
            raise RuntimeError(f"No sequences after 20m-anchored pairing. "
                               f"Check max_date_diff_days={self.max_diff}, delta_days={self.delta}.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        seq = self.samples[idx]
        imgs20, imgs4, labels, masks = [], [], [], []
        for p20, p4, ci, m in seq:
            im20 = _open_rgb(p20)
            if p4 is None:
                im4 = Image.new("RGB", im20.size, (0,0,0))  # Zero-image placeholder
            else:
                im4 = _open_rgb(p4)
            if self.t20 is not None: im20 = self.t20(im20)
            if self.t4  is not None: im4  = self.t4(im4)
            imgs20.append(im20); imgs4.append(im4); labels.append(ci); masks.append(m)
        x20 = torch.stack(imgs20, dim=0)                 # (T,3,H,W)
        x4  = torch.stack(imgs4,  dim=0)                 # (T,3,H,W)
        y   = torch.tensor(labels, dtype=torch.long)     # (T,)
        m4  = torch.tensor(masks,  dtype=torch.float32)  # (T,)
        length = self.lengths[idx]
        return x20, x4, y, length, m4

def collate_fn_dual_mask(batch):
    """
    Input: [(x20, x4, y, length, mask4), ...]
    Output: X20:(B,T,3,H,W)  X4:(B,T,3,H,W)  Y:(B,T)  L:(B,)  M4:(B,T)
    """
    import torch
    max_T = max(L for *_, L, _ in batch)
    X20s, X4s, Ys, Ls, M4s = [], [], [], [], []
    for x20, x4, y, L, m4 in batch:
        if x20.size(0) < max_T:
            pad = max_T - x20.size(0)
            x20 = torch.cat([x20, x20[-1:].repeat(pad,1,1,1)], dim=0)
            x4  = torch.cat([x4 ,  x4[-1:].repeat(pad,1,1,1)], dim=0)
            y   = torch.cat([y  ,   y[-1:].repeat(pad)], dim=0)
            m4  = torch.cat([m4 ,   m4[-1:].repeat(pad)], dim=0)
        X20s.append(x20); X4s.append(x4); Ys.append(y); Ls.append(L); M4s.append(m4)
    return (torch.stack(X20s,0), torch.stack(X4s,0),
            torch.stack(Ys,0), torch.tensor(Ls), torch.stack(M4s,0))
