#!/usr/bin/env python3
"""
Leakage-free KG-SSL baseline for the Basin2Vec temporal-generalization benchmark.

What this script does
---------------------
1) Reimplements the *unsupervised* KG-SSL setting from Ghosh et al. (KDD 2022):
       L = L_rec + L_cont
   with:
     - 365-day driver-response windows
     - one-layer bidirectional LSTM encoder, hidden size 64
     - forward/backward final states added to form the embedding
     - autoregressive LSTM reconstruction decoder
     - same-basin temporal windows as positives, other basins as negatives

2) Uses ONLY 1990-01-01 ... 2009-12-31 driver + observed streamflow data
   to train KG-SSL and construct one fixed KG-SSL embedding per basin.
   No 2010-2014 discharge is ever used to construct the KG-SSL descriptor.

3) Evaluates those fixed descriptors on the SAME Basin2Vec temporal split:
       downstream train: 1990-01-01 ... 2009-12-31
       downstream eval : 2010-01-01 ... 2014-12-31
   using the same 365-day, lead-0, horizon-1 LSTM static-conditioning model
   and the same NSE/KGE calculation used in the supplied Basin2Vec script.

4) By default, reads the exact basin list previously produced by the Basin2Vec
   temporal experiment (common_temporal_basins.csv), so the basin population is
   identical rather than merely "all CAMELS basins available to KG-SSL".

Important fairness note
-----------------------
KG-SSL is allowed historical streamflow only because this is the *gauged temporal*
generalization experiment. The downstream forecasting model itself still receives
NO past discharge as input; the historical discharge is used only once to construct
the fixed KG-SSL basin descriptor.

This is a paper-faithful reimplementation. I could not locate an official public
KG-SSL code repository/checkpoint; the implementation follows the KDD 2022 paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# Defaults copied/aligned with the supplied Basin2Vec temporal benchmark
# -----------------------------------------------------------------------------
# CAMELS-US is read from CAMELS_ROOT (default: $BASIN2VEC_DATA_ROOT/camels_us). The
# shipped results are in evaluation_outputs/kgssl_temporal_baseline; reruns write to a
# separate folder so they are not overwritten.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("BASIN2VEC_DATA_ROOT", REPO_ROOT / "data"))
DEFAULT_CAMELS_ROOT = Path(os.environ.get("CAMELS_ROOT", DATA_ROOT / "camels_us"))
DEFAULT_COMMON_BASINS_CSV = (
    SCRIPT_DIR / "evaluation_outputs" / "final_temporal_split_results" / "common_temporal_basins.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "evaluation_outputs" / "reproduced_kgssl_temporal_baseline"

DATE_COL = "date"
TARGET_COL = "discharge"
DYNAMIC_COLS = ["prcp", "srad", "tmax", "tmin", "vp"]

DATA_START = "1990-01-01"
DATA_END = "2014-12-31"
TRAIN_START = "1990-01-01"
TRAIN_END = "2009-12-31"
EVAL_START = "2010-01-01"
EVAL_END = "2014-12-31"

# KG-SSL paper implementation details
KGSSL_WINDOW = 365
KGSSL_STRIDE = 183
KGSSL_HIDDEN = 64
KGSSL_LAYERS = 1
KGSSL_TEMPERATURE = 0.1
KGSSL_LAMBDA_REC = 1.0
KGSSL_LAMBDA_CONT = 1.0

# Basin2Vec downstream benchmark details
DOWNSTREAM_SEQ_LEN = 365
DOWNSTREAM_HORIZON = 1
DOWNSTREAM_LEAD = 0
DOWNSTREAM_HIDDEN = 128
DOWNSTREAM_LAYERS = 2
DOWNSTREAM_DROPOUT = 0.15
DOWNSTREAM_EPOCHS = 40
DOWNSTREAM_PATIENCE = 5
DOWNSTREAM_LR = 1e-3
DOWNSTREAM_WEIGHT_DECAY = 1e-4
DOWNSTREAM_GRAD_CLIP = 1.0
DEFAULT_DOWNSTREAM_SEEDS = (42, 123, 456)


# -----------------------------------------------------------------------------
# Reproducibility / utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_site_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(8) if len(s) <= 8 else s


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def parse_seed_list(text: str) -> List[int]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one downstream seed is required.")
    return [int(x) for x in vals]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# -----------------------------------------------------------------------------
# CAMELS-US loading -- mirrors the supplied Basin2Vec temporal script
# -----------------------------------------------------------------------------
def detect_camels_forcing_header(path: Path) -> int:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            cols = line.strip().split()
            if len(cols) >= 3:
                if (
                    cols[0].lower() == "year"
                    and cols[1].lower() in ["mnth", "month"]
                    and cols[2].lower() == "day"
                ):
                    return i
    raise RuntimeError(f"Could not detect CAMELS forcing header: {path}")


def read_camels_daymet(daymet_dir: Path, site_id: str) -> pd.DataFrame:
    sid = normalize_site_id(site_id)
    files = list(daymet_dir.rglob(f"{sid}*_forcing_leap.txt"))
    if not files:
        raise FileNotFoundError(f"No Daymet forcing found for {sid}")

    path = files[0]
    header = detect_camels_forcing_header(path)
    df = pd.read_csv(path, sep=r"\s+", skiprows=header)
    df = df.rename(
        columns={
            "Year": "year",
            "Mnth": "month",
            "Month": "month",
            "Day": "day",
            "prcp(mm/day)": "prcp",
            "srad(W/m2)": "srad",
            "swe(mm)": "swe",
            "tmax(C)": "tmax",
            "tmin(C)": "tmin",
            "vp(Pa)": "vp",
        }
    )

    required = ["year", "month", "day"] + DYNAMIC_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing Daymet columns for {sid}: {missing}. Available={df.columns.tolist()}"
        )

    df[DATE_COL] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce"
    )
    out = df[[DATE_COL] + DYNAMIC_COLS].copy()
    for c in DYNAMIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_camels_streamflow(streamflow_dir: Path, site_id: str) -> pd.DataFrame:
    sid = normalize_site_id(site_id)
    files = list(streamflow_dir.rglob(f"{sid}_streamflow_qc.txt"))
    if not files:
        raise FileNotFoundError(f"No streamflow found for {sid}")

    df = pd.read_csv(
        files[0],
        sep=r"\s+",
        header=None,
        names=["site_id", "year", "month", "day", TARGET_COL, "flag"],
    )
    df[DATE_COL] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce"
    )
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df[[DATE_COL, TARGET_COL]].copy()


def read_common_basin_ids(common_csv: Path) -> List[str]:
    if not common_csv.exists():
        raise FileNotFoundError(
            f"Exact Basin2Vec common-basin list not found: {common_csv}\n"
            "Run the original temporal benchmark first, or pass --common-basins-csv "
            "to the common_temporal_basins.csv produced by that experiment."
        )
    df = pd.read_csv(common_csv)
    if "site_id" not in df.columns:
        raise ValueError(f"{common_csv} must contain a 'site_id' column.")
    ids = [normalize_site_id(x) for x in df["site_id"].tolist()]
    ids = sorted({x for x in ids if x is not None})
    if not ids:
        raise RuntimeError(f"No valid site IDs in {common_csv}")
    return ids


def load_camels_data(camels_root: Path, basin_ids: Sequence[str]) -> Dict[str, pd.DataFrame]:
    daymet_dir = camels_root / "basin_mean_forcing" / "daymet"
    streamflow_dir = camels_root / "usgs_streamflow"
    start = pd.Timestamp(DATA_START)
    end = pd.Timestamp(DATA_END)

    basin_data: Dict[str, pd.DataFrame] = {}
    skipped = []

    for sid in tqdm(list(basin_ids), desc="load CAMELS basins"):
        try:
            forcing = read_camels_daymet(daymet_dir, sid)
            flow = read_camels_streamflow(streamflow_dir, sid)
            df = forcing.merge(flow, on=DATE_COL, how="inner")
            df["site_id"] = sid
            df = df.replace([-999, -999.0, -9999, -9999.0], np.nan)
            df = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)]
            df = df.sort_values(DATE_COL).reset_index(drop=True)
            for c in DYNAMIC_COLS + [TARGET_COL]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if len(df):
                basin_data[sid] = df
        except Exception as exc:
            skipped.append({"site_id": sid, "reason": str(exc)})

    missing = sorted(set(basin_ids) - set(basin_data.keys()))
    if missing:
        msg = (
            f"Could not load {len(missing)} basins from the exact comparison list. "
            "For a strict same-basin comparison, fix these files rather than silently dropping them."
        )
        if skipped:
            msg += f" First failure: {skipped[0]}"
        raise RuntimeError(msg)

    return basin_data


# -----------------------------------------------------------------------------
# Train-only scaling for KG-SSL driver-response inputs
# -----------------------------------------------------------------------------
@dataclass
class ChannelScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def fit_kgssl_channel_scaler(basin_data: Dict[str, pd.DataFrame]) -> ChannelScaler:
    start = pd.Timestamp(TRAIN_START)
    end = pd.Timestamp(TRAIN_END)
    rows = []
    for df in basin_data.values():
        sub = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)]
        rows.append(sub[DYNAMIC_COLS + [TARGET_COL]].to_numpy(dtype=np.float32))
    X = np.vstack(rows)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ChannelScaler(mean=mean, std=std)


def contiguous_valid_window_starts(
    df: pd.DataFrame,
    cols: Sequence[str],
    start_date: str,
    end_date: str,
    window: int,
    stride: int = 1,
) -> np.ndarray:
    """Return starts for fully finite, daily-contiguous windows inside [start_date,end_date]."""
    dates = pd.to_datetime(df[DATE_COL]).reset_index(drop=True)
    X = df[list(cols)].to_numpy(dtype=np.float32)
    n = len(df)
    if n < window:
        return np.empty(0, dtype=np.int64)

    valid = np.isfinite(X).all(axis=1)
    cvalid = np.concatenate([[0], np.cumsum(valid.astype(np.int32))])
    ords = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=np.int64)
    period_start = pd.Timestamp(start_date)
    period_end = pd.Timestamp(end_date)

    starts = []
    for i in range(0, n - window + 1, stride):
        j = i + window
        if dates.iloc[i] < period_start or dates.iloc[j - 1] > period_end:
            continue
        if cvalid[j] - cvalid[i] != window:
            continue
        if ords[j - 1] - ords[i] != window - 1:
            continue
        starts.append(i)
    return np.asarray(starts, dtype=np.int64)


# -----------------------------------------------------------------------------
# KG-SSL datasets
# -----------------------------------------------------------------------------
class KGSSLPairDataset(Dataset):
    """
    One dataset item = one basin.
    Each access samples two independent 365-day windows from that same basin.
    A batch therefore contains N basins -> 2N sequences, matching the KG-SSL paper.
    """

    def __init__(
        self,
        basin_data: Dict[str, pd.DataFrame],
        scaler: ChannelScaler,
        window: int,
        pairs_per_basin_per_epoch: int = 1,
    ):
        self.window = int(window)
        self.scaler = scaler
        self.items = []
        self.pairs_per_basin_per_epoch = int(pairs_per_basin_per_epoch)

        for sid, df in basin_data.items():
            starts = contiguous_valid_window_starts(
                df,
                DYNAMIC_COLS + [TARGET_COL],
                TRAIN_START,
                TRAIN_END,
                window=self.window,
                stride=1,
            )
            if len(starts) < 2:
                continue
            raw = df[DYNAMIC_COLS + [TARGET_COL]].to_numpy(dtype=np.float32)
            scaled = scaler.transform(raw)
            self.items.append((sid, scaled, starts))

        if not self.items:
            raise RuntimeError("No basins have at least two valid KG-SSL training windows.")

    @property
    def basin_ids(self) -> List[str]:
        return [x[0] for x in self.items]

    def __len__(self):
        return len(self.items) * self.pairs_per_basin_per_epoch

    def __getitem__(self, idx):
        basin_idx = idx % len(self.items)
        sid, arr, starts = self.items[basin_idx]
        n = len(starts)
        a = int(torch.randint(0, n, (1,)).item())
        b = int(torch.randint(0, n - 1, (1,)).item())
        if b >= a:
            b += 1
        s1 = int(starts[a])
        s2 = int(starts[b])
        x1 = arr[s1 : s1 + self.window]
        x2 = arr[s2 : s2 + self.window]
        return torch.from_numpy(x1).float(), torch.from_numpy(x2).float(), sid


class KGSSLEmbeddingWindowDataset(Dataset):
    """All valid stride-183 training-period windows used to construct final basin embeddings."""

    def __init__(
        self,
        basin_data: Dict[str, pd.DataFrame],
        scaler: ChannelScaler,
        window: int,
        stride: int,
    ):
        self.window = int(window)
        self.samples = []
        self.arrays = {}

        for sid, df in basin_data.items():
            starts = contiguous_valid_window_starts(
                df,
                DYNAMIC_COLS + [TARGET_COL],
                TRAIN_START,
                TRAIN_END,
                window=self.window,
                stride=int(stride),
            )
            if len(starts) == 0:
                continue
            arr = scaler.transform(
                df[DYNAMIC_COLS + [TARGET_COL]].to_numpy(dtype=np.float32)
            )
            self.arrays[sid] = arr
            for s in starts.tolist():
                self.samples.append((sid, int(s)))

        if not self.samples:
            raise RuntimeError("No valid windows available for KG-SSL embedding generation.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, start = self.samples[idx]
        x = self.arrays[sid][start : start + self.window]
        return torch.from_numpy(x).float(), sid


# -----------------------------------------------------------------------------
# KG-SSL model: BiLSTM encoder + autoregressive reconstruction decoder
# -----------------------------------------------------------------------------
class KGSSLSequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        # h_n: [2, B, H] for one-layer bidirectional LSTM.
        # KG-SSL paper adds forward and backward final hidden states.
        return h_n[0] + h_n[1]


class KGSSLAutoregressiveDecoder(nn.Module):
    """
    LSTM sequence generator conditioned on the embedding as its initial hidden state.
    Its previous prediction is fed as the next input, matching the paper description.
    """

    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding: torch.Tensor, seq_len: int) -> torch.Tensor:
        B = embedding.shape[0]
        h = embedding.unsqueeze(0)
        c = torch.zeros_like(h)
        prev = torch.zeros(B, 1, self.output_dim, device=embedding.device, dtype=embedding.dtype)
        outputs = []
        for _ in range(int(seq_len)):
            z, (h, c) = self.lstm(prev, (h, c))
            prev = self.output_head(z)
            outputs.append(prev)
        return torch.cat(outputs, dim=1)


class KGSSL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = KGSSLSequenceEncoder(input_dim, hidden_dim)
        self.decoder = KGSSLAutoregressiveDecoder(input_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        recon = self.decoder(h, x.shape[1])
        return h, recon


def kgssl_info_nce(h1: torch.Tensor, h2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE/NT-Xent: same-basin pair positive, all other basin windows negative.

    IMPORTANT: compute the similarity matrix and cross-entropy in FP32 even when the
    encoder is trained with AMP.  In FP16, masking the diagonal with a large finite
    negative value such as -1e9 overflows (the minimum finite FP16 value is -65504).
    FP32 here is also numerically safer for the softmax/log-sum-exp calculation.
    """
    if h1.shape != h2.shape:
        raise ValueError(f"h1/h2 shape mismatch: {h1.shape} vs {h2.shape}")
    N = h1.shape[0]
    if N < 2:
        raise ValueError("Contrastive batch must contain at least two basins.")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Disable autocast locally: h1/h2 may be FP16 because this function is called
    # from inside the AMP training context, but InfoNCE should be evaluated in FP32.
    with torch.autocast(device_type=h1.device.type, enabled=False):
        z = torch.cat([h1, h2], dim=0).float()
        z = F.normalize(z, dim=1)
        logits = (z @ z.T) / float(temperature)

        self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(self_mask, float("-inf"))

        targets = torch.arange(2 * N, device=z.device)
        targets = torch.where(targets < N, targets + N, targets - N)
        return F.cross_entropy(logits, targets)


@dataclass
class KGSSLTrainConfig:
    epochs: int
    batch_basins: int
    lr: float
    weight_decay: float
    window: int
    hidden: int
    temperature: float
    lambda_rec: float
    lambda_cont: float
    pairs_per_basin_per_epoch: int
    seed: int


def train_kgssl(
    model: KGSSL,
    dataset: KGSSLPairDataset,
    cfg: KGSSLTrainConfig,
    device: torch.device,
    num_workers: int,
    out_dir: Path,
    use_amp: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "kgssl_unsupervised_best.pt"
    history_path = out_dir / "kgssl_pretrain_history.csv"

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = make_loader(
        dataset,
        batch_size=cfg.batch_basins,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,  # keeps contrastive batch size stable
        generator=generator,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_loss = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = rec_total = cont_total = 0.0
        n_batches = 0
        t0 = time.time()

        iterator = tqdm(loader, desc=f"KGSSL epoch {epoch:03d}/{cfg.epochs}", leave=False)
        for x1, x2, _ in iterator:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                h1, r1 = model(x1)
                h2, r2 = model(x2)
                rec_loss = 0.5 * (F.mse_loss(r1, x1) + F.mse_loss(r2, x2))
                cont_loss = kgssl_info_nce(h1, h2, cfg.temperature)
                loss = cfg.lambda_rec * rec_loss + cfg.lambda_cont * cont_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            rec_total += float(rec_loss.item())
            cont_total += float(cont_loss.item())
            n_batches += 1
            iterator.set_postfix(
                total=f"{total/max(n_batches,1):.4f}",
                rec=f"{rec_total/max(n_batches,1):.4f}",
                cont=f"{cont_total/max(n_batches,1):.4f}",
            )

        avg = total / max(n_batches, 1)
        row = {
            "epoch": epoch,
            "loss": avg,
            "rec_loss": rec_total / max(n_batches, 1),
            "cont_loss": cont_total / max(n_batches, 1),
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "input_cols": DYNAMIC_COLS + [TARGET_COL],
                    "train_start": TRAIN_START,
                    "train_end": TRAIN_END,
                    "best_train_loss": best_loss,
                },
                ckpt_path,
            )

        print(
            f"KGSSL epoch={epoch:03d} total={row['loss']:.6f} "
            f"rec={row['rec_loss']:.6f} cont={row['cont_loss']:.6f} "
            f"best={best_loss:.6f}"
        )

    return ckpt_path


@torch.no_grad()
def generate_kgssl_embeddings(
    model: KGSSL,
    dataset: KGSSLEmbeddingWindowDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    out_path: Path,
) -> Dict[str, np.ndarray]:
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval().to(device)

    by_basin: Dict[str, List[np.ndarray]] = {}
    for x, site_ids in tqdm(loader, desc="generate KGSSL embeddings"):
        x = x.to(device, non_blocking=True)
        h = model.encode(x).detach().cpu().numpy().astype(np.float32)
        for sid, z in zip(site_ids, h):
            by_basin.setdefault(str(sid), []).append(z)

    static = {}
    counts = []
    for sid, rows in by_basin.items():
        Z = np.stack(rows, axis=0)
        z = Z.mean(axis=0).astype(np.float32)  # Eq. 7: element-wise mean over sequence embeddings
        static[sid] = z
        counts.append({"site_id": sid, "n_windows": int(len(rows))})

    site_ids = np.asarray(sorted(static.keys()))
    embeddings = np.stack([static[sid] for sid in site_ids], axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        site_ids=site_ids,
        source=np.asarray("KGSSL_unsupervised_Lrec_plus_Lcont"),
        embedding_start=np.asarray(TRAIN_START),
        embedding_end=np.asarray(TRAIN_END),
        window=np.asarray(KGSSL_WINDOW),
        stride=np.asarray(KGSSL_STRIDE),
    )
    pd.DataFrame(counts).to_csv(out_path.with_name("kgssl_embedding_window_counts.csv"), index=False)
    return static


def load_kgssl_embeddings(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    ids = [normalize_site_id(x) for x in data["site_ids"]]
    return {sid: emb[i] for i, sid in enumerate(ids) if sid is not None}


# -----------------------------------------------------------------------------
# Downstream benchmark: same LSTM/static interface as supplied Basin2Vec code
# -----------------------------------------------------------------------------
def fit_downstream_scaler(
    basin_data: Dict[str, pd.DataFrame], basin_ids: Sequence[str]
) -> Dict[str, np.ndarray]:
    xs, ys = [], []
    start, end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    basin_ids = set(basin_ids)
    for sid, df in basin_data.items():
        if sid not in basin_ids:
            continue
        sub = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)]
        xs.append(sub[DYNAMIC_COLS].to_numpy(dtype=np.float32))
        ys.append(sub[[TARGET_COL]].to_numpy(dtype=np.float32))
    X, Y = np.vstack(xs), np.vstack(ys)
    scaler = {
        "x_mean": np.nanmean(X, axis=0).astype(np.float32),
        "x_std": np.nanstd(X, axis=0).astype(np.float32),
        "y_mean": np.nanmean(Y, axis=0).astype(np.float32),
        "y_std": np.nanstd(Y, axis=0).astype(np.float32),
    }
    scaler["x_std"] = np.where(scaler["x_std"] < 1e-6, 1.0, scaler["x_std"])
    scaler["y_std"] = np.where(scaler["y_std"] < 1e-6, 1.0, scaler["y_std"])
    return scaler


def apply_downstream_scaler(
    basin_data: Dict[str, pd.DataFrame], scaler: Dict[str, np.ndarray]
) -> Dict[str, pd.DataFrame]:
    out = {}
    for sid, df in basin_data.items():
        d = df.copy()
        d[DYNAMIC_COLS] = (
            d[DYNAMIC_COLS].to_numpy(dtype=np.float32) - scaler["x_mean"]
        ) / scaler["x_std"]
        d[TARGET_COL] = (
            d[[TARGET_COL]].to_numpy(dtype=np.float32) - scaler["y_mean"]
        ) / scaler["y_std"]
        out[sid] = d
    return out


def inverse_target(arr: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr * np.asarray(scaler["y_std"], dtype=np.float32).reshape(1, 1) + np.asarray(
        scaler["y_mean"], dtype=np.float32
    ).reshape(1, 1)


def standardize_static_embeddings(
    static: Dict[str, np.ndarray], basin_ids: Sequence[str]
) -> Dict[str, np.ndarray]:
    ids = [sid for sid in basin_ids if sid in static]
    M = np.stack([static[sid] for sid in ids], axis=0).astype(np.float32)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {sid: ((static[sid][None, :] - mean) / std)[0].astype(np.float32) for sid in ids}


class StreamflowWindowDataset(Dataset):
    def __init__(
        self,
        basin_data: Dict[str, pd.DataFrame],
        static_dict: Dict[str, np.ndarray],
        seq_len: int,
        start_date: str,
        end_date: str,
        basin_ids: Sequence[str],
    ):
        self.basin_data = {}
        self.static_dict = static_dict
        self.seq_len = int(seq_len)
        self.samples = []

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        for sid in basin_ids:
            if sid not in basin_data or sid not in static_dict:
                continue
            df = basin_data[sid]
            X = df[DYNAMIC_COLS].to_numpy(dtype=np.float32)
            y = df[TARGET_COL].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(df[DATE_COL]).reset_index(drop=True)
            ord_days = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=np.int64)

            valid_x = np.isfinite(X).all(axis=1)
            valid_y = np.isfinite(y)
            cx = np.concatenate([[0], np.cumsum(valid_x.astype(np.int32))])
            cy = np.concatenate([[0], np.cumsum(valid_y.astype(np.int32))])

            self.basin_data[sid] = dict(X=X, y=y, dates=dates, ord_days=ord_days, cx=cx, cy=cy)
            n = len(df)
            first_issue_t = self.seq_len - 1
            # H=1, lead=0, exactly as supplied benchmark
            for issue_t in range(first_issue_t, n):
                target_t = issue_t
                target_date = pd.Timestamp(dates.iloc[target_t])
                if target_date < start_date or target_date > end_date:
                    continue
                x0, x1 = issue_t - self.seq_len + 1, issue_t + 1
                if cx[x1] - cx[x0] != self.seq_len:
                    continue
                if cy[target_t + 1] - cy[target_t] != 1:
                    continue
                if ord_days[x1 - 1] - ord_days[x0] != self.seq_len - 1:
                    continue
                self.samples.append((sid, issue_t, target_date.strftime("%Y-%m-%d")))

        if not self.samples:
            raise RuntimeError(f"No downstream samples for {start_date}..{end_date}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, issue_t, target_date = self.samples[idx]
        obj = self.basin_data[sid]
        x0, x1 = issue_t - self.seq_len + 1, issue_t + 1
        x = obj["X"][x0:x1].astype(np.float32)
        y = np.asarray([obj["y"][issue_t]], dtype=np.float32)
        s = self.static_dict[sid].astype(np.float32)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(s).float(),
            torch.from_numpy(y).float(),
            sid,
            target_date,
        )


class StaticProjector(nn.Module):
    def __init__(self, static_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_static):
        return self.net(torch.nan_to_num(x_static.float(), nan=0.0, posinf=0.0, neginf=0.0))


class LSTMStaticForecastModel(nn.Module):
    """Exact LSTM/static-conditioning architecture from the supplied temporal script."""

    def __init__(
        self,
        dynamic_size: int,
        static_size: int,
        output_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.static_proj = StaticProjector(static_size, hidden_size, dropout)
        self.lstm = nn.LSTM(
            dynamic_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        s = self.static_proj(x_static).unsqueeze(1).expand(-1, x_dyn.shape[1], -1)
        z, _ = self.lstm(torch.cat([x_dyn, s], dim=-1))
        return self.head(z[:, -1])


# Metrics copied from the existing temporal benchmark
def nse(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom < 1e-12:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def rmse(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2)))


def mae(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(obs[valid] - pred[valid])))


def pearson_r(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2 or np.std(obs) < 1e-12 or np.std(pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(obs, pred)[0, 1])


def kge(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return np.nan
    r = pearson_r(obs, pred)
    obs_std, pred_std = np.std(obs), np.std(pred)
    obs_mean, pred_mean = np.mean(obs), np.mean(pred)
    if obs_std < 1e-12 or abs(obs_mean) < 1e-12:
        return np.nan
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def compute_per_basin_metrics(obs, pred, site_ids) -> pd.DataFrame:
    obs = np.asarray(obs).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    site_ids = np.asarray(site_ids).astype(str)
    rows = []
    for sid in sorted(np.unique(site_ids)):
        idx = site_ids == sid
        o, p = obs[idx], pred[idx]
        rows.append(
            {
                "site_id": sid,
                "NSE": nse(o, p),
                "KGE": kge(o, p),
                "RMSE": rmse(o, p),
                "MAE": mae(o, p),
                "Pearson_r": pearson_r(o, p),
                "n_samples": int(idx.sum()),
            }
        )
    return pd.DataFrame(rows)


def summary_from_per_basin(df: pd.DataFrame) -> Dict[str, float]:
    def med(c):
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else np.nan

    def avg(c):
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else np.nan

    return {
        "median_NSE": med("NSE"),
        "mean_NSE": avg("NSE"),
        "median_KGE": med("KGE"),
        "mean_KGE": avg("KGE"),
        "median_RMSE": med("RMSE"),
        "mean_RMSE": avg("RMSE"),
        "median_MAE": med("MAE"),
        "mean_MAE": avg("MAE"),
        "median_Pearson_r": med("Pearson_r"),
        "mean_Pearson_r": avg("Pearson_r"),
        "n_basins": int(df["site_id"].nunique()),
        "n_basins_valid_NSE": int(np.isfinite(df["NSE"]).sum()),
        "n_basins_valid_KGE": int(np.isfinite(df["KGE"]).sum()),
    }


def train_one_downstream_seed(
    seed: int,
    basin_data_scaled: Dict[str, pd.DataFrame],
    static: Dict[str, np.ndarray],
    basin_ids: Sequence[str],
    scaler: Dict[str, np.ndarray],
    device: torch.device,
    num_workers: int,
    batch_size: int,
    out_dir: Path,
    use_amp: bool,
) -> Dict[str, float]:
    set_seed(seed)
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StreamflowWindowDataset(
        basin_data_scaled,
        static,
        DOWNSTREAM_SEQ_LEN,
        TRAIN_START,
        TRAIN_END,
        basin_ids,
    )
    eval_ds = StreamflowWindowDataset(
        basin_data_scaled,
        static,
        DOWNSTREAM_SEQ_LEN,
        EVAL_START,
        EVAL_END,
        basin_ids,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    eval_loader = make_loader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    static_dim = len(next(iter(static.values())))
    model = LSTMStaticForecastModel(
        dynamic_size=len(DYNAMIC_COLS),
        static_size=static_dim,
        output_size=1,
        hidden_size=DOWNSTREAM_HIDDEN,
        num_layers=DOWNSTREAM_LAYERS,
        dropout=DOWNSTREAM_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DOWNSTREAM_LR, weight_decay=DOWNSTREAM_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val = math.inf
    best_epoch = -1
    bad_epochs = 0
    best_path = seed_dir / "best_model.pt"
    history = []

    for epoch in range(1, DOWNSTREAM_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        n_batches = 0
        for x, s, y, *_ in tqdm(train_loader, desc=f"downstream seed {seed} train", leave=False):
            x, s, y = x.to(device), s.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                yhat = model(x, s)
                loss = criterion(yhat, y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), DOWNSTREAM_GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_sum += float(loss.item())
            n_batches += 1

        model.eval()
        mse_sum, n_elem = 0.0, 0
        with torch.no_grad():
            for x, s, y, *_ in eval_loader:
                x, s, y = x.to(device), s.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    yhat = model(x, s)
                mse_sum += float(torch.sum((yhat - y) ** 2).item())
                n_elem += int(y.numel())
        val_loss = mse_sum / max(n_elem, 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_sum / max(n_batches, 1),
                "val_loss_future_years": val_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
        print(
            f"seed={seed} epoch={epoch:03d} train={history[-1]['train_loss']:.6f} "
            f"eval_loss={val_loss:.6f} best={best_val:.6f} best_epoch={best_epoch}"
        )
        if bad_epochs >= DOWNSTREAM_PATIENCE:
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true_scaled, y_pred_scaled, sites, dates = [], [], [], []
    with torch.no_grad():
        for x, s, y, site_ids, target_dates in tqdm(eval_loader, desc=f"evaluate seed {seed}"):
            x, s = x.to(device), s.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                yhat = model(x, s)
            y_true_scaled.append(y.numpy())
            y_pred_scaled.append(yhat.cpu().numpy())
            sites.extend(list(site_ids))
            dates.extend(list(target_dates))

    y_true_scaled = np.vstack(y_true_scaled)
    y_pred_scaled = np.vstack(y_pred_scaled)
    y_true = inverse_target(y_true_scaled, scaler)
    y_pred = inverse_target(y_pred_scaled, scaler)

    per_basin = compute_per_basin_metrics(y_true, y_pred, sites)
    per_basin.to_csv(seed_dir / "val_per_basin_metrics_horizon1.csv", index=False)
    summary = summary_from_per_basin(per_basin)
    summary.update(
        {
            "seed": int(seed),
            "static_source": "kgssl_unsupervised",
            "seq_len": DOWNSTREAM_SEQ_LEN,
            "horizon": 1,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "static_dim": int(static_dim),
            "n_train_samples": int(len(train_ds)),
            "n_eval_samples": int(len(eval_ds)),
        }
    )
    pd.DataFrame([summary]).to_csv(seed_dir / "val_per_basin_summary_horizon1.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "site_id": sites,
            "target_date": dates,
            "observed": y_true[:, 0],
            "predicted": y_pred[:, 0],
            "residual": y_pred[:, 0] - y_true[:, 0],
        }
    )
    pred_df.to_csv(seed_dir / "val_predictions_horizon1.csv", index=False)
    return summary


def aggregate_seed_summaries(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "median_NSE",
        "mean_NSE",
        "median_KGE",
        "mean_KGE",
        "median_RMSE",
        "mean_RMSE",
        "median_MAE",
        "mean_MAE",
        "median_Pearson_r",
        "mean_Pearson_r",
    ]
    out = {
        "static_source": "kgssl_unsupervised",
        "n_seeds": int(len(df)),
        "seeds": ",".join(str(int(x)) for x in df["seed"].tolist()),
    }
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        out[f"{c}_mean_over_seeds"] = float(np.nanmean(vals))
        out[f"{c}_std_over_seeds"] = float(np.nanstd(vals, ddof=0))
    return pd.DataFrame([out])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--camels-root", type=Path, default=DEFAULT_CAMELS_ROOT)
    p.add_argument("--common-basins-csv", type=Path, default=DEFAULT_COMMON_BASINS_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stage", choices=["all", "pretrain", "embed", "downstream"], default="all")

    p.add_argument("--kgssl-epochs", type=int, default=100)
    p.add_argument("--kgssl-batch-basins", type=int, default=32)
    p.add_argument("--kgssl-lr", type=float, default=1e-3)
    p.add_argument("--kgssl-weight-decay", type=float, default=1e-4)
    p.add_argument("--kgssl-pairs-per-basin-per-epoch", type=int, default=1)
    p.add_argument("--kgssl-seed", type=int, default=42)

    p.add_argument("--downstream-seeds", type=str, default="42,123,456")
    p.add_argument("--downstream-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force-pretrain", action="store_true")
    p.add_argument("--force-embed", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exact_basins = read_common_basin_ids(args.common_basins_csv)
    print(f"Exact Basin2Vec temporal comparison basins: {len(exact_basins)}")
    basin_data = load_camels_data(args.camels_root, exact_basins)

    # ------------------------------------------------------------------
    # KG-SSL train-only input normalization
    # ------------------------------------------------------------------
    kg_scaler = fit_kgssl_channel_scaler(basin_data)
    save_json(
        {
            "columns": DYNAMIC_COLS + [TARGET_COL],
            "mean": kg_scaler.mean.tolist(),
            "std": kg_scaler.std.tolist(),
            "fit_start": TRAIN_START,
            "fit_end": TRAIN_END,
        },
        args.out_dir / "kgssl_input_scaler.json",
    )

    kg_cfg = KGSSLTrainConfig(
        epochs=int(args.kgssl_epochs),
        batch_basins=int(args.kgssl_batch_basins),
        lr=float(args.kgssl_lr),
        weight_decay=float(args.kgssl_weight_decay),
        window=KGSSL_WINDOW,
        hidden=KGSSL_HIDDEN,
        temperature=KGSSL_TEMPERATURE,
        lambda_rec=KGSSL_LAMBDA_REC,
        lambda_cont=KGSSL_LAMBDA_CONT,
        pairs_per_basin_per_epoch=int(args.kgssl_pairs_per_basin_per_epoch),
        seed=int(args.kgssl_seed),
    )
    ckpt_path = args.out_dir / "kgssl_unsupervised_best.pt"
    embedding_path = args.out_dir / "kgssl_embeddings_train1990_2009.npz"

    # ------------------------------------------------------------------
    # Stage 1: unsupervised KG-SSL pretraining (1990-2009 only)
    # ------------------------------------------------------------------
    if args.stage in ["all", "pretrain"]:
        if ckpt_path.exists() and not args.force_pretrain:
            print(f"[skip] KGSSL checkpoint already exists: {ckpt_path}")
        else:
            set_seed(kg_cfg.seed)
            pair_ds = KGSSLPairDataset(
                basin_data,
                kg_scaler,
                window=KGSSL_WINDOW,
                pairs_per_basin_per_epoch=kg_cfg.pairs_per_basin_per_epoch,
            )
            missing_pair_basins = sorted(set(exact_basins) - set(pair_ds.basin_ids))
            if missing_pair_basins:
                raise RuntimeError(
                    f"{len(missing_pair_basins)} exact comparison basins lack two complete 365-day "
                    "training windows for KGSSL. Do not silently change the comparison population. "
                    f"Examples: {missing_pair_basins[:10]}"
                )
            model = KGSSL(input_dim=len(DYNAMIC_COLS) + 1, hidden_dim=KGSSL_HIDDEN)
            ckpt_path = train_kgssl(
                model,
                pair_ds,
                kg_cfg,
                device,
                args.num_workers,
                args.out_dir,
                use_amp,
            )

    if args.stage == "pretrain":
        print(f"Done. Checkpoint: {ckpt_path}")
        return

    # ------------------------------------------------------------------
    # Stage 2: fixed descriptor generation (still 1990-2009 only)
    # ------------------------------------------------------------------
    if args.stage in ["all", "embed"]:
        if embedding_path.exists() and not args.force_embed:
            print(f"[skip] KGSSL embeddings already exist: {embedding_path}")
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"KGSSL checkpoint not found: {ckpt_path}")
            model = KGSSL(input_dim=len(DYNAMIC_COLS) + 1, hidden_dim=KGSSL_HIDDEN)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            embed_ds = KGSSLEmbeddingWindowDataset(
                basin_data,
                kg_scaler,
                window=KGSSL_WINDOW,
                stride=KGSSL_STRIDE,
            )
            static = generate_kgssl_embeddings(
                model,
                embed_ds,
                device,
                batch_size=max(1, args.kgssl_batch_basins * 2),
                num_workers=args.num_workers,
                out_path=embedding_path,
            )
            missing_embed_basins = sorted(set(exact_basins) - set(static.keys()))
            if missing_embed_basins:
                raise RuntimeError(
                    f"KGSSL embedding generation missed {len(missing_embed_basins)} exact comparison basins: "
                    f"{missing_embed_basins[:10]}"
                )

    if args.stage == "embed":
        print(f"Done. Embeddings: {embedding_path}")
        return

    # ------------------------------------------------------------------
    # Stage 3: same downstream temporal benchmark
    # ------------------------------------------------------------------
    if args.stage in ["all", "downstream"]:
        if not embedding_path.exists():
            raise FileNotFoundError(f"KGSSL embeddings not found: {embedding_path}")
        static_raw = load_kgssl_embeddings(embedding_path)
        missing = sorted(set(exact_basins) - set(static_raw.keys()))
        if missing:
            raise RuntimeError(
                f"KGSSL descriptor missing for {len(missing)} exact comparison basins. Examples: {missing[:10]}"
            )
        static = standardize_static_embeddings(static_raw, exact_basins)

        downstream_scaler = fit_downstream_scaler(basin_data, exact_basins)
        save_json(
            {k: np.asarray(v).tolist() for k, v in downstream_scaler.items()},
            args.out_dir / "downstream_scaler.json",
        )
        basin_data_scaled = apply_downstream_scaler(basin_data, downstream_scaler)

        seed_rows = []
        for seed in parse_seed_list(args.downstream_seeds):
            seed_rows.append(
                train_one_downstream_seed(
                    seed,
                    basin_data_scaled,
                    static,
                    exact_basins,
                    downstream_scaler,
                    device,
                    args.num_workers,
                    args.downstream_batch_size,
                    args.out_dir / "downstream_lstm",
                    use_amp,
                )
            )

        per_seed = pd.DataFrame(seed_rows)
        per_seed.to_csv(args.out_dir / "kgssl_temporal_results_per_seed.csv", index=False)
        aggregate = aggregate_seed_summaries(seed_rows)
        aggregate.to_csv(args.out_dir / "kgssl_temporal_results_mean_std.csv", index=False)
        print("\nKGSSL temporal results per seed:")
        print(per_seed[["seed", "median_NSE", "median_KGE", "mean_NSE", "mean_KGE"]])
        print("\nMean/std over downstream seeds:")
        print(aggregate.T)

    save_json(
        {
            "method": "KGSSL unsupervised L_rec + L_cont",
            "kgssl_driver_response_columns": DYNAMIC_COLS + [TARGET_COL],
            "kgssl_embedding_information_period": [TRAIN_START, TRAIN_END],
            "kgssl_test_period_discharge_used_for_embedding": False,
            "downstream_model_past_discharge_input": False,
            "downstream_train_period": [TRAIN_START, TRAIN_END],
            "downstream_eval_period": [EVAL_START, EVAL_END],
            "exact_common_basin_csv": str(args.common_basins_csv),
            "n_exact_basins": len(exact_basins),
            "kgssl_config": asdict(kg_cfg),
            "paper_details_followed": {
                "window": 365,
                "stride_for_embedding_generation": 183,
                "encoder": "1-layer BiLSTM",
                "hidden_units": 64,
                "bidirectional_states": "forward + backward final hidden states",
                "loss": "L_rec + L_cont (no pseudo-inverse/static-attribute supervision)",
                "final_basin_embedding": "mean of sequence embeddings from training period only",
            },
        },
        args.out_dir / "KGSSL_PROTOCOL.json",
    )


if __name__ == "__main__":
    main()
