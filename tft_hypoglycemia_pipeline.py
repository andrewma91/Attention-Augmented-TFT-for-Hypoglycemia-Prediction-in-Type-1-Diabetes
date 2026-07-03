# cell 0 - install - !pip install torch scikit-learn pandas numpy matplotlib seaborn tqdm scipy -q


# cell 1 - imports
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE.type == 'cpu':
    print("WARNING: No GPU detected. Training will be slow. "
          "Enable GPU in Settings > Accelerator if on Kaggle/Colab.")

# cell 2 - paths - change to your path
DATA_DIR    = '/path/to/T1DiabetesGranada/'  # set to local dataset path
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# cell 3 - load all 4 files
print("Loading files...")

df_glucose = pd.read_csv(
    f'{DATA_DIR}/Glucose_measurements.csv',
    dtype={'Measurement': np.int16}
)
df_biochem  = pd.read_csv(f'{DATA_DIR}/Biochemical_parameters.csv')
df_patients = pd.read_csv(f'{DATA_DIR}/Patient_info.csv')
df_diag     = pd.read_csv(f'{DATA_DIR}/Diagnostics.csv')

df_glucose['Measurement_date'] = pd.to_datetime(
    df_glucose['Measurement_date'], infer_datetime_format=True)
df_biochem['Reception_date'] = pd.to_datetime(
    df_biochem['Reception_date'], infer_datetime_format=True)

print(f"Glucose measurements : {len(df_glucose):>10,} rows")
print(f"Biochemical params   : {len(df_biochem):>10,} rows")
print(f"Patients             : {len(df_patients):>10,} rows")
print(f"Diagnostics          : {len(df_diag):>10,} rows")


# cell 4 - config
BIOCHEM_FEATURES = [
    'Glycated hemoglobin (A1c)',
    'Total cholesterol',
    'HDL cholesterol',
    'Triglycerides',
    'Creatinine',
]

CONFIG = {
    'hypo_threshold'  : 70.0,
    'cgm_interval_min': 5,
    'lookback_min'    : 60,
    'horizon_min'     : 30,

    'test_frac' : 0.20,
    'val_frac'  : 0.10,

    'max_windows'         : 1_000_000, 

    'batch_size'          : 256,
    'lr'                  : 1e-3,
    'epochs'              : 60,
    'early_stop_patience' : 10,
    'weight_decay'        : 1e-4,

    'hidden_size'   : 64,
    'lstm_layers'   : 2,
    'attn_heads'    : 4,
    'dropout'       : 0.2,
    'tcn_kernel'    : 3,
    'tcn_dilations' : [1, 2, 4, 8],
}

CONFIG['lookback_steps'] = CONFIG['lookback_min'] // CONFIG['cgm_interval_min']
CONFIG['horizon_steps']  = CONFIG['horizon_min']  // CONFIG['cgm_interval_min']

print(f"Lookback : {CONFIG['lookback_steps']} steps ({CONFIG['lookback_min']} min)")
print(f"Horizon  : {CONFIG['horizon_steps']} steps ({CONFIG['horizon_min']} min)")


# cell 5 - patient static features

def build_static(df_patients, df_biochem, df_diag):
    df = df_patients[['Patient_ID', 'Sex', 'Birth_year']].copy()
    df['Age']     = 2018 - df['Birth_year'].astype(int)
    df['Sex_enc'] = (df['Sex'] == 'M').astype(float)

    hba1c = (df_biochem[df_biochem['Name'] == 'Glycated hemoglobin (A1c)']
             .sort_values('Reception_date')
             .groupby('Patient_ID')['Value'].last()
             .rename('HbA1c_latest'))
    df = df.merge(hba1c, on='Patient_ID', how='left')

    n_diag = df_diag.groupby('Patient_ID')['Code'].nunique().rename('N_diagnoses')
    df = df.merge(n_diag, on='Patient_ID', how='left')
    df['N_diagnoses'] = df['N_diagnoses'].fillna(0)

    cols = ['Age', 'Sex_enc', 'HbA1c_latest', 'N_diagnoses']
    df[cols] = df[cols].fillna(df[cols].mean())
    return df.set_index('Patient_ID')[cols]


df_static = build_static(df_patients, df_biochem, df_diag)
print("Static features per patient:")
print(df_static.describe().round(2))


# cell 6 - biochem forward fill table.

def build_biochem_ffill(df_biochem, features):
    df_b = df_biochem[df_biochem['Name'].isin(features)].copy()
    df_b['Date'] = df_b['Reception_date'].dt.date

    pivoted = (df_b.groupby(['Patient_ID', 'Date', 'Name'])['Value']
               .mean()
               .unstack('Name')
               .reset_index())
    pivoted['Date'] = pd.to_datetime(pivoted['Date'])
    pivoted = pivoted.sort_values(['Patient_ID', 'Date'])

    biochem_dict = {}
    for pid, grp in pivoted.groupby('Patient_ID'):
        grp = grp.set_index('Date').drop(columns=['Patient_ID'])
        grp = grp.reindex(
            pd.date_range(grp.index.min(), grp.index.max(), freq='D')
        ).ffill().fillna(method='bfill').fillna(0)
        biochem_dict[pid] = grp

    return biochem_dict, [c for c in pivoted.columns if c not in ('Patient_ID', 'Date')]


biochem_dict, biochem_cols = build_biochem_ffill(df_biochem, BIOCHEM_FEATURES)
print(f"Biochemical feature columns: {biochem_cols}")


# cell 7 - memory safe sliding window extraction
import gc

def build_windows(df_glucose, df_static, biochem_dict, biochem_cols, cfg):
    L      = cfg['lookback_steps']
    H      = cfg['horizon_steps']
    THRESH = cfg['hypo_threshold']
    n_biochem = len(biochem_cols)

    n_patients = df_glucose['Patient_ID'].nunique()
    cap_per_patient = max(500, cfg['max_windows'] // n_patients)
    print(f"Capping at {cap_per_patient:,} windows/patient "
          f"({n_patients} patients -> target ~{cap_per_patient*n_patients:,} total)")

    all_X, all_S, all_y, all_pid = [], [], [], []

    for pid, grp in tqdm(df_glucose.groupby('Patient_ID'), desc='Building windows'):
        grp = grp.sort_values('Measurement_date').reset_index(drop=True)
        g   = grp['Measurement'].values.astype(np.float32)
        n   = len(g)
        if n < L + H + 1:
            continue

        g = pd.Series(g).interpolate('linear', limit=3).values.astype(np.float32)

        roc   = np.gradient(g).astype(np.float32)
        accel = np.gradient(roc).astype(np.float32)
        gs    = pd.Series(g)
        rmean = gs.rolling(6, min_periods=1).mean().values.astype(np.float32)
        rstd  = gs.rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)

        hours = (grp['Measurement_date'].dt.hour +
                 grp['Measurement_date'].dt.minute / 60.0).values.astype(np.float32)
        sin_t = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        cos_t = np.cos(2 * np.pi * hours / 24).astype(np.float32)

        biochem_vals = np.zeros((n, n_biochem), dtype=np.float32)
        if pid in biochem_dict:
            bdf = biochem_dict[pid].reset_index()
            bdf.columns = ['Date'] + list(bdf.columns[1:])
            grp_dates = grp['Measurement_date'].dt.normalize().rename('Date')
            merged = pd.merge_asof(
                grp_dates.to_frame(), bdf, on='Date', direction='backward')
            cols_present = [c for c in biochem_cols if c in merged.columns]
            if cols_present:
                biochem_vals[:, :len(cols_present)] = (
                    merged[cols_present].fillna(0).values.astype(np.float32))

        dyn = np.column_stack(
            [g, roc, accel, rmean, rstd, sin_t, cos_t, biochem_vals])

        static_vec = (df_static.loc[pid].values.astype(np.float32)
                      if pid in df_static.index
                      else np.zeros(df_static.shape[1], dtype=np.float32))

        n_wins = n - L - H + 1
        if n_wins <= 0:
            continue

        row_idx = np.arange(L)[None, :] + np.arange(n_wins)[:, None]
        X_wins  = dyn[row_idx]

        fut_idx  = np.arange(H)[None, :] + (np.arange(n_wins) + L)[:, None]
        future_g = g[fut_idx]
        labels   = (np.nanmin(future_g, axis=1) < THRESH).astype(np.int64)

        nan_mask = np.isnan(X_wins[:, :, 0]).sum(axis=1) <= (L * 0.2)
        X_wins   = X_wins[nan_mask]
        labels   = labels[nan_mask]

        if len(labels) == 0:
            continue

        if len(labels) > cap_per_patient:
            keep = np.random.choice(len(labels), cap_per_patient, replace=False)
            X_wins = X_wins[keep]
            labels = labels[keep]

        all_X.append(X_wins)
        all_S.append(np.tile(static_vec, (len(labels), 1)))
        all_y.append(labels)
        all_pid.extend([pid] * len(labels))

        del dyn, X_wins, labels, biochem_vals

    X    = np.concatenate(all_X,  axis=0).astype(np.float32)
    S    = np.concatenate(all_S,  axis=0).astype(np.float32)
    y    = np.concatenate(all_y,  axis=0).astype(np.int64)
    pids = np.array(all_pid)

    del all_X, all_S, all_y, all_pid
    gc.collect()

    print(f"\nTotal windows : {len(y):,}")
    print(f"Hypoglycemic  : {y.sum():,}  ({100*y.mean():.2f}%)")
    print(f"Dynamic feats : {X.shape[2]}   Static feats: {S.shape[1]}")
    return X, S, y, pids


X_all, S_all, y_all, pids_all = build_windows(
    df_glucose, df_static, biochem_dict, biochem_cols, CONFIG)

del df_glucose
gc.collect()

print(f"Final window count: {len(y_all):,}  ({100*y_all.mean():.2f}% hypo)")



# cell 8 - train/val/test split

def patient_split(X, S, y, pids, cfg):
    unique = np.unique(pids)
    np.random.shuffle(unique)
    n      = len(unique)
    n_test = max(1, int(n * cfg['test_frac']))
    n_val  = max(1, int(n * cfg['val_frac']))

    test_set  = set(unique[:n_test])
    val_set   = set(unique[n_test:n_test + n_val])
    train_set = set(unique[n_test + n_val:])

    def mask(s):
        m = np.array([p in s for p in pids])
        return X[m], S[m], y[m]

    tr, va, te = mask(train_set), mask(val_set), mask(test_set)
    print(f"Train : {len(tr[2]):>8,} windows  ({len(train_set)} patients)")
    print(f"Val   : {len(va[2]):>8,} windows  ({len(val_set)} patients)")
    print(f"Test  : {len(te[2]):>8,} windows  ({len(test_set)} patients)")
    return tr, va, te


train_data, val_data, test_data = patient_split(X_all, S_all, y_all, pids_all, CONFIG)
X_tr, S_tr, y_tr = train_data
X_va, S_va, y_va = val_data
X_te, S_te, y_te = test_data

n_dynamic = X_tr.shape[2]
n_static  = S_tr.shape[1]

dyn_sc, stat_sc = StandardScaler(), StandardScaler()
X_tr = dyn_sc.fit_transform(X_tr.reshape(-1, n_dynamic)).reshape(X_tr.shape)
X_va = dyn_sc.transform(X_va.reshape(-1, n_dynamic)).reshape(X_va.shape)
X_te = dyn_sc.transform(X_te.reshape(-1, n_dynamic)).reshape(X_te.shape)
S_tr = stat_sc.fit_transform(S_tr)
S_va = stat_sc.transform(S_va)
S_te = stat_sc.transform(S_te)

train_data = (X_tr, S_tr, y_tr)
val_data   = (X_va, S_va, y_va)
test_data  = (X_te, S_te, y_te)

pos_weight = torch.tensor(
    [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
    dtype=torch.float32).to(DEVICE)
print(f"\nPos weight (imbalance): {pos_weight.item():.2f}")


# cell 9 - pytorch dataset

class HypoDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]


def make_loaders(tr, va, te, cfg):
    return (
        DataLoader(HypoDataset(*tr), cfg['batch_size'], shuffle=True,  num_workers=0),
        DataLoader(HypoDataset(*va), cfg['batch_size'], shuffle=False, num_workers=0),
        DataLoader(HypoDataset(*te), cfg['batch_size'], shuffle=False, num_workers=0),
    )

train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, CONFIG)


# cell 10 - model parts

class GatedResidualNetwork(nn.Module):
    def __init__(self, in_sz, hid_sz, out_sz, dropout=0.1, ctx_sz=None):
        super().__init__()
        self.fc1  = nn.Linear(in_sz, hid_sz)
        self.fc2  = nn.Linear(hid_sz, out_sz * 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_sz)
        self.skip = nn.Linear(in_sz, out_sz, bias=False) if in_sz != out_sz else nn.Identity()
        self.ctx  = nn.Linear(ctx_sz, hid_sz, bias=False) if ctx_sz else None

    def forward(self, x, context=None):
        r = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.ctx is not None:
            h = h + self.ctx(context)
        h = F.elu(h)
        h = self.drop(h)
        v, g = self.fc2(h).chunk(2, dim=-1)
        return self.norm(r + v * torch.sigmoid(g))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_vars, var_dim, hidden, dropout, ctx_size=None):
        super().__init__()
        self.n_vars  = n_vars
        self.grns    = nn.ModuleList([
            GatedResidualNetwork(var_dim, hidden, hidden, dropout, ctx_size)
            for _ in range(n_vars)
        ])
        self.selector = GatedResidualNetwork(
            n_vars * var_dim, hidden, n_vars, dropout, ctx_size)

    def forward(self, x, context=None):
        B, T, V, D = x.shape
        proc = torch.stack(
            [self.grns[i](x[:, :, i, :], context) for i in range(V)], dim=2)
        flat    = x.reshape(B, T, -1)
        weights = torch.softmax(self.selector(flat, context), dim=-1)
        out     = (proc * weights.unsqueeze(-1)).sum(dim=2)
        return out, weights


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        pad       = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.trim = pad
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU()

    def forward(self, x):
        h = self.conv(x.transpose(1,2))
        if self.trim: h = h[:, :, :-self.trim]
        h = self.act(h).transpose(1,2)
        return self.norm(x + self.drop(h))


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_heads, dropout):
        super().__init__()
        assert hidden % n_heads == 0
        self.d_k   = hidden // n_heads
        self.heads = n_heads
        self.W_q   = nn.Linear(hidden, hidden)
        self.W_k   = nn.Linear(hidden, hidden)
        self.W_v   = nn.Linear(hidden, self.d_k)
        self.W_o   = nn.Linear(hidden, hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        B, T, H = x.shape
        Q = self.W_q(x).reshape(B,T,self.heads,self.d_k).transpose(1,2)
        K = self.W_k(x).reshape(B,T,self.heads,self.d_k).transpose(1,2)
        V = self.W_v(x).unsqueeze(1).expand(-1,self.heads,-1,-1)
        sc   = (Q @ K.transpose(-2,-1)) / (self.d_k**0.5)
        attn = self.drop(torch.softmax(sc, dim=-1))
        out  = (attn @ V).transpose(1,2).reshape(B,T,H)
        return self.W_o(out), attn.mean(dim=1)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_dyn, n_stat, cfg):
        super().__init__()
        H, D, nh = cfg['hidden_size'], cfg['dropout'], cfg['attn_heads']

        self.stat_emb = nn.Linear(n_stat, H)
        self.stat_grn = GatedResidualNetwork(H, H, H, D)

        self.dyn_proj = nn.Linear(1, H)
        self.vsn      = VariableSelectionNetwork(n_dyn, H, H, D, ctx_size=H)

        self.tcn = nn.Sequential(*[
            TCNBlock(H, cfg['tcn_kernel'], d, D) for d in cfg['tcn_dilations']])

        self.lstm = nn.LSTM(H, H, cfg['lstm_layers'], batch_first=True,
                            dropout=D if cfg['lstm_layers']>1 else 0)

        self.attn      = InterpretableMultiHeadAttention(H, nh, D)
        self.attn_grn  = GatedResidualNetwork(H, H, H, D)
        self.attn_norm = nn.LayerNorm(H)

        self.out_grn = GatedResidualNetwork(H, H, H, D)
        self.clf     = nn.Linear(H, 1)

    def forward(self, x_dyn, x_stat):
        B, T, _ = x_dyn.shape
        ctx = self.stat_grn(self.stat_emb(x_stat))
        x_p = self.dyn_proj(x_dyn.unsqueeze(-1))
        ctx_t = ctx.unsqueeze(1).expand(-1, T, -1)
        x_v, vsn_w = self.vsn(x_p, context=ctx_t)
        x_t = self.tcn(x_v)
        h0 = ctx.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        x_l, _ = self.lstm(x_t, (h0, torch.zeros_like(h0)))
        a_out, attn_w = self.attn(x_l)
        x_a = self.attn_norm(x_l + self.attn_grn(a_out))
        x_pool = x_a[:, -CONFIG['horizon_steps']:, :].mean(1)
        logit = self.clf(self.out_grn(x_pool)).squeeze(-1)
        return logit, attn_w, vsn_w


# cell 11 - baseline model

class LSTMBaseline(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers>1 else 0)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x, s=None):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1), None, None


class VanillaTransformer(nn.Module):
    def __init__(self, hidden=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(1, hidden)
        enc = nn.TransformerEncoderLayer(hidden, n_heads,
              dim_feedforward=hidden*2, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layers)
        self.fc  = nn.Linear(hidden, 1)
    def forward(self, x, s=None):
        h = self.enc(self.proj(x))
        return self.fc(h[:,-1,:]).squeeze(-1), None, None


def threshold_baseline(X_te_raw):
    g   = X_te_raw[:, -1, 0]
    roc = X_te_raw[:, -1, 1]
    return ((g < -1.0) & (roc < -0.5)).astype(float)


# cell 12 - training utilities

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience   = patience
        self.best       = None
        self.best_state = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_auc, model):
        if self.best is None or val_auc > self.best + 1e-4:
            self.best       = val_auc
            self.best_state = deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


@torch.no_grad()
def evaluate(model, loader, crit, device, glucose_only=False):
    model.eval()
    logits_all, y_all = [], []
    loss_sum = 0
    for X, S, y in loader:
        X, S, y = X.to(device), S.to(device), y.to(device)
        if glucose_only:
            X = X[:, :, :1]
        logit, _, _ = model(X, S)
        loss_sum += crit(logit, y).item() * len(y)
        logits_all.append(logit.cpu())
        y_all.append(y.cpu())
    logits = torch.cat(logits_all)
    ys     = torch.cat(y_all).numpy().astype(int)
    probs  = torch.sigmoid(logits).numpy()
    a      = roc_auc_score(ys, probs) if ys.sum() > 0 else 0.0
    return loss_sum / len(loader.dataset), a, probs, ys


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    return {
        'AUC'        : round(roc_auc_score(y_true, y_prob) if y_true.sum()>0 else 0, 4),
        'Accuracy'   : round(accuracy_score(y_true, y_pred), 4),
        'Sensitivity': round(tp / (tp+fn+1e-8), 4),
        'Specificity': round(tn / (tn+fp+1e-8), 4),
        'FAR'        : round(fp / (fp+tn+1e-8), 4),
        'TP':int(tp), 'FP':int(fp), 'TN':int(tn), 'FN':int(fn),
    }


def train_model(model, tr_loader, va_loader, cfg, device, pos_weight,
                glucose_only=False):
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    es    = EarlyStopping(cfg['early_stop_patience'])
    hist  = {'tr_loss':[], 'va_loss':[], 'va_auc':[]}

    def _run(loader, training):
        model.train() if training else model.eval()
        ls, lg, ly = 0, [], []
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for X, S, y in loader:
                X, S, y = X.to(device), S.to(device), y.to(device)
                if glucose_only:
                    X = X[:, :, :1]
                if training: opt.zero_grad()
                logit, _, _ = model(X, S)
                loss = crit(logit, y)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                ls += loss.item() * len(y)
                lg.append(logit.detach().cpu())
                ly.append(y.cpu())
        logits = torch.cat(lg); ys = torch.cat(ly).numpy().astype(int)
        probs  = torch.sigmoid(logits).numpy()
        a      = roc_auc_score(ys, probs) if ys.sum() > 0 else 0.0
        return ls / len(loader.dataset), a

    for epoch in range(1, cfg['epochs']+1):
        tr_loss, _        = _run(tr_loader, True)
        va_loss, va_auc    = _run(va_loader, False)
        sched.step()
        es(va_auc, model)
        hist['tr_loss'].append(tr_loss)
        hist['va_loss'].append(va_loss)
        hist['va_auc'].append(va_auc)
        if epoch % 5 == 0 or es.stop:
            print(f"Ep {epoch:3d}  tr={tr_loss:.4f}  va={va_loss:.4f}  "
                  f"vaAUC={va_auc:.4f}" + ("  [STOP]" if es.stop else ""))
        if es.stop: break

    model.load_state_dict(es.best_state)
    print(f"  -> Best val AUC: {es.best:.4f}")
    return model, hist


# cell 13 - train all models
crit_eval   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
all_results = {}
all_hist    = {}
all_probs   = {}

# full TFT
print("\n" + "="*55)
print("1/4  TFT - Full Model")
print("="*55)
tft = TemporalFusionTransformer(n_dynamic, n_static, CONFIG).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in tft.parameters()):,}")
tft, h = train_model(tft, train_loader, val_loader, CONFIG, DEVICE, pos_weight)
_, _, tft_probs, tft_y = evaluate(tft, test_loader, crit_eval, DEVICE)
all_results['TFT (Full)'] = compute_metrics(tft_y, tft_probs)
all_hist['TFT (Full)']    = h
all_probs['TFT (Full)']   = (tft_probs, tft_y)
print("  Metrics:", all_results['TFT (Full)'])

# LSTM baseline (glucose only)
print("\n" + "="*55)
print("2/4  LSTM Baseline (glucose only)")
print("="*55)
lstm = LSTMBaseline().to(DEVICE)
lstm, h = train_model(lstm, train_loader, val_loader,
                      CONFIG, DEVICE, pos_weight, glucose_only=True)
_, _, lstm_probs, lstm_y = evaluate(lstm, test_loader, crit_eval, DEVICE,
                                    glucose_only=True)
all_results['LSTM Baseline'] = compute_metrics(lstm_y, lstm_probs)
all_hist['LSTM Baseline']    = h
all_probs['LSTM Baseline']   = (lstm_probs, lstm_y)
print("  Metrics:", all_results['LSTM Baseline'])

# vanialla transformer (glucose only)
print("\n" + "="*55)
print("3/4  Vanilla Transformer (glucose only)")
print("="*55)
vt = VanillaTransformer().to(DEVICE)
vt, h = train_model(vt, train_loader, val_loader,
                    CONFIG, DEVICE, pos_weight, glucose_only=True)
_, _, vt_probs, vt_y = evaluate(vt, test_loader, crit_eval, DEVICE,
                                glucose_only=True)
all_results['Vanilla Transformer'] = compute_metrics(vt_y, vt_probs)
all_hist['Vanilla Transformer']    = h
all_probs['Vanilla Transformer']   = (vt_probs, vt_y)
print("  Metrics:", all_results['Vanilla Transformer'])

# clinical threshold
thresh_p = threshold_baseline(X_te)
all_results['Clinical Threshold'] = compute_metrics(y_te, thresh_p)
all_probs['Clinical Threshold']   = (thresh_p, y_te)
print("\n4/4  Clinical Threshold:", all_results['Clinical Threshold'])


# cell 14 - ablation study
print("\n" + "="*55)
print("ABLATION STUDY")
print("="*55)

class TFT_NoTCN(nn.Module):
    def __init__(self, n_dyn, n_stat, cfg):
        super().__init__()
        H, D = cfg['hidden_size'], cfg['dropout']
        self.stat_emb  = nn.Linear(n_stat, H)
        self.stat_grn  = GatedResidualNetwork(H,H,H,D)
        self.dyn_proj  = nn.Linear(1, H)
        self.vsn       = VariableSelectionNetwork(n_dyn,H,H,D,H)
        self.lstm      = nn.LSTM(H,H,cfg['lstm_layers'],batch_first=True,
                                 dropout=D if cfg['lstm_layers']>1 else 0)
        self.attn      = InterpretableMultiHeadAttention(H,cfg['attn_heads'],D)
        self.attn_grn  = GatedResidualNetwork(H,H,H,D)
        self.attn_norm = nn.LayerNorm(H)
        self.out_grn   = GatedResidualNetwork(H,H,H,D)
        self.clf       = nn.Linear(H,1)
    def forward(self, x_dyn, x_stat):
        B,T,_ = x_dyn.shape
        ctx   = self.stat_grn(self.stat_emb(x_stat))
        xp    = self.dyn_proj(x_dyn.unsqueeze(-1))
        xv,_  = self.vsn(xp, context=ctx.unsqueeze(1).expand(-1,T,-1))
        h0    = ctx.unsqueeze(0).repeat(self.lstm.num_layers,1,1)
        xl,_  = self.lstm(xv,(h0,torch.zeros_like(h0)))
        ao,aw = self.attn(xl)
        xa    = self.attn_norm(xl + self.attn_grn(ao))
        xp2   = xa[:,-CONFIG['horizon_steps']:,:].mean(1)
        return self.clf(self.out_grn(xp2)).squeeze(-1), aw, None


class TFT_NoStatic(nn.Module):
    def __init__(self, n_dyn, n_stat, cfg):
        super().__init__()
        H, D = cfg['hidden_size'], cfg['dropout']
        self.dyn_proj  = nn.Linear(1, H)
        self.vsn       = VariableSelectionNetwork(n_dyn,H,H,D)
        self.tcn       = nn.Sequential(*[TCNBlock(H,cfg['tcn_kernel'],d,D)
                                         for d in cfg['tcn_dilations']])
        self.lstm      = nn.LSTM(H,H,cfg['lstm_layers'],batch_first=True,
                                 dropout=D if cfg['lstm_layers']>1 else 0)
        self.attn      = InterpretableMultiHeadAttention(H,cfg['attn_heads'],D)
        self.attn_grn  = GatedResidualNetwork(H,H,H,D)
        self.attn_norm = nn.LayerNorm(H)
        self.out_grn   = GatedResidualNetwork(H,H,H,D)
        self.clf       = nn.Linear(H,1)
    def forward(self, x_dyn, x_stat=None):
        B,T,_ = x_dyn.shape
        xp    = self.dyn_proj(x_dyn.unsqueeze(-1))
        xv,_  = self.vsn(xp)
        xt    = self.tcn(xv)
        xl,_  = self.lstm(xt)
        ao,aw = self.attn(xl)
        xa    = self.attn_norm(xl + self.attn_grn(ao))
        xp2   = xa[:,-CONFIG['horizon_steps']:,:].mean(1)
        return self.clf(self.out_grn(xp2)).squeeze(-1), aw, None


ablations = {
    'TFT w/o TCN'   : TFT_NoTCN(n_dynamic, n_static, CONFIG).to(DEVICE),
    'TFT w/o Static': TFT_NoStatic(n_dynamic, n_static, CONFIG).to(DEVICE),
}

for name, model in ablations.items():
    print(f"\n-- {name}")
    model, _ = train_model(model, train_loader, val_loader,
                           CONFIG, DEVICE, pos_weight)
    _, _, probs, yt = evaluate(model, test_loader, crit_eval, DEVICE)
    all_results[name] = compute_metrics(yt, probs)
    all_probs[name]   = (probs, yt)
    print("  Metrics:", all_results[name])


# cell 15 - results table
print("\n" + "="*60)
print("FULL RESULTS")
print("="*60)
res_df = pd.DataFrame(all_results).T[
    ['AUC','Accuracy','Sensitivity','Specificity','FAR','TP','FP','TN','FN']]
print(res_df.to_string())
res_df.to_csv(f'{RESULTS_DIR}/results_table.csv')


# cell 16 - figures table
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
COLORS = {'TFT (Full)':'navy','LSTM Baseline':'crimson',
          'Vanilla Transformer':'green','Clinical Threshold':'darkorange'}

ax = axes[0]
for name, (pr, yt) in all_probs.items():
    if name in COLORS and yt.sum() > 0:
        fpr, tpr, _ = roc_curve(yt, pr)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[name], label=f'{name} (AUC={a:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.4)
ax.set(xlabel='FPR', ylabel='TPR', title='ROC - 30-min Hypoglycemia Prediction')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for name, h in all_hist.items():
    ax.plot(h['va_auc'], label=name,
            color=COLORS.get(name,'gray'),
            ls='--' if 'LSTM' in name or 'Vanilla' in name else '-')
ax.set(xlabel='Epoch', ylabel='Val AUC', title='Training Curves')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
abl_names = ['TFT (Full)','TFT w/o TCN','TFT w/o Static','LSTM Baseline','Clinical Threshold']
abl_names = [n for n in abl_names if n in all_results]
abl_aucs  = [all_results[n]['AUC'] for n in abl_names]
colors    = ['navy'] + ['steelblue']*(len(abl_names)-1)
bars = ax.barh(abl_names, abl_aucs, color=colors)
for b, v in zip(bars, abl_aucs):
    ax.text(b.get_width()+0.003, b.get_y()+b.get_height()/2,
            f'{v:.3f}', va='center', fontsize=9)
ax.set(xlim=(0.5,1.02), xlabel='AUC-ROC', title='Ablation Study')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(f'{RESULTS_DIR}/figures.{ext}', dpi=300, bbox_inches='tight')
plt.show()
print(f"Figures saved -> {RESULTS_DIR}/figures.pdf")


# cell 17 - attention heatmap
tft.eval()
hypo_attn, norm_attn = [], []
with torch.no_grad():
    for X, S, y in test_loader:
        _, attn_w, _ = tft(X.to(DEVICE), S.to(DEVICE))
        last = attn_w[:, -1, :].cpu().numpy()
        hypo_attn.append(last[y.numpy()==1])
        norm_attn.append(last[y.numpy()==0])

mean_hypo = np.concatenate([a for a in hypo_attn if len(a)]).mean(0)
mean_norm = np.concatenate([a for a in norm_attn if len(a)]).mean(0)

labels = [f"t-{(CONFIG['lookback_steps']-i)*CONFIG['cgm_interval_min']}m"
          for i in range(CONFIG['lookback_steps'])]

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(labels, mean_hypo, 'r-o', ms=5, label='Hypoglycemic')
ax2.plot(labels, mean_norm, 'b-o', ms=5, label='Non-hypoglycemic')
ax2.set(xlabel='Minutes before prediction', ylabel='Mean attention weight',
        title='TFT Attention Weights Over Lookback Window')
ax2.legend(); ax2.grid(alpha=0.3)
plt.xticks(rotation=45); plt.tight_layout()
for ext in ('pdf','png'):
    fig2.savefig(f'{RESULTS_DIR}/attention_weights.{ext}',
                 dpi=300, bbox_inches='tight')
plt.show()


# cell 18 - inference speed
tft.eval()
dx = torch.randn(1, CONFIG['lookback_steps'], n_dynamic).to(DEVICE)
ds = torch.randn(1, n_static).to(DEVICE)
for _ in range(20): tft(dx, ds)

N = 500
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N): tft(dx, ds)
ms = (time.perf_counter()-t0)/N*1000
print(f"\nInference: {ms:.2f} ms/prediction  (device: {DEVICE})")


# cell 19 - summary
tft_m  = all_results['TFT (Full)']
lstm_m = all_results['LSTM Baseline']
far_red = (lstm_m['FAR'] - tft_m['FAR']) / (lstm_m['FAR']+1e-8) * 100

summary = {
    'results'         : all_results,
    'inference_ms'    : round(ms, 2),
    'n_dynamic_feats' : int(n_dynamic),
    'n_static_feats'  : int(n_static),
    'train_windows'   : int(len(y_tr)),
    'test_windows'    : int(len(y_te)),
    'hypo_rate_train' : round(float(y_tr.mean()), 4),
    'hypo_rate_test'  : round(float(y_te.mean()), 4),
    'tft_params'      : sum(p.numel() for p in tft.parameters()),
}
with open(f'{RESULTS_DIR}/paper_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*55)
print("COPY THESE INTO LATEX:")
print("="*55)
print(f"TFT AUC             : {tft_m['AUC']:.3f}")
print(f"TFT Accuracy        : {tft_m['Accuracy']*100:.1f}%")
print(f"TFT Sensitivity     : {tft_m['Sensitivity']*100:.1f}%")
print(f"TFT Specificity     : {tft_m['Specificity']*100:.1f}%")
print(f"TFT FAR             : {tft_m['FAR']*100:.1f}%")
print(f"LSTM AUC            : {lstm_m['AUC']:.3f}")
print(f"FAR reduction       : {far_red:.1f}%")
print(f"Inference latency   : {ms:.0f} ms")
print(f"\nOutputs in: {RESULTS_DIR}/")