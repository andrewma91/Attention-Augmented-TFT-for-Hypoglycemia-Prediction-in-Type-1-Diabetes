 # cell 1 - imports

import os, sys, json, time, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.model_selection import Ksplit
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if DEVICE.type == 'cpu':
    print("WARNING. This will take a very long time on CPU. Turn on GPU in settings->accelerator.")



# cell 2 - paths 

DATA_DIR    = '/path/to/T1DiabetesGranada/'  # set to local dataset path
RESULTS_DIR = './results_cv'
os.makedirs(RESULTS_DIR, exist_ok=True)

# checkpoint file
CHECKPOINT_CSV = f'{RESULTS_DIR}/split_results_checkpoint.csv'



# cell 3 - load all 4 files

print("Loading files...")
df_glucose = pd.read_csv(f'{DATA_DIR}/Glucose_measurements.csv', dtype={'Measurement': np.int16})
df_biochem  = pd.read_csv(f'{DATA_DIR}/Biochemical_parameters.csv')
df_patients = pd.read_csv(f'{DATA_DIR}/Patient_info.csv')
df_diag     = pd.read_csv(f'{DATA_DIR}/Diagnostics.csv')

df_glucose['Measurement_date'] = pd.to_datetime(df_glucose['Measurement_date'], infer_datetime_format=True)
df_biochem['Reception_date'] = pd.to_datetime(df_biochem['Reception_date'], infer_datetime_format=True)

print(f"Glucose measurements : {len(df_glucose):>10,} rows")
print(f"Biochemical params   : {len(df_biochem):>10,} rows")
print(f"Patients             : {len(df_patients):>10,} rows")
print(f"Diagnostics          : {len(df_diag):>10,} rows")



# cell 4 - config

BIOCHEM_FEATURES = [
    'Glycated hemoglobin (A1c)', 'Total cholesterol', 'HDL cholesterol',
    'Triglycerides', 'Creatinine',
]

CONFIG = {
    'hypo_threshold'  : 70.0,
    'cgm_interval_min': 5,
    'lookback_min'    : 60,
    'horizon_min'     : 30,

    'max_windows'         : 1_000_000, 

    # cross validation
    'n_splits'             : 5,    # set to 3 for faster run
    'val_frac_within_split': 0.12, 

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
print(f"Lookback: {CONFIG['lookback_steps']} steps, Horizon: {CONFIG['horizon_steps']} steps")
print(f"Cross-validation: {CONFIG['n_splits']}-split, patient-level")



# cell 5 - build static features

def build_static(df_patients, df_biochem, df_diag):
    df = df_patients[['Patient_ID', 'Sex', 'Birth_year']].copy()
    df['Age']     = 2018 - df['Birth_year'].astype(int)
    df['Sex_enc'] = (df['Sex'] == 'M').astype(float)
    hba1c = (df_biochem[df_biochem['Name'] == 'Glycated hemoglobin (A1c)']
             .sort_values('Reception_date').groupby('Patient_ID')['Value'].last()
             .rename('HbA1c_latest'))
    df = df.merge(hba1c, on='Patient_ID', how='left')
    n_diag = df_diag.groupby('Patient_ID')['Code'].nunique().rename('N_diagnoses')
    df = df.merge(n_diag, on='Patient_ID', how='left')
    df['N_diagnoses'] = df['N_diagnoses'].fillna(0)
    cols = ['Age', 'Sex_enc', 'HbA1c_latest', 'N_diagnoses']
    df[cols] = df[cols].fillna(df[cols].mean())
    return df.set_index('Patient_ID')[cols]

df_static = build_static(df_patients, df_biochem, df_diag)
print("Static features built for", len(df_static), "patients")



# cell 6 - biochem forward fill

def build_biochem_ffill(df_biochem, features):
    df_b = df_biochem[df_biochem['Name'].isin(features)].copy()
    df_b['Date'] = df_b['Reception_date'].dt.date
    pivoted = (df_b.groupby(['Patient_ID', 'Date', 'Name'])['Value'].mean()
               .unstack('Name').reset_index())
    pivoted['Date'] = pd.to_datetime(pivoted['Date'])
    pivoted = pivoted.sort_values(['Patient_ID', 'Date'])
    biochem_dict = {}
    for pid, grp in pivoted.groupby('Patient_ID'):
        grp = grp.set_index('Date').drop(columns=['Patient_ID'])
        grp = grp.reindex(pd.date_range(grp.index.min(), grp.index.max(), freq='D')).ffill().fillna(method='bfill').fillna(0)
        biochem_dict[pid] = grp
    return biochem_dict, [c for c in pivoted.columns if c not in ('Patient_ID', 'Date')]

biochem_dict, biochem_cols = build_biochem_ffill(df_biochem, BIOCHEM_FEATURES)
print(f"Biochemical feature columns: {biochem_cols}")



# cell 7 - fast vectorized window extraction

def build_windows(df_glucose, df_static, biochem_dict, biochem_cols, cfg):
    L, H, THRESH = cfg['lookback_steps'], cfg['horizon_steps'], cfg['hypo_threshold']
    n_biochem = len(biochem_cols)
    n_patients = df_glucose['Patient_ID'].nunique()
    cap_per_patient = max(500, cfg['max_windows'] // n_patients)
    print(f"Capping at {cap_per_patient:,} windows/patient")

    all_X, all_S, all_y, all_pid = [], [], [], []
    for pid, grp in tqdm(df_glucose.groupby('Patient_ID'), desc='Building windows'):
        grp = grp.sort_values('Measurement_date').reset_index(drop=True)
        g = grp['Measurement'].values.astype(np.float32)
        n = len(g)
        if n < L + H + 1:
            continue
        g = pd.Series(g).interpolate('linear', limit=3).values.astype(np.float32)
        roc = np.gradient(g).astype(np.float32)
        accel = np.gradient(roc).astype(np.float32)
        gs = pd.Series(g)
        rmean = gs.rolling(6, min_periods=1).mean().values.astype(np.float32)
        rstd = gs.rolling(6, min_periods=1).std().fillna(0).values.astype(np.float32)
        hours = (grp['Measurement_date'].dt.hour + grp['Measurement_date'].dt.minute / 60.0).values.astype(np.float32)
        sin_t = np.sin(2 * np.pi * hours / 24).astype(np.float32)
        cos_t = np.cos(2 * np.pi * hours / 24).astype(np.float32)

        biochem_vals = np.zeros((n, n_biochem), dtype=np.float32)
        if pid in biochem_dict:
            bdf = biochem_dict[pid].reset_index()
            bdf.columns = ['Date'] + list(bdf.columns[1:])
            grp_dates = grp['Measurement_date'].dt.normalize().rename('Date')
            merged = pd.merge_asof(grp_dates.to_frame(), bdf, on='Date', direction='backward')
            cols_present = [c for c in biochem_cols if c in merged.columns]
            if cols_present:
                biochem_vals[:, :len(cols_present)] = merged[cols_present].fillna(0).values.astype(np.float32)

        dyn = np.column_stack([g, roc, accel, rmean, rstd, sin_t, cos_t, biochem_vals])
        static_vec = (df_static.loc[pid].values.astype(np.float32) if pid in df_static.index
                      else np.zeros(df_static.shape[1], dtype=np.float32))

        n_wins = n - L - H + 1
        if n_wins <= 0:
            continue
        row_idx = np.arange(L)[None, :] + np.arange(n_wins)[:, None]
        X_wins = dyn[row_idx]
        fut_idx = np.arange(H)[None, :] + (np.arange(n_wins) + L)[:, None]
        future_g = g[fut_idx]
        labels = (np.nanmin(future_g, axis=1) < THRESH).astype(np.int64)
        nan_mask = np.isnan(X_wins[:, :, 0]).sum(axis=1) <= (L * 0.2)
        X_wins, labels = X_wins[nan_mask], labels[nan_mask]
        if len(labels) == 0:
            continue
        if len(labels) > cap_per_patient:
            keep = np.random.choice(len(labels), cap_per_patient, replace=False)
            X_wins, labels = X_wins[keep], labels[keep]

        all_X.append(X_wins)
        all_S.append(np.tile(static_vec, (len(labels), 1)))
        all_y.append(labels)
        all_pid.extend([pid] * len(labels))
        del dyn, X_wins, labels, biochem_vals

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    S = np.concatenate(all_S, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    pids = np.array(all_pid)
    del all_X, all_S, all_y, all_pid
    gc.collect()
    print(f"\nTotal windows: {len(y):,}  Hypoglycemic: {y.sum():,} ({100*y.mean():.2f}%)")
    return X, S, y, pids

X_all, S_all, y_all, pids_all = build_windows(df_glucose, df_static, biochem_dict, biochem_cols, CONFIG)
del df_glucose
gc.collect()



# cell 8 - model parts

class GatedResidualNetwork(nn.Module):
    def __init__(self, in_sz, hid_sz, out_sz, dropout=0.1, ctx_sz=None):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, hid_sz)
        self.fc2 = nn.Linear(hid_sz, out_sz * 2)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_sz)
        self.skip = nn.Linear(in_sz, out_sz, bias=False) if in_sz != out_sz else nn.Identity()
        self.ctx = nn.Linear(ctx_sz, hid_sz, bias=False) if ctx_sz else None
    def forward(self, x, context=None):
        r = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.ctx is not None:
            h = h + self.ctx(context)
        h = F.elu(h); h = self.drop(h)
        v, g = self.fc2(h).chunk(2, dim=-1)
        return self.norm(r + v * torch.sigmoid(g))

class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_vars, var_dim, hidden, dropout, ctx_size=None):
        super().__init__()
        self.n_vars = n_vars
        self.grns = nn.ModuleList([GatedResidualNetwork(var_dim, hidden, hidden, dropout, ctx_size) for _ in range(n_vars)])
        self.selector = GatedResidualNetwork(n_vars * var_dim, hidden, n_vars, dropout, ctx_size)
    def forward(self, x, context=None):
        B, T, V, D = x.shape
        proc = torch.stack([self.grns[i](x[:, :, i, :], context) for i in range(V)], dim=2)
        flat = x.reshape(B, T, -1)
        weights = torch.softmax(self.selector(flat, context), dim=-1)
        return (proc * weights.unsqueeze(-1)).sum(dim=2), weights

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.trim = pad
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        h = self.conv(x.transpose(1,2))
        if self.trim: h = h[:, :, :-self.trim]
        h = self.act(h).transpose(1,2)
        return self.norm(x + self.drop(h))

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_heads, dropout):
        super().__init__()
        self.d_k = hidden // n_heads; self.heads = n_heads
        self.W_q = nn.Linear(hidden, hidden); self.W_k = nn.Linear(hidden, hidden)
        self.W_v = nn.Linear(hidden, self.d_k); self.W_o = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, H = x.shape
        Q = self.W_q(x).reshape(B,T,self.heads,self.d_k).transpose(1,2)
        K = self.W_k(x).reshape(B,T,self.heads,self.d_k).transpose(1,2)
        V = self.W_v(x).unsqueeze(1).expand(-1,self.heads,-1,-1)
        sc = (Q @ K.transpose(-2,-1)) / (self.d_k**0.5)
        attn = self.drop(torch.softmax(sc, dim=-1))
        out = (attn @ V).transpose(1,2).reshape(B,T,H)
        return self.W_o(out), attn.mean(dim=1)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_dyn, n_stat, cfg):
        super().__init__()
        H, D, nh = cfg['hidden_size'], cfg['dropout'], cfg['attn_heads']
        self.stat_emb = nn.Linear(n_stat, H)
        self.stat_grn = GatedResidualNetwork(H, H, H, D)
        self.dyn_proj = nn.Linear(1, H)
        self.vsn = VariableSelectionNetwork(n_dyn, H, H, D, ctx_size=H)
        self.tcn = nn.Sequential(*[TCNBlock(H, cfg['tcn_kernel'], d, D) for d in cfg['tcn_dilations']])
        self.lstm = nn.LSTM(H, H, cfg['lstm_layers'], batch_first=True, dropout=D if cfg['lstm_layers']>1 else 0)
        self.attn = InterpretableMultiHeadAttention(H, nh, D)
        self.attn_grn = GatedResidualNetwork(H, H, H, D)
        self.attn_norm = nn.LayerNorm(H)
        self.out_grn = GatedResidualNetwork(H, H, H, D)
        self.clf = nn.Linear(H, 1)
        self.horizon = cfg['horizon_steps']
    def forward(self, x_dyn, x_stat):
        B, T, _ = x_dyn.shape
        ctx = self.stat_grn(self.stat_emb(x_stat))
        x_p = self.dyn_proj(x_dyn.unsqueeze(-1))
        ctx_t = ctx.unsqueeze(1).expand(-1, T, -1)
        x_v, _ = self.vsn(x_p, context=ctx_t)
        x_t = self.tcn(x_v)
        h0 = ctx.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        x_l, _ = self.lstm(x_t, (h0, torch.zeros_like(h0)))
        a_out, attn_w = self.attn(x_l)
        x_a = self.attn_norm(x_l + self.attn_grn(a_out))
        x_pool = x_a[:, -self.horizon:, :].mean(1)
        return self.clf(self.out_grn(x_pool)).squeeze(-1), attn_w

class LSTMBaseline(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout if layers>1 else 0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x, s=None):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1), None



# cell 9 - dataset, training, eval

class HypoDataset(Dataset):
    def __init__(self, X, S, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.S[i], self.y[i]

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience, self.best, self.best_state, self.counter, self.stop = patience, None, None, 0, False
    def __call__(self, val_auc, model):
        if self.best is None or val_auc > self.best + 1e-4:
            self.best, self.best_state, self.counter = val_auc, deepcopy(model.state_dict()), 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    precision = tp / (tp+fp+1e-8)
    recall = tp / (tp+fn+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    return {
        'AUC': roc_auc_score(y_true, y_prob) if y_true.sum()>0 else 0.0,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall, 'Specificity': tn/(tn+fp+1e-8),
        'Precision': precision, 'F1': f1,
        'FAR': fp/(fp+tn+1e-8),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }

def train_model(model, tr_loader, va_loader, cfg, device, pos_weight, glucose_only=False, verbose_prefix=""):
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    es = EarlyStopping(cfg['early_stop_patience'])

    def _run(loader, training):
        model.train() if training else model.eval()
        ls, lg, ly = 0, [], []
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X, S, y in loader:
                X, S, y = X.to(device), S.to(device), y.to(device)
                if glucose_only: X = X[:, :, :1]
                if training: opt.zero_grad()
                logit, _ = model(X, S)
                loss = crit(logit, y)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                ls += loss.item() * len(y)
                lg.append(logit.detach().cpu()); ly.append(y.cpu())
        logits, ys = torch.cat(lg), torch.cat(ly).numpy().astype(int)
        probs = torch.sigmoid(logits).numpy()
        a = roc_auc_score(ys, probs) if ys.sum() > 0 else 0.0
        return ls / len(loader.dataset), a

    for epoch in range(1, cfg['epochs']+1):
        tr_loss, _ = _run(tr_loader, True)
        va_loss, va_auc = _run(va_loader, False)
        sched.step()
        es(va_auc, model)
        if epoch % 10 == 0 or es.stop:
            print(f"{verbose_prefix} Ep {epoch:3d}  va_AUC={va_auc:.4f}" + ("  [STOP]" if es.stop else ""))
        if es.stop: break

    model.load_state_dict(es.best_state)
    return model

@torch.no_grad()
def evaluate(model, loader, device, glucose_only=False):
    model.eval()
    lg, ly = [], []
    for X, S, y in loader:
        X, S = X.to(device), S.to(device)
        if glucose_only: X = X[:, :, :1]
        logit, _ = model(X, S)
        lg.append(logit.cpu()); ly.append(y)
    logits, ys = torch.cat(lg), torch.cat(ly).numpy().astype(int)
    probs = torch.sigmoid(logits).numpy()
    return probs, ys



# cell 10 - per-split train+eval function

def run_split(split_idx, train_patients, test_patients, X_all, S_all, y_all, pids_all, cfg, device):
    rng = np.random.RandomState(42 + split_idx)
    train_patients = np.array(list(train_patients))
    rng.shuffle(train_patients)
    n_val = max(1, int(len(train_patients) * cfg['val_frac_within_split']))
    val_patients = set(train_patients[:n_val])
    actual_train_patients = set(train_patients[n_val:])
    test_patients = set(test_patients)

    def mask(patient_set):
        m = np.isin(pids_all, list(patient_set))
        return X_all[m], S_all[m], y_all[m]

    X_tr, S_tr, y_tr = mask(actual_train_patients)
    X_va, S_va, y_va = mask(val_patients)
    X_te, S_te, y_te = mask(test_patients)

    print(f"\n[split {split_idx}] train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,} "
          f"({len(actual_train_patients)}/{len(val_patients)}/{len(test_patients)} patients)")

    n_dynamic, n_static = X_tr.shape[2], S_tr.shape[1]
    dyn_sc, stat_sc = StandardScaler(), StandardScaler()
    X_tr = dyn_sc.fit_transform(X_tr.reshape(-1, n_dynamic)).reshape(X_tr.shape)
    X_va = dyn_sc.transform(X_va.reshape(-1, n_dynamic)).reshape(X_va.shape)
    X_te = dyn_sc.transform(X_te.reshape(-1, n_dynamic)).reshape(X_te.shape)
    S_tr = stat_sc.fit_transform(S_tr); S_va = stat_sc.transform(S_va); S_te = stat_sc.transform(S_te)

    pos_weight = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)], dtype=torch.float32).to(device)

    train_loader = DataLoader(HypoDataset(X_tr, S_tr, y_tr), cfg['batch_size'], shuffle=True)
    val_loader   = DataLoader(HypoDataset(X_va, S_va, y_va), cfg['batch_size'], shuffle=False)
    test_loader  = DataLoader(HypoDataset(X_te, S_te, y_te), cfg['batch_size'], shuffle=False)

    split_results = {}

    # TFT
    t0 = time.time()
    tft = TemporalFusionTransformer(n_dynamic, n_static, cfg).to(device)
    tft = train_model(tft, train_loader, val_loader, cfg, device, pos_weight, verbose_prefix=f"[split {split_idx} TFT]")
    probs, ys = evaluate(tft, test_loader, device)
    split_results['TFT'] = compute_metrics(ys, probs)
    split_results['TFT']['train_time_s'] = time.time() - t0
    print(f"[split {split_idx}] TFT  AUC={split_results['TFT']['AUC']:.4f}  "
          f"({split_results['TFT']['train_time_s']:.0f}s)")
    del tft

    # LSTM baseline
    t0 = time.time()
    lstm = LSTMBaseline().to(device)
    lstm = train_model(lstm, train_loader, val_loader, cfg, device, pos_weight,
                       glucose_only=True, verbose_prefix=f"[split {split_idx} LSTM]")
    probs, ys = evaluate(lstm, test_loader, device, glucose_only=True)
    split_results['LSTM'] = compute_metrics(ys, probs)
    split_results['LSTM']['train_time_s'] = time.time() - t0
    print(f"[split {split_idx}] LSTM AUC={split_results['LSTM']['AUC']:.4f}  "
          f"({split_results['LSTM']['train_time_s']:.0f}s)")
    del lstm

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()
    return split_results



# cell 11 - run k split cross val

unique_patients = np.unique(pids_all)
print(f"\nTotal unique patients: {len(unique_patients)}")

kf = Ksplit(n_splits=CONFIG['n_splits'], shuffle=True, random_state=42)

completed_splits = set()
if os.path.exists(CHECKPOINT_CSV):
    existing = pd.read_csv(CHECKPOINT_CSV)
    completed_splits = set(existing['split'].unique())
    print(f"Resuming: splits already completed: {sorted(completed_splits)}")
else:
    pd.DataFrame(columns=['split','model','AUC','Accuracy','Sensitivity','Specificity',
                          'Precision','F1','FAR','TP','FP','TN','FN','train_time_s']
                ).to_csv(CHECKPOINT_CSV, index=False)

overall_start = time.time()
for split_idx, (train_idx, test_idx) in enumerate(kf.split(unique_patients), 1):
    if split_idx in completed_splits:
        print(f"Skipping split {split_idx} (already completed)")
        continue

    train_patients = unique_patients[train_idx]
    test_patients  = unique_patients[test_idx]

    split_results = run_split(split_idx, train_patients, test_patients,
                            X_all, S_all, y_all, pids_all, CONFIG, DEVICE)

    rows = []
    for model_name, metrics in split_results.items():
        row = {'split': split_idx, 'model': model_name, **metrics}
        rows.append(row)
    pd.DataFrame(rows).to_csv(CHECKPOINT_CSV, mode='a', header=False, index=False)
    print(f"[split {split_idx}] Checkpoint saved. Elapsed total: {(time.time()-overall_start)/60:.1f} min")

print(f"\nAll splits complete. Total time: {(time.time()-overall_start)/60:.1f} min")



# cell 12 - aggrevate results across splits + significance test

results_df = pd.read_csv(CHECKPOINT_CSV)
print("\nPer-split results:")
print(results_df[['split','model','AUC','Accuracy','Sensitivity','Specificity','F1','FAR']].to_string(index=False))

summary = results_df.groupby('model')[['AUC','Accuracy','Sensitivity','Specificity','Precision','F1','FAR']].agg(['mean','std'])
print("\n" + "="*70)
print("MEAN ± STD ACROSS splitS")
print("="*70)
print(summary.to_string())

# paired significance test
tft_aucs  = results_df[results_df.model=='TFT'].sort_values('split')['AUC'].values
lstm_aucs = results_df[results_df.model=='LSTM'].sort_values('split')['AUC'].values

t_stat, p_value = scipy_stats.ttest_rel(tft_aucs, lstm_aucs)
print("\n" + "="*70)
print("PAIRED SIGNIFICANCE TEST (TFT vs LSTM AUC, paired t-test across splits)")
print("="*70)
print(f"TFT split AUCs:  {np.round(tft_aucs, 4)}")
print(f"LSTM split AUCs: {np.round(lstm_aucs, 4)}")
print(f"Mean difference: {np.mean(tft_aucs - lstm_aucs):.4f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.5f}")
print(f"{'SIGNIFICANT at alpha=0.05' if p_value < 0.05 else 'NOT significant at alpha=0.05'}")
print("\nNOTE: with n_splits={} this test has limited statistical power; report".format(CONFIG['n_splits']))
print("the exact p-value and n in the paper rather than just 'p<0.05'.")

summary.to_csv(f'{RESULTS_DIR}/cv_summary.csv')
with open(f'{RESULTS_DIR}/significance_test.json', 'w') as f:
    json.dump({
        'tft_split_aucs': tft_aucs.tolist(), 'lstm_split_aucs': lstm_aucs.tolist(),
        'mean_diff': float(np.mean(tft_aucs - lstm_aucs)),
        't_statistic': float(t_stat), 'p_value': float(p_value),
        'n_splits': CONFIG['n_splits'],
    }, f, indent=2)



# cell 13 - fig: split level AUC distribution

fig, ax = plt.subplots(figsize=(6, 5))
positions = [1, 2]
data = [tft_aucs, lstm_aucs]
bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
for i, d in enumerate(data):
    ax.scatter([positions[i]]*len(d), d, color='navy', zorder=3, s=40)
ax.set_xticks(positions)
ax.set_xticklabels(['TFT', 'LSTM Baseline'])
ax.set_ylabel('AUC-ROC')
ax.set_title(f'{CONFIG["n_splits"]}-split CV: AUC Distribution\n(p={p_value:.4f}, paired t-test)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{RESULTS_DIR}/cv_auc_distribution.{ext}', dpi=300, bbox_inches='tight')
plt.show()



# cell 14 - summary

print("\n" + "="*70)
print("="*70)
for model in ['TFT', 'LSTM']:
    row = summary.loc[model]
    print(f"\n{model}:")
    for metric in ['AUC','Accuracy','Sensitivity','Specificity','Precision','F1','FAR']:
        mean, std = row[(metric,'mean')], row[(metric,'std')]
        print(f"  {metric}: {mean:.4f} +/- {std:.4f}")
print(f"\nPaired t-test (TFT vs LSTM AUC, n={CONFIG['n_splits']} splits): "
      f"t={t_stat:.3f}, p={p_value:.5f}")
print(f"\nOutputs saved in: {RESULTS_DIR}/")