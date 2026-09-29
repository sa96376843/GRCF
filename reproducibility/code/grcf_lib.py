#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GRCF core library: data loading, leakage-safe feature builder, EW-GRA,
models (baselines + GRCF), training loops and metrics.

Datasets (both real and public):
  XTX  : XuetangX course-enrollment data (AAAI'19 HRL dataset, moocdata.cn)
  MARS : Mandarine Academy Recommender System dataset (Harvard Dataverse,
         doi:10.7910/DVN/BMY3UD)
"""
import os, json, time, math, random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss
from scipy import sparse

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
WORK = os.environ.get('GRCF_WORK', BASE)   # override for scratch execution
RAW = os.path.join(BASE, 'data', 'raw')
RES = os.path.join(WORK, 'results'); os.makedirs(RES, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def remap(a, n):  # id -> internal index with index n reserved for unknown
    return np.minimum(a, n - 0)  # ids already dense in our preprocessing

# ------------------------------------------------------------------ data ----
def load_xtx():
    df = pd.read_csv(os.path.join(RAW, 'xuetangx/data.csv'), encoding='gbk')
    df.columns = ['user', 'time', 'course', 'name', 'cat', 'cat_id']
    df['ts'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')
    df = df.dropna(subset=['cat_id'])
    df['cat_id'] = df['cat_id'].astype(int)
    # dense remap
    umap = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    cmap = {c: i for i, c in enumerate(sorted(df.course.unique()))}
    df['user'] = df.user.map(umap); df['course'] = df.course.map(cmap)
    courses = df.groupby('course').agg(name=('name', 'first'), cat_id=('cat_id', 'first')).reset_index()
    courses['text'] = courses['name'].fillna('')
    inter = df[['user', 'course', 'ts']].drop_duplicates()
    inter['rating'] = 1.0
    return inter, courses, None

def load_mars():
    it = pd.read_csv(os.path.join(RAW, 'mars_items_fr.tab'), sep='\t', quoting=3)
    us = pd.read_csv(os.path.join(RAW, 'mars_users_fr.tab'), sep='\t', quoting=3)
    ex = pd.read_csv(os.path.join(RAW, 'mars_explicit_fr.tab'), sep='\t', quoting=3)
    clean = lambda s: s.astype(str).str.strip('"')
    it = it.rename(columns={'Difficulty': 'difficulty', 'Software': 'software',
                            'Theme': 'theme', 'type': 'itype'})
    for c in ['language', 'name', 'difficulty', 'Job', 'software', 'theme', 'itype']:
        it[c] = clean(it[c])
    it['duration'] = pd.to_numeric(it['duration'], errors='coerce')
    it['nb_views'] = pd.to_numeric(it['nb_views'], errors='coerce')
    it['difficulty'] = it['difficulty'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}).fillna(2)
    us['job'] = clean(us['job'])
    ex['ts'] = pd.to_datetime(ex['created_at'])
    # keep users/items present in explicit ratings
    uids = sorted(ex.user_id.unique()); iids = sorted(ex.item_id.unique())
    umap = {u: i for i, u in enumerate(uids)}; imap = {i: k for k, i in enumerate(iids)}
    ex['user'] = ex.user_id.map(umap); ex['course'] = ex.item_id.map(imap)
    it = it[it.item_id.isin(iids)].copy(); it['course'] = it.item_id.map(imap)
    it = it.sort_values('course').reset_index(drop=True)
    it['cat_id'] = it['theme'].astype('category').cat.codes
    it['text'] = (it['name'].fillna('') + ' ' + it['description'].fillna('')).str.slice(0, 1500)
    us = us[us.user_id.isin(uids)].copy(); us['user'] = us.user_id.map(umap)
    us = us.sort_values('user').reset_index(drop=True)
    inter = ex[['user', 'course', 'ts', 'rating']].copy()
    return inter, it, us

# ---------------------------------------------------------------- splits ----
def split_interactions(inter, kind='random', seed=0, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    if kind == 'random':
        p = rng.permutation(len(inter))
        n1 = int(ratios[0] * len(inter)); n2 = n1 + int(ratios[1] * len(inter))
        return inter.iloc[p[:n1]], inter.iloc[p[n1:n2]], inter.iloc[p[n2:]]
    if kind == 'temporal':
        s = inter.sort_values('ts')
        n1 = int(ratios[0] * len(s)); n2 = n1 + int(ratios[1] * len(s))
        return s.iloc[:n1], s.iloc[n1:n2], s.iloc[n2:]
    if kind == 'user_disjoint':
        us = np.array(sorted(inter.user.unique())); rng.shuffle(us)
        n1 = int(ratios[0] * len(us)); n2 = n1 + int(ratios[1] * len(us))
        tr_u, va_u, te_u = set(us[:n1]), set(us[n1:n2]), set(us[n2:])
        return (inter[inter.user.isin(tr_u)], inter[inter.user.isin(va_u)],
                inter[inter.user.isin(te_u)])
    raise ValueError(kind)

def make_pairs(tr, va, te, n_items, neg_ratio=2, seed=0, pos_label=1.0):
    """Add negative samples (not in the user's history) for each split."""
    rng = np.random.default_rng(seed)
    hist = tr.groupby('user')['course'].apply(set).to_dict()
    all_pos = pd.concat([tr, va, te]).groupby('user')['course'].apply(set).to_dict()
    out = []
    for split in (tr, va, te):
        rows = []
        for u, c in zip(split.user.values, split.course.values):
            rows.append((u, c, 1))
            seen = all_pos.get(u, set())
            k = 0
            while k < neg_ratio:
                cn = rng.integers(0, n_items)
                if cn not in seen:
                    rows.append((u, cn, 0)); k += 1
        out.append(pd.DataFrame(rows, columns=['user', 'course', 'y']))
    return out[0], out[1], out[2]

# ------------------------------------------------------- feature builder ----
NUM_CLIP = 1e-6

class FeatureBuilder:
    """Leakage-safe pair features organised in three views.
    View 1 (interaction), View 2 (attribute), View 3 (text)."""
    def __init__(self, svd_dim=16):
        self.svd_dim = svd_dim

    def _text_matrix(self, texts):
        self.tfidf = TfidfVectorizer(max_features=6000, analyzer='char_wb',
                                     ngram_range=(2, 4), min_df=2)
        T = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.svd_dim, T.shape[1] - 1), random_state=0)
        Z = self.svd.fit_transform(T)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z

    def fit(self, tr_pairs, tr_inter, courses, users=None):
        """Fit all statistics on the TRAIN split only."""
        self.n_items = len(courses)
        self.Z = self._text_matrix(courses.text.tolist())
        self.item_cat = courses['cat_id'].values if 'cat_id' in courses else np.zeros(len(courses), int)
        # category positive-rate target encoding
        df = tr_pairs.copy(); df['cat'] = self.item_cat[df.course.values]
        g = df.groupby('cat').y.agg(['mean', 'count'])
        glob = df.y.mean()
        self.cat_te = (g['mean'] * g['count'] + glob * 25) / (g['count'] + 25)
        self.cat_te_default = float(glob)
        # user stats from positive train interactions
        pos = tr_pairs[tr_pairs.y == 1]
        self.u_cnt = pos.groupby('user').size()
        self.u_last = tr_inter.groupby('user').ts.max() if tr_inter is not None else None
        self.u_first = tr_inter.groupby('user').ts.min() if tr_inter is not None else None
        self.i_cnt = pos.groupby('course').size()
        self.u_cat = pos.assign(cat=self.item_cat[pos.course.values]).groupby(['user', 'cat']).size()
        self.u_cat_tot = self.u_cat.groupby('user').sum()
        self.glob_pop = np.log1p(np.bincount(pos.course.values, minlength=self.n_items))
        # optional numeric course attributes (MARS)
        self.attrs = {}
        for col in ['difficulty', 'duration', 'nb_views']:
            if col in courses.columns:
                v = courses[col].astype(float).values
                self.attrs[col] = v
        # categorical target encodings for item attribute columns
        self.item_attr_te = {}
        for col in ['itype', 'software', 'theme']:
            if col in courses.columns:
                codes = courses[col].astype('category').cat.codes.values
                df['a'] = codes[df.course.values]
                g = df.groupby('a').y.agg(['mean', 'count'])
                self.item_attr_te[col] = (codes, ((g['mean'] * g['count'] + glob * 25) / (g['count'] + 25)).to_dict(), float(glob))
        # user job target encoding (MARS)
        self.job_te = None
        if users is not None and 'job' in users.columns:
            codes = users['job'].astype('category')
            self.job_cat = {u: c for u, c in zip(users.user.values, codes.cat.codes.values)}
            df['j'] = df.user.map(self.job_cat).fillna(-1).astype(int)
            g = df[df.j >= 0].groupby('j').y.agg(['mean', 'count'])
            self.job_te = ((g['mean'] * g['count'] + glob * 25) / (g['count'] + 25)).to_dict()
            self.job_te_default = float(glob)
        # reference time for recency
        self.t_ref = tr_inter.ts.max() if tr_inter is not None else pd.Timestamp('2018-01-01')
        n_users_dense = int(tr_pairs.user.max()) + 1
        if self.u_last is not None:
            la = np.zeros(n_users_dense, dtype='datetime64[ns]'); la[:] = np.datetime64(self.t_ref)
            fa = np.zeros(n_users_dense, dtype='datetime64[ns]'); fa[:] = np.datetime64(self.t_ref)
            idx = self.u_last.index.values.astype(int); la[idx] = self.u_last.values
            fa[self.u_first.index.values.astype(int)] = self.u_first.values
            self.u_last_arr = la; self.u_first_arr = fa
        self.ucat_dict = {k: v for k, v in self.u_cat.items()}
        self.ucattot_dict = self.u_cat_tot.to_dict()
        return self

def fb_fit(fb, tr_pairs, tr_inter, courses, users=None):
    fb.fit(tr_pairs, tr_inter, courses, users)
    fb.texts = courses.text.tolist()
    # dense user-text profile matrix (vectorised lookup at transform time)
    d = fb.Z.shape[1]
    n_users = max(tr_pairs.user.max() + 1, (users.user.max() + 1) if users is not None else 0)
    n_users = int(max(n_users, tr_pairs.user.max() + 1))
    U = np.zeros((n_users, d), dtype=np.float32)
    pos = tr_pairs[tr_pairs.y == 1]
    cnt = np.bincount(pos.user.values, minlength=n_users).astype(np.float32)
    np.add.at(U, np.repeat(pos.user.values, 1), fb.Z[pos.course.values])
    U = U / np.maximum(cnt, 1)[:, None]
    fb.Utext = U
    fb.tlen_arr = np.log1p(np.array([len(t) for t in fb.texts], float))
    return fb

def _map_dict(arr, mp, default=0.0):
    if isinstance(mp, pd.Series): mp = mp.to_dict()
    return np.array([mp.get(int(x), default) for x in arr], dtype=np.float64)

def fb_transform(fb, pairs):
    u = pairs.user.values.astype(np.int64); c = pairs.course.values.astype(np.int64)
    cat = fb.item_cat[c]
    u_cnt = np.log1p(_map_dict(u, fb.u_cnt))
    i_pop = fb.glob_pop[c]
    ucat = np.array([fb.ucat_dict.get(k, 0) for k in zip(u.tolist(), cat.tolist())], float)
    utot = _map_dict(u, fb.ucattot_dict)
    affinity = ucat / (utot + 1.0)
    uu = np.minimum(u, len(fb.u_last_arr) - 1) if hasattr(fb, 'u_last_arr') else u
    if hasattr(fb, 'u_last_arr'):
        last_days = (np.datetime64(fb.t_ref) - fb.u_last_arr[uu]) / np.timedelta64(1, 'D')
        span_days = (fb.u_last_arr[uu] - fb.u_first_arr[uu]) / np.timedelta64(1, 'D')
    else:
        last_days = np.zeros(len(u)); span_days = np.zeros(len(u))
    recency = np.log1p(np.clip(last_days, 0, None))
    span = np.log1p(np.clip(span_days, 0, None))
    cat_te = _map_dict(cat, fb.cat_te, fb.cat_te_default)
    feats_v2 = [cat_te]
    for col, v in fb.attrs.items():
        feats_v2.append(np.log1p(np.nan_to_num(v.astype(float)[c])))
    for col, (codes, mp, dft) in fb.item_attr_te.items():
        feats_v2.append(_map_dict(codes[c], mp, dft))
    if fb.job_te is not None:
        jcodes = np.array([fb.job_cat.get(int(x), -1) for x in u])
        feats_v2.append(_map_dict(jcodes, fb.job_te, fb.job_te_default))
    sim = np.einsum('ij,ij->i', fb.Utext[np.minimum(u, fb.Utext.shape[0] - 1)], fb.Z[c])
    tlen = fb.tlen_arr[c]
    V1 = np.stack([u_cnt, i_pop, affinity, recency, span], 1).astype(np.float32)
    V2 = np.stack(feats_v2, 1).astype(np.float32)
    V3 = np.stack([sim, tlen], 1).astype(np.float32)
    return V1, V2, V3

FEAT_NAMES_V1 = ['u_n_enroll', 'i_popularity', 'u_cat_affinity', 'u_recency', 'u_span']
FEAT_NAMES_V3 = ['text_sim', 'text_len']

def pack_views(V1, V2, V3, v2_names):
    X = np.concatenate([V1, V2, V3], 1)
    names = FEAT_NAMES_V1 + v2_names + FEAT_NAMES_V3
    views = (slice(0, V1.shape[1]),
             slice(V1.shape[1], V1.shape[1] + V2.shape[1]),
             slice(V1.shape[1] + V2.shape[1], X.shape[1]))
    return X, names, views

# ------------------------------------------------------------------ EW-GRA --
class EWGRA:
    """Entropy-weighted Grey Relational Analysis against a label-aware ideal
    reference profile. Produces the relational degree rho, per-feature
    coefficients xi, and sample weights w = rho ** gamma."""
    def __init__(self, zeta=0.5, gamma=1.0):
        self.zeta = zeta; self.gamma = gamma

    def fit(self, X, y):
        self.lo = X.min(0); self.hi = X.max(0)
        Xn = self._norm(X)
        # entropy weights (constant features carry no information -> zero weight)
        P = Xn / (Xn.sum(0, keepdims=True) + 1e-12)
        e = -(P * np.log(P + 1e-12)).sum(0) / math.log(len(Xn))
        d = (1 - e) * (X.std(0) > 1e-8)
        self.w_ = d / (d.sum() + 1e-12)
        # label-aware reference direction
        yc = y - y.mean()
        corr = np.array([np.corrcoef(Xn[:, k], yc)[0, 1] if Xn[:, k].std() > 0 else 0
                         for k in range(Xn.shape[1])])
        corr = np.nan_to_num(corr)
        self.ref = (corr > 0).astype(float)
        D = np.abs(self.ref[None, :] - Xn)
        self.dmin = float(D.min()); self.dmax = float(D.max())
        return self

    def _norm(self, X):
        return (X - self.lo) / (self.hi - self.lo + 1e-9)

    def xi(self, X):
        Xn = self._norm(X)
        D = np.abs(self.ref[None, :] - Xn)
        return (self.dmin + self.zeta * self.dmax) / (D + self.zeta * self.dmax + 1e-12)

    def degree(self, X):
        return (self.xi(X) * self.w_[None, :]).sum(1)

    def weight(self, X, gamma=None):
        g = self.gamma if gamma is None else gamma
        return np.power(np.clip(self.degree(X), 1e-6, 1.0), g)

# ------------------------------------------------------------------ models --
class MLPModel(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, 1))
    def forward(self, X, u=None, i=None):
        return torch.sigmoid(self.net(X)).squeeze(-1)

class WideDeep(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.wide = nn.Linear(d, 1)
        self.deep = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))
    def forward(self, X, u=None, i=None):
        return torch.sigmoid(self.wide(X) + self.deep(X)).squeeze(-1)

class NCF(nn.Module):
    def __init__(self, d, nu, ni, emb=32):
        super().__init__()
        self.p = nn.Embedding(nu, emb); self.q = nn.Embedding(ni, emb)
        self.mlp_u = nn.Embedding(nu, emb); self.mlp_i = nn.Embedding(ni, emb)
        self.mlp = nn.Sequential(nn.Linear(2 * emb, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.out = nn.Linear(emb + 32, 1)
    def forward(self, X, u, i):
        gmf = self.p(u) * self.q(i)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], -1))
        return torch.sigmoid(self.out(torch.cat([gmf, mlp], -1))).squeeze(-1)

class LightGCN(nn.Module):
    def __init__(self, d, nu, ni, emb=48, layers=3):
        super().__init__()
        self.nu, self.ni, self.layers = nu, ni, layers
        self.eu = nn.Embedding(nu, emb); self.ei = nn.Embedding(ni, emb)
        nn.init.normal_(self.eu.weight, std=0.05); nn.init.normal_(self.ei.weight, std=0.05)
        self.register_buffer('adj', torch.sparse_coo_tensor((2, 2)))  # set later
    def set_graph(self, u, i):
        n_u, n_i = self.nu, self.ni
        row = np.concatenate([u, i + n_u]); col = np.concatenate([i + n_u, u])
        deg = np.bincount(row, minlength=n_u + n_i).astype(np.float64)
        dinv = np.power(deg + 1e-9, -0.5)
        val = dinv[row] * dinv[col]
        A = torch.sparse_coo_tensor(torch.tensor(np.stack([row, col])),
                                    torch.tensor(val, dtype=torch.float32),
                                    (n_u + n_i, n_i + n_u)).coalesce().to_sparse_csr()
        self.adj = A
    def _prop(self):
        E = torch.cat([self.eu.weight, self.ei.weight], 0)
        out = [E]
        for _ in range(self.layers):
            E = torch.sparse.mm(self.adj, E); out.append(E)
        E = torch.stack(out, 0).mean(0)
        return E[:self.nu], E[self.nu:]
    def forward(self, X, u, i):
        eu, ei = self._prop()
        return torch.sigmoid((eu[u] * ei[i]).sum(-1))

class SGL(LightGCN):
    def ssl_loss(self, u, i, tau=0.2, lam=0.1):
        def perturbed():
            E = torch.cat([self.eu.weight, self.ei.weight], 0)
            for _ in range(self.layers):
                E = torch.sparse.mm(self.adj, E)
                E = Fn.dropout(E, 0.1, training=True)
            return E[:self.nu], E[self.nu:]
        u1, i1 = perturbed(); u2, i2 = perturbed()
        def infonce(a, b):
            a = Fn.normalize(a, dim=-1); b = Fn.normalize(b, dim=-1)
            pos = (a * b).sum(-1) / tau
            neg = a @ b.T / tau
            return -torch.mean(pos - torch.logsumexp(neg, dim=-1))
        idx = torch.randperm(self.nu)[:4096]
        jdx = torch.randperm(self.ni)[:2048]
        return lam * (infonce(u1[idx], u2[idx]) + infonce(i1[jdx], i2[jdx]))

class GRAMLOF(nn.Module):
    """Faithful re-implementation of the baseline paper's core idea:
    features augmented with the GRA relational degree feed an MLP."""
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, 1))
    def forward(self, X, u=None, i=None, rho=None):
        return torch.sigmoid(self.net(torch.cat([X, rho[:, None]], -1))).squeeze(-1)

class GRCF(nn.Module):
    """Our model: dual-gate view encoders + ID factorisation + evidential head
    + cross-view contrastive alignment."""
    def __init__(self, views_dims, nu, ni, emb=32, vh=48, lam_cl=0.1, drop=0.1,
                 use_cl=True, use_gate=True, use_evd=True, use_views=(True, True, True)):
        super().__init__()
        self.use_cl, self.use_gate, self.use_evd = use_cl, use_gate, use_evd
        self.use_views = use_views
        self.lam_cl = lam_cl
        self.encs = nn.ModuleList([nn.Sequential(nn.Linear(d, vh), nn.ReLU(), nn.Dropout(drop),
                                                 nn.Linear(vh, emb))
                                   for d in views_dims])
        self.gates = nn.ModuleList([nn.Linear(d, emb) for d in views_dims])
        self.p = nn.Embedding(nu, emb); self.q = nn.Embedding(ni, emb)
        nn.init.normal_(self.p.weight, std=0.05); nn.init.normal_(self.q.weight, std=0.05)
        self.proj = nn.ModuleList([nn.Linear(emb, emb) for _ in views_dims])
        self.head = nn.Sequential(nn.Linear(2 * emb + 1, 64), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(64, 2))
    def _embed(self, X, views, u, i):
        es, gs = [], []
        for vi, sl in enumerate(views):
            if not self.use_views[vi]:
                es.append(torch.zeros(X.shape[0], self.p.weight.shape[1], device=X.device)); continue
            e = self.encs[vi](X[:, sl])
            if self.use_gate:
                g = torch.sigmoid(self.gates[vi](X[:, sl]))
            else:
                g = torch.ones_like(e)
            es.append(e); gs.append(g)
        wsum = torch.stack(gs, 0).sum(0) + 1e-9
        z = sum(g * e for g, e in zip(gs, es)) / wsum
        return z, es
    def forward(self, X, u, i, views=None, rho=None, ret_all=False):
        z, es = self._embed(X, views, u, i)
        idp = self.p(u) * self.q(i)
        r = rho[:, None] if rho is not None else torch.zeros(X.shape[0], 1)
        h = torch.cat([z, idp, r], -1)
        e = Fn.softplus(self.head(h))
        alpha = e + 1.0                      # class order: [negative, positive]
        S = alpha.sum(-1)
        p = alpha[:, 1] / S
        unc = 2.0 / S
        if ret_all:
            return p, unc, es, alpha
        return p
    def cl_loss(self, es, tau=0.2, max_n=768):
        if not self.use_cl or self.lam_cl == 0:
            return 0.0
        if es[0].shape[0] > max_n:
            idx = torch.randperm(es[0].shape[0], device=es[0].device)[:max_n]
            es = [e[idx] for e in es]
        hs = [Fn.normalize(pr(e), dim=-1) for pr, e in zip(self.proj, es)]
        loss, cnt = 0.0, 0
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                if not (self.use_views[a] and self.use_views[b]): continue
                pos = (hs[a] * hs[b]).sum(-1) / tau
                neg = hs[a] @ hs[b].T / tau
                loss = loss - torch.mean(pos - torch.logsumexp(neg, dim=-1)); cnt += 1
        return self.lam_cl * loss / max(cnt, 1)

def edl_loss(alpha, y, w=None, lam_kl=0.1, epoch=1):
    """Evidential classification loss (Sensoy et al. 2018) with KL annealing."""
    S = alpha.sum(-1, keepdim=True)
    yoh = torch.stack([1 - y, y], -1)
    A = ((yoh - alpha / S) ** 2).sum(-1) + (alpha * (S - alpha) / (S * S * (S + 1))).sum(-1)
    kl_a = alpha * (1 - yoh) + 1.0
    kl = torch.lgamma(kl_a.sum(-1)) - torch.lgamma(torch.tensor(2.0)) \
        - (torch.lgamma(kl_a)).sum(-1) + ((kl_a - 1) * (torch.digamma(kl_a) - torch.digamma(kl_a.sum(-1, keepdim=True)))).sum(-1)
    anneal = min(1.0, epoch / 10.0)
    L = A + lam_kl * anneal * kl
    if w is not None:
        L = L * w
    return L.mean()

# --------------------------------------------------------------- training ---
def torch_data(X, y, u, i, sw=None, rho=None, bs=16384, shuffle=True):
    t = [torch.tensor(X), torch.tensor(y, dtype=torch.float32),
         torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long)]
    if sw is not None:
        t.append(torch.tensor(sw, dtype=torch.float32))
    if rho is not None:
        t.append(torch.tensor(rho, dtype=torch.float32))
    return DataLoader(TensorDataset(*t), batch_size=bs, shuffle=shuffle)

def train_torch(model, dat, views, seed=0, epochs=12, patience=3, lr=1e-3,
                sw_tr=None, rho=None, lam_kl=0.05, verbose=False, bs=None):
    """rho: optional dict with 'tr'/'va' relational degrees (GRCF evidential head)."""
    set_seed(seed)
    (Xtr, ytr, utr, itr), (Xva, yva, uva, iva), _ = dat
    big = len(ytr) > 300000
    if bs is None:
        if isinstance(model, LightGCN):
            bs = 65536 if big else 2048   # full-graph propagation per batch is costly
        else:
            bs = 8192 if big else 2048
    if isinstance(model, LightGCN) and big:
        if isinstance(model, SGL):
            epochs, patience = max(epochs, 20), max(patience, 4)
        else:
            epochs, patience = max(epochs, 30), max(patience, 5)
    rtr = rho['tr'] if (rho is not None and isinstance(model, GRCF)) else None
    rva = rho['va'] if (rho is not None and isinstance(model, GRCF)) else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    best, bad, best_state = -1, 0, None
    for ep in range(epochs):
        model.train()
        for batch in torch_data(Xtr, ytr, utr, itr, sw_tr, rtr, bs=bs):
            Xb, yb, ub, ib = batch[0], batch[1], batch[2], batch[3]
            wb, rb = None, None
            k = 4
            if sw_tr is not None:
                wb = batch[k]; k += 1
            if rtr is not None:
                rb = batch[k]; k += 1
            opt.zero_grad()
            if isinstance(model, (LightGCN,)):
                p = model(Xb, ub, ib)
                loss = Fn.binary_cross_entropy(p, yb, weight=wb)
                if isinstance(model, SGL):
                    loss = loss + model.ssl_loss(ub, ib)
            elif isinstance(model, GRAMLOF):
                raise RuntimeError('use train_gramlof')
            elif isinstance(model, GRCF):
                p, unc, es, alpha = model(Xb, ub, ib, views, rho=rb, ret_all=True)
                loss = edl_loss(alpha, yb, wb, lam_kl=lam_kl, epoch=ep + 1) + model.cl_loss(es)
            else:
                p = model(Xb, ub, ib)
                loss = Fn.binary_cross_entropy(p, yb, weight=wb)
            loss.backward(); opt.step()
        va_auc = evaluate(model, (Xva, yva, uva, iva), views, rho=rva)['auc']
        if verbose: print(f'  ep{ep} val_auc={va_auc:.4f}')
        if va_auc > best:
            best, bad = va_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, time.time() - t0

def train_gramlof(model, dat, rho, seed=0, epochs=12, patience=3, lr=1e-3, sw_tr=None):
    set_seed(seed)
    (Xtr, ytr, utr, itr), (Xva, yva, uva, iva), _ = dat
    bs = 8192 if len(ytr) > 300000 else 2048
    rtr, rva = rho['tr'], rho['va']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time(); best, bad, best_state = -1, 0, None
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32),
                       torch.tensor(rtr, dtype=torch.float32),
                       torch.tensor(sw_tr, dtype=torch.float32) if sw_tr is not None else torch.ones(len(ytr)))
    for ep in range(epochs):
        model.train()
        for Xb, yb, rb, wb in DataLoader(ds, batch_size=bs, shuffle=True):
            opt.zero_grad()
            p = model(Xb, None, None, rho=rb)
            loss = Fn.binary_cross_entropy(p, yb, weight=wb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(Xva), None, None, rho=torch.tensor(rva, dtype=torch.float32))
        va_auc = roc_auc_score(yva, pv.numpy())
        if va_auc > best: best, bad = va_auc, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, time.time() - t0

@torch.no_grad()
def predict_torch(model, dat_split, views, bs=65536, rho=None):
    X, y, u, i = dat_split
    model.eval(); ps = []
    for s in range(0, len(X), bs):
        Xb = torch.tensor(X[s:s + bs]); ub = torch.tensor(u[s:s + bs], dtype=torch.long)
        ib = torch.tensor(i[s:s + bs], dtype=torch.long)
        if isinstance(model, GRAMLOF):
            p = model(Xb, None, None, rho=torch.tensor(rho[s:s + bs], dtype=torch.float32))
        elif isinstance(model, GRCF):
            rb = torch.tensor(rho[s:s + bs], dtype=torch.float32) if rho is not None else None
            p = model(Xb, ub, ib, views, rho=rb)
        else:
            p = model(Xb, ub, ib)
        ps.append(p.cpu().numpy())
    return np.concatenate(ps)

def evaluate(model, dat_split, views, rho=None):
    X, y, u, i = dat_split
    p = predict_torch(model, dat_split, views, rho=rho)
    return dict(auc=float(roc_auc_score(y, p)),
                f1=float(f1_score(y, p > 0.5)),
                brier=float(brier_score_loss(y, p)))

def ece(y, p, M=10):
    bins = np.linspace(0, 1, M + 1); out = 0.0
    for b in range(M):
        m = (p > bins[b]) & (p <= bins[b + 1]) if b else (p >= 0) & (p <= bins[1])
        if m.sum() > 0:
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)

# ------------------------------------------------------------ checkpoint ----
def ckpt(name):
    fp = os.path.join(RES, name)
    return json.load(open(fp)) if os.path.exists(fp) else {}

def save_ck(name, obj):
    json.dump(obj, open(os.path.join(RES, name), 'w'))
