#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GRCF experiment runner. All stages checkpoint to results/*.json and are
resumable. Usage: python run_experiments.py [main|robustness|ablation|...|all]
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grcf_lib as G
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, log_loss
from scipy.optimize import minimize_scalar

CACHE = os.path.join(G.WORK, 'cache'); os.makedirs(CACHE, exist_ok=True)
METHODS = ['LR', 'RF', 'MLP', 'WideDeep', 'NCF', 'LightGCN', 'SGL', 'GRA-MLOF', 'GRCF']

# ---------------------------------------------------------------- context ---
def build_context(ds, kind, seed):
    """Returns dict with splits, features, views, EW-GRA weights. Cached."""
    fp = os.path.join(CACHE, f'{ds}_{kind}_{seed}.pkl')
    if os.path.exists(fp):
        ctx = pickle.load(open(fp, 'rb'))
        # sample weights always (re)derived with the default gamma = 0.5
        ctx['sw'] = ctx['gra'].weight(ctx['dat'][0][0], gamma=0.5).astype(np.float32)
        return ctx
    G.set_seed(seed)
    if ds == 'XTX':
        inter, courses, users = G.load_xtx()
        pos_label = None
    else:
        inter, courses, users = G.load_mars()
        inter['y_true'] = (inter.rating >= 6).astype(int)
    tr_i, va_i, te_i = G.split_interactions(inter, kind, seed)
    if ds == 'XTX':
        trp, vap, tep = G.make_pairs(tr_i, va_i, te_i, len(courses), neg_ratio=2, seed=seed)
    else:
        trp = tr_i[['user', 'course']].assign(y=tr_i.y_true.values)
        vap = va_i[['user', 'course']].assign(y=va_i.y_true.values)
        tep = te_i[['user', 'course']].assign(y=te_i.y_true.values)
    fb = G.fb_fit(G.FeatureBuilder(), trp, tr_i, courses, users)
    V = {}
    for nm, sp in [('tr', trp), ('va', vap), ('te', tep)]:
        V1, V2, V3 = G.fb_transform(fb, sp)
        V[nm] = (V1, V2, V3)
    v2_names = ['cat_te']
    for col in fb.attrs: v2_names.append(col)
    for col in fb.item_attr_te: v2_names.append(col + '_te')
    if fb.job_te is not None: v2_names.append('job_te')
    Xs, views = {}, None
    names = None
    for nm in V:
        X, names, views = G.pack_views(V[nm][0], V[nm][1], V[nm][2], v2_names)
        Xs[nm] = X
    def pack(sp, X):
        return (X, sp.y.values.astype(np.float32),
                sp.user.values.astype(np.int64), sp.course.values.astype(np.int64))
    dat = (pack(trp, Xs['tr']), pack(vap, Xs['va']), pack(tep, Xs['te']))
    gra = G.EWGRA(zeta=0.5, gamma=0.5).fit(Xs['tr'], trp.y.values.astype(np.float32))
    sw = gra.weight(Xs['tr'], gamma=0.5).astype(np.float32)
    rho = {nm: gra.degree(Xs[nm]).astype(np.float32) for nm in Xs}
    nu = int(max(trp.user.max(), vap.user.max(), tep.user.max())) + 1
    ni = len(courses)
    ctx = dict(ds=ds, kind=kind, seed=seed, dat=dat, views=views, names=names,
               gra=gra, sw=sw, rho=rho, nu=nu, ni=ni, fb=fb, v2names=v2_names,
               tr_inter=tr_i, tep=tep, trp=trp, courses=courses,
               vdims=(Xs['tr'][:, views[0]].shape[1], Xs['tr'][:, views[1]].shape[1],
                      Xs['tr'][:, views[2]].shape[1]))
    pickle.dump(ctx, open(fp, 'wb'))
    return ctx

def ep_budget(ctx):
    """Epoch cap / early-stop patience. The evidential and neural models
    converge more slowly on the smaller, class-imbalanced MARS data, so all
    methods receive a larger (still identical) budget there."""
    return (20, 4) if ctx['ds'] == 'MARS' else (12, 3)

def get_model(m, ctx, variant='full'):
    d = ctx['dat'][0][0].shape[1]; nu, ni = ctx['nu'], ctx['ni']
    if m == 'MLP': return G.MLPModel(d)
    if m == 'WideDeep': return G.WideDeep(d)
    if m == 'NCF': return G.NCF(d, nu, ni)
    if m in ('LightGCN', 'SGL'):
        mod = (G.LightGCN if m == 'LightGCN' else G.SGL)(d, nu, ni)
        pos = ctx['trp'][ctx['trp'].y == 1]
        mod.set_graph(pos.user.values, pos.course.values)
        return mod
    if m == 'GRA-MLOF': return G.GRAMLOF(d)
    if m == 'GRCF':
        kw = dict(use_cl=True, use_gate=True, use_evd=True, use_views=(True, True, True))
        if variant == 'no_cl': kw['use_cl'] = False
        if variant == 'no_gate': kw['use_gate'] = False
        if variant == 'no_evd': kw['use_evd'] = False
        if variant == 'no_v1': kw['use_views'] = (False, True, True)
        if variant == 'no_v2': kw['use_views'] = (True, False, True)
        if variant == 'no_v3': kw['use_views'] = (True, True, False)
        return G.GRCF(ctx['vdims'], nu, ni, **kw)
    raise ValueError(m)

def run_one(m, ctx, seed):
    """Train + evaluate one method. Returns metrics dict."""
    dat, views = ctx['dat'], ctx['views']
    t0 = time.time()
    if m in ('LR', 'RF'):
        from sklearn.preprocessing import StandardScaler
        X, y = dat[0][0], dat[0][1]
        if m == 'LR':
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(sc.transform(X), y)
            p = clf.predict_proba(sc.transform(dat[2][0]))[:, 1]
            tt = time.time() - t0
            yv = dat[2][1]
            return dict(auc=float(roc_auc_score(yv, p)), f1=float(f1_score(yv, p > 0.5)),
                        brier=float(brier_score_loss(yv, p)),
                        nll=float(log_loss(yv, np.clip(p, 1e-7, 1 - 1e-7))),
                        ece=G.ece(yv, p), time=tt)
        clf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=2, random_state=seed)
        clf.fit(X, y)
        p = clf.predict_proba(dat[2][0])[:, 1]
        tt = time.time() - t0
    elif m == 'GRA-MLOF':
        mod = get_model(m, ctx)
        ep, pa = ep_budget(ctx)
        mod, tt = G.train_gramlof(mod, dat, ctx['rho'], seed=seed, sw_tr=ctx['sw'],
                                  epochs=ep, patience=pa)
        p = G.predict_torch(mod, dat[2], views, rho=ctx['rho']['te'])
    else:
        mod = get_model(m, ctx)
        ep, pa = ep_budget(ctx)
        sw = ctx['sw'] if m == 'GRCF' else None
        rho = ctx['rho'] if m == 'GRCF' else None
        mod, tt = G.train_torch(mod, dat, views, seed=seed, sw_tr=sw, rho=rho,
                                epochs=ep, patience=pa)
        p = G.predict_torch(mod, dat[2], views,
                            rho=ctx['rho']['te'] if m == 'GRCF' else None)
    y = dat[2][1]
    return dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p > 0.5)),
                brier=float(brier_score_loss(y, p)),
                nll=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
                ece=G.ece(y, p), time=tt)

# ------------------------------------------------------------------ stages --
def stage_main():
    res = G.ckpt('main.json')
    for ds in ['XTX', 'MARS']:
        for seed in [0, 1, 2]:
            ctx = build_context(ds, 'random', seed)
            rk = res.setdefault(ds, {}).setdefault(str(seed), {})
            for m in METHODS:
                if m in rk: continue
                rk[m] = run_one(m, ctx, seed)
                G.save_ck('main.json', res)
                print('[main]', ds, seed, m, round(rk[m]['auc'], 4), flush=True)

def stage_robustness():
    res = G.ckpt('robustness.json')
    for kind in ['user_disjoint', 'temporal']:
        for seed in [0, 1]:
            ctx = build_context('XTX', kind, seed)
            rk = res.setdefault(kind, {}).setdefault(str(seed), {})
            for m in ['MLP', 'NCF', 'LightGCN', 'SGL', 'GRCF']:
                if m in rk: continue
                rk[m] = run_one(m, ctx, seed)
                G.save_ck('robustness.json', res)
                print('[rob]', kind, seed, m, round(rk[m]['auc'], 4), flush=True)

def stage_ablation():
    res = G.ckpt('ablation.json')
    variants = ['full', 'no_cl', 'no_gate', 'no_evd', 'no_v1', 'no_v2', 'no_v3', 'no_ewgra']
    for seed in [0, 1]:
        ctx = build_context('XTX', 'random', seed)
        for v in variants:
            key = f'{v}|{seed}'
            if key in res: continue
            mod = get_model('GRCF', ctx, v)
            sw = None if v == 'no_ewgra' else ctx['sw']
            rho = None if v == 'no_ewgra' else ctx['rho']
            mod, tt = G.train_torch(mod, ctx['dat'], ctx['views'], seed=seed,
                                    sw_tr=sw, rho=rho)
            p = G.predict_torch(mod, ctx['dat'][2], ctx['views'],
                                rho=None if v == 'no_ewgra' else ctx['rho']['te'])
            y = ctx['dat'][2][1]
            res[key] = dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p > 0.5)))
            G.save_ck('ablation.json', res)
            print('[abl]', key, round(res[key]['auc'], 4), flush=True)

def stage_sensitivity():
    res = G.ckpt('sens.json')
    ctx = build_context('XTX', 'random', 0)
    Xtr, ytr = ctx['dat'][0][0], ctx['dat'][0][1]
    Xva, Xte = ctx['dat'][1][0], ctx['dat'][2][0]
    for zeta in [0.3, 0.5, 0.7, 0.9]:
        gra = G.EWGRA(zeta=zeta, gamma=0.5).fit(Xtr, ytr)
        rho_s = {'tr': gra.degree(Xtr).astype(np.float32),
                 'va': gra.degree(Xva).astype(np.float32),
                 'te': gra.degree(Xte).astype(np.float32)}
        for lam in [0.0, 0.1, 0.2]:
            key = f'{zeta}|{lam}'
            if key in res: continue
            mod = G.GRCF(ctx['vdims'], ctx['nu'], ctx['ni'], lam_cl=lam)
            mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=0,
                                   sw_tr=gra.weight(Xtr, gamma=0.5).astype(np.float32),
                                   rho=rho_s, epochs=10, patience=2)
            p = G.predict_torch(mod, ctx['dat'][2], ctx['views'], rho=rho_s['te'])
            y = ctx['dat'][2][1]
            res[key] = dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p > 0.5)))
            G.save_ck('sens.json', res)
            print('[sens]', key, round(res[key]['auc'], 4), flush=True)
    for gamma in [0.5, 1.0, 2.0]:
        key = f'g|{gamma}'
        if key in res: continue
        mod = G.GRCF(ctx['vdims'], ctx['nu'], ctx['ni'])
        mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=0,
                               sw_tr=ctx['gra'].weight(Xtr, gamma=gamma).astype(np.float32),
                               rho=ctx['rho'], epochs=10, patience=2)
        p = G.predict_torch(mod, ctx['dat'][2], ctx['views'], rho=ctx['rho']['te'])
        y = ctx['dat'][2][1]
        res[key] = dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p > 0.5)))
        G.save_ck('sens.json', res)
        print('[sens]', key, round(res[key]['auc'], 4), flush=True)

def stage_coldstart():
    res = G.ckpt('coldstart.json')
    bins = [(0, 0, '0 (cold)'), (1, 3, '1-3'), (4, 10, '4-10'), (11, 10 ** 9, '>10')]
    for seed in [0, 1]:
        ctx = build_context('XTX', 'random', seed)
        tep = ctx['tep']
        ntr = ctx['trp'][ctx['trp'].y == 1].groupby('user').size()
        n_tr = np.array([ntr.get(x, 0) for x in tep.user.values])
        y = ctx['dat'][2][1]
        for m in ['MLP', 'NCF', 'LightGCN', 'GRCF']:
            mod = get_model(m, ctx)
            if m == 'GRCF':
                mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=seed,
                                       sw_tr=ctx['sw'], rho=ctx['rho'])
            else:
                mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=seed)
            p = G.predict_torch(mod, ctx['dat'][2], ctx['views'],
                                rho=ctx['rho']['te'] if m == 'GRCF' else None)
            for lo, hi, lab in bins:
                mk = (n_tr >= lo) & (n_tr <= hi)
                if mk.sum() < 50 or len(np.unique(y[mk])) < 2: continue
                res.setdefault(f'{m}|{lab}', []).append(float(roc_auc_score(y[mk], p[mk])))
        G.save_ck('coldstart.json', res)
        print('[cold]', seed, flush=True)

def stage_calibration():
    res = G.ckpt('calib.json')
    ctx = build_context('XTX', 'random', 0)
    probs = {}
    for m in ['MLP', 'NCF', 'GRCF']:
        if m in res: continue
        mod = get_model(m, ctx)
        mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=0,
                               sw_tr=ctx['sw'] if m == 'GRCF' else None,
                               rho=ctx['rho'] if m == 'GRCF' else None)
        pv = G.predict_torch(mod, ctx['dat'][1], ctx['views'],
                             rho=ctx['rho']['va'] if m == 'GRCF' else None)
        pt = G.predict_torch(mod, ctx['dat'][2], ctx['views'],
                             rho=ctx['rho']['te'] if m == 'GRCF' else None)
        yva, yte = ctx['dat'][1][1], ctx['dat'][2][1]
        zva = np.log(np.clip(pv, 1e-7, 1 - 1e-7) / (1 - np.clip(pv, 1e-7, 1 - 1e-7)))
        T = float(np.exp(minimize_scalar(
            lambda lt: log_loss(yva, 1 / (1 + np.exp(-zva * np.exp(lt)))),
            bounds=(-3, 3), method='bounded').x))
        zte = np.log(np.clip(pt, 1e-7, 1 - 1e-7) / (1 - np.clip(pt, 1e-7, 1 - 1e-7)))
        pT = 1 / (1 + np.exp(-zte / T))
        res[m] = dict(ece_before=G.ece(yte, pt), ece_after=G.ece(yte, pT),
                      brier=float(brier_score_loss(yte, pt)),
                      nll=float(log_loss(yte, np.clip(pt, 1e-7, 1 - 1e-7))), T=T)
        probs[f'p_{m.lower()}'] = pt
        if m == 'GRCF':
            probs['pt_grcf'] = pT
            with torch.no_grad():
                _, _, es, alpha = mod(torch.tensor(ctx['dat'][2][0][:200000]),
                                      torch.tensor(ctx['dat'][2][2][:200000], dtype=torch.long),
                                      torch.tensor(ctx['dat'][2][3][:200000], dtype=torch.long),
                                      ctx['views'],
                                      rho=torch.tensor(ctx['rho']['te'][:200000],
                                                       dtype=torch.float32),
                                      ret_all=True)
                probs['u_grcf'] = (2.0 / alpha.sum(-1)).numpy()
        probs['y'] = yte
        G.save_ck('calib.json', res)
        print('[calib]', m, res[m], flush=True)
    np.savez_compressed(os.path.join(G.RES, 'probs_calib.npz'), **probs)

def stage_ranking():
    res = G.ckpt('ranking.json')
    for seed in [0, 1]:
        ctx = build_context('XTX', 'user_disjoint', seed)
        rng = np.random.default_rng(seed)
        te_pos = ctx['tep'][ctx['tep'].y == 1]
        ue = rng.choice(sorted(te_pos.user.unique()),
                        size=min(150, te_pos.user.nunique()), replace=False)
        tr_hist = ctx['trp'][ctx['trp'].y == 1].groupby('user')['course'].apply(set).to_dict()
        rows, labels = [], []
        for u in ue:
            pos = te_pos[te_pos.user == u].course.tolist()
            seen = tr_hist.get(u, set()) | set(pos)
            negs = []
            while len(negs) < 99:
                cn = rng.integers(0, ctx['ni'])
                if cn not in seen: negs.append(cn)
            rows += [(u, c) for c in pos + negs]; labels += [1] * len(pos) + [0] * len(negs)
        df = pd.DataFrame(rows, columns=['user', 'course'])
        V1, V2, V3 = G.fb_transform(ctx['fb'], df)
        X, _, _ = G.pack_views(V1, V2, V3, ctx['v2names'])
        up = df.user.values; y = np.array(labels)
        uu = df.user.values.astype(np.int64); ii = df.course.values.astype(np.int64)

        def rank_metrics(pc):
            nd, hr, mr = [], [], []
            for u in ue:
                mk = up == u; yu, pu = y[mk], pc[mk]
                order = np.argsort(-pu)[:10]; rel = yu[order]
                ideal = min(10, int(yu.sum()))
                idcg = (1 / np.log2(np.arange(2, ideal + 2))).sum() if ideal > 0 else 1e-9
                nd.append(float((rel / np.log2(np.arange(2, 12))).sum() / idcg))
                hr.append(float(rel.sum() > 0))
                mr.append(float(1.0 / (np.where(rel > 0)[0][0] + 1)) if rel.sum() > 0 else 0.0)
            return [float(np.mean(nd)), float(np.mean(hr)), float(np.mean(mr))]

        key = f'Pop|{seed}'
        if key not in res:
            popc = ctx['trp'][ctx['trp'].y == 1].groupby('course').size()
            pop_score = df.course.map(popc).fillna(0.0).values.astype(np.float64)
            res[key] = rank_metrics(pop_score)
            G.save_ck('ranking.json', res)
            print('[rank]', key, res[key], flush=True)
        for m in ['MLP', 'NCF', 'LightGCN', 'GRCF']:
            key = f'{m}|{seed}'
            if key in res: continue
            mod = get_model(m, ctx)
            mod, _ = G.train_torch(mod, ctx['dat'], ctx['views'], seed=seed,
                                   sw_tr=ctx['sw'] if m == 'GRCF' else None,
                                   rho=ctx['rho'] if m == 'GRCF' else None)
            rho_c = ctx['gra'].degree(X).astype(np.float32) if m == 'GRCF' else None
            pc = G.predict_torch(mod, (X, y, uu, ii), ctx['views'], rho=rho_c)
            res[key] = rank_metrics(pc)
            G.save_ck('ranking.json', res)
            print('[rank]', key, res[key], flush=True)

def stage_efficiency():
    res = G.ckpt('efficiency.json')
    ctx = build_context('XTX', 'random', 0)
    dat = ctx['dat']
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(dat[0][1]))
    for frac in [0.25, 0.5, 0.75, 1.0]:
        key = str(frac)
        if key in res: continue
        n = int(len(dat[0][1]) * frac)
        idx = perm[:n]
        sub = (dat[0][0][idx], dat[0][1][idx], dat[0][2][idx], dat[0][3][idx])
        dats = (sub, dat[1], dat[2])
        row = {}
        for m in ['MLP', 'GRCF']:
            mod = get_model(m, ctx)
            rho_sub = {'tr': ctx['rho']['tr'][idx], 'va': ctx['rho']['va']} \
                if m == 'GRCF' else None
            mod, tt = G.train_torch(mod, dats, ctx['views'], seed=0, epochs=8, patience=2,
                                    sw_tr=ctx['sw'][idx] if m == 'GRCF' else None,
                                    rho=rho_sub)
            t0 = time.time()
            G.predict_torch(mod, dat[2], ctx['views'],
                            rho=ctx['rho']['te'] if m == 'GRCF' else None)
            tinf = time.time() - t0
            row[m] = dict(n_train=int(n), train_s=float(tt), infer_s=float(tinf))
        res[key] = row
        G.save_ck('efficiency.json', res)
        print('[eff]', key, flush=True)

STAGES = dict(main=stage_main, robustness=stage_robustness, ablation=stage_ablation,
              sensitivity=stage_sensitivity, coldstart=stage_coldstart,
              calibration=stage_calibration, ranking=stage_ranking,
              efficiency=stage_efficiency)

if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('GRCF_THREADS', '2')))
    which = sys.argv[1:] or ['main']
    if which == ['all']: which = list(STAGES)
    for w in which:
        print('=== stage', w, flush=True)
        STAGES[w]()
