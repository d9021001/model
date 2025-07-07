# coding: utf-8
"""Green-Learning feature extraction for the 3x3 app-usage 'image' + heuristic
domain features.

Green Learning core = Saab transform (Subspace Approximation with Adjusted Bias,
Kuo et al.): per-neighborhood DC (mean) + AC (PCA) coefficients, computed
feed-forward with NO back-prop.  On a 3x3 input we run a 2x2 PixelHop unit then a
2x2 second hop, plus a global Saab on the full 9-vector.  All statistics are fit
on the TRAINING data only (pass train then transform test) -> leakage-free.
"""
import numpy as np


class SaabHop:
    """One PixelHop/Saab unit over (N,H,W,C) with a (w x w) window, stride 1."""
    def __init__(self, win=2, n_ac=None):
        self.win = win
        self.n_ac = n_ac

    def _patches(self, X):
        N, H, W, C = X.shape
        w = self.win
        Ho, Wo = H - w + 1, W - w + 1
        out = np.empty((N, Ho, Wo, w * w * C), dtype=np.float64)
        for i in range(Ho):
            for j in range(Wo):
                out[:, i, j, :] = X[:, i:i + w, j:j + w, :].reshape(N, -1)
        return out, Ho, Wo

    def fit(self, X):
        P, Ho, Wo = self._patches(X)
        flat = P.reshape(-1, P.shape[-1])
        # DC = mean over feature dims; AC = residual
        self.dc_mean_ = flat.mean(axis=1, keepdims=True)
        ac = flat - self.dc_mean_
        self.feat_mean_ = ac.mean(axis=0, keepdims=True)
        acc = ac - self.feat_mean_
        # PCA on AC
        U, S, Vt = np.linalg.svd(acc, full_matrices=False)
        k = self.n_ac if self.n_ac is not None else Vt.shape[0]
        self.components_ = Vt[:k]
        self.Ho_, self.Wo_ = Ho, Wo
        return self

    def transform(self, X):
        P, Ho, Wo = self._patches(X)
        N = X.shape[0]
        flat = P.reshape(-1, P.shape[-1])
        dc = flat.mean(axis=1, keepdims=True)
        ac = (flat - dc) - self.feat_mean_
        coef = ac @ self.components_.T            # AC coefficients
        feats = np.hstack([dc, coef])             # 1 DC + k AC
        K = feats.shape[1]
        return feats.reshape(N, Ho, Wo, K)


def _pool(X4):
    """Spatial max + mean pooling over (N,H,W,K) -> (N, 2K)."""
    N = X4.shape[0]
    flat = X4.reshape(N, -1, X4.shape[-1])
    return np.hstack([flat.max(axis=1), flat.mean(axis=1)])


def saab_features(Xtr_img, Xts_img):
    """Two-hop Saab + global Saab. Fit on train, transform both."""
    h1 = SaabHop(win=2).fit(Xtr_img)
    t1, s1 = h1.transform(Xtr_img), h1.transform(Xts_img)
    h2 = SaabHop(win=2).fit(t1)
    t2, s2 = h2.transform(t1), h2.transform(s1)

    def feats(img, hop1, hop2):
        f1 = _pool(hop1)
        f2 = hop2.reshape(hop2.shape[0], -1)
        return np.hstack([f1, f2])
    Ftr = feats(Xtr_img, t1, t2)
    Fts = feats(Xts_img, s1, s2)
    return Ftr, Fts


def heuristic_features(X9):
    """Domain features from the 7 app-usage durations (cols 0..6; 7,8 are zero pad)."""
    v = X9[:, :7].astype(np.float64)
    eps = 1e-9
    total = v.sum(1, keepdims=True)
    srt = np.sort(v, axis=1)[:, ::-1]                      # descending
    p = v / (total + eps)
    entropy = -np.sum(np.where(p > 0, p * np.log(p + eps), 0), axis=1, keepdims=True)
    nz = (v > 0).sum(1, keepdims=True).astype(float)
    mean = v.mean(1, keepdims=True)
    std = v.std(1, keepdims=True)
    mx = v.max(1, keepdims=True)
    gini = (np.abs(v[:, :, None] - v[:, None, :]).sum((1, 2)) /
            (2 * 7 * (total[:, 0] + eps)))[:, None]
    feats = np.hstack([
        total, mx, mean, std,
        std / (mean + eps),                                # coeff of variation
        srt,                                               # 7 sorted values
        srt[:, :1] / (total + eps),                        # top-1 share
        (srt[:, :2].sum(1, keepdims=True)) / (total + eps),# top-2 share
        (srt[:, :3].sum(1, keepdims=True)) / (total + eps),# top-3 share
        entropy, nz, gini,
        np.log1p(total), np.log1p(mx), np.log1p(v),        # log-scaled
    ])
    return feats


def build_features(Xtr_img, Xts_img, use_saab=True, use_heur=True, use_raw=True, extra=None):
    Xtr9 = Xtr_img.reshape(len(Xtr_img), -1)
    Xts9 = Xts_img.reshape(len(Xts_img), -1)
    parts_tr, parts_ts = [], []
    if use_raw:
        parts_tr.append(Xtr9); parts_ts.append(Xts9)
    if use_heur:
        parts_tr.append(heuristic_features(Xtr9)); parts_ts.append(heuristic_features(Xts9))
    if use_saab:
        sa_tr, sa_ts = saab_features(Xtr_img, Xts_img)
        parts_tr.append(sa_tr); parts_ts.append(sa_ts)
    if extra is not None:
        parts_tr.append(extra[0]); parts_ts.append(extra[1])
    return np.hstack(parts_tr), np.hstack(parts_ts)
