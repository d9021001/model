# coding: utf-8
"""Shared data loading + embedding + PCA for the TPX reproduction.
Mirrors the loading in tpx24w_apptime.py exactly (incl. the range(0,n-1) drop-last
and the /266500.0 normalisation)."""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from tpx_embed import get_embeddings

NORM = 266500.0


def load_file(path, label):
    """Replicate the original loader: read A:I, iterate range(0, n-1) -> drops last row."""
    u1 = pd.read_excel(path, usecols="A:I", sheet_name="Sheet1")
    n = int(u1.size / 9)
    rows = []
    for j in range(0, n - 1):
        rows.append(u1[j:j + 1].astype("uint32").values.reshape(3, 3, 1))
    X = np.array(rows, dtype="float64") / NORM
    y = np.full(len(X), label, dtype="uint8")
    return X, y


def load_all():
    x0, y0 = load_file("tr0.xlsx", 0)
    x1, y1 = load_file("tr1.xlsx", 1)
    x_tr = np.concatenate([x0, x1]); y_tr = np.concatenate([y0, y1])
    ts0x, ts0y = load_file("ts0.xlsx", 0)
    ts1x, ts1y = load_file("ts1.xlsx", 1)
    return x_tr, y_tr, ts0x, ts0y, ts1x, ts1y


def load_pca():
    with open("pca_model.pkl", "rb") as f:
        return pickle.load(f)


def embed_pca(X, pca):
    # Apply PCA manually (the 0.24.2-pickled estimator's .transform() is incompatible
    # with the installed sklearn). For whiten=False this is exactly sklearn's transform.
    emb = get_embeddings(X)
    return (emb - pca.mean_) @ pca.components_.T


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    x_tr, y_tr, ts0x, ts0y, ts1x, ts1y = load_all()
    print(f"x_tr: {x_tr.shape}  (class0={np.sum(y_tr==0)}, class1={np.sum(y_tr==1)})")
    print(f"ts0: {ts0x.shape}, ts1: {ts1x.shape}")
    pca = load_pca()
    d_tr = embed_pca(x_tr, pca)
    print(f"PCA features: {d_tr.shape}")
    # separability check in 3D PCA space
    m0 = d_tr[y_tr == 0].mean(0); m1 = d_tr[y_tr == 1].mean(0)
    print("class0 PCA mean:", np.round(m0, 4))
    print("class1 PCA mean:", np.round(m1, 4))
    print("centroid distance:", round(float(np.linalg.norm(m1 - m0)), 4))
    print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 4),
          "sum=", round(float(pca.explained_variance_ratio_.sum()), 4))
