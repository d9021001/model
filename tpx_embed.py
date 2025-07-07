# coding: utf-8
"""TF-free reimplementation of the Triplet-CNN base-network forward pass.

The base network (create_base_network in tpx24w_apptime.py) is:
  Conv2D(32,3x3,same) -> ReLU -> BN
  Conv2D(32,3x3,same) -> ReLU -> BN
  Dropout(0.5)              # identity at inference
  Conv2D(64,3x3,same) -> ReLU -> BN
  Conv2D(64,3x3,same) -> ReLU -> BN
  Dropout(0.5)              # identity at inference
  Flatten (channels_last, C-order)
  Dense(9, linear)

Weights are read directly from triplet.h5 (group model_weights/sequential_1),
so this is independent of any TensorFlow/Keras version.
"""
import numpy as np
import h5py

BN_EPS = 1e-3


def _load_weights(h5path="triplet.h5"):
    w = {}
    with h5py.File(h5path, "r") as f:
        seq = f["model_weights"]["sequential_1"]

        def walk(g, pfx=""):
            for k in g.keys():
                item = g[k]
                if isinstance(item, h5py.Dataset):
                    w[pfx + k] = item[()]
                else:
                    walk(item, pfx + k + "/")
        walk(seq)
    return w


def _conv2d_same(x, kernel, bias):
    """x: (N,H,W,Cin), kernel: (kh,kw,Cin,Cout), stride 1, 'same' padding.
    Keras uses cross-correlation (no kernel flip)."""
    N, H, W, Cin = x.shape
    kh, kw, _, Cout = kernel.shape
    ph, pw = kh // 2, kw // 2  # =1 for 3x3 -> symmetric same padding, stride 1
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")
    out = np.zeros((N, H, W, Cout), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            patch = xp[:, i:i + kh, j:j + kw, :]          # (N,kh,kw,Cin)
            out[:, i, j, :] = np.tensordot(patch, kernel, axes=([1, 2, 3], [0, 1, 2]))
    return out + bias


def _bn(x, gamma, beta, mean, var, eps=BN_EPS):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _relu(x):
    return np.maximum(x, 0.0)


def get_embeddings(X, h5path="triplet.h5"):
    """X: (N,3,3,1) float -> embeddings (N,9)."""
    w = _load_weights(h5path)
    x = np.asarray(X, dtype=np.float64)

    def conv_relu_bn(x, ci, bi):
        x = _conv2d_same(x, w[f"conv2d_{ci}/kernel:0"], w[f"conv2d_{ci}/bias:0"])
        x = _relu(x)
        x = _bn(x, w[f"batch_normalization_{bi}/gamma:0"], w[f"batch_normalization_{bi}/beta:0"],
                w[f"batch_normalization_{bi}/moving_mean:0"], w[f"batch_normalization_{bi}/moving_variance:0"])
        return x

    x = conv_relu_bn(x, 1, 1)
    x = conv_relu_bn(x, 2, 2)
    # Dropout(0.5) -> identity at inference
    x = conv_relu_bn(x, 3, 3)
    x = conv_relu_bn(x, 4, 4)
    # Flatten channels_last, C-order: (N,H,W,C) -> (N, H*W*C)
    N = x.shape[0]
    x = x.reshape(N, -1)                       # C-order matches Keras Flatten
    x = x @ w["dense_1/kernel:0"] + w["dense_1/bias:0"]   # Dense(9), linear
    return x


if __name__ == "__main__":
    import pandas as pd
    # quick smoke test on a few tr0 rows
    d = pd.read_excel("tr0.xlsx", usecols="A:I", sheet_name="Sheet1").head(5)
    X = (d.values.astype("float64").reshape(-1, 3, 3, 1)) / 266500.0
    emb = get_embeddings(X)
    print("emb shape:", emb.shape)
    print("emb[0]:", np.round(emb[0], 4))
    print("min/max/mean:", emb.min(), emb.max(), emb.mean())
