"""Worker functions for nb04's encoding analyses."""
from __future__ import annotations

import numpy as np
from scipy import stats


def fourier_degree(v, y, seed, degrees=(1, 2, 3, 4), gain=0.02, folds=5):
    """Smallest truncated-Fourier degree beyond which cross-validated AUC stops
    improving by more than `gain`. Uses the labels (LABEL-DEPENDENT)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    th = 2 * np.pi * stats.rankdata(v) / (len(v) + 1)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    sc = []
    for d in degrees:
        B = np.column_stack([g(k * th) for k in range(1, d + 1) for g in (np.cos, np.sin)])
        sc.append(cross_val_score(LogisticRegression(max_iter=2000), B, y, cv=cv,
                                  scoring="roc_auc").mean())
    r = degrees[0]
    for i in range(1, len(degrees)):
        if sc[i] - sc[i - 1] > gain:
            r = degrees[i]
    return r, sc


def fourier_task(task):
    """Unpack (feature, values, labels, seed) for pmap."""
    f, v, y, seed = task
    r, sc = fourier_degree(v, y, seed)
    return {"feature": f, "seed": seed, "r": r, "scores": sc}


def iqp_states(angles):
    """PennyLane IQPEmbedding (n_repeats=1) statevectors, one per row of `angles`."""
    import pennylane as qml

    n = angles.shape[1]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def state(x):
        qml.IQPEmbedding(x, wires=range(n), n_repeats=1)
        return qml.state()

    return np.array([state(x) for x in angles])


def kernel_spread(task):
    """Off-diagonal fidelity-kernel std and mean for one (n_qubits, bandwidth)."""
    angles, n, band = task
    S = iqp_states(angles[:, :n] * band)
    G = np.abs(S.conj() @ S.T) ** 2
    o = G[~np.eye(len(G), dtype=bool)]
    return {"n_qubits": n, "band": band, "std": float(o.std()), "mean": float(o.mean())}
