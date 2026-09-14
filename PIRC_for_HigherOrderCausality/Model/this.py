from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.fft import fft, ifft
from collections import defaultdict
from joblib import Parallel, delayed

def this(X, Y, ooi, dmax, lam=0.1, rho=1.0, niter=10):

    if X.shape != Y.shape:
        print("Dimensions of states and derivatives do not match.")
        return None

    # Retrieve monomials
    theta, d = get_thetad(X, dmax)
    idx_mon = {}
    for i in range(d.shape[0]):
        mon = d[i, d[i, :] != 0].astype(int).tolist()
        if len(mon) == len(set(mon)):
            idx_mon[i] = sorted(mon)

    # Run SINDy
    coeff, err = my_sindy(theta, Y, lam, rho, niter)
    relerr = err / np.linalg.norm(Y, ord=1)

    # print("THIS completed.")

    # Reconstruct adjacency tensors
    Ainf = {o: np.zeros((0, o + 1)) for o in range(1, dmax + 2)}
    idx_coeff = {}
    nz_idx = []

    for i in idx_mon.keys():
        # agents where coeff[:, i] is nonzero and not in idx_mon[i]
        nonzero_agents = np.where(np.abs(coeff[:, i]) > 1e-8)[0] + 1  # +1 if you want Julia-like 1-based agent ids
        aaa = [a for a in nonzero_agents.tolist() if a not in idx_mon[i]]
        if len(aaa) > 0:
            nz_idx.append(i)
            idx_coeff[i] = aaa

    for idx in nz_idx:
        ii = idx_coeff[idx]
        jj = idx_mon[idx]
        o = len(jj) + 1

        ii_col = np.array(ii).reshape(-1, 1)
        jj_rep = np.tile(np.array(jj).reshape(1, -1), (len(ii), 1))
        c_col = coeff[np.array(ii) - 1, idx].reshape(-1, 1)  # -1 for Python indexing

        row_block = np.hstack([ii_col-1, jj_rep-1, c_col])
        Ainf[o] = np.vstack([Ainf[o], row_block])

    # print("Dictionary of inferred adjacency tensors built.")
    return Ainf, coeff, relerr

def my_sindy(theta, Y, lam=0.1, rho=1.0, niter=10):
    """
    Own implementation of SINDy.
    """

    n, T = Y.shape
    m, T2 = theta.shape
    if T != T2:
        raise ValueError("Y and theta must have the same number of columns (time steps).")

    I = np.eye(m)
    Xi = Y @ theta.T @ np.linalg.inv(theta @ theta.T + rho * I) #XI=W

    nz = 1_000_000
    k = 1

    while k < niter and np.sum(np.abs(Xi) > 1e-6) != nz:
        k += 1
        nz = np.sum(np.abs(Xi) > 1e-6)

        smallinds = np.abs(Xi) < lam
        Xi[smallinds] = 0.0

        for i in range(n):
            biginds = ~smallinds[i, :]
            if np.sum(biginds) == 0:
                continue

            th = theta[biginds, :]
            I_big = np.eye(np.sum(biginds))
            # Xi[i, biginds] = (
            #     Y[i:i+1, :] @ th.T @ np.linalg.pinv(th @ th.T + rho * I_big)
            # ).ravel()

            A = th @ th.T + rho * I_big  # (D, D)
            B = th @ Y[i:i + 1, :].T  # (D, 1)

            Xi[i, biginds] = np.linalg.solve(A, B).ravel()

    err = np.linalg.norm(Y - Xi @ theta, ord=1)
    return Xi, err

def get_θd(X: np.ndarray, dmax: int, i0: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape

    θ = np.ones((1, T), dtype=np.float64)
    d = np.zeros((1, dmax), dtype=np.int64)

    if dmax == 0:
        return θ, d

    for i in range(1, n + 1):  # Julia: i in 1:n
        θ0, d0 = get_θd(X[i - 1 : n, :], dmax - 1, i)

        Xi_rep = np.repeat(X[[i - 1], :], repeats=θ0.shape[0], axis=0)

        θ = np.vstack([θ, Xi_rep * θ0])

        d = np.vstack([d, np.hstack([d0, i * np.ones((d0.shape[0], 1), dtype=np.int64)])])

    d = d + (i0 - 1) * (d > 0).astype(np.int64)

    return θ, d


def get_thetad(X: np.ndarray, dmax: int, i0: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    return get_θd(X, dmax, i0)

def main():
    np.random.seed(42)

    n, T = 4, 200
    t = np.linspace(0, 10, T)

    X = np.zeros((n, T))
    X[0, :] = np.sin(t)
    X[1, :] = np.cos(t)
    X[2, :] = np.sin(2 * t)
    X[3, :] = 0.5 * np.cos(0.5 * t)

    Y = np.zeros((n, T))
    Y[0, :] = 1 + 1.2 * X[0, :] - 0.8 * X[1, :] + 0.5 * X[0, :] * X[2, :]
    Y[1, :] = -0.7 * X[1, :] + 0.3 * X[0, :] * X[3, :]
    Y[2, :] = 0.9 * X[2, :]
    Y[3, :] = -0.4 * X[3, :] + 0.2 * X[1, :] * X[2, :]

    ooi = [1, 2]
    dmax = 2

    # 3) Run THIS
    Ainf, coeff, relerr = this(X, Y, ooi, dmax, lam=0.1, rho=1.0, niter=20)

    # 4) Print result summary
    print("==== TEST RESULT ====")
    print("coeff:", coeff)
    print("relative error:", relerr)
    for order, mat in Ainf.items():
        print(f"Ainf[{order}] shape = {mat.shape}")
        if mat.shape[0] > 0:
            print(mat[: min(5, mat.shape[0]), :])  # preview first rows

# ================================
# average_over_zones
# ================================
def average_over_zones(s2signal: dict, s2z: dict):
    """
    s2signal: dict[str, np.ndarray]
    s2z: dict[str, int]
    """
    l = max(s2z.values())  # number of zones
    T = len(next(iter(s2signal.values())))

    c = np.zeros(l)
    asig = np.zeros((l, T))

    for k in s2signal:
        z = s2z[k] - 1  # ⚠️ Python 从0开始
        c[z] += 1
        asig[z, :] = asig[z, :] * (c[z] - 1) / c[z] + s2signal[k] / c[z]

    return asig


# ================================
# denoise_fourier (vector)
# ================================
def denoise_fourier_1d(x: np.ndarray, nmodes: int):
    T = len(x)
    f = fft(x)

    g = np.zeros(T, dtype=complex)
    g[:nmodes + 1] = f[:nmodes + 1]
    g[T - nmodes:T] = f[T - nmodes:T]

    return np.real(ifft(g))


# ================================
# denoise_fourier (matrix)
# ================================
def denoise_fourier(X: np.ndarray, nmodes: int):
    n, T = X.shape
    Y = np.zeros((n, T))

    for i in range(n):
        Y[i, :] = denoise_fourier_1d(X[i, :], nmodes)

    return Y


# ================================
# list_all_subjects
# ================================
def list_all_subjects(n: int):
    subjects = []

    for i in range(1, min(9, n) + 1):
        subjects.append(f"{i:03d}")

    for i in range(10, min(99, n) + 1):
        subjects.append(f"{i:03d}")

    for i in range(100, min(999, n) + 1):
        subjects.append(f"{i:03d}")

    return subjects


# ================================
# read_eeg
# ================================
def read_eeg(file):

    import pyedflib

    f = pyedflib.EdfReader(file)
    n = f.signals_in_file

    s2signal = {}

    for i in range(n):
        label = f.getLabel(i)
        if "annot" in label.lower():
            continue
        signal = f.readSignal(i)
        s2signal[label] = signal.astype(np.float64)
        # print(signal[:10])
        # print(np.max(signal), np.min(signal))
    f.close()
    return s2signal


# ================================
# load_and_save_eeg_avg
# ================================
def load_and_save_eeg_avg(subjects, states, nz):
    sensors = pd.read_csv(f"eeg-data/sensors-{nz}.csv", header=None).values.flatten()
    zones = pd.read_csv(f"eeg-data/zones-{nz}.csv", header=None).values.flatten()

    s2z = {sensors[i]: int(zones[i]) for i in range(len(sensors))}

    for su in subjects:
        for st in states:
            print(f"Working on S{su}R{st}")

            s2signal = read_eeg(f"eeg-data/S{su}R{st}.edf")
            asig = average_over_zones(s2signal, s2z)

            np.savetxt(f"eeg-data/S{su}R{st}-X.csv", asig, delimiter=",")


def restrict_box_size(X: np.ndarray, nstep: int):
    n, T = X.shape
    ns = min(T - 1, nstep)

    mX = np.median(X, axis=1, keepdims=True)
    dX = X - mX
    nX = np.sum(dX ** 2, axis=0)

    sX = np.sort(nX)
    th = (sX[ns] + sX[ns - 1]) / 2  # ⚠️ Python index

    ids = np.where(nX < th)[0]

    return X[:, ids], ids

def apply_threshold(Ainf, th):
    Aout = {}
    for o, A in Ainf.items():
        if A.shape[0] == 0:
            continue
        mask = np.abs(A[:, -1]) > th
        if np.sum(mask) > 0:
            Aout[o] = A[mask]
    return Aout


def compute_rho(Ainf, X, orders=[2,3,4]):
    N, T = X.shape
    rho = {l: np.zeros(T) for l in orders}
    for t in range(T):
        x = X[:, t]
        denom = 0.0
        numerators = {}
        for l in orders:
            if l not in Ainf or Ainf[l].shape[0] == 0:
                numerators[l] = 0.0
                continue
            A = Ainf[l]
            val = 0.0
            for row in A:
                idx = row[:-1].astype(int)
                c = row[-1]
                # 只乘 i2,...,iℓ
                prod = 1.0
                for j in idx[1:]:
                    prod *= x[j]
                val += abs(c * prod)
            numerators[l] = val
            denom += val
        denom = max(denom, 1e-12)
        for l in orders:
            rho[l][t] = numerators[l] / denom
    return rho

def compute_quantiles(rho_all):
    median = np.median(rho_all, axis=0)
    q25 = np.percentile(rho_all, 25, axis=0)
    q75 = np.percentile(rho_all, 75, axis=0)
    q10 = np.percentile(rho_all, 10, axis=0)
    q90 = np.percentile(rho_all, 90, axis=0)
    return median, q25, q75, q10, q90

def plot_panel(ax, x, median, q25, q75, q10, q90, color, title):
    ax.fill_between(x, q10, q90, color=color, alpha=0.2)
    ax.fill_between(x, q25, q75, color=color, alpha=0.5)
    ax.plot(x, median, color=color, linewidth=2)
    ax.axvline(0.1, color='k', linestyle='--')

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylim(0, 1)
import numpy as np

def restrict_hypercube_size(X, delta, center=None):
    """
    Keep all samples inside a hypercube.

    Parameters
    ----------
    X : ndarray, shape (N, T)
        N variables, T samples (each column is one sample).
    delta : float
        Side length of the hypercube.
    center : ndarray of shape (N,), optional
        Center of the hypercube.
        If None, the median of each variable is used.

    Returns
    -------
    X_new : ndarray
        Samples inside the hypercube.
    ids : ndarray
        Indices of retained samples.
    """

    N, T = X.shape

    # Default center: median of each dimension
    if center is None:
        center = np.median(X, axis=1, keepdims=True)
    else:
        center = np.asarray(center).reshape(N, 1)

    # Offset from the center
    dX = X - center

    # Check whether every coordinate lies inside [-delta/2, delta/2]
    inside = np.all(np.abs(dX) <= delta / 2, axis=0)

    ids = np.where(inside)[0]

    return X[:, ids], ids

def this_par_python(X, Y, ooi, dmax, lam=0.1, rho=1.0, niter=10, n_jobs=-1):
    if X.shape != Y.shape:
        print("Dimensions of states and derivatives do not match.")
        return None

    n, T = X.shape

    theta, d = get_thetad(X, dmax)

    idx_mon = {}
    for k in range(d.shape[0]):
        mon = d[k, d[k, :] != 0].astype(int).tolist()
        if len(mon) == len(set(mon)):
            idx_mon[k] = sorted(mon)

    def run_one_node(i):
        coeff_i, err_i = my_sindy(
            theta,
            Y[i:i+1, :],
            lam=lam,
            rho=rho,
            niter=niter
        )
        return i, coeff_i.reshape(-1), err_i

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_node)(i)
        for i in range(n)
    )

    coeff = {}
    err = 0.0

    for i, coeff_i, err_i in results:
        coeff[i] = coeff_i
        err += float(err_i)

    relerr = err / (np.linalg.norm(Y, ord=1) + 1e-12)

    Ainf = {
        o: np.zeros((0, o + 1))
        for o in range(1, dmax + 2)
    }

    for i in range(n):
        c = coeff[i]

        for j in range(len(c)):
            if j not in idx_mon:
                continue

            mon = idx_mon[j]

            if i in mon:
                continue

            weight = c[j]
            if abs(weight) <= 1e-8:
                continue

            order = len(mon) + 1

            row = np.array([i] + mon + [weight], dtype=float).reshape(1, -1)
            Ainf[order] = np.vstack([Ainf[order], row])

    return Ainf, coeff, relerr

if __name__ == "__main__":
    main()