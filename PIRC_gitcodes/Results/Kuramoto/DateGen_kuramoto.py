import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import os
import json
#20-nodes Kuramoto model with pairwise and three-way interactions
def generate_kuramoto_data_fast(
        n=8, dt=0.01, steps=5000,
        Pair_strength=0, Tri_strength=0,
        a2=None, a3=None, P_probability=0.5, T_probability=0.5):

    # ---------- a2 ----------
    if a2 is None:
        mask = np.random.rand(n, n) < P_probability
        a2 = Pair_strength * mask.astype(float)
        np.fill_diagonal(a2, 0)
    else:
        a2 = np.array(a2, dtype=float)

    # ---------- a3 ----------
    if a3 is None:
        a3 = build_a3(n, Tri_strength, T_probability)
    else:
        a3 = np.array(a3, dtype=float)

    # ---------- 初值 ----------
    theta = np.random.rand(n) * 2 * np.pi
    sign = np.random.choice([-1, 1], size=n)
    omega = sign * (0.3 + np.random.rand(n))

    theta_hist = np.empty((steps, n))

    substeps = 10
    ddt = dt / substeps

    for t in range(steps):
        for _ in range(substeps):
            theta = rk4_step_vec(theta, ddt, omega, a2, a3)
        theta_hist[t] = theta

    return a2, a3, theta_hist


def build_a3(n, Tri_strength, probability):
    mask = np.random.rand(n, n, n) < probability
    a3 = np.zeros((n, n, n))

    # 只保留 j<k
    upper = np.triu(np.ones((n, n)), k=1).astype(bool)
    upper3 = upper[None, :, :]              # (1,n,n)

    a3[mask & upper3] = Tri_strength

    # 去掉 i=j 或 i=k
    idx = np.arange(n)
    a3[idx, idx, :] = 0
    a3[idx, :, idx] = 0
    a3[:, idx, idx] = 0

    return a3

def rk4_step_vec(theta, dt, omega, a2, a3):
    k1 = kuramoto_model_vec(theta, omega, a2, a3)
    k2 = kuramoto_model_vec(theta + 0.5*dt*k1, omega, a2, a3)
    k3 = kuramoto_model_vec(theta + 0.5*dt*k2, omega, a2, a3)
    k4 = kuramoto_model_vec(theta + dt*k3, omega, a2, a3)
    return theta + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def kuramoto_model_vec(theta, omega, a2, a3):
    """
    theta: (n,)
    omega: (n,)
    a2:    (n,n)
    a3:    (n,n,n)  只在 j<k 位置有值
    """
    # ---------- 二阶项 ----------
    # θj - θi
    diff = theta[None, :] - theta[:, None]        # (n,n)
    pair_term = np.sum(a2 * np.sin(diff), axis=1) # (n,)

    # ---------- 三阶项 ----------
    theta_i = theta[:, None, None]                # (n,1,1)
    theta_j = theta[None, :, None]                # (1,n,1)
    theta_k = theta[None, None, :]                # (1,1,n)

    tri_phase = theta_j + theta_k - 2 * theta_i   # (n,n,n)
    tri_term = np.sum(a3 * np.sin(tri_phase), axis=(1, 2))  # (n,)

    return omega + pair_term + tri_term


def kuramoto_model(theta, omega, a2, a3):
    """
        Kuramoto model with pairwise and three-way interactions.

        Parameters:
            theta : ndarray, shape (n,)
                The phases of n oscillators.
            omega : ndarray, shape (n,)
                The natural frequencies of n oscillators.
            a2 : ndarray, shape (n, n)
                Coupling matrix for pairwise interactions.
            a3 : ndarray, shape (n, n, n)
                Coupling tensor for three-way interactions.

        Returns:
            dtheta : ndarray, shape (n,)
                Time derivatives of the phases.
        """
    n = len(theta)
    dtheta = np.zeros(n)
    for i in range(n):
        pairwise_sum = np.sum(a2[i, :] * np.sin(theta - theta[i]))
        threeway_sum = 0.0
        for j in range(n):
            for k in range(n):
                threeway_sum += a3[i, j, k] * np.sin(theta[j] + theta[k] - 2 * theta[i])
        dtheta[i] = omega[i] + pairwise_sum + threeway_sum
    return dtheta


def rk4_step(f, y, dt, *args):
    k1 = f(y, *args)
    k2 = f(y + 0.5 * dt * k1, *args)
    k3 = f(y + 0.5 * dt * k2, *args)
    k4 = f(y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

def generate_kuramoto_data(n=8, dt=0.01, steps=5000, Pair_strength = 0, Tri_strength = 0, a2=None, a3=None, probability=0.5):
    if a2 is None:
        a2 = np.where(np.random.rand(n, n)<probability, Pair_strength, 0)
        # a2 = np.random.rand(n, n)
        np.fill_diagonal(a2, 0)
        # a2 = np.where(a2 > 1-probability, Pair_strength, 0)
    else:
        # a2 = np.where(np.array(a2, dtype=float) > 0, Pair_strength, 0)
        a2 = np.array(a2, dtype=float)
    if a3 is None:
        a3 = np.where(np.random.rand(n, n, n)<probability, Tri_strength, 0)
        # a3 = np.random.rand(n, n, n)
        for i in range(n):
            a3[i, i, :] = 0
            a3[i, :, i] = 0
            a3[:, i, i] = 0
            for j in range(n):
                for k in range(n):
                    if not (j < k):
                        a3[i, j, k] = 0
        # a3 = np.where(a3 > 1-probability, Tri_strength, 0)
    else:
        # a3 = np.where(np.array(a3, dtype=float) > 0, Tri_strength, 0)
        a3 = np.array(a3, dtype=float)
    # print("a2:\n", a2)
    # print("a3:\n", a3)
    theta = np.random.rand(n) * 2 * np.pi
    # omega = np.random.normal(0, 0.5, n)
    sign = np.random.choice([-1, 1], size=n)
    omega = sign * (0.3 + np.random.rand(n))
    theta_unwrapped_hist = np.zeros((steps, n))
    substeps = 10
    ddt = dt / substeps
    for t in range(steps):
        for _ in range(substeps):
            theta = rk4_step(kuramoto_model, theta, ddt, omega, a2, a3)
        # theta = np.mod(theta, 2*np.pi)
        theta_unwrapped_hist[t] = theta
    return a2, a3, theta_unwrapped_hist

def load_or_generate_kuramoto(args,
                              cache_dir='data'):
    """
    Load Kuramoto data and interaction matrices if cached;
    otherwise generate and save them.

    Returns
    -------
    data : np.ndarray
        Kuramoto phase trajectories
    a2 : np.ndarray
        Pairwise interaction matrix
    a3 : np.ndarray
        Higher-order (triadic) interaction tensor
    """
    cache_name=f"kuramoto_n{args.node_num}_P{args.Pair_strength}_T{args.Tri_strength}_Ns{args.N_start}_Nw{args.N_washout}_Ntr{args.N_train}_Nte{args.N_test}.npz"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # =========================
    # 1. Load cache if exists
    # =========================
    if os.path.exists(cache_path):
        print(f"[Kuramoto] Loading cached data from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)

        # ---- sanity check ----
        if _check_args_consistency(cache, args):
            data = cache['data']
            a2 = cache['a2']
            a3 = cache['a3']
            return data, a2, a3

    # =========================
    # 2. Generate from scratch
    # =========================
    print("[Kuramoto] data is Generating...")

    a2, a3, data = generate_kuramoto_data_fast(
        n=args.node_num,
        dt=args.dt,
        steps=args.N_start + args.N_washout + args.N_train + args.N_test + 1,
        Pair_strength=args.Pair_strength,
        Tri_strength=args.Tri_strength,
        a2=None,
        a3=None,
        P_probability=args.P_probability,
        T_probability=args.T_probability
    )

    # =========================
    # 3. Save cache
    # =========================
    np.savez(
        cache_path,
        data=data,
        a2=a2,
        a3=a3,
        node_num=args.node_num,
        Pair_strength=args.Pair_strength,
        Tri_strength=args.Tri_strength,
        dt=args.dt,
        N_start=args.N_start,
        N_washout=args.N_washout,
        N_train=args.N_train,
        N_test=args.N_test,
        P_probability=args.P_probability,
        T_probability=args.T_probability
    )

    print(f"[Kuramoto] Data generated and cached at {cache_path}")
    return data, a2, a3


def _check_args_consistency(cache, args):
    """
    Prevent silent bugs: loading data generated with different parameters.
    """
    keys = [
        'node_num',
        'Pair_strength',
        'Tri_strength',
        'dt',
        'N_start',
        'N_washout',
        'N_train',
        'N_test',
        'P_probability',
        'T_probability'
    ]

    for k in keys:
        if k not in cache:
            print(f"[Kuramoto][Warning] Missing key '{k}' in cache.")
            continue

        if cache[k] != getattr(args, k):
            print(
                f"[Kuramoto] Cache mismatch on '{k}': "
                f"cache={cache[k]}, args={getattr(args, k)}"
                "data will regenerate."
            )
            return False
    return True

