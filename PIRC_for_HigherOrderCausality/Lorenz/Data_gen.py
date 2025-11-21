import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle
import os

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz_data(
    t_span=(0.0, 50.0),
    dt=0.01,
    initial_state=(1.0, 1.0, 1.0),
    sigma=10.0,
    rho=28.0,
    beta=8/3
):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lorenz,
        t_span,
        initial_state,
        t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-9,
        atol=1e-9,
        method="RK45"
    )
    return sol.t, sol.y  # t, (3, N)

def plot_lorenz(t, xyz, show=True, save_prefix=None):
    x, y, z = xyz
    fig = plt.figure(figsize=(12, 8))

    # 3D 轨迹
    ax3d = fig.add_subplot(2, 2, (1, 3), projection='3d')
    ax3d.plot(x, y, z, lw=0.6, color='darkblue')
    ax3d.set_title("Lorenz Attractor (3D)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # X-Z 投影
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_xz.plot(x, z, color='tomato', lw=0.5)
    ax_xz.set_title("Projection: X-Z")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")

    # 时间序列（只画 x，可扩展）
    ax_ts = fig.add_subplot(2, 2, 4)
    ax_ts.plot(t, x, color='green', lw=0.5)
    ax_ts.set_title("Time Series (X)")
    ax_ts.set_xlabel("t")
    ax_ts.set_ylabel("X")

    plt.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_lorenz.png", dpi=200)

    if show:
        plt.show()

if __name__ == "__main__":
    t, xyz = generate_lorenz_data(
        t_span=(0, 1000),
        dt=0.01,
        initial_state=(1.0, 1.0, 1.0),
        sigma=10.0,
        rho=28.0,
        beta=8/3
    )
    # 只画三维轨迹
    x, y, z = xyz
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(x, y, z, lw=0.6, color='darkblue')
    ax3d.set_title("Lorenz Attractor (3D)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    plt.tight_layout()
    plt.show()

    os.makedirs('data', exist_ok=True)
    with open('data/lorenz_data.pkl', 'wb') as f:
        pickle.dump((t, xyz), f)


