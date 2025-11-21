import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *

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

def generate_kuramoto_data(n=8, dt=0.01, steps=5000, Pair_strength = 0, Tri_strength = 0, a2=None, a3=None, probability=0.7):
    if a2 is None:
        a2 = np.where(np.random.rand(n, n)<probability, Pair_strength, 0)
        # a2 = np.random.rand(n, n)
        np.fill_diagonal(a2, 0)
        # a2 = np.where(a2 > 1-probability, Pair_strength, 0)
    else:
        a2 = np.where(np.array(a2, dtype=float) > 0, Pair_strength, 0)
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
        a3 = np.where(np.array(a3, dtype=float) > 0, Tri_strength, 0)
    # print("a2:\n", a2)
    # print("a3:\n", a3)
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.normal(0, 0.5, n)
    theta_history = np.zeros((steps, n))
    for t in range(steps):
        theta = rk4_step(kuramoto_model, theta, dt, omega, a2, a3)
        theta = np.mod(theta, 2*np.pi)
        theta_history[t] = theta
    return a2, a3, theta_history

if __name__ == "__main__":
    a2 = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.1],
        [0.1, 0.0, 0.0]
    ]

    a3 = [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
    ]

    # a2, a3, data = generate_kuramoto_data(n=3, dt=0.01, steps=100000, Pair_strength = 0.1, Tri_strength = 0.1, a2=a2, a3=a3)
    a2, a3, data = generate_kuramoto_data(n=3, dt=0.01, steps=20000, Pair_strength=0.1, Tri_strength=0.1, a2=a2, a3=a3)
    os.makedirs('data', exist_ok=True)
    with open('data/NodeNum_3_PairStrength_0.1_TriStrength_0.1.pkl', 'wb') as f:
        pickle.dump((a2,a3,data), f)

    # 可视化，每个节点一个子图
    n = data.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    for i in range(n):
        axes[i].plot(data[:5000, i])
        axes[i].set_ylabel(f'Node {i} Phase')
        axes[i].set_title(f'Node {i}')
    axes[-1].set_xlabel('Time step')
    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/Phase.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f'Node {i}')
    plt.xlabel('Time step')
    plt.ylabel('Phase')
    plt.title('Kuramoto Model Phases Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()