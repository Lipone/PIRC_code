import numpy as np
from scipy.integrate import solve_ivp


def lorenz_nonpairwise(t, state, A2, A3, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz oscillators with pairwise and triadic interactions.

    state: shape (3*n,)
           [x1,...,xn, y1,...,yn, z1,...,zn]
    A2: shape (n,n)
        pairwise adjacency matrix a_ij^(2)
    A3: shape (n,n,n)
        triadic adjacency tensor a_ijk^(3)
    """

    n = A2.shape[0]

    x = state[:n]
    y = state[n:2*n]
    z = state[2*n:3*n]

    # intrinsic Lorenz dynamics
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    # pairwise coupling: sum_j a_ij^(2) (x_j - x_i)
    pairwise = A2 @ x - x * A2.sum(axis=1)

    # triadic coupling: sum_{j,k} a_ijk^(3) (x_j x_k^2 - x_i^3)
    triadic_drive = np.einsum("ijk,j,k->i", A3, x, x**2)
    triadic_self = x**3 * A3.sum(axis=(1, 2))
    triadic = triadic_drive - triadic_self

    dx += pairwise + triadic

    return np.concatenate([dx, dy, dz])


# # =========================
# # Example usage
# # =========================
#
# n = 10
# T = 50
# dt = 0.01
#
# # random pairwise adjacency
# A2 = np.random.rand(n, n) < 0.2
# A2 = A2.astype(float)
# np.fill_diagonal(A2, 0.0)
#
# # random triadic adjacency tensor
# A3 = np.random.rand(n, n, n) < 0.02
# A3 = A3.astype(float)
#
# # remove self-related triadic terms if desired
# for i in range(n):
#     A3[i, i, :] = 0.0
#     A3[i, :, i] = 0.0
#
# # coupling strengths
# gamma2 = 0.1
# gamma3 = 0.01
# A2 *= gamma2
# A3 *= gamma3
#
# # initial condition
# state0 = np.random.randn(3 * n)
#
# t_eval = np.arange(0, T, dt)
#
# sol = solve_ivp(
#     lorenz_nonpairwise,
#     t_span=(0, T),
#     y0=state0,
#     t_eval=t_eval,
#     args=(A2, A3),
#     method="RK45",
#     rtol=1e-8,
#     atol=1e-10
# )
#
# X = sol.y[:n].T
# Y = sol.y[n:2*n].T
# Z = sol.y[2*n:3*n].T
#
# print(X.shape)  # (time_steps, n)