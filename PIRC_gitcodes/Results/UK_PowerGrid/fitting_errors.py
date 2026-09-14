import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# 1. Unified metric: node-wise RRMSE
# ============================================================
def rrmse_per_node(Y_true, Y_pred, eps=1e-12):
    """
    Y_true, Y_pred: shape (N, T)
    return: node-wise RRMSE, shape (N,)
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=1))
    scale = np.sqrt(np.mean(Y_true ** 2, axis=1)) + eps

    return rmse / scale

# ============================================================
# 2. PIRC: node-wise training RRMSE
# ============================================================
@torch.no_grad()
def compute_pirc_rrmse(pirc, X_washout, X_train, Y_train):
    """
    Compute node-wise training RRMSE for PIRC.

    X_train, Y_train: shape (B, T, O)
    Here B corresponds to target nodes.
    """
    B = X_train.shape[0]
    N_units = pirc.n_units

    if pirc.Expand == 2:
        Y_target_diff = Y_train[:, :, 0:1]
    else:
        Y_target_diff = (Y_train[:, :, 0:1] - X_train[:, :, 0:1]) / pirc.dt

    rf_washout = torch.zeros(
        B,
        pirc.ExpandNodes * N_units,
        device=pirc.device
    )

    for t in range(X_washout.shape[1]):
        rf_washout = pirc.step(rf_washout, X_washout[:, t, :])

    R = pirc.open_loop(X_train, rf_washout, noise=None)
    R_train = R[:, 1:, :]

    LHS = R_train.transpose(1, 2) @ R_train
    LHS.diagonal(dim1=-2, dim2=-1).add_(pirc.tikh)

    RHS = R_train.transpose(1, 2) @ Y_target_diff

    try:
        pirc.Wout = torch.linalg.solve(LHS, RHS)
    except RuntimeError:
        pirc.Wout = torch.linalg.pinv(LHS) @ RHS

    Y_pred = R_train @ pirc.Wout  # (B, T, 1)

    Y_true_np = Y_target_diff.squeeze(-1).detach().cpu().numpy()
    Y_pred_np = Y_pred.squeeze(-1).detach().cpu().numpy()

    pirc_rrmse = rrmse_per_node(Y_true_np, Y_pred_np)

    train_mse = np.mean((Y_true_np - Y_pred_np) ** 2)

    return pirc_rrmse, Y_pred_np, train_mse

# ============================================================
# 3. ARNI: node-wise training RRMSE
# ============================================================
def compute_arni_rrmse(
    X,
    Y,
    args,
    base="polynomial",
    basis_order=2,
    expand_type="mul",
    th=1e-4
):
    """
    X: shape (T, N)
    Y: shape (T, N), derivative
    return:
        node-wise RRMSE, shape (N,)
        Y_pred, shape (N, T)
    """
    N = args.node_num
    Y_T = Y.T  # (N, T)

    Y_pred = np.zeros_like(Y_T)

    for NODE in range(N):
        X_expanded = expand_nodes(
            X.copy(),
            NODE,
            type=expand_type
        )

        X_expanded_T = X_expanded.T  # (ExpandedNodes, T)

        llist, cost, vec = ARNI(
            X_expanded_T,
            Y_T,
            base,
            basis_order,
            NODE,
            th=th
        )

        X_basis = basis_expansion(
            X_expanded_T,
            basis_order,
            base,
            NODE
        )  # (num_basis, T, ExpandedNodes)

        if len(llist) == 0:
            Y_pred[NODE, :] = 0.0
            continue

        R = np.vstack([
            X_basis[:, :, idx]
            for idx in llist
        ])  # (num_selected_basis, T)

        coef, *_ = np.linalg.lstsq(
            R.T,
            Y_T[NODE, :],
            rcond=None
        )

        Y_pred[NODE, :] = R.T @ coef

    arni_rrmse = rrmse_per_node(Y_T, Y_pred)

    return arni_rrmse, Y_pred

# ============================================================
# 4. THIS: node-wise training RRMSE
# ============================================================
def compute_this_rrmse(
    X_this,
    Y_this,
    dmax=2,
    lam=0.1,
    rho=1.0,
    niter=10
):
    """
    X_this: shape (N, T)
    Y_this: shape (N, T), derivative

    return:
        node-wise RRMSE, shape (N,)
        Y_pred, shape (N, T)
        Ainf, coeff
    """
    Ainf, coeff, relerr = this(
        X_this,
        Y_this,
        ooi=[],
        dmax=dmax,
        lam=lam,
        rho=rho,
        niter=niter
    )

    theta, _ = get_thetad(X_this, dmax)
    Y_pred = coeff @ theta

    this_rrmse = rrmse_per_node(Y_this, Y_pred)

    return this_rrmse, Y_pred, Ainf, coeff

# ============================================================
# 5. Violin plot
# ============================================================
def plot_violin_rrmse(
    error_dict,
    save_path="nodewise_training_rrmse_violin.svg"
):
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "DejaVu Serif"
    mpl.rcParams["mathtext.fontset"] = "stix"

    labels = list(error_dict.keys())

    data = [
        np.asarray(error_dict[k], dtype=float)
        for k in labels
    ]

    data = [
        x[np.isfinite(x)]
        for x in data
    ]

    colors = [
        "#8B1E1E",  # PIRC
        "#1F4E79",  # THIS
        "#B8860B",  # ARNI1
        "#2E6B3E",  # ARNI2
    ]

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=600)

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor("none")
        body.set_alpha(0.65)

    parts["cmedians"].set_color("#333333")
    parts["cmedians"].set_linewidth(2.0)

    rng = np.random.default_rng(42)

    for i, y in enumerate(data, start=1):
        x = rng.normal(i, 0.035, size=len(y))
        ax.scatter(
            x,
            y,
            s=20,
            color="#333333",
            alpha=0.65,
            edgecolors="none",
            zorder=3
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=15)

    ax.set_ylabel(
        "Node-wise training RRMSE",
        fontsize=16,
        fontweight="bold"
    )

    ax.tick_params(axis="y", labelsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        transparent=True
    )
    plt.show()

# ============================================================
# 6. Example usage
# ============================================================

# # ---------- ARNI ----------
# arni1_rrmse, arni1_pred = compute_arni_rrmse(
#     X,
#     Y,
#     args,
#     base="polynomial",
#     basis_order=2,
#     expand_type="mul",
#     th=1e-4
# )
#
# arni2_rrmse, arni2_pred = compute_arni_rrmse(
#     X,
#     Y,
#     args,
#     base="fourier",
#     basis_order=2,
#     expand_type="add",
#     th=1e-4
# )
#
# # ---------- THIS ----------
# this_rrmse, this_pred, this_Ainf, this_coeff = compute_this_rrmse(
#     X.T,
#     Y.T,
#     dmax=2,
#     lam=0.1,
#     rho=1.0,
#     niter=10
# )
#
# # ---------- PIRC ----------
# # 这里假设你已经构建好了：
# # pirc, X_washout, X_train, Y_train
#
# pirc_rrmse, pirc_pred, pirc_train_mse = compute_pirc_rrmse(
#     pirc,
#     X_washout,
#     X_train,
#     Y_train
# )
#
# # ---------- Plot ----------
# plot_violin_rrmse({
#     "PIRC": pirc_rrmse,
#     "THIS": this_rrmse,
#     "ARNI$^{1}$": arni1_rrmse,
#     "ARNI$^{2}$": arni2_rrmse,
# })
#
