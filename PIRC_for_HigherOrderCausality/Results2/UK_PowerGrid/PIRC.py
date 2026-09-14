import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import math
import torch

class PIRC_flatten: # in this version, we mask the reservoir states
    def __init__(self, n_units, in_dim, out_dim, Win, Wres, sigma_in=0.5, rho=0.9, alpha=0.9, tikh=1e-4,option=1, device= None, block_dim = 1,mode = 0, bias=0, dt=0.01, I_type=0, Expand=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_units = n_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_in = sigma_in
        self.rho = rho
        self.tikh = tikh
        self.alpha = alpha
        self.option = option
        self.block_dim = block_dim #block_dim=1,2,4
        self.Expand = Expand
        self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2) + 1
        self.Wres = Wres.to(device)
        self.Win = Win.to(device)
        eigs = torch.linalg.eigvals(self.Wres)
        if Wres.dim() == 2:
            self.Wres = self.Wres * (rho / eigs.abs().max())
        elif Wres.dim() == 3:
            rho_raw = torch.max(torch.abs(eigs), dim=1, keepdim=True)[0]  # (E,1)
            rho_raw = torch.clamp(rho_raw, min=1e-6)  # 防止除0
            rho_raw = rho_raw.unsqueeze(-1)  # (E,1,1)
            self.Wres = Wres * (rho / rho_raw)
        self.n_units = self.Wres.shape[1]
        self.mode = mode
        self.bias = bias
        self.dt = dt
        self.I_type = I_type

    # @torch.no_grad()
    # def dataProcess(self, X):
    #     B = X.shape[0]
    #     bias = torch.ones(B, 1, device=X.device)
    #     X = torch.remainder(X, 2 * torch.pi)
    #     x1 = X[:, 0:1]  # (B,1)
    #     if self.option == 1: #1,x1,x1x2,x1x3,x1x2x3
    #         rest = X[:, 1:]
    #         idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
    #         f3 = rest[:, idx[0]] * rest[:, idx[1]] * x1  # (B,K)
    #         rest = X[:, 1:] * x1
    #         v = torch.cat([bias, x1, rest, f3], dim=1)  # (B,E)
    #         X_proc = v  # (B,E)
    #     elif self.option == 2: #1,x1,x1x2,x1x3,x1x1x2x3
    #         rest = X[:, 1:] * x1
    #         idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
    #         f3 = rest[:, idx[0]] * rest[:, idx[1]]   # (B,K)
    #         v = torch.cat([bias, x1, rest, f3], dim=1)  # (B,E)
    #         X_proc = v  # (B,E)
    #     elif self.option == 3:
    #         rest = x1 * X[:, 1:]  # (B, n-1)
    #         idx2 = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
    #         f3 = rest[:, idx2[0]] * rest[:, idx2[1]]  # (B, C(n-1,2))
    #         idx3 = torch.combinations(torch.arange(rest.shape[1], device=X.device), r=3)
    #         f4 = rest[:, idx3[:, 0]] * rest[:, idx3[:, 1]] * rest[:, idx3[:, 2]]
    #         v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3, f4], dim=1)  # (B,E)
    #         X_proc = v  # (B,E)
    #     return X_proc
    def dataProcess(self, X):
        B,D = X.shape
        bias = torch.ones(B, 1, device=X.device)
        X = torch.remainder(X, 2 * torch.pi)
        x1 = X[:, 0:1]  # (B,1)
        if self.option == 1: #1,x1,x1x2,x1x3,x1x2x3
            rest = X[:, 1:]
            idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
            f3 = rest[:, idx[0]] * rest[:, idx[1]] * x1  # (B,K)
            rest = X[:, 1:] * x1
            v = torch.cat([bias, x1, rest, f3], dim=1)  # (B,E)
            X_proc = v  # (B,E)
        elif self.option == 2:  # (1,1,x1,x1x1,x2,x1x2,x3,x1x3,x2x3,x1x1x2x3)
            rest = X[:, 1:]  # (B,D-1)
            M = D - 1  # M = D-1
            x1_sq = x1 * x1
            pair = torch.stack([rest, x1 * rest], dim=2)
            pair_flat = pair.reshape(B, -1)
            idx = torch.triu_indices(M, M, offset=1, device=X.device)
            f_rest_cross = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
            f_high = x1_sq * f_rest_cross
            tri = torch.stack([f_high, f_high], dim=2)
            tri_flat = tri.reshape(B, -1)
            v = torch.cat([bias, bias, x1, x1_sq, pair_flat, tri_flat], dim=1)
            X_proc = v
        elif self.option == 3:  # (1,1,x1,x1x1,x2,x1x2,x3,x1x3,x2x3,x1x2x3)
            rest = X[:, 1:]  # (B,D-1)
            M = D - 1  # M = D-1
            x1_sq = x1 * x1
            pair = torch.stack([ rest, x1 *rest], dim=2)
            pair_flat = pair.reshape(B, -1)
            idx = torch.triu_indices(M, M, offset=1, device=X.device)
            f_rest_cross = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
            f_high = x1 * f_rest_cross
            tri = torch.stack([f_rest_cross, f_high], dim=2)
            tri_flat = tri.reshape(B, -1)
            v = torch.cat([bias, bias, x1_sq, x1, pair_flat, tri_flat], dim=1)
            X_proc = v
        return X_proc

    @torch.no_grad()
    def step(self, r_pre, x, A=None, noise=None): #r_pre B,E*N
        B = x.shape[0]
        x = self.dataProcess(x)  # expect (B, Din)
        if A is not None:
            x = x * A
        scaled_x = (x * self.sigma_in).reshape(B,self.ExpandNodes,self.block_dim)
        B, E, D = scaled_x.shape
        N = self.n_units
        chunk = E # 建议 16 / 32 / 64，根据显存调
        r_new = torch.empty(B, E, N, device=scaled_x.device)
        for n0 in range(0, E, chunk):
            n1 = min(n0 + chunk, E)
            r_new[:, n0:n1, :] = torch.einsum(
                "bed,edn->ben",
                scaled_x[:, n0:n1, :],
                self.Win[n0:n1, :, :]
            )
        if self.Wres.dim() == 2:
            R = r_pre.reshape(B,E,N) @ self.Wres.T + self.bias  # (B, E, N)
        else:
            R = torch.einsum("ben,enn->ben",r_pre.reshape(B,E,N), self.Wres.transpose(1,2))+ self.bias
        if noise:
            nonlinear = torch.tanh(r_new + R + noise * torch.randn_like(R)) # (B, E, N)
        else:
            nonlinear = torch.tanh(r_new + R)  # (B, E, N)
        r_post = self.alpha * nonlinear.reshape(B, E*N) + (1 - self.alpha) * r_pre
        return r_post

    @torch.no_grad()
    def open_loop(self, X, r0, noise=None):
        B, T, Din = X.shape
        N = self.n_units
        R = torch.empty((B, T + 1, self.ExpandNodes*N), device=self.device, dtype=r0.dtype)
        R[:, 0, :] = r0
        for t in range(1, T + 1):
            R[:, t, :] = self.step(R[:, t - 1, :], X[:, t - 1, :], noise=noise)
        return R

    @torch.no_grad()
    def train(self, X_washout, X_train, Y_train):
        B = X_train.shape[0]
        N = self.n_units
        O = self.out_dim
        if self.Expand == 2:
            Y_target_diff = Y_train[:, :, 0:1]
        else:
            Y_target_diff = (Y_train[:, :, 0:1] - X_train[:, :, 0:1]) / self.dt  # (B, T, 1)
        rf_washout = torch.zeros(B, self.ExpandNodes * N, device=self.device)
        for t in range(X_washout.shape[1]):
            rf_washout = self.step(rf_washout, X_washout[:, t, :])  # (B, E*N)
        R = self.open_loop(X_train, rf_washout, noise=None)  # (B, T+1, E*N)
        R_train = R[:, 1:, :]  # (B, T, E*N)
        LHS = R_train.transpose(1, 2) @ R_train  # (B, E*N, E*N)
        LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
        RHS = R_train.transpose(1, 2) @ Y_target_diff  # (B, E*N, O)
        try:
            self.Wout = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            self.Wout = torch.linalg.pinv(LHS) @ RHS  # (B, N, O)
        Y_diff_pred = R_train @ self.Wout

        Y_pred_np = Y_diff_pred.squeeze(-1).detach().cpu().numpy()
        Y_true_np = Y_target_diff.squeeze(-1).detach().cpu().numpy()
        Y_true = np.asarray(Y_true_np)
        Y_pred = np.asarray(Y_pred_np)
        eps = 1e-12
        rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=1))
        scale = np.sqrt(np.mean(Y_true ** 2, axis=1)) + eps
        pirc_rrmse = rmse / scale

        if self.Expand == 2:
            train_loss = (Y_diff_pred - Y_target_diff).pow(2).mean()
        else:
            pred = X_train[:, :, :O] + Y_diff_pred[:, :, :O] * self.dt
            train_loss = (Y_train[:, :, :O] - pred).pow(2).mean()
        return R_train[:, -1, :], rmse, pirc_rrmse

    @torch.no_grad()
    def apply_wout(self,r):
        if self.Wout.dim() == 2:
            return r @ self.Wout
        elif self.Wout.dim() == 3:
            if r.dim() == 2: #(B,N) evolve
                return (r.unsqueeze(1) @ self.Wout).squeeze(1)
            elif r.dim() == 3: #多实验并行的edge removal
                return torch.einsum("ebn,bno->ebo", r, self.Wout)

    @torch.no_grad()
    def evolve_with_edge_removal(self, r0, N_evo, Y_train=None, Y_test=None):
        B, EN = r0.shape
        O = self.out_dim
        E = self.ExpandNodes
        N = self.n_units
        device = r0.device
        dtype = r0.dtype
        r_base = r0.clone()
        perm = torch.empty((O, O), dtype=torch.long, device=device)
        for b in range(O):
            perm[b] = torch.tensor(
                [b] + [i for i in range(O) if i != b],
                device=device
            )
        x_base = Y_train[:, -1, :].clone()
        x_global = x_base[:, 0].clone()
        Prediction_base = torch.zeros((B, N_evo, 1), device=device, dtype=dtype)
        Prediction_int = torch.zeros((E, B, N_evo, 1), device=device, dtype=dtype)
        mask_state = torch.ones((E, E, N), device=device, dtype=dtype)
        for e in range(E):
            mask_state[e, e, :] = 0.0
        mask_state = mask_state.reshape(E, 1, E * N)
        for t in range(N_evo):
            r_base = self.step(r_base, x_base)
            y_base = self.apply_wout(r_base).squeeze(-1)  # (B,)
            x_global_next = x_global + y_base * self.dt # (B,)
            x_base_next = x_global_next[perm]
            Prediction_base[:, t, 0] = x_global_next
            r_int = r_base.unsqueeze(0) * mask_state
            y_int = self.apply_wout(r_int).squeeze(-1)  # (E,B)
            x_int = x_global.unsqueeze(0) + y_int * self.dt # （E,B)
            Prediction_int[:, :, t, 0] = x_int
            x_global = x_global_next
            x_base = x_base_next
        return Prediction_base, Prediction_int

    @torch.no_grad()
    def evolve_with_edge_removal2(self, r0, N_evo, Y_train=None, Y_test=None):
        B, EN = r0.shape
        E = self.ExpandNodes
        N = self.n_units
        device = r0.device
        dtype = r0.dtype
        r_base = r0.clone()
        Prediction_base = torch.zeros((B, N_evo, 1), device=device, dtype=dtype)
        Prediction_int = torch.zeros((E, B, N_evo, 1), device=device, dtype=dtype)
        mask_state = torch.ones((E, E, N), device=device, dtype=dtype)
        for e in range(E):
            mask_state[e, e, :] = 0.0
        mask_state = mask_state.reshape(E, 1, E * N)
        x_input = Y_train[:, -1, :].clone()
        for t in range(N_evo):
            r_base = self.step(r_base, x_input)
            y_base = self.apply_wout(r_base).squeeze(-1)
            x_base_pred = x_input[:, 0] + y_base * self.dt
            Prediction_base[:, t, 0] = x_base_pred
            r_int = r_base.unsqueeze(0) * mask_state
            y_int = self.apply_wout(r_int).squeeze(-1)
            x_int_pred = x_input[:, 0].unsqueeze(0) + y_int * self.dt
            Prediction_int[:, :, t, 0] = x_int_pred
            x_input = Y_test[:, t, :].clone()
        return Prediction_base, Prediction_int

    @torch.no_grad()
    def Prediction(self, R, N_test=100, Y_train=None, Y_test=None):
        B = R.shape[0]
        E = self.ExpandNodes
        Prediction = torch.zeros((E + 1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        Prediction[0, :, :, :], Prediction[1:, :, :, :] = self.evolve_with_edge_removal(r0,N_test,Y_train=Y_train,Y_test=Y_test)
        return Prediction

    def Prediction2(self, R, N_test=100, Y_train=None, Y_test=None):
        B = R.shape[0]
        E = self.ExpandNodes
        Prediction = torch.zeros((E + 1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        Prediction[0, :, :, :], Prediction[1:, :, :, :] = self.evolve_with_edge_removal2(r0,N_test,Y_train=Y_train,Y_test=Y_test)
        return Prediction


