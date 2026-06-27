import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import math
import torch

class PIRC_flatten:
    def __init__(self, n_units, in_dim, out_dim,
                 Win, Wres,
                 sigma_in=0.5, rho=0.9, alpha=0.9, tikh=1e-4,
                 option=1, device= None, block_dim = 1,
                 mode = 0, bias=1, dt=0.01,
                 I_type=0, Expand=1):
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
        if Expand == 0:
            self.ExpandNodes = self.in_dim + 1
        elif Expand == 1:
            self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2)+1
        elif Expand == 2:
            self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2) + math.comb(self.in_dim - 1, 3)+1
        self.ExpandDim =self.ExpandNodes * self.block_dim
        self.Wres = Wres.to(device)
        self.Win = Win.to(device)
        eigs = torch.linalg.eigvals(self.Wres)
        self.Wres = self.Wres * (rho / eigs.abs().max())
        self.n_units = self.Wres.shape[1]
        self.mode = mode
        self.bias = bias
        self.dt = dt
        self.I_type = I_type

    @torch.no_grad()
    def dataProcess(self, X):
        B = X.shape[0]
        if self.Expand==2:
            X = X
        else:
            X = torch.remainder(X, 2 * torch.pi)  # 直接修改 X 本身
        if self.Expand==1: #3-order
            x1 = X[:, 0:1]  # (B,1)
            if self.option == 1: # 1,x1,x2,x3,x2x3
                rest = X[:, 1:]  # (B,D-1)
                idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
                f3 = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
            elif self.option == 2: #1,x1,x1x2,x1x3,x1x1x2x3
                # rest = X[:, 1:]
                rest = X[:, 1:] * x1
                idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
                # f3 = rest[:, idx[0]] * rest[:, idx[1]] * x1  # (B,K)
                f3 = rest[:, idx[0]] * rest[:, idx[1]]   # (B,K)
                # rest = X[:, 1:] * x1
            v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3], dim=1)  # (B,E)
            if self.block_dim == 2: #use sin-cos encoding
                sin_v = torch.sin(v)
                cos_v = torch.cos(v)
                X_proc = torch.stack((sin_v, cos_v), dim=2).reshape(B, -1)
            else:
                X_proc = v  # (B,E)
        elif self.Expand==2: #4-order
            x1 = X[:, 0:1]  # (B,1)
            rest = x1 * X[:, 1:]  # (B, n-1)
            idx2 = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
            f3 = rest[:, idx2[0]] * rest[:, idx2[1]]  # (B, C(n-1,2))
            idx3 = torch.combinations(torch.arange(rest.shape[1], device=X.device), r=3)
            f4 = rest[:, idx3[:, 0]] * rest[:, idx3[:, 1]] * rest[:, idx3[:, 2]]
            v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3, f4], dim=1)  # (B,E)
            X_proc = v  # (B,E)
        else: #2-order 1,x1,x1x2,x1x3
            x1 = X[:, 0:1]
            rest = X[:, 1:] * x1
            v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest], dim=1)  # (B,E)
            if self.block_dim == 2:
                sin_v = torch.sin(v)
                cos_v = torch.cos(v)
                X_proc = torch.stack((sin_v, cos_v), dim=2).reshape(B, -1)
            else:
                X_proc = v  # (B,E)
        return X_proc

    @torch.no_grad()
    def step(self, r_pre, x, A=None, noise=0): #r_pre B,E*N
        B = x.shape[0]
        x = self.dataProcess(x)  # expect (B, Din)
        if A is not None:
            x = x * A
        scaled_x = (x * self.sigma_in).view(B,self.ExpandNodes,self.block_dim)
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
        R = r_pre.view(B,E,N) @ self.Wres.T + self.bias  # (B, E, N)
        if noise:
            nonlinear = torch.tanh(r_new + R + noise * torch.randn_like(R)) # (B, E, N)
        else:
            nonlinear = torch.tanh(r_new + R)  # (B, E, N)
        r_post = self.alpha * nonlinear.view(B, E*N) + (1 - self.alpha) * r_pre
        return r_post

    @torch.no_grad()
    def open_loop(self, X, r0, noise=0):
        B, T, Din = X.shape
        N = self.n_units
        R = torch.empty((B, T + 1, self.ExpandNodes*N), device=self.device, dtype=r0.dtype)
        R[:, 0, :] = r0
        for t in range(1, T + 1):
            R[:, t, :] = self.step(R[:, t - 1, :], X[:, t - 1, :], noise=noise)  # 注意这里 X[:, t-1, :]，因为 step 内部会对输入进行处理
        return R

    @torch.no_grad()
    def train(self, X_washout, X_train, Y_train):
        B = X_train.shape[0]
        N = self.n_units
        O = self.out_dim
        if self.Expand==2: #4-order predict the original value instead of the difference
            Y_target_diff = Y_train[:, :, :O]
        else:
            Y_target_diff = (Y_train[:, :, :O] - X_train[:, :, :O])/self.dt # (B, T, O)
        rf_washout = torch.zeros(B, self.ExpandNodes*N, device=self.device)
        for t in range(X_washout.shape[1]):
            rf_washout = self.step(rf_washout, X_washout[:, t, :])  # (B, E*N)
        R = self.open_loop(X_train, rf_washout, noise=0.0)  # (B, T+1, E*N)
        R_train = R[:, 1:, :]  # (B, T, E*N)
        # R_train += 1e-6 * torch.randn_like(R_train)
        # I = torch.eye(self.ExpandNodes * N, device=self.device)  # (E*N, E*N)
        LHS = R_train.transpose(1, 2) @ R_train   # (B, E*N, E*N)
        LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
        RHS = R_train.transpose(1, 2) @ Y_target_diff  # (B, E*N, O)
        try:
            self.Wout = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            self.Wout = torch.linalg.pinv(LHS) @ RHS  # (B, N, O)
        Y_diff_pred = R_train @ self.Wout
        if self.Expand == 2:
            train_loss = (Y_diff_pred[:,:,0] - Y_target_diff[:,:,0]).pow(2).mean()
        else:
            pred = X_train[:, :, :O] + Y_diff_pred[:, :, :O] * self.dt
            train_loss = (Y_train[:, :, :O] - pred).pow(2).mean()
        return R_train[:, -1, :], train_loss

    @torch.no_grad()
    def train_memEff(self, X_washout, X_train, Y_train):
        B, T, Din = X_train.shape
        N = self.n_units
        O = self.out_dim
        D = self.ExpandNodes * N
        r_post = torch.zeros(B, D, device=self.device)
        for t in range(X_washout.shape[1]):
            r_post = self.step(r_post, X_washout[:, t, :])
        Wout = torch.zeros(B, D, O, device=self.device)
        # I = torch.eye(D, device=self.device)
        chunk = B
        for i in range(0, B, chunk):
            b_idx = slice(i, min(i + chunk, B))
            r = r_post[b_idx]
            chunk_size = r.shape[0]
            LHS = torch.zeros(chunk_size, D, D, device=self.device)
            RHS = torch.zeros(chunk_size, D, O, device=self.device)
            for t in range(T):
                r = self.step(r, X_train[b_idx, t])
                y_diff = (Y_train[b_idx, t, :O] - X_train[b_idx, t, :O]) / self.dt
                LHS += torch.bmm(r.unsqueeze(2), r.unsqueeze(1))
                RHS += torch.bmm(r.unsqueeze(2), y_diff.unsqueeze(1))
            LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
            try:
                Wout[b_idx] = torch.linalg.solve(LHS, RHS)
            except RuntimeError:
                Wout[b_idx] = torch.linalg.pinv(LHS) @ RHS
            r_post[b_idx] = r
        self.Wout = Wout
        return r_post

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
    def evolve(self, r0, N_evo, Y_train=None):
        B = r0.shape[0]
        Yh = torch.zeros((B, N_evo + 1, self.in_dim), device=self.device, dtype=r0.dtype)
        if self.Expand == 2:
            Yh[:, 0, :] = self.apply_wout(r0)
        else:
            Yh[:, 0, :] = self.apply_wout(r0)*self.dt+Y_train[:,-2,:]  # (B, out_dim) ; 注意这里假设 out_dim == in_dim
        for t in range(1, N_evo + 1):
            r0 = self.step(r0, Yh[:, t - 1, :])
            if self.Expand == 2:
                Yh[:, t, :] = self.apply_wout(r0)
            else:
                Yh[:, t, :] = self.apply_wout(r0) * self.dt + Yh[:, t - 1, :]
        return Yh[:, 1:, 0:1]

    @torch.no_grad()
    def evolve_edge_removal(self, r0, A_in, N_evo, Y_train=None):
        B = r0.shape[0]
        N = self.n_units
        E = A_in.shape[0]
        device = r0.device
        R_base = r0.unsqueeze(0).expand(E, B, E*N).clone()  # (E,B,E*N)
        R_int = r0.unsqueeze(0).expand(E, B, E*N).clone()  # (E,B,E*N)
        Aout = torch.ones(self.out_dim, device=device)
        Aout[0] = 0
        Aint = torch.zeros(self.out_dim, device=device)
        Aint[0] = 1
        Y_int_collect = torch.zeros((E, B, N_evo + 1, 1), device=device)
        if self.Expand==2:
            y0 = self.apply_wout(R_base)
        else:
            y0 = self.apply_wout(R_base)* self.dt + Y_train[:, -2, :]
        Y_prev_base = y0 # (E,B,O)
        Y_prev_int = y0 # (E,B,O)
        A_expand = A_in.repeat_interleave(B, dim=0) #E,D -> EB,D
        if self.mode == 2:
            mask_state = torch.ones((E, E, N), device=self.device)  # (edge, group, N)
            for i in range(E):
                mask_state[i, i, :] = 0  # 删除第 i 个 group
        for k in range(1, N_evo + 1):
            R_base_new = self.step(
                R_base.reshape(E * B, E*N),
                Y_prev_base.reshape(E * B, -1)
            ).reshape(E, B, E*N)
            y_base = self.apply_wout(R_base_new)
            # if self.mode == 1:
            if self.I_type == 1:
                R_int = self.step(
                    R_base.reshape(E * B, E*N),
                    Y_prev_base.reshape(E * B, -1),
                    A=A_expand
                ).reshape(E, B, E*N)
            elif self.I_type == 2:
                R_int = self.step(
                    R_int.reshape(E * B, E*N),
                    Y_prev_int.reshape(E * B, -1),
                    A=A_expand
                ).reshape(E, B, E*N)
            # if self.mode == 2:
            #     R_int = R_base_new * mask_state.view(E, 1, E * N)
            R_base= R_base_new
            y_int = self.apply_wout(R_int)
            y_mix = y_base * Aout + y_int * Aint
            if self.Expand == 2:
                y_mix = y_mix
                y_ture = y_base
            else:
                y_mix = y_mix * self.dt +  Y_prev_int
                y_ture= y_base * self.dt +  Y_prev_base
            if self.I_type==1:
                Y_prev_base = y_ture
            if self.I_type==2:
                Y_prev_base = y_mix
            Y_int_collect[:, :, k, :] = y_mix[:,:, :1] #只收集第一个维度的预测结果用于后续的AUC计算
            Y_prev_int = y_mix
        return Y_int_collect[:, :, 1:, :]

    @torch.no_grad()
    def Prediction(self, R, N_test=100, Y_train=None):
        B = Y_train.shape[0]
        E = self.ExpandNodes
        Prediction = torch.zeros((E+1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        Prediction[0,:,:,:] = self.evolve(r0, N_test, Y_train)  # baseline
        group_in = (torch.arange(self.ExpandDim, device=self.device) // self.block_dim).clamp_max(E - 1)  # (D,)
        A_in = (group_in.unsqueeze(0) != torch.arange(E, device=self.device).unsqueeze(1)).to(torch.float32)  # (E, D)
        Prediction[1:,:,:,:] = self.evolve_edge_removal(r0, A_in, N_test, Y_train)
        return Prediction

    @torch.no_grad()
    def Validation(self, R, X_test=None, Y_test=None):
        B = X_test.shape[0]
        test_length = X_test.shape[1]
        E = self.ExpandNodes
        r0 = R
        Yh = torch.zeros((B, test_length + 1, self.in_dim), device=self.device, dtype=r0.dtype)
        if self.Expand == 2:
            Yh[:, 0, :] = self.apply_wout(r0)
        for t in range(1, test_length + 1):
            r0 = self.step(r0, Yh[:, t - 1, :])
            if self.Expand == 2:
                Yh[:, t, :] = self.apply_wout(r0)
            else:
                Yh[:, t, :] = self.apply_wout(r0) * self.dt + Yh[:, t - 1, :]
        validation_loss=(Y_test[:, :, 0:1] - Yh[:, 1:, 0:1]).pow(2).mean()
        return Yh[:, 1:, :], validation_loss


