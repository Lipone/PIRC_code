import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
import math
import torch

class PIRC_flatten:
    def __init__(self, n_units, in_dim, out_dim,Win, Wres,sigma_in=0.5, rho=0.9, alpha=0.9, tikh=1e-4,option=1, device= None, block_dim = 1,mode = 0, bias=1, dt=0.01,I_type=0, Expand=1):
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
        if self.Expand==1:
            x1 = X[:, 0:1]  # (B,1)
            if self.option == 1: #1,x1,x2,x3,x2x3
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
            if self.block_dim == 2:
                sin_v = torch.sin(v)
                cos_v = torch.cos(v)
                X_proc = torch.stack((sin_v, cos_v), dim=2).reshape(B, -1)
            else:
                X_proc = v  # (B,E)
        elif self.Expand==2:
            x1 = X[:, 0:1]  # (B,1)
            rest = x1 * X[:, 1:]  # (B, n-1)
            idx2 = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
            f3 = rest[:, idx2[0]] * rest[:, idx2[1]]  # (B, C(n-1,2))
            idx3 = torch.combinations(torch.arange(rest.shape[1], device=X.device), r=3)
            f4 = rest[:, idx3[:, 0]] * rest[:, idx3[:, 1]] * rest[:, idx3[:, 2]]
            v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3, f4], dim=1)  # (B,E)
            X_proc = v  # (B,E)
        else:
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
        if self.Expand == 2:
            Y_target_diff = Y_train[:, :, 0:1]
        else:
            Y_target_diff = (Y_train[:, :, 0:1] - X_train[:, :, 0:1]) / self.dt  # (B, T, O)
        rf_washout = torch.zeros(B, self.ExpandNodes * N, device=self.device)
        for t in range(X_washout.shape[1]):
            rf_washout = self.step(rf_washout, X_washout[:, t, :])  # (B, E*N)
        R = self.open_loop(X_train, rf_washout, noise=0.0)  # (B, T+1, E*N)
        R_train = R[:, 1:, :]  # (B, T, E*N)
        LHS = R_train.transpose(1, 2) @ R_train  # (B, E*N, E*N)
        LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
        RHS = R_train.transpose(1, 2) @ Y_target_diff  # (B, E*N, O)
        try:
            self.Wout = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            self.Wout = torch.linalg.pinv(LHS) @ RHS  # (B, N, O)
        Y_diff_pred = R_train @ self.Wout
        if self.Expand == 2:
            train_loss = (Y_diff_pred - Y_target_diff).pow(2).mean()
        else:
            pred = X_train[:, :, :O] + Y_diff_pred[:, :, :O] * self.dt
            train_loss = (Y_train[:, :, :O] - pred).pow(2).mean()
        return R_train[:, -1, :], train_loss

    @torch.no_grad()
    # def train_multi_series(self, X_washout, X_train, Y_train):
    #     B = X_train.shape[0]
    #     N = self.n_units
    #     O = self.out_dim
    #     train_loss = 0
    #     R_train_list = []
    #     for dim_t in range(O):
    #         order = [dim_t] + [i for i in range(O) if i != dim_t]
    #         Y_target_diff = Y_train[:, :, dim_t:dim_t + 1]
    #         X_washout_new = X_washout[:, :, order]
    #         X_train_new = X_train[:, :, order]
    #         rf_washout = torch.zeros(B, self.ExpandNodes * N, device=self.device)
    #         for t in range(X_washout.shape[1]):
    #             rf_washout = self.step(rf_washout, X_washout_new[:, t, :])
    #         R = self.open_loop(X_train_new, rf_washout, noise=0.0)
    #         R_train = R[:, 1:, :]
    #         LHS = R_train.transpose(1, 2) @ R_train  # (B, E*N, E*N)
    #         LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
    #         RHS = R_train.transpose(1, 2) @ Y_target_diff  # (B, E*N, O)
    #         try:
    #             Wout = torch.linalg.solve(LHS, RHS)
    #         except RuntimeError:
    #             Wout = torch.linalg.pinv(LHS) @ RHS  # (B, N, O)
    #         Y_diff_pred = R_train @ Wout
    #         train_loss += (Y_diff_pred - Y_target_diff).pow(2).mean()
    #         # self.Wout.append(Wout)
    #         R_train_list.append(R_train[:, -1, :])
    #     return R_train_list, train_loss / O

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
        O = self.out_dim
        Yh = torch.zeros((1,N_evo + 1,O), device=self.device, dtype=r0.dtype)
        Yh_base = torch.zeros((B,N_evo + 1,O), device=self.device, dtype=r0.dtype)
        for b in range(O):
            Yh[0,0,b:b+1]=r0[b:b+1,:]@ self.Wout[b,:,:]
        for t in range(1, N_evo + 1):
            for b in range(O):
                order = [b] + [i for i in range(O) if i != b]
                r0[b:b+1,:] = self.step(r0[b:b+1,:], Yh[:, t - 1, order])
                Yh[:, t, b:b+1]=r0[b:b+1,:]@ self.Wout[b,:,:]
        for b in range(O):
            order = [b] + [i for i in range(O) if i != b]
            Yh_base[b:b+1,:,:]=Yh[:,:,order]
        return Yh[:, 1:, :].permute(2,1,0), Yh_base

    @torch.no_grad()
    def evolve_edge_removal(self, r0, A_in, N_evo, Y_train=None, Yh_base=None):
        B = r0.shape[0]
        N = self.n_units
        E = A_in.shape[0]
        O = self.out_dim
        device = r0.device
        R_base = r0 # (B,E*N)
        Y_int_collect = torch.zeros((E, B, N_evo + 1, 1), device=device)
        y0=Yh_base[:,0,:] # (B,O)
        Y_prev_base = y0 # (B,O)
        # Y_prev_int = y0.expand(E, B, O).clone() # (E,B,O)
        A_expand = A_in.repeat_interleave(B, dim=0) #E,D -> EB,D
        for k in range(1, N_evo + 1):
            R_base_new = self.step(R_base,Y_prev_base) # (B,E*N)
            y_base = Yh_base[:,k,:] #(B,O)
            R_int = self.step(
                R_base.expand(E, B, E*N).reshape(E * B, E*N),
                Y_prev_base.expand(E, B, O).reshape(E * B, -1),
                A=A_expand
            ).reshape(E, B, E*N)
            R_base= R_base_new
            y_int = self.apply_wout(R_int) # (E,B,1)
            y_mix = torch.cat([y_int, y_base.expand(E, B, O)[:, :, 1:]], dim=2)
            if self.Expand == 2:
                y_mix = y_mix
                y_ture = y_base
            # else:
            #     y_mix = y_mix * self.dt +  Y_prev_int
            #     y_ture= y_base * self.dt +  Y_prev_base
            if self.I_type==1:
                Y_prev_base = y_ture
            Y_int_collect[:, :, k, :] = y_mix[:,:, :1] #只收集第一个维度的预测结果用于后续的AUC计算
            # Y_prev_int = y_mix
        return Y_int_collect[:, :, 1:, :]

    @torch.no_grad()
    def Prediction(self, R, N_test=100, Y_train=None):
        B = Y_train.shape[0]
        E = self.ExpandNodes
        Prediction = torch.zeros((E + 1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        Prediction[0, :, :, :],Yh_base = self.evolve(r0, N_test, Y_train)  # baseline
        group_in = (torch.arange(self.ExpandDim, device=self.device) // self.block_dim).clamp_max(E - 1)  # (D,)
        A_in = (group_in.unsqueeze(0) != torch.arange(E, device=self.device).unsqueeze(1)).to(torch.float32)  # (E, D)
        Prediction[1:, :, :, :] = self.evolve_edge_removal(r0, A_in, N_test, Y_train, Yh_base)
        return Prediction



