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
                 I_type=0, Expand=1, use_sin=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_units = n_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma_in = sigma_in
        self.rho = rho
        self.tikh = tikh
        self.alpha = alpha
        self.option = option
        self.use_sin = use_sin
        self.block_dim = block_dim #block_dim=1,2,4
        self.Expand = Expand
        if Expand == 0:
            self.ExpandNodes = self.in_dim + 1
        elif Expand == 1:
            self.ExpandNodes = (self.in_dim + math.comb(self.in_dim - 1, 2)+1)*2
        elif Expand == 2:
            self.ExpandNodes = self.in_dim + math.comb(self.in_dim - 1, 2) + math.comb(self.in_dim - 1, 3)+1
        self.ExpandDim =self.ExpandNodes * self.block_dim
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
        self.mode = mode
        self.bias = bias
        self.dt = dt
        self.I_type = I_type

    @torch.no_grad()
    def dataProcess(self, X):
        B,D = X.shape
        if self.use_sin==0:
            X = X
        else:
            X = torch.remainder(X, 2 * torch.pi)  # 直接修改 X 本身
        if self.Expand==1:
            x1 = X[:, 0:1]  # (B,1)
            if self.option == 1: #(1,x1,x2,x3,x2x3)
                rest = X[:, 1:]  # (B,D-1)
                idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
                f3 = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
                v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3], dim=1)  # (B,E)
            elif self.option == 2: #(1,x1,x1x2,x1x3,x1x1x2x3)
                rest = X[:, 1:] * x1
                idx = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
                f3 = rest[:, idx[0]] * rest[:, idx[1]]   # (B,K)
                v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3], dim=1)  # (B,E)
            elif self.option == 3: # (1,1,x1,x1x1,x2,x1x2,x3,x1x3,x2x3,x1x1x2x3)
                x1 = X[:, 0:1]  # (B,1)
                rest = X[:, 1:]  # (B,D-1)
                M = D-1  # M = D-1
                ones = torch.ones(B, 1, device=X.device)
                x1_sq = x1 * x1
                pair = torch.stack([rest, x1 * rest], dim=2)
                pair_flat = pair.reshape(B, -1)
                idx = torch.triu_indices(M, M, offset=1, device=X.device)
                f_rest_cross = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
                f_high = x1_sq * f_rest_cross
                v = torch.cat([ones, ones, x1,  x1_sq,  pair_flat,  f_rest_cross,  f_high], dim=1)
            elif self.option == 4: # (1,1,x1,x1x1,x1x2,x1x2,x1x3,x1x3,x1x2x3,x1x1x2x3)
                x1 = X[:, 0:1]  # (B,1)
                rest = X[:, 1:]  # (B,D-1)
                M = D-1  # M = D-1
                ones = torch.ones(B, 1, device=X.device)
                x1_sq = x1 * x1
                pair = torch.stack([x1 *rest, x1 * rest], dim=2)
                pair_flat = pair.reshape(B, -1)
                idx = torch.triu_indices(M, M, offset=1, device=X.device)
                f_rest_cross = x1 *rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
                f_high = x1 * f_rest_cross
                high_order= torch.stack([f_high, f_high], dim=2)
                high_order = high_order.reshape(B, -1)
                v = torch.cat([ones, ones, x1,  x1,  pair_flat,  high_order], dim=1)
            elif self.option == 5: # (1,1,x1,x1x1,x2,x1x2,x3,x1x3,x2x3,x1x2x3)
                x1 = X[:, 0:1]  # (B,1)
                rest = X[:, 1:]  # (B,D-1)
                M = D-1  # M = D-1
                ones = torch.ones(B, 1, device=X.device)
                x1_sq = x1 * x1
                pair = torch.stack([rest, x1 * rest], dim=2)
                pair_flat = pair.reshape(B, -1)
                idx = torch.triu_indices(M, M, offset=1, device=X.device)
                f_rest_cross = rest[:, idx[0]] * rest[:, idx[1]]  # (B,K)
                f_high = x1 * f_rest_cross
                high_order= torch.stack([f_rest_cross, f_high], dim=2)
                high_order = high_order.reshape(B, -1)
                v = torch.cat([ones, ones, x1,  x1_sq,  pair_flat,  high_order], dim=1)
            if self.use_sin == 1:
                sin_v = torch.sin(v)
                cos_v = torch.cos(v)
                X_proc = torch.stack((sin_v, cos_v), dim=2).reshape(B, -1)
            else:
                X_proc = v  # (B,E)
        elif self.Expand==2: # 4 order
            x1 = X[:, 0:1]  # (B,1)
            rest = x1 * X[:, 1:]  # (B, n-1)
            idx2 = torch.triu_indices(rest.shape[1], rest.shape[1], offset=1, device=X.device)
            f3 = rest[:, idx2[0]] * rest[:, idx2[1]]  # (B, C(n-1,2))
            idx3 = torch.combinations(torch.arange(rest.shape[1], device=X.device), r=3)
            f4 = rest[:, idx3[:, 0]] * rest[:, idx3[:, 1]] * rest[:, idx3[:, 2]]
            v = torch.cat([torch.ones(B, 1, device=X.device), x1, rest, f3, f4], dim=1)  # (B,E)
            X_proc = v  # (B,E)
        else: # Pair-wise only
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
        # r_new=torch.einsum("bed,dn->ben", scaled_x, self.Win)
        chunk = E  # 建议 16 / 32 / 64，根据显存调
        r_new = torch.empty(B, E, N, device=scaled_x.device)
        for n0 in range(0, E, chunk):
            n1 = min(n0 + chunk, E)
            r_new[:, n0:n1, :] = torch.einsum(
                "bed,edn->ben",
                scaled_x[:, n0:n1, :],
                self.Win[n0:n1, :, :]
            )
        if self.Wres.dim() == 2:
            R = r_pre.view(B,E,N) @ self.Wres.T + self.bias  # (B, E, N)
        else:
            R = torch.einsum("ben,enn->ben",r_pre.view(B,E,N), self.Wres.transpose(1,2))+ self.bias
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
        if self.Expand==2:
            Y_target_diff = Y_train[:, :, :O]
        else:
            Y_target_diff = (Y_train[:, :, :O] - X_train[:, :, :O])/self.dt # (B, T, O)
        rf_washout = torch.zeros(B, self.ExpandNodes*N, device=self.device)
        for t in range(X_washout.shape[1]):
            rf_washout = self.step(rf_washout, X_washout[:, t, :])  # (B, E*N)
        R = self.open_loop(X_train, rf_washout, noise=0.0)  # (B, T+1, E*N)
        R_train = R[:, 1:, :]  # (B, T, E*N)
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
                # r = r + 1e-6 * torch.randn_like(r)
                y_diff = (Y_train[b_idx, t, :O] - X_train[b_idx, t, :O]) / self.dt
                LHS += torch.bmm(r.unsqueeze(2), r.unsqueeze(1))
                RHS += torch.bmm(r.unsqueeze(2), y_diff.unsqueeze(1))
                # LHS += torch.einsum("bi,bj->bij", r, r)
                # RHS += torch.einsum("bi,bo->bio", r, y_diff)
            # try:
            #     Wout[b_idx] = torch.linalg.solve(
            #         LHS + self.tikh * I,
            #         RHS
            #     )
            # except RuntimeError:
            #     Wout[b_idx] = torch.linalg.lstsq(LHS + self.tikh * I, RHS).solution
            # idx = torch.arange(D, device=self.device)
            # LHS[:, idx, idx] += self.tikh
            LHS.diagonal(dim1=-2, dim2=-1).add_(self.tikh)
            try:
                # L = torch.linalg.cholesky(LHS)
                # Wout[b_idx] = torch.linalg.solve(RHS, LHS)
                Wout[b_idx] = torch.linalg.solve(LHS, RHS)
            except RuntimeError:
                Wout[b_idx] = torch.linalg.pinv(LHS) @ RHS
                # for j in range(chunk_size):
                #     Wout[b_idx.start + j] = torch.linalg.solve(LHS[j], RHS[j])
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
                # return torch.einsum("bn,bno->bo", r, self.Wout)
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
        R_base = r0.unsqueeze(0).expand(E, B, self.ExpandNodes*self.n_units).clone()  # (E,B,E*N)
        R_int = r0.unsqueeze(0).expand(E, B, self.ExpandNodes*self.n_units).clone()  # (E,B,E*N)
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
        # if self.mode == 2:
        #     mask_state = torch.ones((E, E, N), device=self.device)  # (edge, group, N)
        #     for i in range(E):
        #         mask_state[i, i, :] = 0  # 删除第 i 个 group
        for k in range(1, N_evo + 1):
            R_base_new = self.step(
                R_base.reshape(E * B, self.ExpandNodes*self.n_units),
                Y_prev_base.reshape(E * B, -1)
            ).reshape(E, B, self.ExpandNodes*self.n_units)
            y_base = self.apply_wout(R_base_new)
            # if self.mode == 1:
            if self.I_type == 1:
                R_int = self.step(
                    R_base.reshape(E * B, self.ExpandNodes*self.n_units),
                    Y_prev_base.reshape(E * B, -1),
                    A=A_expand
                ).reshape(E, B, self.ExpandNodes*self.n_units)
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
        E = self.ExpandNodes//2
        Prediction = torch.zeros((E+1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        # r1 = r0.clone()
        Prediction[0,:,:,:] = self.evolve(r0, N_test, Y_train)  # baseline
        # r2 = r0.clone()
        group_in = (torch.arange(self.ExpandDim, device=self.device) // (self.block_dim*2)).clamp_max(E - 1)  # (D,)
        A_in = (group_in.unsqueeze(0) != torch.arange(E, device=self.device).unsqueeze(1)).to(torch.float32)  # (E, D)
        Prediction[1:,:,:,:] = self.evolve_edge_removal(r0, A_in, N_test, Y_train)
        return Prediction

    @torch.no_grad()
    def train2(self, X_washout, X_train, Y_train):
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
        # Y_diff_pred = R_train @ self.Wout
        # if self.Expand == 2:
        #     train_loss = (Y_diff_pred - Y_target_diff).pow(2).mean()
        # else:
        #     diff=Y_diff_pred[:, :, :O] * self.dt
        #     pred = X_train[:, :, :O] + diff
        #     train_loss = (Y_train[:, :, :O] - pred).pow(2).mean()
        return R_train[:, -1, :], 0

    @torch.no_grad()
    def Prediction2(self, R, N_test=100, Y_train=None):
        B = Y_train.shape[0]
        E = self.ExpandNodes//2
        Prediction = torch.zeros((E + 1, B, N_test, 1), device=self.device, dtype=R.dtype)
        r0 = R
        r1=r0.clone()
        Prediction[0, :, :, :], Yh_base = self.evolve2(r0, N_test, Y_train)  # baseline
        r2=r0.clone()
        group_in = (torch.arange(self.ExpandDim, device=self.device) // (self.block_dim*2)).clamp_max(E - 1)  # (D,)
        A_in = (group_in.unsqueeze(0) != torch.arange(E, device=self.device).unsqueeze(1)).to(torch.float32)  # (E, D)
        Prediction[1:, :, :, :] = self.evolve_edge_removal2(r0, A_in, N_test, Y_train, Yh_base)
        return Prediction

    @torch.no_grad()
    def evolve2(self, r0, N_evo, Y_train=None):
        B = r0.shape[0]
        O = self.out_dim
        Yh = torch.zeros((1, N_evo + 1, O), device=self.device, dtype=r0.dtype)
        Yh_base = torch.zeros((B, N_evo + 1, O), device=self.device, dtype=r0.dtype)
        # Yh[0,0,:] = (torch.einsum('bn,bnk->bk', r0, self.Wout)).squeeze(-1) * self.dt + Y_train[0,-2,:]
        Yh[0,0,:] = Y_train[0,-1,:]
        r_1 = r0.clone()
        R_base_new = self.step(r0, Y_train[:,-1,:])
        r_2 = r0.clone()
        diff_base=self.apply_wout(R_base_new)
        test = (self.apply_wout(R_base_new) * self.dt).squeeze(-1) + Yh[0,0,:]
        for t in range(1, N_evo + 1):
            for b in range(O):
                order = [b] + [i for i in range(O) if i != b]
                r_2[b:b + 1, :] = self.step(r_2[b:b + 1, :], Yh[:, t - 1, order])
                diff=r_2[b:b + 1, :] @ self.Wout[b, :, :]
                if self.Expand == 2:
                    Yh[:, t, b:b + 1] = diff
                else:
                    Yh[:, t, b:b + 1] = diff*self.dt+Yh[:,t-1,b:b + 1]
        for b in range(O):
            order = [b] + [i for i in range(O) if i != b]
            Yh_base[b:b + 1, :, :] = Yh[:, :, order]
        Yh_prediction = Yh[:, 1:, :].permute(2, 1, 0)
        return Yh_prediction, Yh_base

    @torch.no_grad()
    def evolve_edge_removal2(self, r0, A_in, N_evo, Y_train=None, Yh_base=None):
        B = r0.shape[0]
        N = self.n_units
        E = A_in.shape[0]
        O = self.out_dim
        device = r0.device
        R_base = r0  # (B,E*N)
        Y_int_collect = torch.zeros((E, B, N_evo + 1, 1), device=device)
        y0 = Yh_base[:, 0, :]  # (B,O)
        Y_prev_base = y0  # (B,O)
        A_expand = A_in.repeat_interleave(B, dim=0)  # E,D -> EB,D
        for k in range(1, N_evo + 1):
            R_base_new = self.step(R_base, Y_prev_base)  # (B,E*N)
            # y_base_test = torch.zeros_like(Y_prev_base)
            # for b in range(O):
            #     order = [b] + [i for i in range(O) if i != b]
            #     diff_base=self.apply_wout(R_base_new)
            #     y_base_test[b:b+1,:] =(diff_base*self.dt).transpose(0,1)[:,order]+Y_prev_base[b:b+1,:]
            y_base = Yh_base[:, k, :]  # (B,O)
            R_int = self.step(
                R_base.expand(E, B, self.ExpandNodes*self.n_units).reshape(E * B, self.ExpandNodes*self.n_units),
                Y_prev_base.expand(E, B, O).reshape(E * B, -1),
                A=A_expand
            ).reshape(E, B, self.ExpandNodes*self.n_units)
            R_base = R_base_new
            Y_prev_base = y_base
            y_int = self.apply_wout(R_int)*self.dt+Y_prev_base.expand(E,B,O)[:,:,0:1]  # (E,B,1)
            Y_int_collect[:, :, k, :] = y_int  # 只收集第一个维度的预测结果用于后续的AUC计算
        return Y_int_collect[:, :, 1:, 0:1]





