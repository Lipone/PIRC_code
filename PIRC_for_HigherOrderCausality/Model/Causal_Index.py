import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Library import *

def cal_CausalIndex_PIRC(pirc, X_test, n_rep, n_washout, n_evo, rng=rng, unique_offsets=True):
    n_nodes=X_test.shape[1]
    Expand_nodes_num= n_nodes+math.comb(n_nodes-1, 2)
    Error = np.zeros((n_rep, Expand_nodes_num))
    PIRC_index = np.zeros(Expand_nodes_num)

    low = n_washout
    high = X_test.shape[0] - n_evo
    if unique_offsets:
        if high - low < n_rep:
            raise ValueError("可选范围不足以采样互不重复的 n_rep 个 offset")
        offsets = rng.choice(np.arange(low, high), size=n_rep, replace=False)
    else:
        offsets = rng.integers(low, high, size=n_rep)

    for rep, offset in enumerate(offsets):
        X_test_washout = X_test[offset - n_washout: offset]
        r0 = pirc.open_loop(X_test_washout, np.zeros(pirc.n_units))[-1]
        Yh = pirc.evolve(r0, n_evo)
        E = np.zeros(Expand_nodes_num)
        for j in range(Expand_nodes_num):
            Yhji = pirc.evolve_edge_removal(r0, j, 0, n_evo)
            d = (Yh - Yhji)[:, 0]
            E[j]=np.sum(d ** 2) / n_evo # MSE
        Error[rep]= E


    for j in range(Expand_nodes_num):
        t = Error[:, j]
        e = t[t > 0]
        if e.shape[0] == 0:
            # print(f'warning: too large threshold, j:{j} no error data')
            PIRC_index[j] = 0
        else:
            PIRC_index[j] = np.mean(e)
    return PIRC_index



