import os
import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *
from Model.PIRC_torch import PIRC_Torch as PIRC
from Data_gen import generate_kuramoto_data

def worker(node_id, gpu_id, args_dict, data_file, Pair_strength, Tri_strength):
    """
    Run an optuna study for a single node_id on a specific gpu_id.
    Parameters passed are simple picklable objects (dicts, arrays).
    """
    with open(data_file, "rb") as f:
        a2, a3, data = pickle.load(f)
    data_np = np.asarray(data, dtype=np.float32)

    try:
        # Avoid CPU thread oversubscription
        # os.environ.setdefault("OMP_NUM_THREADS", str(args_dict.get("OMP_NUM_THREADS", "1")))
        # os.environ.setdefault("MKL_NUM_THREADS", str(args_dict.get("MKL_NUM_THREADS", "1")))

        # Device selection
        if torch.cuda.is_available() and gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
            try:
                torch.cuda.set_device(device)
            except Exception:
                # set_device may fail in some environments; ignore
                pass
        else:
            device = torch.device("cpu")
            if gpu_id is not None and args_dict.get("warn_no_cuda", True):
                print(f"[worker {node_id}] Warning: CUDA not available or disabled; running on CPU.")

        # Recreate data on CPU first to avoid cross-process GPU tensor pickling issues.
        X_cpu = torch.from_numpy(np.asarray(data_np, dtype=np.float32))  # CPU tensor
        # Ensure shift is done on CPU (more robust) and then move to device.
        X_shifted_cpu = shift_column_to_first(X_cpu.clone(), node_id)
        X = X_shifted_cpu.to(device)

        # Prepare Optuna storage for per-node study to avoid concurrency issues.
        optuna_dir = Path(args_dict.get("optuna_dir", "optuna_studies"))
        node_dir = optuna_dir / f"N_train_{args_dict.get('N_train')}_Node_num_{args_dict.get('node_num')}_PairStrength_{Pair_strength}_TriStrength_{Tri_strength}"
        node_dir.mkdir(parents=True, exist_ok=True)
        storage_path = node_dir / f"study_node_{node_id}.db"
        storage = f"sqlite:///{storage_path}"

        # Create / load study
        study = optuna.create_study(
            storage=storage,
            study_name=f"node_{node_id}_study",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True
        )

        # Run optimization
        n_trials = int(args_dict.get("n_trials", 500))
        # objective must accept (trial, X)
        study.optimize(lambda trial: objective(
            trial, X,
            args_dict.get("in_dim"),
            args_dict.get("N_start"),
            args_dict.get("N_washout"),
            args_dict.get("N_test"),
            args_dict.get("N_rep"),
            args_dict.get("block_dim"),
            args_dict.get("ExpandNodes"),
            args_dict.get("use_Sin"),
            args_dict.get("N_train"),
            device), n_trials=n_trials,
                       callbacks=[stop_when_low_enough] if args_dict.get("use_callback", True) else None)

        # Logging (FileLock to avoid race)
        today = args_dict.get("today", date.today())
        log_dir = Path("Log") / f"{today}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"N_train_{args_dict.get('N_train')}_NodeNum_{args_dict.get('node_num')}_PairStrength_{Pair_strength}_TriStrength_{Tri_strength}_TrainLog.txt"
        lock = FileLock(str(log_file) + ".lock")
        with lock:
            with log_file.open("a", encoding="utf-8") as lf:
                params_json = json.dumps(study.best_params, ensure_ascii=False, default=str)
                lf.write(f"Best Hyperparameters for Node{node_id} (GPU {gpu_id}): {params_json}\n")
                lf.write(f"Lowest Loss for Node{node_id}: {study.best_value}\n")

        # Save parameter file
        params_dir = Path("Parameters") / f"{today}"
        params_dir.mkdir(parents=True, exist_ok=True)
        best_params = study.best_params if study.best_trial is not None else {}
        best_out_dim = best_params.get("out_dim", None)
        best_Normalization = best_params.get("Normalization", None)
        param_file = f"Parameters/{today}/N_train_{args_dict.get('N_train')}_NodeNum_{args_dict.get('node_num')}_node_{node_id}_PairStrength_{Pair_strength}_TriStrength_{Tri_strength}_out_dim_{best_out_dim}_UseSin_{args_dict.get('use_Sin')}_Normalization_{best_Normalization}_block_dim{args_dict.get('block_dim')}.pkl"
        with open(param_file, "wb") as pf:
            pickle.dump((best_params), pf)

        # Clear GPU cache if on CUDA
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return {"Pair_strength": Pair_strength, "Tri_strength": Tri_strength, "node_id": node_id, "gpu_id": gpu_id, "best_value": study.best_value, "best_params": best_params, "status": "ok"}
    except Exception as e:
        # Return error info to caller
        return {"Pair_strength": Pair_strength, "Tri_strength": Tri_strength, "node_id": node_id, "gpu_id": gpu_id, "error": str(e), "status": "error"}


# %% hyperparameter optimization with optuna
def trainLoss_single_node(device, in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, N_test, alpha, sigma_in, rho, option, tikh, block_dim, ExpandNodes, use_Sin = False):
    Structured_W = generate_structured_w_v4(in_dim, ExpandNodes, device, block_dim = block_dim)
    Win, Wres = Partitioned_Win_Wres(n_units, ExpandNodes, device, block_dim = block_dim)
    pirc = PIRC(n_units, in_dim, out_dim, Structured_W, Win, Wres, sigma_in, rho, alpha, tikh, option, device, block_dim = block_dim, use_Sin=use_Sin)
    R, Training_Loss, X_predict=pirc.train(X_washout, X_train, Y_train)
    Y_test_predict, Test_Loss = pirc.Prediction(Y_test, R, N_test, Y_train)
    return R, Training_Loss, X_predict, Y_test_predict, Test_Loss

def objective(trial, X, in_dim, N_start, N_washout, N_test, N_rep, block_dim, ExpandNodes, use_Sin, N_train, device):
    n_units = trial.suggest_int('n_units', 1000, 5000, step=1000)
    alpha = trial.suggest_float('alpha', 0.1,1, step=0.1)
    sigma_in = trial.suggest_float('sigma_in', 0.1,1, step=0.1)
    rho=trial.suggest_float('rho',  0.1,1, step=0.1)
    tikh=trial.suggest_categorical('tikh', [1e-1,1e-2,1e-3,1e-4,1e-5])
    option = trial.suggest_categorical('option', [3,4,5])
    # trial.set_user_attr("Structured_W", Structured_W.tolist())
    # trial.set_user_attr("Win", Win.tolist())
    # trial.set_user_attr("Wres", Wres.tolist())
    # N_train =  trial.suggest_int('N_train', 5000, 50000, step=5000)
    # N_train = trial.suggest_categorical('N_train', [10000])
    # out_dim = trial.suggest_categorical('out_dim', [3])
    Normalization = trial.suggest_categorical('Normalization', [1])

    N_train= N_train
    out_dim = in_dim

    std = X.std(dim=0)
    std[std == 0] = 1.0
    X = (X - X.mean(dim=0)) / std if Normalization == 1 else X  # 标准化

    X_washout, X_train, Y_train, Y_test = split_dataset(X, N_start, N_washout, N_train, N_test, in_dim=in_dim, out_dim=in_dim)
    test_losses = []
    for rep in range(N_rep):
        R, Training_Loss, Y_train_predict, Y_test_predict, Test_Loss = trainLoss_single_node(device, in_dim, out_dim, n_units, X_washout, X_train, Y_train, Y_test, N_test, alpha, sigma_in, rho, option, tikh, block_dim, ExpandNodes, use_Sin)
        test_losses.append(Test_Loss[0])  # 保存每次的 Test_Loss[0]
        del R, Training_Loss, Y_test_predict, Test_Loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # print("Test Loss for each rep: {}".format(test_losses))
    return sum(test_losses) / len(test_losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help='Comma separated GPU ids to use, e.g. "0,1,2,3". If omitted uses CPU.')
    parser.add_argument("--node_num", type=int, default=3, help="Number of nodes (node_id range)")
    # parser.add_argument("--Pair_strength", type=float, default=0.1)
    # parser.add_argument("--Tri_strength", type=float, default=0.1)
    parser.add_argument("--block_dim", type=int, default=2)
    parser.add_argument("--use_Sin", action="store_true")
    parser.add_argument("--n_trials", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Max concurrent workers; defaults to number of GPUs (or 1 if CPU).")
    # parser.add_argument("--OMP_NUM_THREADS", type=int, default=4, help="Per-process OMP_NUM_THREADS")
    # parser.add_argument("--MKL_NUM_THREADS", type=int, default=4, help="Per-process MKL_NUM_THREADS")
    parser.add_argument("--N_train", type=int, default=10000)
    args = parser.parse_args()
    in_dim = args.node_num
    ExpandNodes = in_dim + math.comb(in_dim - 1, 2)

    # Date
    today = date.today()

    # GPUs list
    if args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    elif torch.cuda.is_available():
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = []  # empty => CPU-only

    # Determine worker count
    if args.n_workers is not None:
        max_workers = args.n_workers
    else:
        max_workers = max(1, len(gpus))  # at most number of GPUs if available, else 1

    # Pack args for workers (simple serializable dict)
    args_dict = {
        "n_trials": args.n_trials,
        "use_Sin": args.use_Sin,
        "block_dim": args.block_dim,
        "node_num": args.node_num,
        # "Pair_strength": args.Pair_strength,
        # "Tri_strength": args.Tri_strength,
        "today": today,
        "optuna_dir": "optuna_studies",
        # "OMP_NUM_THREADS": args.OMP_NUM_THREADS,
        # "MKL_NUM_THREADS": args.MKL_NUM_THREADS,
        "use_callback": True,
        "warn_no_cuda": True,
        "in_dim": in_dim,
        "ExpandNodes": ExpandNodes,
        "N_washout": 100,
        "N_rep": 10,
        "N_test": 10,
        "N_start": 1000,
        "N_train": args.N_train
    }

    node_ids = list(range(args.node_num))

    print(f"Launching {len(node_ids)} tasks with up to {max_workers} concurrent workers. GPUs: {gpus or 'CPU only'}")

    results = []
    data_cache = {}

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        for Pair_strength in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
            for Tri_strength in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
                key = (Pair_strength, Tri_strength)
                data_file = Path(
                    f"data/NodeNum_{args.node_num}_PairStrength_{Pair_strength}_TriStrength_{Tri_strength}.pkl")
                if key not in data_cache:
                    # Data file path
                    if data_file.exists():
                        with open(data_file, "rb") as f:
                            a2, a3, data = pickle.load(f)
                    else:
                        # generate_kuramoto_data must be available
                        a2, a3, data = generate_kuramoto_data(n=args.node_num, dt=0.01, steps=15000,
                                                              Pair_strength=Pair_strength,
                                                              Tri_strength=Tri_strength)
                        with open(data_file, "wb") as f:
                            pickle.dump((a2, a3, data), f)
                    data_cache[key] = str(data_file)
                    print(f"[Cache] Ready {data_file} | a2.shape={a2.shape}, a3.shape={a3.shape}")

                for node_id in node_ids:
                    # Assign GPU to a node_id in round-robin fashion if GPUs available, else None (CPU).
                    gpu_id = gpus[node_id % len(gpus)] if gpus else None
                    fut = exe.submit(worker, node_id, gpu_id, args_dict, data_cache[key], Pair_strength, Tri_strength)
                    futures[fut] = (node_id, gpu_id, Pair_strength, Tri_strength)

        for fut in as_completed(futures):
            node_id, gpu_id, Pair_strength, Tri_strength = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"Pair_strength": Pair_strength, "Tri_strength": Tri_strength, "node_id": node_id, "gpu_id": gpu_id, "error": str(e), "status": "error"}
            results.append(res)
            if res.get("status") == "ok":
                print(f"[Main] when Pair_strength is {res['Pair_strength']} and Tri_strength is {res['Tri_strength']}, Finished node {res['node_id']} on GPU {res['gpu_id']}: best_value={res['best_value']}")
            else:
                print(f"[Main]  when Pair_strength = {res['Pair_strength']} and Tri_strength = {res['Tri_strength']}, Node {node_id} failed on GPU {gpu_id}: {res.get('error')}")

    print("All tasks completed. Summary:")
    for r in results:
        print(r)

