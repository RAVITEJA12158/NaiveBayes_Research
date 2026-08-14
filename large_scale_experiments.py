import json
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from aewee_nb import StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB, StaticTopKNB

RNG = 42


def make_large_datasets():
    ds = {}

    # A: large N, moderate features, signal concentrated in a minority of features
    # (the regime the method's pitch says should work best)
    X, y = make_classification(
        n_samples=50000, n_features=200, n_informative=20, n_redundant=20,
        n_repeated=0, n_classes=5, n_clusters_per_class=2,
        class_sep=1.5, random_state=RNG,
    )
    ds["large_concentrated_signal"] = (X, y, 200, 5,
        "50k samples, 200 features, only 20 informative -- signal concentrated")

    # B: large N, many features, diffuse signal (the regime the method's pitch
    # says should work worst -- included so the comparison isn't cherry-picked)
    X, y = make_classification(
        n_samples=50000, n_features=500, n_informative=200, n_redundant=100,
        n_repeated=0, n_classes=8, n_clusters_per_class=2,
        class_sep=1.0, random_state=RNG,
    )
    ds["large_diffuse_signal"] = (X, y, 500, 8,
        "50k samples, 500 features, 200 informative -- diffuse signal")

    # C: very high dimensional, sparse signal, fewer samples (stress test on
    # dimensionality rather than N -- closer in spirit to text/bag-of-words)
    X, y = make_classification(
        n_samples=15000, n_features=2000, n_informative=30, n_redundant=30,
        n_repeated=0, n_classes=10, n_clusters_per_class=1,
        class_sep=1.8, random_state=RNG,
    )
    ds["highdim_sparse_signal"] = (X, y, 2000, 10,
        "15k samples, 2000 features, only 30 informative -- high-dim, sparse signal")

    return ds


def eval_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def run_one(name, X, y, nf, nc, desc, adaptive_grid, adaptive_eval_n=3000):
    print(f"\n=== {name}: {desc} ===")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    result = {"dataset": name, "description": desc, "n_features": nf, "n_classes": nc,
              "n_train": len(Xtr), "n_test": len(Xte)}

    # Standard baseline -- fully vectorized, run on the FULL test set
    t0 = time.perf_counter()
    std = StandardGaussianNB().fit(Xtr, ytr)
    pred_std, ev_std = std.predict(Xte)
    t_std = time.perf_counter() - t0
    acc_std = eval_acc(yte, pred_std)
    result["standard"] = {"accuracy": acc_std, "avg_evals": float(ev_std.mean()), "time_sec": t_std,
                           "n_eval_samples": len(Xte)}
    print(f"  standard (full test, n={len(Xte)}): acc={acc_std:.4f} evals={ev_std.mean():.0f} time={t_std:.1f}s")

    # AEWEE-NB grid -- the per-sample Python loop is the bottleneck at this
    # scale, so we evaluate it on a stratified subsample of the test set.
    # Standard and static-top-k above/below stay on the FULL test set since
    # they're vectorized and cheap regardless of N.
    if len(Xte) > adaptive_eval_n:
        Xte_sub, _, yte_sub, _ = train_test_split(
            Xte, yte, train_size=adaptive_eval_n, random_state=RNG, stratify=yte)
    else:
        Xte_sub, yte_sub = Xte, yte

    best_by_budget = []
    for tau, delta in adaptive_grid:
        t0 = time.perf_counter()
        ad = AdaptiveEvidenceWeightedEarlyExitNB(
            tau=tau, delta=delta, use_ordering=True, use_threshold=True, use_early_exit=True,
        ).fit(Xtr, ytr)
        pred_ad, ev_ad, _ = ad.predict(Xte_sub)
        t_ad = time.perf_counter() - t0
        acc_ad = eval_acc(yte_sub, pred_ad)
        pct = float(ev_ad.mean()) / nf * 100.0
        best_by_budget.append({
            "tau": tau, "delta": delta, "accuracy": acc_ad,
            "avg_evals": float(ev_ad.mean()), "pct_evaluated": pct, "time_sec": t_ad,
            "n_eval_samples": len(Xte_sub),
        })
        print(f"  adaptive tau={tau} delta={delta} (n={len(Xte_sub)}): acc={acc_ad:.4f} "
              f"evals={ev_ad.mean():.1f} ({pct:.1f}%) time={t_ad:.1f}s")
    result["adaptive_grid"] = best_by_budget

    # Static top-k baseline, matched to each adaptive grid point's budget,
    # and evaluated on the SAME subsample as the adaptive grid for an
    # apples-to-apples accuracy comparison at matched N.
    static_results = []
    ks_to_test = sorted(set(
        [max(1, int(round(g["avg_evals"]))) for g in best_by_budget] +
        [max(1, nf // 20), max(1, nf // 10), max(1, nf // 4), nf // 2]
    ))
    for k in ks_to_test:
        k = min(k, nf)
        t0 = time.perf_counter()
        stk = StaticTopKNB(k=k).fit(Xtr, ytr)
        pred_stk, ev_stk = stk.predict(Xte_sub)
        t_stk = time.perf_counter() - t0
        acc_stk = eval_acc(yte_sub, pred_stk)
        static_results.append({
            "k": k, "accuracy": acc_stk, "avg_evals": float(k), "pct_evaluated": k / nf * 100.0,
            "time_sec": t_stk, "n_eval_samples": len(Xte_sub),
        })
        print(f"  static-top-{k} (n={len(Xte_sub)}): acc={acc_stk:.4f} ({k/nf*100:.1f}%) time={t_stk:.2f}s")
    result["static_topk_grid"] = static_results

    return result


def main():
    import sys, os
    only = sys.argv[1] if len(sys.argv) > 1 else None
    datasets = make_large_datasets()
    grids = {
        "large_concentrated_signal": [(0.0, 3.0), (0.0, 8.0), (0.2, 3.0), (0.2, 8.0)],
        "large_diffuse_signal": [(0.0, 3.0), (0.0, 8.0), (0.2, 3.0), (0.2, 8.0)],
        "highdim_sparse_signal": [(0.0, 5.0), (0.0, 12.0)],
    }
    eval_ns = {
        "large_concentrated_signal": 3000,
        "large_diffuse_signal": 3000,
        "highdim_sparse_signal": 1500,
    }
    out_path = "/home/claude/aewee_nb/large_scale_results.json"
    all_results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    done = {r["dataset"] for r in all_results}

    for name, (X, y, nf, nc, desc) in datasets.items():
        if only is not None and name != only:
            continue
        if name in done:
            print(f"skip {name} (done)")
            continue
        res = run_one(name, X, y, nf, nc, desc, grids[name], eval_ns[name])
        all_results.append(res)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
    print("\nSaved large_scale_results.json")


if __name__ == "__main__":
    main()
