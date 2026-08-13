import json
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from aewee_nb import StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB

RNG = 42


def load_datasets():
    ds = {}

    bc = load_breast_cancer()
    ds['breast_cancer'] = (bc.data, bc.target, bc.data.shape[1], len(np.unique(bc.target)))

    wine = load_wine()
    ds['wine'] = (wine.data, wine.target, wine.data.shape[1], len(np.unique(wine.target)))

    digits = load_digits()
    ds['digits'] = (digits.data, digits.target, digits.data.shape[1], len(np.unique(digits.target)))

    # Higher-dimensional synthetic dataset (covtype download is blocked by the
    # sandbox's network allowlist, so we use make_classification instead to
    # probe behavior as feature count grows -- Section 12 of the concept doc
    # explicitly asks for this).
    from sklearn.datasets import make_classification
    Xs, ys = make_classification(
        n_samples=4000, n_features=100, n_informative=25, n_redundant=25,
        n_repeated=0, n_classes=4, n_clusters_per_class=2,
        class_sep=1.2, random_state=RNG,
    )
    ds['synthetic_100f_4c'] = (Xs, ys, Xs.shape[1], len(np.unique(ys)))

    return ds


def eval_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, p, r, f1


def run_dataset(name, X, y, n_features, n_classes, thresholds, deltas):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    results = {"dataset": name, "n_features": n_features, "n_classes": n_classes,
               "n_train": len(Xtr), "n_test": len(Xte)}

    # ---- Standard baseline ----
    std = StandardGaussianNB().fit(Xtr, ytr)
    t0 = time.perf_counter()
    pred_std, evals_std = std.predict(Xte)
    t_std = time.perf_counter() - t0
    acc, p, r, f1 = eval_predictions(yte, pred_std)
    results["standard"] = {
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        "avg_feature_evals": float(evals_std.mean()),
        "time_sec": t_std,
    }

    # ---- Ablations across a grid of (tau, delta) ----
    ablation_results = []
    variants = [
        ("ordering_only", dict(use_ordering=True, use_threshold=False, use_early_exit=False)),
        ("threshold_only", dict(use_ordering=False, use_threshold=True, use_early_exit=False)),
        ("early_exit_only", dict(use_ordering=False, use_threshold=False, use_early_exit=True)),
        ("full_method", dict(use_ordering=True, use_threshold=True, use_early_exit=True)),
    ]

    for variant_name, flags in variants:
        for tau in thresholds:
            for delta in deltas:
                kwargs = dict(flags)
                # threshold_only / full_method use tau; early_exit variants use delta
                model = AdaptiveEvidenceWeightedEarlyExitNB(
                    tau=tau if flags["use_threshold"] else 0.0,
                    delta=delta if flags["use_early_exit"] else None,
                    **flags,
                )
                model.fit(Xtr, ytr)
                t0 = time.perf_counter()
                pred, evals, skips = model.predict(Xte)
                t_ad = time.perf_counter() - t0
                acc_a, p_a, r_a, f1_a = eval_predictions(yte, pred)
                ablation_results.append({
                    "variant": variant_name,
                    "tau": tau,
                    "delta": delta,
                    "accuracy": acc_a,
                    "precision": p_a,
                    "recall": r_a,
                    "f1": f1_a,
                    "avg_feature_evals": float(evals.mean()),
                    "avg_features_skipped": float(skips.mean()),
                    "pct_features_evaluated": float(evals.mean()) / n_features * 100.0,
                    "speedup_vs_standard": float(evals_std.mean()) / float(evals.mean()) if evals.mean() > 0 else float('nan'),
                    "accuracy_delta_vs_standard": acc_a - acc,
                    "time_sec": t_ad,
                })
    results["ablations"] = ablation_results
    return results


def main():
    import sys
    import os

    datasets = load_datasets()
    thresholds = [0.0, 0.05, 0.2, 0.5]
    deltas = [2.0, 5.0, 10.0]

    only = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = "/home/claude/aewee_nb/results.json"
    all_results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    done_names = {r["dataset"] for r in all_results}

    for name, (X, y, nf, nc) in datasets.items():
        if only is not None and name != only:
            continue
        if name in done_names:
            print(f"Skipping {name} (already done)")
            continue
        print(f"Running {name} (n_features={nf}, n_classes={nc}, n={len(X)}) ...")
        t0 = time.perf_counter()
        res = run_dataset(name, X, y, nf, nc, thresholds, deltas)
        print(f"  done in {time.perf_counter()-t0:.1f}s  standard acc={res['standard']['accuracy']:.4f}  "
              f"avg_evals={res['standard']['avg_feature_evals']:.1f}")
        all_results.append(res)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print("Saved results.json")


if __name__ == "__main__":
    main()
