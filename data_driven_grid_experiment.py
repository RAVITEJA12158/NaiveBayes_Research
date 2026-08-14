import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification

from aewee_nb import (StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB,
                       diagnostic_scan, data_driven_grid)

RNG = 42


def run_dataset(name, X, y, nf, nc, diagnostic_sample_size=300):
    print(f"\n{'='*80}\n{name}: n_features={nf}, n_classes={nc}, n={len(X)}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # further split training into fit-set + validation-set for grid selection
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.3, random_state=RNG, stratify=ytr)

    std = StandardGaussianNB().fit(Xfit, yfit)
    pred_std, ev_std = std.predict(Xte)
    acc_std = accuracy_score(yte, pred_std)
    print(f"  standard baseline: acc={acc_std:.4f} evals={ev_std.mean():.0f}")

    # ---- Step 1: fit a reference model (ordering only) to get feature_order_,
    #      theta_, var_ -- needed before we can scan w_i / gap distributions ----
    ref = AdaptiveEvidenceWeightedEarlyExitNB(use_ordering=True, use_threshold=False,
                                                use_early_exit=False).fit(Xfit, yfit)

    # ---- Step 2: DATA-DRIVEN step -- actually look at the real w_i / gap
    #      distributions this model produces on validation data, instead of
    #      guessing round numbers ----
    diag_X = Xval[:diagnostic_sample_size]
    w_values, gap_values = diagnostic_scan(ref, diag_X)
    print(f"  w_i distribution:  min={w_values.min():.4f} p25={np.percentile(w_values,25):.4f} "
          f"p50={np.percentile(w_values,50):.4f} p75={np.percentile(w_values,75):.4f} "
          f"p90={np.percentile(w_values,90):.4f} max={w_values.max():.4f}")
    print(f"  gap distribution:  min={gap_values.min():.4f} p25={np.percentile(gap_values,25):.4f} "
          f"p50={np.percentile(gap_values,50):.4f} p75={np.percentile(gap_values,75):.4f} "
          f"max={gap_values.max():.4f}")

    tau_grid, delta_grid = data_driven_grid(w_values, gap_values)
    print(f"  DATA-DRIVEN tau candidates:   {tau_grid}")
    print(f"  DATA-DRIVEN delta candidates: {delta_grid}")

    # ---- Step 3: run the grid, exactly as before, but using the data-driven
    #      candidate lists instead of a fixed [0.0, 0.05, 0.2, 0.5] guess ----
    grid_results = []
    for tau in tau_grid:
        for delta in delta_grid:
            model = AdaptiveEvidenceWeightedEarlyExitNB(
                tau=tau, delta=delta, use_ordering=True, use_threshold=True, use_early_exit=True,
            ).fit(Xfit, yfit)
            # select using VALIDATION set
            pred_val, ev_val, _ = model.predict(Xval)
            acc_val = accuracy_score(yval, pred_val)
            grid_results.append({"tau": tau, "delta": delta, "val_accuracy": acc_val,
                                  "val_avg_evals": float(ev_val.mean())})

    # pick best: smallest avg evals subject to val accuracy loss <= 0.5pp vs standard
    std_val_pred, std_val_ev = std.predict(Xval)
    std_val_acc = accuracy_score(yval, std_val_pred)
    safe = [g for g in grid_results if g["val_accuracy"] >= std_val_acc - 0.005]
    pool = safe if safe else grid_results
    best = min(pool, key=lambda g: g["val_avg_evals"])
    print(f"  BEST (val-selected): tau={best['tau']} delta={best['delta']} "
          f"val_acc={best['val_accuracy']:.4f} val_evals={best['val_avg_evals']:.1f}")

    # ---- Step 4: touch the TEST set exactly once, with the winning config ----
    final = AdaptiveEvidenceWeightedEarlyExitNB(
        tau=best["tau"], delta=best["delta"],
        use_ordering=True, use_threshold=True, use_early_exit=True,
    ).fit(Xtr, ytr)  # refit on full train (fit+val) before final test evaluation
    pred_test, ev_test, _ = final.predict(Xte)
    acc_test = accuracy_score(yte, pred_test)
    print(f"  FINAL TEST: acc={acc_test:.4f} (std={acc_std:.4f})  "
          f"evals={ev_test.mean():.2f}/{nf} ({ev_test.mean()/nf*100:.1f}%)  "
          f"speedup={ev_std.mean()/ev_test.mean():.2f}x")

    return {
        "dataset": name, "n_features": nf, "n_classes": nc,
        "w_distribution": {"min": float(w_values.min()), "p25": float(np.percentile(w_values,25)),
                            "p50": float(np.percentile(w_values,50)), "p75": float(np.percentile(w_values,75)),
                            "p90": float(np.percentile(w_values,90)), "max": float(w_values.max())},
        "gap_distribution": {"min": float(gap_values.min()), "p25": float(np.percentile(gap_values,25)),
                              "p50": float(np.percentile(gap_values,50)), "p75": float(np.percentile(gap_values,75)),
                              "max": float(gap_values.max())},
        "tau_grid_tried": tau_grid, "delta_grid_tried": delta_grid,
        "best_tau": best["tau"], "best_delta": best["delta"],
        "standard_accuracy": acc_std, "standard_avg_evals": float(ev_std.mean()),
        "final_test_accuracy": acc_test, "final_test_avg_evals": float(ev_test.mean()),
        "final_speedup": float(ev_std.mean() / ev_test.mean()),
    }


def main():
    results = []

    bc = load_breast_cancer()
    results.append(run_dataset("breast_cancer", bc.data, bc.target, 30, 2))

    wine = load_wine()
    results.append(run_dataset("wine", wine.data, wine.target, 13, 3))

    digits = load_digits()
    results.append(run_dataset("digits", digits.data, digits.target, 64, 10))

    # THE CRITICAL TEST: the large diffuse-signal dataset where a fixed,
    # non-data-driven tau=0.2 previously collapsed accuracy to near-random.
    # Does the data-driven grid avoid ever proposing a tau that bad?
    Xd, yd = make_classification(
        n_samples=50000, n_features=500, n_informative=200, n_redundant=100,
        n_repeated=0, n_classes=8, n_clusters_per_class=2,
        class_sep=1.0, random_state=RNG,
    )
    # subsample for tractability of the per-sample python loop in diagnostics/predict
    idx = np.random.RandomState(RNG).choice(len(Xd), size=6000, replace=False)
    results.append(run_dataset("large_diffuse_signal_subsampled", Xd[idx], yd[idx], 500, 8,
                                diagnostic_sample_size=300))

    with open("/home/claude/aewee_nb/data_driven_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved data_driven_grid_results.json")


if __name__ == "__main__":
    main()
