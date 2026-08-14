import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification

from aewee_nb import (StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB,
                       diagnostic_scan, data_driven_grid)

RNG = 42


def cv_evaluate(tau, delta, X, y, n_folds, use_threshold=True, use_early_exit=True):
    """Run n_folds stratified CV for one (tau, delta) candidate. Returns mean
    accuracy and mean avg-feature-evals across folds -- much more trustworthy
    than a single small validation split."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG)
    accs, evals_list = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = AdaptiveEvidenceWeightedEarlyExitNB(
            tau=tau, delta=delta, use_ordering=True,
            use_threshold=use_threshold, use_early_exit=use_early_exit,
        ).fit(X[train_idx], y[train_idx])
        pred, ev, _ = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], pred))
        evals_list.append(ev.mean())
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(evals_list))


def cv_evaluate_standard(X, y, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RNG)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        model = StandardGaussianNB().fit(X[train_idx], y[train_idx])
        pred, ev = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], pred))
    return float(np.mean(accs))


def run_dataset(name, X, y, nf, nc, n_folds=5, diagnostic_sample_size=400):
    print(f"\n{'='*80}\n{name}: n_features={nf}, n_classes={nc}, n={len(X)}  ({n_folds}-fold CV)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    std = StandardGaussianNB().fit(Xtr, ytr)
    pred_std_test, ev_std_test = std.predict(Xte)
    acc_std_test = accuracy_score(yte, pred_std_test)
    std_cv_acc = cv_evaluate_standard(Xtr, ytr, n_folds)
    print(f"  standard: CV acc={std_cv_acc:.4f}  |  held-out test acc={acc_std_test:.4f} evals={ev_std_test.mean():.0f}")

    # Step 1+2: data-driven diagnostic scan (on the training set, not test)
    ref = AdaptiveEvidenceWeightedEarlyExitNB(use_ordering=True, use_threshold=False,
                                                use_early_exit=False).fit(Xtr, ytr)
    diag_X = Xtr[:diagnostic_sample_size]
    w_values, gap_values = diagnostic_scan(ref, diag_X)
    tau_grid, delta_grid = data_driven_grid(w_values, gap_values)
    print(f"  data-driven tau candidates:   {tau_grid}")
    print(f"  data-driven delta candidates: {delta_grid}")

    # Step 3: evaluate every (tau, delta) candidate via k-fold CV on TRAIN only
    grid_results = []
    for tau in tau_grid:
        for delta in delta_grid:
            mean_acc, std_acc, mean_evals = cv_evaluate(tau, delta, Xtr, ytr, n_folds)
            grid_results.append({"tau": tau, "delta": delta, "cv_mean_acc": mean_acc,
                                  "cv_std_acc": std_acc, "cv_mean_evals": mean_evals})
    for g in sorted(grid_results, key=lambda g: -g["cv_mean_acc"])[:3]:
        print(f"    tau={g['tau']} delta={g['delta']}: CV acc={g['cv_mean_acc']:.4f}"
              f"\u00b1{g['cv_std_acc']:.4f} evals={g['cv_mean_evals']:.1f}")

    # Step 4: select using a PESSIMISTIC lower-bound test: a candidate's
    # accuracy minus its OWN uncertainty must still clear the bar. This is
    # the opposite of the naive version (which let noisy candidates in more
    # easily) -- noisy candidates should need to prove themselves MORE, not
    # less, before we trust them enough to deploy.
    margin = 0.01
    safe = [g for g in grid_results
            if (g["cv_mean_acc"] - g["cv_std_acc"]) >= (std_cv_acc - margin)]
    if safe:
        best = min(safe, key=lambda g: g["cv_mean_evals"])
        selection_mode = "certified-safe (picked fastest among certified-safe candidates)"
    else:
        # Nothing cleared the bar -- this dataset/CV setup is too noisy to
        # certify ANY (tau, delta) as safe. The honest fallback is to pick
        # whichever candidate has the best (least-bad) lower-bound accuracy
        # estimate, NOT the fastest one -- picking the fastest here would be
        # actively choosing the option we have the LEAST evidence is safe.
        best = max(grid_results, key=lambda g: g["cv_mean_acc"] - g["cv_std_acc"])
        selection_mode = "UNCERTIFIED (no candidate cleared the safety bar -- picked best available lower bound, not fastest)"
    print(f"  selection mode: {selection_mode}")
    print(f"  SELECTED (CV-safe): tau={best['tau']} delta={best['delta']} "
          f"CV acc={best['cv_mean_acc']:.4f}\u00b1{best['cv_std_acc']:.4f} evals={best['cv_mean_evals']:.1f}")

    # Step 5: touch the held-out TEST set exactly once, with the winning config
    final = AdaptiveEvidenceWeightedEarlyExitNB(
        tau=best["tau"], delta=best["delta"],
        use_ordering=True, use_threshold=True, use_early_exit=True,
    ).fit(Xtr, ytr)
    pred_test, ev_test, _ = final.predict(Xte)
    acc_test = accuracy_score(yte, pred_test)
    print(f"  FINAL HELD-OUT TEST: acc={acc_test:.4f} (standard={acc_std_test:.4f}, "
          f"\u0394={acc_test-acc_std_test:+.4f})  evals={ev_test.mean():.2f}/{nf} "
          f"({ev_test.mean()/nf*100:.1f}%)  speedup={ev_std_test.mean()/ev_test.mean():.2f}x")

    return {
        "dataset": name, "n_features": nf, "n_classes": nc, "n_folds": n_folds,
        "tau_grid_tried": tau_grid, "delta_grid_tried": delta_grid,
        "selected_tau": best["tau"], "selected_delta": best["delta"],
        "selection_cv_accuracy": best["cv_mean_acc"], "selection_cv_std": best["cv_std_acc"],
        "selection_mode": selection_mode,
        "standard_test_accuracy": acc_std_test, "standard_test_evals": float(ev_std_test.mean()),
        "final_test_accuracy": acc_test, "final_test_evals": float(ev_test.mean()),
        "final_test_delta_accuracy": acc_test - acc_std_test,
        "final_speedup": float(ev_std_test.mean() / ev_test.mean()),
    }


def main():
    results = []

    bc = load_breast_cancer()
    results.append(run_dataset("breast_cancer", bc.data, bc.target, 30, 2, n_folds=5))

    wine = load_wine()
    results.append(run_dataset("wine", wine.data, wine.target, 13, 3, n_folds=5))

    digits = load_digits()
    results.append(run_dataset("digits", digits.data, digits.target, 64, 10, n_folds=5))

    Xd, yd = make_classification(
        n_samples=50000, n_features=500, n_informative=200, n_redundant=100,
        n_repeated=0, n_classes=8, n_clusters_per_class=2,
        class_sep=1.0, random_state=RNG,
    )
    idx = np.random.RandomState(RNG).choice(len(Xd), size=3000, replace=False)
    results.append(run_dataset("large_diffuse_signal_subsampled", Xd[idx], yd[idx], 500, 8,
                                n_folds=3, diagnostic_sample_size=300))

    with open("/home/claude/aewee_nb/cv_grid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved cv_grid_results.json")


if __name__ == "__main__":
    main()
