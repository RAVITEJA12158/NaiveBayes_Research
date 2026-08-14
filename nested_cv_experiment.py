"""
Fixes the grid-leakage issue: previously, diagnostic_scan() (which produces
the tau/delta candidate grid) was run on the FULL training set, which
includes samples that later become each CV fold's validation split. That
means the candidate grid itself was informed by data the fold-selection step
was supposed to treat as held-out.

This version generates the grid from ONLY the inner-training portion of each
outer fold (folds != current), evaluates strictly on the untouched held-out
fold, and only concretizes final (tau, delta) numbers from the FULL
development set at the very end, after the outer-fold nested estimate has
already selected WHICH percentile-based candidate to use. The true test set
is never touched until the single final evaluation.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, make_classification

from aewee_nb import StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB, diagnostic_scan

RNG = 42
TAU_PCTS = (0, 25, 50, 75, 90)
DELTA_PCTS = (10, 25, 50, 75)


def grid_from_percentiles(w_values, gap_values):
    """Concrete (tau, delta) values from percentile LABELS, computed from
    whatever data is passed in. Percentile labels are the stable identifiers
    used to compare across folds even though the concrete numbers differ
    slightly fold to fold."""
    tau_vals = {p: (0.0 if p == 0 else round(float(np.percentile(w_values, p)), 4)) for p in TAU_PCTS}
    delta_vals = {p: round(float(np.percentile(gap_values, p)), 3) for p in DELTA_PCTS}
    return tau_vals, delta_vals


def run_dataset(name, X, y, nf, nc, n_outer_folds=5, diagnostic_sample_size=300):
    print(f"\n{'='*90}\n{name}: n_features={nf}, n_classes={nc}, n={len(X)}  "
          f"({n_outer_folds}-fold NESTED CV, grid leakage fixed)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # ---- outer nested loop: candidate grid computed from inner-train ONLY,
    #      evaluated strictly on the untouched outer validation fold ----
    skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=RNG)
    # per (tau_pct, delta_pct) label: list of (accuracy, avg_evals) across outer folds
    candidate_results = {(tp, dp): {"acc": [], "evals": []} for tp in TAU_PCTS for dp in DELTA_PCTS}
    std_fold_accs = []

    for fold_i, (inner_idx, outer_idx) in enumerate(skf.split(Xtr, ytr)):
        X_inner, y_inner = Xtr[inner_idx], ytr[inner_idx]
        X_outer, y_outer = Xtr[outer_idx], ytr[outer_idx]

        # standard baseline, same fold, for a matched nested comparison
        std_fold = StandardGaussianNB().fit(X_inner, y_inner)
        pred_std_fold, _ = std_fold.predict(X_outer)
        std_fold_accs.append(accuracy_score(y_outer, pred_std_fold))

        # grid generated from INNER-TRAIN ONLY -- this is the fix
        ref = AdaptiveEvidenceWeightedEarlyExitNB(use_ordering=True, use_threshold=False,
                                                    use_early_exit=False).fit(X_inner, y_inner)
        diag_n = min(diagnostic_sample_size, len(X_inner))
        w_values, gap_values = diagnostic_scan(ref, X_inner[:diag_n])
        tau_vals, delta_vals = grid_from_percentiles(w_values, gap_values)

        for tp in TAU_PCTS:
            for dp in DELTA_PCTS:
                tau = tau_vals[tp]
                delta = delta_vals[dp]
                model = AdaptiveEvidenceWeightedEarlyExitNB(
                    tau=tau, delta=delta, use_ordering=True,
                    use_threshold=True, use_early_exit=True,
                ).fit(X_inner, y_inner)
                # evaluated on the outer fold this candidate's grid NEVER saw
                pred_outer, ev_outer, _ = model.predict(X_outer)
                acc = accuracy_score(y_outer, pred_outer)
                candidate_results[(tp, dp)]["acc"].append(acc)
                candidate_results[(tp, dp)]["evals"].append(float(ev_outer.mean()))
        print(f"  outer fold {fold_i+1}/{n_outer_folds} done "
              f"(inner_train={len(X_inner)}, outer_val={len(X_outer)})")

    std_nested_acc = float(np.mean(std_fold_accs))
    print(f"  standard NB nested CV accuracy: {std_nested_acc:.4f} (\u00b1{np.std(std_fold_accs):.4f})")

    # aggregate: for each percentile-label, mean/std accuracy & evals ACROSS
    # OUTER FOLDS -- this is now a genuinely leakage-free nested estimate
    summary = []
    for (tp, dp), vals in candidate_results.items():
        mean_acc = float(np.mean(vals["acc"]))
        std_acc = float(np.std(vals["acc"]))
        mean_evals = float(np.mean(vals["evals"]))
        summary.append({"tau_pct": tp, "delta_pct": dp, "nested_mean_acc": mean_acc,
                         "nested_std_acc": std_acc, "nested_mean_evals": mean_evals})
    for s in sorted(summary, key=lambda s: -(s["nested_mean_acc"] - s["nested_std_acc"]))[:3]:
        print(f"    tau_pct={s['tau_pct']:>3} delta_pct={s['delta_pct']:>3}: "
              f"nested acc={s['nested_mean_acc']:.4f}\u00b1{s['nested_std_acc']:.4f} "
              f"evals={s['nested_mean_evals']:.1f}")

    # pessimistic lower-bound selection, same logic as before, now on a
    # properly leakage-free nested estimate
    margin = 0.01
    safe = [s for s in summary if (s["nested_mean_acc"] - s["nested_std_acc"]) >= (std_nested_acc - margin)]
    if safe:
        best = min(safe, key=lambda s: s["nested_mean_evals"])
        selection_mode = "certified-safe (nested)"
    else:
        best = max(summary, key=lambda s: s["nested_mean_acc"] - s["nested_std_acc"])
        selection_mode = "UNCERTIFIED (nested) -- fell back to best available lower bound"
    print(f"  SELECTED percentile-label: tau_pct={best['tau_pct']} delta_pct={best['delta_pct']}  "
          f"[{selection_mode}]")

    # ---- concretize final (tau, delta) numbers from the FULL development
    #      set (Xtr) -- legitimate now, since no further held-out data needs
    #      protecting; the true test set Xte was never part of Xtr ----
    ref_full = AdaptiveEvidenceWeightedEarlyExitNB(use_ordering=True, use_threshold=False,
                                                     use_early_exit=False).fit(Xtr, ytr)
    diag_n_full = min(diagnostic_sample_size, len(Xtr))
    w_full, gap_full = diagnostic_scan(ref_full, Xtr[:diag_n_full])
    tau_vals_full, delta_vals_full = grid_from_percentiles(w_full, gap_full)
    final_tau = tau_vals_full[best["tau_pct"]]
    final_delta = delta_vals_full[best["delta_pct"]]
    print(f"  concretized on FULL dev set: tau={final_tau}  delta={final_delta}")

    # ---- final model: fit on ALL of Xtr, touch Xte exactly once ----
    final_model = AdaptiveEvidenceWeightedEarlyExitNB(
        tau=final_tau, delta=final_delta, use_ordering=True,
        use_threshold=True, use_early_exit=True,
    ).fit(Xtr, ytr)
    pred_test, ev_test, _ = final_model.predict(Xte)
    acc_test = accuracy_score(yte, pred_test)

    std_final = StandardGaussianNB().fit(Xtr, ytr)
    pred_std_test, ev_std_test = std_final.predict(Xte)
    acc_std_test = accuracy_score(yte, pred_std_test)

    print(f"  FINAL HELD-OUT TEST: AEWEE-NB acc={acc_test:.4f} evals={ev_test.mean():.2f}/{nf} "
          f"({ev_test.mean()/nf*100:.1f}%)  |  standard acc={acc_std_test:.4f} evals={ev_std_test.mean():.0f}  "
          f"|  \u0394acc={acc_test-acc_std_test:+.4f}  speedup={ev_std_test.mean()/ev_test.mean():.2f}x")

    return {
        "dataset": name, "n_features": nf, "n_classes": nc, "n_outer_folds": n_outer_folds,
        "standard_nested_cv_accuracy": std_nested_acc,
        "selected_tau_pct": best["tau_pct"], "selected_delta_pct": best["delta_pct"],
        "selection_mode": selection_mode,
        "nested_selection_accuracy": best["nested_mean_acc"], "nested_selection_std": best["nested_std_acc"],
        "final_tau": final_tau, "final_delta": final_delta,
        "standard_test_accuracy": acc_std_test, "standard_test_evals": float(ev_std_test.mean()),
        "final_test_accuracy": acc_test, "final_test_evals": float(ev_test.mean()),
        "final_test_delta_accuracy": acc_test - acc_std_test,
        "final_speedup": float(ev_std_test.mean() / ev_test.mean()),
        "candidate_grid_summary": summary,
    }


def main():
    results = []

    bc = load_breast_cancer()
    results.append(run_dataset("breast_cancer", bc.data, bc.target, 30, 2, n_outer_folds=5))

    wine = load_wine()
    results.append(run_dataset("wine", wine.data, wine.target, 13, 3, n_outer_folds=5))

    digits = load_digits()
    results.append(run_dataset("digits", digits.data, digits.target, 64, 10, n_outer_folds=5))

    with open("/home/claude/aewee_nb/nested_cv_results_small.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved nested_cv_results_small.json")


if __name__ == "__main__":
    main()
