import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from aewee_nb import StandardGaussianNB, AdaptiveEvidenceWeightedEarlyExitNB
from run_experiments import load_datasets

N_SPLITS = 10

# (tau, delta) chosen from the grid search as a good acc-preserving operating
# point per dataset (see results.json "BEST full_method (acc loss <=0.5pp)")
RECOMMENDED = {
    "breast_cancer": (0.2, 10.0),
    "wine": (0.0, 10.0),
    "digits": (0.0, 5.0),
    "synthetic_100f_4c": (0.0, 5.0),
}


def eval_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, p, r, f1


def main():
    datasets = load_datasets()
    summary = {}
    for name, (X, y, nf, nc) in datasets.items():
        tau, delta = RECOMMENDED[name]
        accs_std, accs_ad = [], []
        evals_std, evals_ad = [], []
        for split in range(N_SPLITS):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=split, stratify=y)
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            std = StandardGaussianNB().fit(Xtr_s, ytr)
            pred_std, ev_std = std.predict(Xte_s)
            a_std, *_ = eval_predictions(yte, pred_std)

            ad = AdaptiveEvidenceWeightedEarlyExitNB(
                tau=tau, delta=delta,
                use_ordering=True, use_threshold=True, use_early_exit=True,
            ).fit(Xtr_s, ytr)
            pred_ad, ev_ad, _ = ad.predict(Xte_s)
            a_ad, *_ = eval_predictions(yte, pred_ad)

            accs_std.append(a_std)
            accs_ad.append(a_ad)
            evals_std.append(ev_std.mean())
            evals_ad.append(ev_ad.mean())

        summary[name] = {
            "n_features": nf, "n_classes": nc, "tau": tau, "delta": delta,
            "standard_acc_mean": float(np.mean(accs_std)), "standard_acc_std": float(np.std(accs_std)),
            "adaptive_acc_mean": float(np.mean(accs_ad)), "adaptive_acc_std": float(np.std(accs_ad)),
            "standard_evals_mean": float(np.mean(evals_std)),
            "adaptive_evals_mean": float(np.mean(evals_ad)), "adaptive_evals_std": float(np.std(evals_ad)),
            "speedup_mean": float(np.mean(evals_std) / np.mean(evals_ad)),
            "acc_delta_mean": float(np.mean(accs_ad) - np.mean(accs_std)),
        }
        print(name, summary[name])

    with open("/home/claude/aewee_nb/robustness_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
