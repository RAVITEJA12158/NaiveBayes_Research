version2-https://doi.org/10.5281/zenodo.21939169
version1-https://doi.org/10.13140/RG.2.2.23508.97921
# Adaptive Evidence-Weighted Early-Exit Naive Bayes (AEWEE-NB)

**Author:** Tollamadugu Siva Naga Venkata Raviteja

Can Naive Bayes skip features it doesn't need and stop early once it's confident? Yes — and at small
scale it works well. At larger scale, a much simpler idea (just keep a fixed set of the best features)
matches or beats it. This repo has the full implementation, experiments, and writeup for both results.

## TL;DR

- Built an inference-time modification to Gaussian Naive Bayes: order features by how discriminative
  they are, skip ones that turn out uninformative for a given input, and stop early once the model is
  confident (a Naive-Bayes application of [Wald's Sequential Probability Ratio Test](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test)).
- **Small/moderate scale (13–100 features):** 1.9×–5.4× fewer feature evaluations per prediction, no
  accuracy loss (and a +6pp accuracy gain on one dataset).
- **Larger scale (200–2,000 features, up to 50k samples):** benchmarked against the dead-simple
  alternative — a *fixed* top-k feature subset with zero adaptivity — and the adaptive method did **not**
  clearly beat it under the original hyperparameter-selection process.
- **Then found and fixed two real bugs in that selection process itself:** (1) the safety check let
  *noisier* hyperparameter candidates through *more* easily instead of less, and (2) when no candidate
  could be certified safe, the fallback silently picked the most aggressive untested option instead of
  the safest one.
- **An external review then caught a third, subtler issue:** the candidate grid itself was generated
  from the full training set *before* it was split into cross-validation folds — a mild form of grid
  leakage across folds (not test-set leakage, but not clean nested CV either). Fixed by generating the
  grid strictly from each fold's inner-training data and evaluating only on that fold's untouched
  outer validation split — proper nested cross-validation.
- **Final, methodologically clean comparison against static top-k** (see
  `nested_cv_final_comparison_vs_statictopk.json`): AEWEE-NB wins on Wine and Digits, static top-k
  wins on the large diffuse-signal dataset, and it's a tie on Breast Cancer. A genuine mixed result,
  not a clean win or a clean loss either way — and the win/loss pattern is driven mainly by whether the
  safety-aware selection process can certify an aggressive configuration, not by the adaptive-inference
  machinery being inherently cheaper.
- Ablation study found the raw speedup comes almost entirely from the early-exit rule; the "smart"
  ordering and thresholding pieces contribute no computation savings on their own — they just make
  early-exit safer to use aggressively.

**Read the full paper:** [`AEWEE_NB_Paper.docx`](./AEWEE_NB_Paper.docx) (also exported as PDF) — Section 7
of the paper now documents all three corrections (data-driven grid, two selection bugs, and the nested-CV
grid-leakage fix) with fully consistent final numbers throughout.

## Why this is here

This started as an idea for cutting Naive Bayes inference cost. Rather than just writing it up, I
implemented it, tested it rigorously (including a baseline I initially didn't think to include), and
reported what actually happened — including the parts that don't flatter the original idea. The paper
walks through the full arc: motivation → method → small-scale results → a bug found and fixed →
large-scale results → the honest conclusion that a simpler approach currently wins at scale.

## Repo contents

| File | What it is |
|---|---|
| `aewee_nb.py` | Core implementation: `StandardGaussianNB` (baseline), `AdaptiveEvidenceWeightedEarlyExitNB` (the proposed method, with ablation switches), `StaticTopKNB` (the simple baseline), plus `diagnostic_scan`/`data_driven_grid` for data-driven hyperparameter grid selection |
| `run_experiments.py` | Small/moderate-scale experiment grid (Wine, Breast Cancer, Digits, synthetic) |
| `robustness_check.py` | 10-split repeated-random-split robustness check |
| `large_scale_experiments.py` | Large synthetic-dataset experiments + static top-k comparison |
| `data_driven_grid_experiment.py` | First attempt at data-driven (τ, δ) grid selection using a single validation split — **superseded**, kept for the record; exposed the need for CV |
| `cv_grid_experiment.py` | Second version: k-fold CV selection with a pessimistic safety margin — **superseded**, kept for the record; had a subtle grid-leakage issue caught by external review |
| `nested_cv_experiment.py` | **Current/correct** version: proper nested CV — the (τ, δ) candidate grid is generated strictly from each fold's inner-training data, never the data it's evaluated against |
| `results.json`, `robustness_results.json`, `large_scale_results.json` | Raw output from the original (paper) experiments |
| `data_driven_grid_results.json`, `cv_grid_results_final.json` | Raw output from the superseded (non-nested) hyperparameter-selection experiments |
| `nested_cv_results_small.json`, `nested_cv_results_large.json` | Raw output from the current, leakage-free nested CV experiments |
| `final_comparison_vs_statictopk.json` | Superseded head-to-head vs. static top-k (pre-nested-CV) |
| `nested_cv_final_comparison_vs_statictopk.json` | **Current** head-to-head vs. static top-k, using the leakage-free nested CV results |
| `AEWEE_NB_Paper.docx` / `.pdf` | Full writeup: method, related work, results, ablations, all three selection-process corrections (Section 7), honest limitations |
| `build_paper.js` | Script that generates the paper document from the results |

## Reproducing the results

```bash
pip install numpy scikit-learn matplotlib
python run_experiments.py          # small/moderate-scale grid
python robustness_check.py         # 10-split robustness check
python large_scale_experiments.py  # large-scale + static top-k comparison
```

Each script writes its results to a `.json` file; nothing here needs a GPU or takes more than a few
minutes to run.

## Method, briefly

For each feature, at training time, compute a Fisher-score-style separability measure. At inference
time, visit features in that order; for each one, compute how much it currently distinguishes between
the leading candidate classes for *this specific input*, skip it if that's below a threshold τ, and stop
entirely once the gap between the best and second-best class score exceeds a confidence margin δ. Full
derivation and the multiclass generalization are in the paper (Section 3).

## What I'd do next

- Test on real high-dimensional sparse data (text classification with Multinomial NB) instead of only
  synthetic data — the sandboxed environment used here couldn't reach real dataset servers.
- Find a genuinely cheap way to decide whether to skip a feature, since currently the threshold check
  costs as much as just scoring the feature (a limitation the ablation study surfaced).
- Try per-instance rather than global feature ordering.

## License

MIT — see [`LICENSE`](./LICENSE).
