"""
Adaptive Evidence-Weighted Early-Exit Naive Bayes (AEWEE-NB)
Reference implementation + standard Gaussian NB baseline, built from the
same numpy primitives so per-feature evaluation counts are directly comparable.
"""
import numpy as np


class StandardGaussianNB:
    """Textbook Gaussian Naive Bayes. Evaluates all n features for all
    C classes on every test sample. Used as the baseline."""

    def fit(self, X, y, var_smoothing=1e-9):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        C = len(self.classes_)
        self.theta_ = np.zeros((C, n_features))   # per-class means
        self.var_ = np.zeros((C, n_features))     # per-class variances
        self.class_prior_ = np.zeros(C)
        eps = var_smoothing * X.var(axis=0).max()
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[idx] = Xc.mean(axis=0)
            self.var_[idx] = Xc.var(axis=0) + eps
            self.class_prior_[idx] = Xc.shape[0] / X.shape[0]
        self.n_features_ = n_features
        return self

    def _log_gauss(self, x, k):
        """log N(x_i | mean_ki, var_ki) for every feature i, class k."""
        mu = self.theta_[k]
        var = self.var_[k]
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var

    def predict_one(self, x):
        C = len(self.classes_)
        scores = np.log(self.class_prior_).copy()
        n_features = self.n_features_
        for k in range(C):
            scores[k] += self._log_gauss(x, k).sum()
        # "Feature evaluations" = number of features whose class-conditional
        # densities were computed. Each such evaluation costs C density
        # computations (one per class) in BOTH the standard and adaptive
        # models, so counting per-feature (not per-feature-per-class) keeps
        # the comparison apples-to-apples with AdaptiveEvidenceWeightedEarlyExitNB,
        # which counts identically.
        evals = n_features
        return self.classes_[int(np.argmax(scores))], evals

    def predict(self, X):
        preds, evals = [], []
        for x in X:
            p, e = self.predict_one(x)
            preds.append(p)
            evals.append(e)
        return np.array(preds), np.array(evals)


class AdaptiveEvidenceWeightedEarlyExitNB:
    """
    Implements the method described in the concept document:
      - global training-time feature ordering (Fisher-score style
        class-separability ranking)
      - per-sample, per-feature discriminative weight w_i, generalized to
        C classes as the variance (across classes) of the feature's
        log-density for the current sample
      - threshold tau: features with w_i < tau are skipped
      - early exit: stop once the gap between the leading and
        second-place class log-score exceeds delta, OR once the
        softmax posterior of the leading class exceeds min_confidence
      - ablation switches so each component can be tested in isolation
    """

    def __init__(self, tau=0.0, delta=None, min_confidence=None,
                 use_ordering=True, use_threshold=True, use_early_exit=True):
        self.tau = tau
        self.delta = delta
        self.min_confidence = min_confidence
        self.use_ordering = use_ordering
        self.use_threshold = use_threshold
        self.use_early_exit = use_early_exit

    def fit(self, X, y, var_smoothing=1e-9):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        C = len(self.classes_)
        self.theta_ = np.zeros((C, n_features))
        self.var_ = np.zeros((C, n_features))
        self.class_prior_ = np.zeros(C)
        eps = var_smoothing * X.var(axis=0).max()
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[idx] = Xc.mean(axis=0)
            self.var_[idx] = Xc.var(axis=0) + eps
            self.class_prior_[idx] = Xc.shape[0] / X.shape[0]
        self.n_features_ = n_features

        # global training-time feature ordering: Fisher-score style
        # class separability = variance of per-class means / mean of
        # per-class variances (per feature)
        mean_var_across_classes = np.var(self.theta_, axis=0)
        avg_within_class_var = np.mean(self.var_, axis=0)
        fisher_score = mean_var_across_classes / (avg_within_class_var + 1e-12)
        if self.use_ordering:
            self.feature_order_ = np.argsort(-fisher_score)
        else:
            self.feature_order_ = np.arange(n_features)  # natural order
        self.fisher_score_ = fisher_score
        return self

    def _log_gauss_all_classes(self, xi, i):
        """log density of feature i's value for every class -> shape (C,)"""
        mu = self.theta_[:, i]
        var = self.var_[:, i]
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((xi - mu) ** 2) / var

    def predict_one(self, x):
        C = len(self.classes_)
        scores = np.log(self.class_prior_).copy()
        evals = 0
        skipped = 0

        for i in self.feature_order_:
            log_dens = self._log_gauss_all_classes(x[i], i)  # per-class density
            evals += 1  # one feature "processed" (cost ~ C density evals either way)

            if self.use_threshold:
                w_i = np.var(log_dens)  # generalization of |log ratio| to C classes
                if w_i < self.tau:
                    skipped += 1
                    continue  # feature contributes nothing; treated as non-discriminative

            scores = scores + log_dens

            if self.use_early_exit:
                sorted_scores = np.sort(scores)[::-1]
                gap = sorted_scores[0] - sorted_scores[1]
                stop = False
                if self.delta is not None and gap > self.delta:
                    stop = True
                if self.min_confidence is not None:
                    shifted = scores - scores.max()
                    post = np.exp(shifted)
                    post = post / post.sum()
                    if post.max() > self.min_confidence:
                        stop = True
                if stop:
                    break

        pred = self.classes_[int(np.argmax(scores))]
        return pred, evals, skipped

    def predict(self, X):
        preds, evals, skips = [], [], []
        for x in X:
            p, e, s = self.predict_one(x)
            preds.append(p)
            evals.append(e)
            skips.append(s)
        return np.array(preds), np.array(evals), np.array(skips)
