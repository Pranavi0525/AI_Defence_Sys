"""
Autoencoder anomaly detector -- hand-rolled numpy, no new dependencies,
same convention as gcn.py (verified on a toy problem before touching
real data).

WHY THIS MODEL, AND WHY IT'S DIFFERENT FROM STAGE 2 / STAGE 3
----------------------------------------------------------------
Stage 2 (XGBoost) and Stage 3 (GCN) are both SUPERVISED -- they learn
"what does labeled fraud look like" (Stage 2: from a trace's own
features; Stage 3: from its position in the cross-customer graph).
Both are therefore bounded by what's already in the labeled attack
corpora.

This model is UNSUPERVISED. It is trained ONLY on legitimate traces,
and it learns one thing: "what does normal behavior look like." At
inference time it tries to reconstruct a trace's feature vector through
a compressed bottleneck; a trace that reconstructs poorly is one whose
feature pattern doesn't fit the shape of normal behavior the model
learned, whether or not that pattern resembles any fraud the model has
ever been shown a label for. That is a genuinely different detection
capability, not "components have to be a separate neural network for
their own sake" -- it is included here because it addresses attacks the
supervised path cannot claim to be shaped by.

ARCHITECTURE
------------
    Z1 = X @ W1 + b1
    H  = ReLU(Z1)              (compressed bottleneck, hidden_dim < in_dim)
    Z2 = H @ W2 + b2
    X_hat = Z2                 (linear reconstruction, no output activation
                                 -- features are standardized, so they are
                                 not bounded to [0, 1])
    reconstruction_error = mean((X - X_hat)^2, axis=1)   (per-row MSE)

Trained with plain mini-batch-free full-batch gradient descent on MSE,
gradients derived by hand (chain rule through both linear layers and
the ReLU), same spirit as OneLayerGCN.backward() in gcn.py.
"""
import numpy as np


class Autoencoder:
    def __init__(self, in_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, in_dim))
        self.b2 = np.zeros(in_dim)

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H = np.maximum(self.Z1, 0)  # ReLU
        self.Z2 = self.H @ self.W2 + self.b2  # linear reconstruction
        return self.Z2

    def backward(self, lr=0.05):
        n = self.X.shape[0]
        X_hat = self.Z2
        # dL/dZ2 for L = mean over rows AND cols of (X - X_hat)^2
        dZ2 = -2.0 * (self.X - X_hat) / (n * self.X.shape[1])

        dW2 = self.H.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dH = dZ2 @ self.W2.T
        dZ1 = dH * (self.Z1 > 0)  # ReLU derivative
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def loss(self):
        return float(np.mean((self.X - self.Z2) ** 2))

    def reconstruction_error(self, X):
        """Per-row MSE reconstruction error -- the anomaly score, higher
        = less like the normal behavior this model was trained on."""
        X_hat = self.forward(X)
        return np.mean((X - X_hat) ** 2, axis=1)


def train(model, X, epochs=300, lr=0.05):
    """Full-batch training. X should already be standardized and should
    contain ONLY normal/legitimate rows -- this is an unsupervised
    detector, it must never see fraud rows during fitting or it would
    just learn to reconstruct fraud well too, defeating the point."""
    losses = []
    for _ in range(epochs):
        model.forward(X)
        losses.append(model.loss())
        model.backward(lr=lr)
    return losses


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def error_to_score(errors, normal_train_errors, high_percentile=95.0):
    """Turn a raw reconstruction error into a (0, 1) anomaly score that's
    calibrated against the model's OWN normal-training-data error
    distribution, not some arbitrary fixed cutoff:

        threshold = the high_percentile-th percentile of reconstruction
                    error on the held-in normal training rows (e.g. the
                    95th percentile -- "worse than 95% of normal
                    behavior this model has seen")
        scale     = std of that same normal-error distribution, so the
                    sigmoid's steepness is set by how noisy normal
                    reconstruction error actually is for this model,
                    not a hand-picked constant.

    score ~= 0.5 right at the threshold, climbs toward 1 for errors well
    above it, falls toward 0 for errors well below it.
    """
    threshold = np.percentile(normal_train_errors, high_percentile)
    scale = max(float(np.std(normal_train_errors)), 1e-6)
    z = (errors - threshold) / scale
    return sigmoid(z)


if __name__ == "__main__":
    # ---- TOY VERIFICATION ----
    # "Normal" = a single Gaussian blob in 6D. The autoencoder is
    # trained ONLY on a training slice of that blob, then scored on:
    #   (a) a HELD-OUT slice of the same normal blob (should reconstruct
    #       about as well as training data -- it's the same distribution)
    #   (b) "anomalies" = points from a DIFFERENT region of feature
    #       space entirely (large offset + different scale/correlation
    #       structure), never seen during training. A correctly-working
    #       autoencoder should reconstruct these MUCH worse, because its
    #       bottleneck only learned to compress the normal blob's shape.
    rng = np.random.default_rng(0)
    in_dim = 6
    hidden_dim = 2  # tight bottleneck -- forces the model to actually
                     # learn the normal blob's shape, not just memorize

    n_normal = 400
    mean_normal = np.zeros(in_dim)
    # correlated normal cluster (not axis-aligned noise -- gives the
    # bottleneck actual structure to learn instead of nothing to lose
    # by ignoring)
    base = rng.normal(0, 1, size=(in_dim, in_dim))
    cov_normal = base @ base.T / in_dim + 0.3 * np.eye(in_dim)
    X_normal = rng.multivariate_normal(mean_normal, cov_normal, size=n_normal)

    split = 300
    X_train, X_held_out_normal = X_normal[:split], X_normal[split:]

    n_anomaly = 60
    X_anomaly = rng.normal(8.0, 3.0, size=(n_anomaly, in_dim))  # far-offset,
    # different scale, no relation to the normal blob's covariance

    model = Autoencoder(in_dim=in_dim, hidden_dim=hidden_dim, seed=0)
    losses = train(model, X_train, epochs=800, lr=0.02)
    print(f"Toy test -- loss: start={losses[0]:.4f} -> end={losses[-1]:.4f}")

    train_err = model.reconstruction_error(X_train)
    held_out_err = model.reconstruction_error(X_held_out_normal)
    anomaly_err = model.reconstruction_error(X_anomaly)

    print(f"Toy test -- mean reconstruction error: "
          f"train_normal={train_err.mean():.4f}, "
          f"held_out_normal={held_out_err.mean():.4f}, "
          f"anomaly={anomaly_err.mean():.4f} "
          f"({anomaly_err.mean() / max(held_out_err.mean(), 1e-9):.1f}x worse "
          f"than held-out normal)")

    scores_normal = error_to_score(held_out_err, train_err, high_percentile=95.0)
    scores_anomaly = error_to_score(anomaly_err, train_err, high_percentile=95.0)
    recall_at_95pct_threshold = float((scores_anomaly >= 0.5).mean())
    false_flag_rate_on_held_out_normal = float((scores_normal >= 0.5).mean())
    print(f"Toy test -- at the 95th-percentile-of-normal threshold: "
          f"anomaly recall={recall_at_95pct_threshold*100:.1f}%, "
          f"held-out normal false-flag rate={false_flag_rate_on_held_out_normal*100:.1f}% "
          f"(false-flag rate should land near 5%, since the threshold IS "
          f"the 95th percentile of normal error by construction)")

    assert anomaly_err.mean() > 10 * held_out_err.mean(), (
        "Anomalies should reconstruct far worse than held-out normal data -- "
        "if this fails the bottleneck isn't learning the normal manifold."
    )
    assert recall_at_95pct_threshold > 0.8, (
        "Anomaly recall at the calibrated threshold is too low -- "
        "reconstruction error isn't separating anomalies from normal."
    )
    print("Toy test -- PASSED (math verified before touching real data).")
