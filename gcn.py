"""
Minimal 1-layer Graph Convolutional Network (Kipf & Welling formulation),
implemented by hand in numpy -- no PyTorch, nothing new to install.

Architecture:
    M = A_hat @ X          (message passing: average each node's own
                             features with its neighbors' features)
    Z = M @ W1              (linear graph-conv layer)
    H = ReLU(Z)
    logit = H @ w2 + b2     (readout to a single fraud score per node)
    p = sigmoid(logit)

A_hat = D^-1/2 (A + I) D^-1/2  -- symmetric-normalized adjacency with
self-loops, exactly the Kipf & Welling propagation rule.
"""
import numpy as np


def normalize_adjacency(A):
    n = A.shape[0]
    A_hat = A + np.eye(n)
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class OneLayerGCN:
    def __init__(self, in_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden_dim))
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim,))
        self.b2 = 0.0

    def forward(self, M):
        """M = A_hat @ X, precomputed once (A_hat doesn't change during training)."""
        self.M = M
        self.Z = M @ self.W1
        self.H = np.maximum(self.Z, 0)  # ReLU
        self.logit = self.H @ self.w2 + self.b2
        self.p = sigmoid(self.logit)
        return self.p

    def backward(self, y, train_mask, lr=0.05):
        n_train = train_mask.sum()
        dlogit = np.zeros_like(self.p)
        dlogit[train_mask] = (self.p[train_mask] - y[train_mask]) / n_train

        dw2 = self.H.T @ dlogit
        db2 = dlogit.sum()
        dH = np.outer(dlogit, self.w2)
        dZ = dH * (self.Z > 0)
        dW1 = self.M.T @ dZ

        self.W1 -= lr * dW1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

    def loss(self, y, train_mask, eps=1e-9):
        p = np.clip(self.p[train_mask], eps, 1 - eps)
        yt = y[train_mask]
        return -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))


def train(model, M, y, train_mask, epochs=300, lr=0.1):
    losses = []
    for _ in range(epochs):
        model.forward(M)
        losses.append(model.loss(y, train_mask))
        model.backward(y, train_mask, lr=lr)
    return losses


if __name__ == "__main__":
    # ---- TOY VERIFICATION ----
    # 2 clusters of 10 nodes each. Cluster A (label 1) is densely
    # interconnected. Cluster B (label 0) is densely interconnected.
    # No edges between clusters. Node features are NOISE (uninformative
    # on their own) -- so a model with NO graph propagation should fail,
    # and a model that correctly uses the graph should succeed, since
    # each node's neighbors all share its true label.
    rng = np.random.default_rng(1)
    n_per_cluster = 10
    n = n_per_cluster * 2
    y = np.array([1] * n_per_cluster + [0] * n_per_cluster, dtype=float)

    X = rng.normal(0, 1, size=(n, 4))  # pure noise features, no signal

    A = np.zeros((n, n))
    for i in range(n_per_cluster):
        for j in range(n_per_cluster):
            if i != j:
                A[i, j] = 1
                A[n_per_cluster + i, n_per_cluster + j] = 1

    A_hat = normalize_adjacency(A)
    M = A_hat @ X

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:6] = True
    train_mask[n_per_cluster:n_per_cluster + 6] = True
    test_mask = ~train_mask

    model = OneLayerGCN(in_dim=4, hidden_dim=8, seed=0)
    losses = train(model, M, y, train_mask, epochs=400, lr=0.3)

    print(f"Toy test -- loss: start={losses[0]:.4f} -> end={losses[-1]:.4f}")
    preds = (model.p >= 0.5).astype(int)
    test_acc = (preds[test_mask] == y[test_mask]).mean()
    print(f"Toy test -- held-out node accuracy: {test_acc*100:.1f}% "
          f"(features are PURE NOISE -- any accuracy above 50% here is "
          f"coming entirely from graph propagation, proving the message "
          f"passing math is doing real work, not just memorizing features)")

    # No-graph ablation on the same toy data: M = X directly (no propagation)
    model_ablation = OneLayerGCN(in_dim=4, hidden_dim=8, seed=0)
    train(model_ablation, X, y, train_mask, epochs=400, lr=0.3)
    preds_ablation = (model_ablation.p >= 0.5).astype(int)
    ablation_acc = (preds_ablation[test_mask] == y[test_mask]).mean()
    print(f"Toy test -- SAME model, NO graph propagation, held-out accuracy: "
          f"{ablation_acc*100:.1f}% (should be ~50%, random guessing, since "
          f"the features carry zero signal without the graph)")
