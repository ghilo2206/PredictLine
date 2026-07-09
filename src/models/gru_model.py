"""
gru_model.py

A from-scratch NumPy implementation of a GRU (Gated Recurrent Unit) binary
sequence classifier, used for fault-risk scoring on windowed telemetry.

WHY NUMPY AND NOT PYTORCH/TENSORFLOW: this repository was built in an
environment without internet access to install ML frameworks. The GRU math
below is the same as a framework implementation (same gate equations,
real backpropagation-through-time, real gradient descent) — it is not a
placeholder. Before the bootcamp / ZCHPC CCE training run, this should be
swapped for an equivalent PyTorch nn.GRU for GPU speed and easier
hyperparameter search; the model architecture and evaluation harness stay
the same, only the execution backend changes. This is flagged explicitly
in the AI4I proposal (Section 3.3) and README.

Gate equations (standard GRU):
    z_t = sigmoid(Wz x_t + Uz h_{t-1} + bz)      update gate
    r_t = sigmoid(Wr x_t + Ur h_{t-1} + br)      reset gate
    h~_t = tanh(Wh x_t + Uh (r_t * h_{t-1}) + bh)  candidate state
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t        new hidden state

Output: sigmoid(Wo h_T + bo) -> fault-risk probability at end of window.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def dsigmoid(y):
    # derivative w.r.t. pre-activation, given y = sigmoid(x)
    return y * (1 - y)


def dtanh(y):
    # derivative w.r.t. pre-activation, given y = tanh(x)
    return 1 - y ** 2


class SimpleGRUClassifier:
    def __init__(self, n_features, hidden_size=8, seed=42):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(hidden_size)
        self.h = hidden_size
        self.n = n_features

        def mat(rows, cols):
            return rng.uniform(-scale, scale, size=(rows, cols))

        # Input-to-hidden and hidden-to-hidden weights for each gate
        self.Wz, self.Uz, self.bz = mat(n_features, hidden_size), mat(hidden_size, hidden_size), np.zeros(hidden_size)
        self.Wr, self.Ur, self.br = mat(n_features, hidden_size), mat(hidden_size, hidden_size), np.zeros(hidden_size)
        self.Wh, self.Uh, self.bh = mat(n_features, hidden_size), mat(hidden_size, hidden_size), np.zeros(hidden_size)

        # Output layer
        self.Wo = mat(hidden_size, 1)
        self.bo = np.zeros(1)

    def forward(self, X):
        """
        X: (batch, window, n_features)
        Returns: y_hat (batch,), and cache needed for backprop.
        """
        batch, window, _ = X.shape
        h = np.zeros((batch, self.h))
        cache = {"x": [], "h": [h], "z": [], "r": [], "hcand": []}

        for t in range(window):
            x_t = X[:, t, :]
            z_t = sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r_t = sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            hcand_t = np.tanh(x_t @ self.Wh + (r_t * h) @ self.Uh + self.bh)
            h = (1 - z_t) * h + z_t * hcand_t

            cache["x"].append(x_t)
            cache["z"].append(z_t)
            cache["r"].append(r_t)
            cache["hcand"].append(hcand_t)
            cache["h"].append(h)

        logits = h @ self.Wo + self.bo
        y_hat = sigmoid(logits).ravel()
        cache["y_hat"] = y_hat
        return y_hat, cache

    def backward(self, X, y_true, cache, lr=0.05):
        """
        Full backpropagation-through-time. Updates weights in place.
        Returns the batch's binary cross-entropy loss (pre-update).
        """
        batch, window, _ = X.shape
        y_hat = cache["y_hat"]
        eps = 1e-8
        loss = -np.mean(y_true * np.log(y_hat + eps) + (1 - y_true) * np.log(1 - y_hat + eps))

        # Output layer gradient
        dlogits = (y_hat - y_true).reshape(-1, 1) / batch  # dL/dlogits
        h_final = cache["h"][-1]
        dWo = h_final.T @ dlogits
        dbo = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wo.T  # gradient flowing into last hidden state

        grads = {k: np.zeros_like(getattr(self, k)) for k in
                 ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh"]}

        for t in reversed(range(window)):
            x_t = cache["x"][t]
            h_prev = cache["h"][t]
            z_t = cache["z"][t]
            r_t = cache["r"][t]
            hcand_t = cache["hcand"][t]

            dh = dh_next  # gradient w.r.t. h_t from downstream

            dz = dh * (hcand_t - h_prev)
            dhcand = dh * z_t
            dh_prev_direct = dh * (1 - z_t)

            dz_pre = dz * dsigmoid(z_t)
            dhcand_pre = dhcand * dtanh(hcand_t)

            # r_t affects hcand via (r_t * h_prev) @ Uh
            drh = dhcand_pre @ self.Uh.T  # gradient w.r.t. (r_t * h_prev)
            dr = drh * h_prev
            dr_pre = dr * dsigmoid(r_t)
            dh_prev_via_r = drh * r_t

            grads["Wz"] += x_t.T @ dz_pre
            grads["Uz"] += h_prev.T @ dz_pre
            grads["bz"] += dz_pre.sum(axis=0)

            grads["Wr"] += x_t.T @ dr_pre
            grads["Ur"] += h_prev.T @ dr_pre
            grads["br"] += dr_pre.sum(axis=0)

            grads["Wh"] += x_t.T @ dhcand_pre
            grads["Uh"] += (r_t * h_prev).T @ dhcand_pre
            grads["bh"] += dhcand_pre.sum(axis=0)

            dh_prev_from_z = dz_pre @ self.Uz.T
            dh_prev_from_r = dr_pre @ self.Ur.T

            dh_next = dh_prev_direct + dh_prev_from_z + dh_prev_from_r + dh_prev_via_r

        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * np.clip(g, -5, 5))
        self.Wo -= lr * np.clip(dWo, -5, 5)
        self.bo -= lr * np.clip(dbo, -5, 5)

        return loss

    def fit(self, X, y, epochs=30, batch_size=64, lr=0.05, verbose=True):
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]
                _, cache = self.forward(Xb)
                loss = self.backward(Xb, yb, cache, lr=lr)
                epoch_loss += loss
                n_batches += 1
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/n_batches:.4f}")

    def predict_proba(self, X):
        y_hat, _ = self.forward(X)
        return y_hat
