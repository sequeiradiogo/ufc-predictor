"""
pytorch_mlp.py -- sklearn-compatible PyTorch MLP with dropout and batch normalization.

Drop-in replacement for sklearn MLPClassifier. Implements fit() and predict_proba()
so no changes are needed in train_v1_models.py, predict.py, or the ensemble.

Key improvements over sklearn MLPClassifier:
  - Dropout for regularization (sklearn has none)
  - BatchNorm1d for faster convergence and implicit regularization
  - AdamW optimizer with proper weight decay (decoupled from gradient)
  - Cosine annealing LR schedule
  - Early stopping on held-out validation loss (last 15% of training data)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ResidualBlock(nn.Module):
    """Linear + LayerNorm + ReLU + Dropout with a learned shortcut."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, layer_norm: bool):
        super().__init__()
        self.linear   = nn.Linear(in_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.drop     = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.relu(self.norm(self.linear(x)))) + self.shortcut(x)


class _UFCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple, dropout: float, batch_norm: bool):
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            blocks.append(_ResidualBlock(in_dim, h, dropout, batch_norm))
            in_dim = h
        self.blocks = nn.Sequential(*blocks)
        self.head   = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x)).squeeze(1)


class PyTorchMLP:
    """
    sklearn-compatible wrapper for a PyTorch MLP binary classifier.

    Implements predict_proba(X) -> ndarray of shape (n, 2) so it can be used
    anywhere sklearn MLPClassifier is used, including joblib serialization.
    """

    def __init__(
        self,
        hidden_sizes: tuple = (128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 20,
        batch_norm: bool = True,
        random_state: int = 42,
    ):
        self.hidden_sizes  = hidden_sizes
        self.dropout       = dropout
        self.lr            = lr
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.batch_norm    = batch_norm
        self.random_state  = random_state
        self._net: _UFCNet | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchMLP":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        # Last 15% of rows (chronological order preserved by caller) for early stopping
        val_cut  = max(int(len(X) * 0.85), len(X) - 500)
        X_tr, X_val = X[:val_cut], X[val_cut:]
        y_tr, y_val = y[:val_cut], y[val_cut:]

        loader  = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        self._net  = _UFCNet(self.n_features_in_, self.hidden_sizes, self.dropout, self.batch_norm)
        optimizer  = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        criterion  = nn.BCEWithLogitsLoss()

        best_val_loss    = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for _epoch in range(self.max_epochs):
            self._net.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                criterion(self._net(X_batch), y_batch).backward()
                optimizer.step()
            scheduler.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss    = val_loss
                best_state       = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict_proba().")
        self._net.eval()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            logits = self._net(X_t).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
