"""Simple feedforward autoencoder for tabular feature compression.

Architecture
------------
Encoder: input_dim → h1 → h2 → ... → hN   (GELU activations, Dropout)
Decoder: hN → ... → h2 → h1 → input_dim   (GELU activations, linear output)

The last element of ``hidden_dims`` is the bottleneck (latent) dimension.
Default follows the paper: hidden_dims = [32, 16, 8].
The encoder half is used as a feature extractor.

Config keys (under ``feature_pipeline.autoencoder``)
-----------------------------------------------------
    hidden_dims: [32, 16, 8]  # per-layer widths; last = bottleneck
    epochs: 50
    batch_size: 512
    lr: 1e-3
    weight_decay: 1e-5
    dropout: 0.1              # dropout applied after each hidden layer
    replace_features: false   # if true, output only latent cols; if false, append them
    device: null              # null = cpu; "mps" for Apple Silicon
    denoising: false          # if true, use DenoisingTabularAutoencoder instead
    noise_scale: 0.1          # std-scale of Student-t corruption noise (denoising only)
    noise_df: 3.0             # degrees of freedom for Student-t           (denoising only)
    variational: false        # if true, use VariationalTabularAutoencoder (VAE) instead
    beta: 1.0                 # KL weight; beta>1 → beta-VAE (more disentangled) (VAE only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

class _AE:
    """Thin wrapper so we can import torch lazily and avoid hard dependency."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        import torch
        import torch.nn as nn

        def _build_layers(dims: list[int], activate_last: bool) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2 or activate_last:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        enc_dims = [input_dim] + hidden_dims
        dec_dims = list(reversed(hidden_dims)) + [input_dim]

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _build_layers(enc_dims, activate_last=False)
                self.decoder = _build_layers(dec_dims, activate_last=False)

            def forward(self, x):
                return self.decoder(self.encoder(x))

            def encode(self, x):
                return self.encoder(x)

        self.net = _Net()
        self._torch = torch

    def fit(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: str,
        noise_fn=None,
    ) -> None:
        """Train the autoencoder.

        If ``noise_fn`` is provided it is called on each clean batch to produce
        a corrupted version; the network learns to reconstruct the *clean* batch
        from the corrupted one (denoising objective).
        """
        import torch

        net = self.net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        n = len(X_t)

        net.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = X_t[idx]
                noisy = noise_fn(batch) if noise_fn is not None else batch
                optimizer.zero_grad()
                loss = criterion(net(noisy), batch)  # reconstruct clean target
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"    [Autoencoder] epoch {epoch + 1:>4}/{epochs}  "
                    f"loss={epoch_loss / n_batches:.6f}"
                )

        net.eval()
        self.net = net.cpu()

    def encode(self, X: np.ndarray, device: str) -> np.ndarray:
        import torch

        net = self.net.to(device)
        net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            latent = net.encode(X_t).cpu().numpy()
        self.net = net.cpu()
        return latent


# ---------------------------------------------------------------------------
# sklearn-style wrapper used by FeaturePipeline
# ---------------------------------------------------------------------------

class TabularAutoencoder:
    """Fit an autoencoder on train features and return latent representations.

    Parameters (all come from the config dict)
    ------------------------------------------
    hidden_dims : list[int], default [32, 16, 8]
        Per-layer widths for the encoder; last element is the bottleneck.
        Decoder mirrors these in reverse.
    epochs     : int, default 50
    batch_size : int, default 512
    lr         : float, default 1e-3
    weight_decay : float, default 1e-5
    dropout    : float, default 0.1
    replace_features : bool, default False
        If True, return only the latent columns.
        If False, append latent columns to the existing features.
    device     : str | None, default None (→ cpu)
    """

    def __init__(self, cfg: dict) -> None:
        self.hidden_dims: list[int] = [int(d) for d in cfg.get("hidden_dims", [32, 16, 8])]
        self.epochs: int = int(cfg.get("epochs", 50))
        self.batch_size: int = int(cfg.get("batch_size", 512))
        self.lr: float = float(cfg.get("lr", 1e-3))
        self.weight_decay: float = float(cfg.get("weight_decay", 1e-5))
        self.dropout: float = float(cfg.get("dropout", 0.1))
        self.replace_features: bool = bool(cfg.get("replace_features", False))
        self.device: str = cfg.get("device") or "cpu"

        self._ae: _AE | None = None
        self._input_cols: list[str] | None = None

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit autoencoder on X and return transformed DataFrame."""
        self._input_cols = X.columns.tolist()
        X_arr = X.fillna(0).values.astype(np.float32)

        dims_str = "→".join(str(d) for d in self.hidden_dims)
        print(
            f"  [Autoencoder] Training  input={X_arr.shape[1]}  "
            f"dims={dims_str}  epochs={self.epochs}  device={self.device}"
        )
        self._ae = _AE(
            input_dim=X_arr.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self._ae.fit(
            X_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
        )
        return self._append_latent(X, X_arr)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode X using the fitted autoencoder."""
        if self._ae is None:
            raise RuntimeError("Call fit_transform() before transform().")
        X_arr = X[self._input_cols].fillna(0).values.astype(np.float32)
        return self._append_latent(X, X_arr)

    # ------------------------------------------------------------------
    def _append_latent(self, X: pd.DataFrame, X_arr: np.ndarray) -> pd.DataFrame:
        latent = self._ae.encode(X_arr, self.device)
        latent_cols = [f"ae_{i}" for i in range(self.hidden_dims[-1])]
        latent_df = pd.DataFrame(latent, index=X.index, columns=latent_cols)
        if self.replace_features:
            return latent_df
        return pd.concat([X, latent_df], axis=1)


class DenoisingTabularAutoencoder(TabularAutoencoder):
    """Autoencoder trained with fat-tail (Student-t) input corruption.

    During training each batch is corrupted with Student-t noise before being
    passed to the encoder; the decoder must reconstruct the *clean* input.
    At inference time the encoder receives the clean input directly, so the
    latent representation is noise-free.

    Additional config keys (on top of TabularAutoencoder)
    ------------------------------------------------------
    noise_scale : float, default 0.1
        Multiplier applied to the sampled noise.
    noise_df    : float, default 3.0
        Degrees of freedom for the Student-t distribution.
        Lower values → heavier tails (df=1 is Cauchy; df→∞ → Gaussian).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.noise_scale: float = float(cfg.get("noise_scale", 0.1))
        self.noise_df: float = float(cfg.get("noise_df", 3.0))

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit denoising autoencoder on X and return transformed DataFrame."""
        import torch

        self._input_cols = X.columns.tolist()
        X_arr = X.fillna(0).values.astype(np.float32)

        dims_str = "\u2192".join(str(d) for d in self.hidden_dims)
        print(
            f"  [DenoisingAutoencoder] Training  input={X_arr.shape[1]}  "
            f"dims={dims_str}  "
            f"noise=student_t(df={self.noise_df}, scale={self.noise_scale})  "
            f"epochs={self.epochs}  device={self.device}"
        )
        self._ae = _AE(
            input_dim=X_arr.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

        dist = torch.distributions.StudentT(df=self.noise_df)
        noise_scale = self.noise_scale

        def noise_fn(x: "torch.Tensor") -> "torch.Tensor":
            return x + dist.sample(x.shape).to(x.device) * noise_scale

        self._ae.fit(
            X_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            noise_fn=noise_fn,
        )
        return self._append_latent(X, X_arr)


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------

class _VAE:
    """Lazy-torch VAE: encoder outputs (mu, log_var); decoder reconstructs input."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        import torch.nn as nn

        latent_dim = hidden_dims[-1]
        enc_body_dims = [input_dim] + hidden_dims[:-1]

        def _mlp(dims: list[int]) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        # Decoder: latent_dim → reversed hidden body → input_dim
        # Must start from latent_dim so shapes match after reparameterisation.
        dec_body_dims = [latent_dim] + list(reversed(hidden_dims[:-1]))

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_body = _mlp(enc_body_dims)
                pre_latent = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
                self.fc_mu  = nn.Linear(pre_latent, latent_dim)
                self.fc_var = nn.Linear(pre_latent, latent_dim)
                self.decoder_body = _mlp(dec_body_dims)
                self.decoder_out  = nn.Linear(dec_body_dims[-1], input_dim)

            def encode(self, x):
                h = self.encoder_body(x)
                return self.fc_mu(h), self.fc_var(h)

            def reparameterise(self, mu, log_var):
                import torch
                std = (0.5 * log_var).exp()
                return mu + std * torch.randn_like(std)

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterise(mu, log_var)
                return self.decoder_out(self.decoder_body(z)), mu, log_var

        self.net = _Net()

    def fit(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        beta: float,
        device: str,
    ) -> None:
        import torch

        net = self.net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        n = len(X_t)

        net.train()
        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = X_t[idx]
                recon, mu, log_var = net(batch)
                recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="mean")
                kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
                loss = recon_loss + beta * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"    [VAE] epoch {epoch + 1:>4}/{epochs}  "
                    f"loss={epoch_loss / n_batches:.6f}"
                )

        net.eval()
        self.net = net.cpu()

    def encode(self, X: np.ndarray, device: str) -> np.ndarray:
        """Return the posterior mean (mu) — deterministic at inference."""
        import torch

        net = self.net.to(device)
        net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            mu, _ = net.encode(X_t)
            latent = mu.cpu().numpy()
        self.net = net.cpu()
        return latent


class VariationalTabularAutoencoder(TabularAutoencoder):
    """VAE variant: encoder outputs a Gaussian posterior; uses ELBO loss.

    At inference, the posterior *mean* (mu) is returned as the latent
    representation — this is deterministic and gives a stable feature vector.

    Additional config keys (on top of TabularAutoencoder)
    ------------------------------------------------------
    beta : float, default 1.0
        Weight on the KL-divergence term.  beta=1 → standard VAE.
        beta>1 → beta-VAE (encourages more disentangled representations).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.beta: float = float(cfg.get("beta", 1.0))

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._input_cols = X.columns.tolist()
        X_arr = X.fillna(0).values.astype(np.float32)

        dims_str = "\u2192".join(str(d) for d in self.hidden_dims)
        print(
            f"  [VAE] Training  input={X_arr.shape[1]}  "
            f"dims={dims_str}  beta={self.beta}  "
            f"epochs={self.epochs}  device={self.device}"
        )
        self._ae = _VAE(
            input_dim=X_arr.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self._ae.fit(
            X_arr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            beta=self.beta,
            device=self.device,
        )
        return self._append_latent(X, X_arr)
