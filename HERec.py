import numpy as np
import torch
import torch.nn as nn

class HECrec(nn.Module):
    """
    HECrec with personalized non-linear fusion g(·).
    - Embedding fusion (Eq. 6–8) with σ non-linearity and per-user/per-item weights w.
    - Rating predictor (Eq. 5): r̂_{u,i} = x_u^T y_i + α e_u^{(U)T} γ_i^{(I)} + β γ_u^{(U)T} e_i^{(I)}.
    - Loss (Eq. 9): squared error on observed entries + L2 regularization.
    """

    def __init__(
        self,
        user_embeds_list,  # list[dict[str->np.ndarray]]  |P^U| meta-paths for users
        item_embeds_list,  # list[dict[str->np.ndarray]]  |P^I| meta-paths for items
        user_ids,          # list of user node ids matching rows of R (e.g., ["U0", ...])
        item_ids,          # list of item node ids matching cols of R (e.g., ["M0", ...])
        D_mf=64,              # latent dimension for MF
        D_emb= 64,      # dimensions of the embeddings
        alpha=1.0,         # α in Eq. (5)
        beta=1.0,          # β in Eq. (5)
        seed=42,
        device=None
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Cache node index order ----
        self.user_ids = list(user_ids)
        self.item_ids = list(item_ids)
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        self.D_mf = D_mf
        self.D_emb = D_emb
        self.alpha = alpha
        self.beta = beta

        # ---- Stack precomputed HIN embeddings (kept fixed) ----
        # users: list length |P^U|, each tensor shape (n_users, d_l)
        self.user_meta = []
        for emb_dict in user_embeds_list:
            d_l = len(next(iter(emb_dict.values())))
            mat = np.stack([emb_dict[uid] for uid in self.user_ids], axis=0)
            self.user_meta.append(torch.tensor(mat, dtype=torch.float32, device=self.device))  # (U, d_l)

        # items: list length |P^I|, each tensor shape (n_items, d_l)
        self.item_meta = []
        for emb_dict in item_embeds_list:
            d_l = len(next(iter(emb_dict.values())))
            mat = np.stack([emb_dict[iid] for iid in self.item_ids], axis=0)
            self.item_meta.append(torch.tensor(mat, dtype=torch.float32, device=self.device))  # (I, d_l)

        self.num_user_paths = len(self.user_meta)
        self.num_item_paths = len(self.item_meta)

        # ---- Parameters of matrix factorization (Eq. 4) ----
        # x_u and y_i in R^D
        self.X = nn.Parameter(torch.randn(self.n_users, D_mf, generator=g) * 0.01)
        self.Y = nn.Parameter(torch.randn(self.n_items, D_mf, generator=g) * 0.01)

        # ---- Parameters paired with fused embeddings (Eq. 5) ----
        # γ_u^(U), γ_i^(I) in R^D
        self.Gamma_u = nn.Parameter(torch.randn(self.n_users, D_emb, generator=g) * 0.01)
        self.Gamma_i = nn.Parameter(torch.randn(self.n_items, D_emb, generator=g) * 0.01)

        # ---- Fusion function parameters Θ^(U), Θ^(I) (Eq. 6–8) ----
        # For each meta-path l: M^(l) ∈ R^{D×d_l}, b^(l) ∈ R^D.
        # Personalized weights: w_u^(l) for users; w_i^(l) for items.
        self.user_M = nn.ParameterList()
        self.user_b = nn.ParameterList()
        for l, meta in enumerate(self.user_meta):
            d_l = meta.shape[1]
            self.user_M.append(nn.Parameter(torch.randn(D_emb, d_l, generator=g) * 0.01))
            self.user_b.append(nn.Parameter(torch.zeros(D_emb)))

        self.item_M = nn.ParameterList()
        self.item_b = nn.ParameterList()
        for l, meta in enumerate(self.item_meta):
            d_l = meta.shape[1]
            self.item_M.append(nn.Parameter(torch.randn(D_emb, d_l, generator=g) * 0.01))
            self.item_b.append(nn.Parameter(torch.zeros(D_emb)))

        # Personalized meta-path weights w (one scalar per (user, l) / (item, l))
        if self.num_user_paths > 0:
            self.user_w = nn.Parameter(torch.ones(self.n_users, self.num_user_paths) / max(1, self.num_user_paths))
        else:
            self.user_w = None

        if self.num_item_paths > 0:
            self.item_w = nn.Parameter(torch.ones(self.n_items, self.num_item_paths) / max(1, self.num_item_paths))
        else:
            self.item_w = None

        self.to(self.device)

    #   g({e_u^(l)}) = σ( Σ_l w_u^(l) · σ( M^(l) e_u^(l) + b^(l) ) )
    # Same for items.
    def fuse_users(self):
        if self.num_user_paths == 0:
            return torch.zeros(self.n_users, self.D, device=self.device)
        parts = []
        for l, meta in enumerate(self.user_meta):
            # h_u^(l) = σ( M^(l) e_u^(l) + b^(l) ) ∈ R^D
            h = torch.sigmoid(meta @ self.user_M[l].T + self.user_b[l])  # (U,D)
            parts.append(h * self.user_w[:, l:l+1])                      # apply personalized weight
        fused = torch.sigmoid(torch.stack(parts, dim=0).sum(dim=0))      # (U,D)
        return fused

    def fuse_items(self):
        if self.num_item_paths == 0:
            return torch.zeros(self.n_items, self.D, device=self.device)
        parts = []
        for l, meta in enumerate(self.item_meta):
            h = torch.sigmoid(meta @ self.item_M[l].T + self.item_b[l])  # (I,D)
            parts.append(h * self.item_w[:, l:l+1])
        fused = torch.sigmoid(torch.stack(parts, dim=0).sum(dim=0))      # (I,D)
        return fused

    # ---------- Rating predictor (Eq. 5) ----------
    # r̂_{u,i} = x_u^T y_i + α e_u^{(U)T} γ_i^{(I)} + β γ_u^{(U)T} e_i^{(I)}
    def predict_full(self):
        eU = self.fuse_users()   # (U,D)
        eI = self.fuse_items()   # (I,D)
        term_mf = self.X @ self.Y.T                                 # (U,I)
        term_u = self.alpha * (eU @ self.Gamma_i.T)                  # (U,I)
        term_i = self.beta  * (self.Gamma_u @ eI.T)                  # (U,I)
        return term_mf + term_u + term_i

    # Convenience for a mini-batch of (u_idx, i_idx)
    def predict_pairs(self, u_idx, i_idx, eU=None, eI=None):
        if eU is None: eU = self.fuse_users()
        if eI is None: eI = self.fuse_items()
        # Gather rows/vectors
        x_u = self.X[u_idx]               # (B,D)
        y_i = self.Y[i_idx]               # (B,D)
        eu  = eU[u_idx]                   # (B,D)
        gi  = self.Gamma_i[i_idx]         # (B,D)
        gu  = self.Gamma_u[u_idx]         # (B,D)
        ei  = eI[i_idx]                   # (B,D)
        # Eq. (5)
        return (x_u * y_i).sum(-1) + self.alpha * (eu * gi).sum(-1) + self.beta * (gu * ei).sum(-1)

    # ---------- Fit with classic SGD over observed entries ----------
    def fit(
        self,
        R,                    # numpy array (U×I), np.nan for missing
        lr=0.01,              # learning rate η
        epochs=30,
        batch_size=4096,
        # Regularization (Eq. 9): λ terms
        lambda_xy=1e-3,       # ||X||^2 + ||Y||^2
        lambda_gamma_u=1e-3,  # ||Γ^(U)||^2
        lambda_gamma_i=1e-3,  # ||Γ^(I)||^2
        lambda_theta_u=1e-3,  # sum_l (||M_u^(l)||^2 + ||b_u^(l)||^2 + ||w_u^(l)||^2)
        lambda_theta_i=1e-3,  # sum_l (||M_i^(l)||^2 + ||b_i^(l)||^2 + ||w_i^(l)||^2)
        verbose=True
    ):
        self.train()
        R = np.asarray(R, dtype=np.float32)
        obs = np.where(~np.isnan(R))
        u_idx_all = torch.tensor(obs[0], dtype=torch.long, device=self.device)
        i_idx_all = torch.tensor(obs[1], dtype=torch.long, device=self.device)
        r_all = torch.tensor(R[obs], dtype=torch.float32, device=self.device)

        # opt = torch.optim.SGD(self.parameters(), lr=lr)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        n = len(r_all)
        train_losses, val_losses = [], []

        for ep in range(epochs):
            # Precompute fused embeddings for this epoch (they depend on Θ and w, so require grad)
            # We’ll recompute inside batches to keep the graph correct for autograd.
            perm = torch.randperm(n, device=self.device)
            total_loss = 0.0

            for start in range(0, n, batch_size):
                idx = perm[start:start+batch_size]
                u_b = u_idx_all[idx]
                i_b = i_idx_all[idx]
                r_b = r_all[idx]

                opt.zero_grad()

                # ---- Forward (Eq. 5) ----
                # We compute fused embeddings used in this batch so gradients flow into Θ and w.
                eU = self.fuse_users()
                eI = self.fuse_items()
                r_hat = self.predict_pairs(u_b, i_b, eU=eU, eI=eI)

                # ---- Data-fitting loss (first line of Eq. 9) ----
                mse = torch.mean((r_b - r_hat) ** 2)

                # ---- Regularization terms (second line of Eq. 9) ----
                reg = lambda_xy * (self.X.pow(2).sum() + self.Y.pow(2).sum())
                reg = reg + lambda_gamma_u * self.Gamma_u.pow(2).sum() + lambda_gamma_i * self.Gamma_i.pow(2).sum()

                if self.num_user_paths > 0:
                    reg_u = sum(M.pow(2).sum() + b.pow(2).sum() for M, b in zip(self.user_M, self.user_b))
                    reg_u = reg_u + self.user_w.pow(2).sum()
                    reg = reg + lambda_theta_u * reg_u

                if self.num_item_paths > 0:
                    reg_i = sum(M.pow(2).sum() + b.pow(2).sum() for M, b in zip(self.item_M, self.item_b))
                    reg_i = reg_i + self.item_w.pow(2).sum()
                    reg = reg + lambda_theta_i * reg_i

                loss = mse + reg
                loss.backward()   # <-- This backpropagates through Eq. (8) fusion to update Θ and w (Eq. 10–14).
                opt.step()

                total_loss += loss.item() * len(idx)
                # ---- Compute average training loss for this epoch ----
                avg_train_loss = total_loss / n
                train_losses.append(avg_train_loss)

                # # ---- Simple validation (on a random subset of observed entries) ----
                # with torch.no_grad():
                #     sample_idx = torch.randperm(n, device=self.device)[:min(5000, n)]
                #     u_val, i_val, r_val = u_idx_all[sample_idx], i_idx_all[sample_idx], r_all[sample_idx]
                #     eU, eI = self.fuse_users(), self.fuse_items()
                #     r_hat_val = self.predict_pairs(u_val, i_val, eU=eU, eI=eI)
                #     val_loss = torch.mean((r_hat_val - r_val) ** 2).item()
                #     val_losses.append(val_loss)

            if verbose:
                print(f"Epoch {ep+1:02d}/{epochs}  |  MSE+Reg: {total_loss/n:.6f}")

        self.eval()
        if verbose:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6,4))
            plt.plot(train_losses, label="Train MSE")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("HECrec Training Progress")
            plt.legend()
            plt.grid(True, alpha=0.3)
            # plt.show()

        return self

    # ---------- Public predict API ----------
    def predict(self):
        """Return the full predicted ratings matrix (U×I) as numpy array."""
        with torch.no_grad():
            pred = self.predict_full().cpu().numpy()
        return pred
