import numpy as np

EPS = 1e-10


def wiener_mask(W, H, source_idx):
    WH_full = W @ H + EPS
    WH_src  = W[:, source_idx] @ H[source_idx, :]
    return WH_src / WH_full


class _BaseNMF:
    def __init__(self, n_components=50, max_iter=300, tol=1e-5, random_state=0):
        self.n_components = n_components
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.W = None
        self.H = None

    def _init_matrices(self, X, W_init):
        rng = np.random.default_rng(self.random_state)
        F, T = X.shape
        K = self.n_components
        if W_init is not None:
            W = W_init.copy().astype(np.float64)
            assert W.shape == (F, K), f"W_init shape mismatch: {W.shape} vs ({F},{K})"
        else:
            W = rng.uniform(0.01, 1.0, (F, K))
        H = rng.uniform(0.01, 1.0, (K, T))
        col_norms = W.sum(axis=0, keepdims=True) + EPS
        W /= col_norms
        H *= col_norms.T
        return W, H


class EuclideanNMF(_BaseNMF):
    def fit(self, X, W_init=None, fix_W=False, max_iter=None, tol=None):
        max_iter = max_iter or self.max_iter
        tol      = tol      or self.tol
        X = X.astype(np.float64)
        W, H = self._init_matrices(X, W_init)

        prev_cost = None
        for _ in range(max_iter):
            H *= (W.T @ X) / (W.T @ (W @ H) + EPS)
            if not fix_W:
                W *= (X @ H.T) / ((W @ H) @ H.T + EPS)
            cost = np.sum((X - W @ H) ** 2)
            if prev_cost is not None and abs(prev_cost - cost) / (prev_cost + EPS) < tol:
                break
            prev_cost = cost

        self.W, self.H = W, H
        return self


class ISNMF(_BaseNMF):
    def fit(self, X, W_init=None, fix_W=False, max_iter=None, tol=None):
        max_iter = max_iter or self.max_iter
        tol      = tol      or self.tol
        X = np.maximum(X.astype(np.float64), EPS)
        W, H = self._init_matrices(X, W_init)

        prev_cost = None
        for _ in range(max_iter):
            Y  = np.maximum(W @ H, EPS)
            Y2 = Y ** (-2)
            Y1 = Y ** (-1)
            H *= (W.T @ (Y2 * X)) / (W.T @ Y1 + EPS)
            if not fix_W:
                Y  = np.maximum(W @ H, EPS)
                Y2 = Y ** (-2)
                Y1 = Y ** (-1)
                W *= ((Y2 * X) @ H.T) / (Y1 @ H.T + EPS)
            Y = np.maximum(W @ H, EPS)
            ratio = X / Y
            cost  = np.sum(ratio - np.log(ratio) - 1)
            if prev_cost is not None and abs(prev_cost - cost) / (abs(prev_cost) + EPS) < tol:
                break
            prev_cost = cost

        self.W, self.H = W, H
        return self


class SparseNMF(_BaseNMF):
    def __init__(self, n_components=50, lam=0.1, max_iter=300, tol=1e-5, random_state=0):
        super().__init__(n_components, max_iter, tol, random_state)
        self.lam = lam

    def fit(self, X, W_init=None, fix_W=False, max_iter=None, tol=None):
        max_iter = max_iter or self.max_iter
        tol      = tol      or self.tol
        X = X.astype(np.float64)
        W, H = self._init_matrices(X, W_init)

        prev_cost = None
        for _ in range(max_iter):
            H *= (W.T @ X) / (W.T @ (W @ H) + self.lam + EPS)
            if not fix_W:
                W *= (X @ H.T) / ((W @ H) @ H.T + EPS)
            cost = np.sum((X - W @ H) ** 2) + self.lam * H.sum()
            if prev_cost is not None and abs(prev_cost - cost) / (prev_cost + EPS) < tol:
                break
            prev_cost = cost

        self.W, self.H = W, H
        return self


class ConvolutiveNMF(_BaseNMF):
    def __init__(self, n_components=50, n_frames=10, max_iter=300, tol=1e-5, random_state=0):
        super().__init__(n_components, max_iter, tol, random_state)
        self.n_frames = n_frames

    def fit(self, X, W_init=None, fix_W=False, max_iter=None, tol=None):
        max_iter = max_iter or self.max_iter
        tol      = tol      or self.tol
        X = np.maximum(X.astype(np.float64), EPS)
        F, T = X.shape
        K = self.n_components
        P = self.n_frames
        rng = np.random.default_rng(self.random_state)

        if W_init is not None:
            W_conv = np.stack([W_init.copy() for _ in range(P)], axis=0).astype(np.float64)
        else:
            W_conv = rng.uniform(0.01, 1.0, (P, F, K))
            for tau in range(P):
                W_conv[tau] /= W_conv[tau].sum(axis=0, keepdims=True) + EPS

        H = rng.uniform(0.01, 1.0, (K, T))

        def _shift_right(M, tau):
            if tau == 0:
                return M
            return np.concatenate([np.zeros((M.shape[0], tau)), M[:, :T - tau]], axis=1)

        def _shift_left(M, tau):
            if tau == 0:
                return M
            return np.concatenate([M[:, tau:], np.zeros((M.shape[0], tau))], axis=1)

        def _model():
            Y = np.zeros((F, T))
            for tau in range(P):
                Y += W_conv[tau] @ _shift_right(H, tau)
            return np.maximum(Y, EPS)

        prev_cost = None
        for _ in range(max_iter):
            Y = _model()
            ratio = X / Y
            num_H = sum(W_conv[tau].T @ _shift_left(ratio, tau) for tau in range(P))
            den_H = sum(W_conv[tau].T @ np.ones((F, T)) for tau in range(P))
            H *= num_H / (den_H + EPS)

            if not fix_W:
                Y = _model()
                ratio = X / Y
                for tau in range(P):
                    Hs = _shift_right(H, tau)
                    W_conv[tau] *= (ratio @ Hs.T) / (np.ones((F, T)) @ Hs.T + EPS)

            Y = _model()
            cost = np.sum(X * np.log(X / Y + EPS) - X + Y)
            if prev_cost is not None and abs(prev_cost - cost) / (abs(prev_cost) + EPS) < tol:
                break
            prev_cost = cost

        self.W_conv = W_conv
        self.W = W_conv[0]
        self.H = H
        return self

    def get_model(self):
        K, T = self.H.shape
        P    = self.n_frames
        Y = np.zeros((self.W_conv[0].shape[0], T))
        for tau in range(P):
            Hs = np.concatenate([np.zeros((K, tau)), self.H[:, :T - tau]], axis=1) if tau > 0 else self.H
            Y += self.W_conv[tau] @ Hs
        return Y
