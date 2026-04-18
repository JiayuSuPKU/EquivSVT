from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv, lu_factor, lu_solve
from scipy.sparse.linalg import splu
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

__all__ = ["Kernel", "MatrixKernel"]


class Kernel(ABC):
    """
    Abstract base class for spatial kernels.

    Handles dense, sparse, and implicit (operator-based) kernels. Switches between
    an explicit representation (the kernel matrix ``K`` is stored and used directly)
    and an implicit representation (the precision matrix ``M = K^{-1}`` is stored and
    linear systems are solved on demand) based on problem size.

    Attributes
    ----------
    n : int
        Number of observations (samples).
    method : str
        Kernel method (``'gaussian'``, ``'matern'``, ``'moran'``, ``'graph_laplacian'``,
        ``'car'``).
    params : dict
        Resolved kernel parameters after defaults have been merged with user overrides
        (e.g., ``bandwidth``, ``nu``, ``rho``, ``k_neighbors``).
    stores_precision : bool
        If ``True``, the kernel is represented implicitly via its precision matrix and
        linear solves are used for :meth:`xtKx` and trace estimation. If ``False``,
        the realized kernel matrix is stored and used directly.

    Notes
    -----
    The internal buffer ``_K`` stores the kernel matrix when ``stores_precision=False`` and
    the precision matrix ``K^{-1}`` when ``stores_precision=True``. Public methods
    (:meth:`xtKx`, :meth:`trace`, :meth:`square_trace`, :meth:`eigenvalues`) transparently
    handle both cases; callers should not access ``_K`` directly.
    """

    def __init__(self, n: int, method: str = "gaussian", **kwargs) -> None:
        """
        Initialize the Kernel.

        Parameters
        ----------
        n : int
            Number of observations.
        method : str, default 'gaussian'
            Kernel method to use.
        **kwargs : dict
            Additional kernel-specific parameters stored in ``self.params``.
        """
        self.n: int = n
        """Number of observations (samples)."""
        self.method: str = method
        """Kernel method name."""
        self.params: dict = kwargs
        """Resolved kernel parameters after defaults are merged with user overrides."""

        # Threshold (in samples) for switching to the implicit representation.
        self._implicit_threshold = 5000
        self.stores_precision: bool = False
        """Whether the kernel is stored in precision form (``True``) or as the realized kernel matrix (``False``)."""
        self._lu = None  # Cache for sparse LU factorization if needed
        self._lu_lock = threading.Lock()  # Thread safety for lazy LU init

        # _K stores the kernel matrix when stores_precision=False and the precision
        # matrix K^{-1} when stores_precision=True (see class Notes).
        self._K = self._build_kernel()
        self._spectrum = None  # Lazy-evaluated eigenvalues cache (access via eigenvalues())

    @abstractmethod
    def _build_kernel(self):
        """Constructs the kernel matrix or its inverse operator."""
        pass

    def _format_params(self):
        """Format kernel params safely without dumping large arrays/matrices."""
        if not self.params:
            return "None"
        parts = []
        for k, v in self.params.items():
            try:
                if isinstance(v, np.ndarray):
                    parts.append(f"{k}=array(shape={v.shape}, dtype={v.dtype})")
                elif sp.issparse(v):
                    parts.append(f"{k}=sparse(shape={v.shape}, nnz={v.nnz})")
                else:
                    parts.append(f"{k}={v}")
            except Exception:
                parts.append(f"{k}=?")
        return ", ".join(parts)

    def __repr__(self):
        return (
            f"<Kernel method={self.method} n={self.n} implicit={self.stores_precision} "
            f"threshold={self._implicit_threshold} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        return (
            "Kernel\n"
            f"- Method: {self.method}\n"
            f"- Samples: {self.n}\n"
            f"- Implicit: {self.stores_precision} (threshold={self._implicit_threshold})\n"
            f"- Params: {self._format_params()}"
        )

    def realization(self) -> np.ndarray:
        """
        Return the realized (N, N) kernel matrix.

        Returns
        -------
        np.ndarray
            Dense (N, N) kernel matrix.

        Notes
        -----
        If ``stores_precision`` is True, this forces expensive dense inversion of the
        precision matrix. Prefer :meth:`xtKx` and :meth:`trace` for implicit kernels.
        """
        if self.stores_precision:
            # _K is M = K^-1. We need to invert it.
            if sp.issparse(self._K):
                return inv(self._K.toarray())
            else:
                return inv(self._K)
        return self._K

    def eigenvalues(self, k: int | None = None) -> np.ndarray:
        """
        Compute the k largest eigenvalues of the kernel matrix.

        Results are cached internally; subsequent calls reuse the cached spectrum
        when it contains enough values to satisfy the request.

        Parameters
        ----------
        k : int, optional
            Number of largest eigenvalues to return. If None, returns all.

        Returns
        -------
        np.ndarray
            Eigenvalues sorted in descending order, shape (k,) or (n,).
        """
        if self._spectrum is not None:
            # check if we have enough cached (spectrum is always descending)
            if k is None and len(self._spectrum) == self.n:
                return self._spectrum
            elif k is not None and len(self._spectrum) >= k:
                return self._spectrum[:k]

        k_orig = k  # preserve original before internal modification

        if self.stores_precision:
            # Implicit case with kernel inverse: Use sparse methods
            from scipy.sparse.linalg import eigsh

            k = k if k is not None else max(6, self.n - 2)
            vals, _ = eigsh(self._K, k=k, which="SM")  # Smallest magnitude of K^-1 = largest of K
            vals = np.real(1.0 / vals)
        else:
            # Handle kernel matrix directly
            if sp.issparse(self._K):
                from scipy.sparse.linalg import eigsh

                k = k if k is not None else max(6, self.n - 2)
                vals, _ = eigsh(self._K, k=k, which="LM")
                vals = np.real(vals)
            else:
                vals = np.linalg.eigvalsh(self._K)  # ascending order

        self._spectrum = np.sort(vals)[::-1]  # always store descending
        return self._spectrum if k_orig is None else self._spectrum[:k_orig]

    # ------------------------------------------------------------------
    # Internal primitive: compute ``K @ x`` as a dense 2D block
    # ------------------------------------------------------------------
    def _apply_K_dense(self, x_2d: np.ndarray) -> np.ndarray:
        """Compute ``K @ x_2d`` and return a dense ``(N, M)`` ndarray.

        Used as the shared kernel of :meth:`Kx`, :meth:`xtKx`, :meth:`xtKy`.
        Expects a dense ``(N, M)`` input; sparse inputs must be densified by
        the caller. Implicit precision solves go through a cached LU; explicit
        kernels dispatch to the underlying sparse / dense matmul.
        """
        if self.stores_precision:
            if sp.issparse(self._K):
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = splu(self._K.tocsc())
                y = self._lu.solve(x_2d)
            else:
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = lu_factor(self._K)
                y = lu_solve(self._lu, x_2d)
            return np.asarray(y)
        # Explicit: sparse K → K.dot(dense) returns dense. Dense K → dense @ dense.
        y = self._K.dot(x_2d)
        if sp.issparse(y):  # pragma: no cover — current scipy always returns dense
            y = np.asarray(y.todense())
        return np.asarray(y)

    @staticmethod
    def _to_2d(x: np.ndarray | sp.spmatrix) -> tuple[Any, bool]:
        """Normalize ``x`` to a 2D ``(N, M)`` (sparse or dense) and report whether
        the caller passed a 1D vector. Does *not* densify sparse input.
        """
        if sp.issparse(x):
            if x.ndim == 1 or x.shape[1] == 1 and x.shape[0] == 1:
                # scipy rarely exposes 1D sparse; reshape if someone slipped it in.
                return x.reshape(-1, 1), True
            if x.shape[1] == 1 and x.shape[0] > 1:
                return x, False  # already a (N, 1) sparse column
            return x, False
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.reshape(-1, 1), True
        return arr, False

    def Kx(self, x: np.ndarray | sp.spmatrix) -> np.ndarray:
        """
        Apply the kernel operator to ``x``, returning ``K @ x``.

        Single public primitive for kernel–vector products. Handles explicit
        (dense or sparse ``K``) and implicit (precision matrix + cached LU) cases
        uniformly.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(N,)`` or ``(N, M)``. Sparse inputs are densified internally because
            ``scipy.linalg.lu_solve`` / ``splu.solve`` require dense RHS and
            ``K @ x`` typically returns dense anyway.

        Returns
        -------
        np.ndarray
            ``(N,)`` if ``x`` was 1D, else ``(N, M)``.

        Examples
        --------
        >>> import numpy as np
        >>> from quadsv import MatrixKernel
        >>> rng = np.random.default_rng(0)
        >>> coords = rng.standard_normal((40, 2))
        >>> kernel = MatrixKernel.from_coordinates(coords, method="matern")
        >>> kernel.Kx(rng.standard_normal(40)).shape
        (40,)
        """
        x_2d, squeeze = self._to_2d(x)
        if sp.issparse(x_2d):
            x_2d = x_2d.toarray()
        y = self._apply_K_dense(x_2d)
        return y.ravel() if squeeze else y

    def _xtKy_from_Ky(
        self,
        x: np.ndarray | sp.spmatrix,
        Ky: np.ndarray,
        n_cols: int,
    ) -> float | np.ndarray:
        """Given sparse-or-dense ``x`` (``(N, M)``) and dense ``Ky`` (``(N, M)``),
        return the paired diagonal ``sum(x_i * Ky_i, axis=0)``.

        Preserves sparsity of ``x`` — ``x.multiply(Ky).sum(axis=0)`` iterates only
        x's non-zeros. Falls back to ``np.sum(x * Ky, axis=0)`` when ``x`` is dense.
        """
        if sp.issparse(x):
            result = np.asarray(x.multiply(Ky).sum(axis=0)).ravel()
        else:
            result = np.sum(x * Ky, axis=0)
        if n_cols == 1:
            return float(result.item())
        return result

    def xtKy(
        self, x: np.ndarray | sp.spmatrix, y: np.ndarray | sp.spmatrix
    ) -> float | np.ndarray:
        """
        Bilinear form ``x^T K y`` (paired diagonal for batched inputs).

        For ``(N, M)`` batches returns ``(M,)`` — the diagonal of ``X^T K Y``
        in the same column order, matching :func:`quadsv.spatial_r_test`.
        Sparse ``x`` is preserved; only ``K @ y`` is densified.

        Parameters
        ----------
        x, y : np.ndarray or scipy.sparse matrix
            ``(N,)`` or ``(N, M)`` (must share the M).

        Returns
        -------
        float or np.ndarray
            Scalar if 1D inputs; ``(M,)`` if batched.

        Examples
        --------
        >>> import numpy as np
        >>> from quadsv import MatrixKernel
        >>> rng = np.random.default_rng(0)
        >>> coords = rng.standard_normal((40, 2))
        >>> kernel = MatrixKernel.from_coordinates(coords, method="matern")
        >>> x = rng.standard_normal(40)
        >>> y = rng.standard_normal(40)
        >>> isinstance(kernel.xtKy(x, y), float)
        True
        """
        x_2d, x_squeeze = self._to_2d(x)
        y_2d, y_squeeze = self._to_2d(y)
        squeeze = x_squeeze and y_squeeze
        n_cols = x_2d.shape[1]
        y_dense = y_2d.toarray() if sp.issparse(y_2d) else y_2d
        Ky = self._apply_K_dense(y_dense)
        return self._xtKy_from_Ky(x_2d, Ky, 1 if squeeze else n_cols)

    def xtKx(self, x: np.ndarray | sp.spmatrix) -> float | np.ndarray:
        """
        Quadratic form ``x^T K x`` (paired diagonal for batched inputs).

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(N,)`` or ``(N, M)``. Sparse ``x`` is preserved through the
            final ``x^T (K x)`` contraction — only the right side ``K @ x``
            needs a dense RHS for the solver / BLAS call.

        Returns
        -------
        float or np.ndarray
            Scalar if 1D input, ``(M,)`` if batched.
        """
        x_2d, squeeze = self._to_2d(x)
        n_cols = x_2d.shape[1]
        x_dense = x_2d.toarray() if sp.issparse(x_2d) else x_2d
        Kx = self._apply_K_dense(x_dense)
        return self._xtKy_from_Ky(x_2d, Kx, 1 if squeeze else n_cols)

    # ------------------------------------------------------------------
    # Sparsity-preserving standardized quadratic form
    # ------------------------------------------------------------------
    def _K_column_sums(self) -> tuple[np.ndarray, float]:
        """Return (``K @ 1_N``, ``1_N^T K 1_N``), computed once and cached.

        Used by :meth:`xtKx_standardized` to evaluate the mean-centering
        correction without densifying sparse inputs.
        """
        cache = getattr(self, "_K_col_sum_cache", None)
        if cache is not None:
            return cache
        ones = np.ones((self.n, 1))
        K_sum = self._apply_K_dense(ones).ravel()  # (N,)
        K_total = float(K_sum.sum())
        self._K_col_sum_cache = (K_sum, K_total)
        return self._K_col_sum_cache

    def xtKx_standardized(
        self,
        x: np.ndarray | sp.spmatrix,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """
        Compute ``z^T K z`` where ``z = (x - means) / stds`` *without* densifying
        sparse ``x``.

        Uses the expansion
        ``z^T K z = (x^T K x - 2 μ (K·1)^T x + μ² (1^T K 1)) / σ²``
        and the cached row sums of ``K`` to compute every term sparse-aware.
        This is the fast path for standardizing large sparse feature matrices
        (e.g. scRNA-seq counts) before a Q-test.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(N,)`` or ``(N, M)``. Columns correspond to features.
        means, stds : np.ndarray
            ``(M,)`` per-feature mean and std (``ddof=1`` to match
            :func:`quadsv.statistics.spatial_q_test`).

        Returns
        -------
        np.ndarray
            ``(M,)`` standardized quadratic form values. Columns with
            ``std <= 0`` are returned as zero.
        """
        x_2d, _ = self._to_2d(x)
        n_cols = x_2d.shape[1]
        means = np.asarray(means, dtype=float).reshape(-1)
        stds = np.asarray(stds, dtype=float).reshape(-1)
        if means.shape[0] != n_cols or stds.shape[0] != n_cols:
            raise ValueError(
                f"means/stds shape {means.shape}/{stds.shape} "
                f"inconsistent with x columns ({n_cols})."
            )

        K_sum, K_total = self._K_column_sums()  # K_sum: (N,), K_total: scalar

        # Term 1: x^T K x — reuse xtKx (handles sparse/dense uniformly).
        q_raw = np.atleast_1d(np.asarray(self.xtKx(x_2d))).astype(float)

        # Term 2: (K·1)^T x → (M,). x^T @ K_sum, preserving x's sparsity.
        if sp.issparse(x_2d):
            ksum_x = np.asarray(x_2d.T @ K_sum).ravel()
        else:
            ksum_x = x_2d.T @ K_sum

        # Standardized quadratic form
        q_centered = q_raw - 2.0 * means * ksum_x + (means**2) * K_total
        valid = stds > 1e-12
        out = np.zeros(n_cols, dtype=float)
        out[valid] = q_centered[valid] / (stds[valid] ** 2)
        return out

    def _get_rvs_trace_cache(self, n_vectors=15):
        """Generate random vectors for trace estimation caching."""
        if not self.stores_precision:
            raise RuntimeError("Trace caching is only for implicit kernels.")

        # Check if cache exists
        if hasattr(self, "_trace_rvs_cache"):
            if self._trace_rvs_cache["n_vectors"] == n_vectors:
                return self._trace_rvs_cache
            else:
                warnings.warn(
                    "Updating trace random vectors cache with different n_vectors.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Hutchinson's trick random vectors
        rvs = np.random.choice([-1, 1], size=(self.n, n_vectors))
        # Batched Solve: Solve M * Y = rvs
        # spsolve can handle multiple RHS if passed as dense 2D array
        if sp.issparse(self._K):
            if self._lu is None:
                self._lu = splu(self._K.tocsc())

            Y = self._lu.solve(rvs)
            # Ensure Y is 2D even if n_vectors=1
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
        else:
            if self._lu is None:
                self._lu = lu_factor(self._K)
            Y = lu_solve(self._lu, rvs)

        # Cache for future use
        self._trace_rvs_cache = {"n_vectors": n_vectors, "rvs": rvs, "Y": Y}
        return self._trace_rvs_cache

    def trace(self) -> float:
        """
        Compute the trace of the kernel matrix Tr(K).

        For implicit kernels, uses Hutchinson's trick with random ±1 vectors
        for efficient O(N) estimation instead of O(N³) eigendecomposition.

        Returns
        -------
        float
            Trace of the kernel matrix.

        Notes
        -----
        For implicit kernels the result is a stochastic estimate using a fixed
        ``n_vectors=15`` Hutchinson probes; repeated calls reuse the cached probes
        so the returned value is deterministic within a given instance.
        """
        if self.stores_precision:
            # Trace estimation using Hutchinson's trick
            n_vectors = 15
            cache = self._get_rvs_trace_cache(n_vectors)

            # Hutchinson's estimator is (1/n_vectors) * sum(v_i^T * y_i)
            rvs = cache["rvs"]
            Y = cache["Y"]
            return np.sum(rvs * Y) / n_vectors

        elif sp.issparse(self._K):
            return self._K.diagonal().sum()
        else:
            return np.trace(self._K)

    def square_trace(self) -> float:
        """
        Compute the trace of the squared kernel Tr(K²).

        Used for variance estimation in statistical tests. For implicit kernels,
        uses Hutchinson's trick for efficient O(N) estimation.

        Returns
        -------
        float
            Trace of K squared.

        Notes
        -----
        For implicit kernels the result is a stochastic estimate using a fixed
        ``n_vectors=15`` Hutchinson probes (shared with :meth:`trace`).
        """
        if self.stores_precision:
            # Trace(K^2) estimation
            n_vectors = 15
            cache = self._get_rvs_trace_cache(n_vectors)

            # Trace(K^2) ~= (1/m) * sum( ||K v_i||^2 )
            # Since Y = K * rvs, we just need sum(Y^2)
            Y = cache["Y"]
            return np.sum(Y**2) / n_vectors

        elif sp.issparse(self._K):
            return self._K.power(2).sum()
        else:
            return np.sum(self._K**2)

    def _compute_inv_diag(self, M):
        """Compute diagonal of K = M^{-1} using batched solves to save memory.

        For sparse M, uses batched splu solves on chunks of the identity matrix.
        This avoids allocating a dense N x N inverse matrix.
        """
        n = self.n

        if sp.issparse(M):
            # Factorize once
            if self._lu is None:
                self._lu = splu(M.tocsc())
            lu = self._lu

            # Result array for the diagonal
            diag_vals = np.zeros(n)

            # Determine batch size (100-1000 is usually optimal for cache locality)
            batch_size = 128
            with tqdm(
                total=n,
                desc="Computing diagonal of K",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as pbar:
                for i in range(0, n, batch_size):
                    end = min(i + batch_size, n)
                    current_batch_size = end - i

                    # Create the RHS block: A slice of the Identity matrix.
                    # This corresponds to columns i through end of I.
                    # Shape: (n, current_batch_size)
                    # We construct it directly to save memory.
                    b = np.zeros((n, current_batch_size))

                    # Fill the specific rows that correspond to the diagonal 1s
                    # For the k-th column in this batch (which corresponds to global column i+k),
                    # the 1 is at row i+k.
                    b[i:end, :] = np.eye(current_batch_size)

                    # Solve M * x = b  =>  x = M^{-1} * b
                    x = lu.solve(b)

                    # We only need the diagonal elements of M^{-1}.
                    # In the result x (shape n, batch), the diagonal elements of M^{-1}
                    # are located at x[i, 0], x[i+1, 1], ..., x[end-1, end-1-i]
                    # This corresponds to the diagonal of the square block starting at row i.
                    diag_vals[i:end] = x[i:end, :].diagonal()

                    # Update the progress bar by the actual number of items processed in this batch
                    pbar.update(current_batch_size)

            return diag_vals

        else:
            # Dense case: Direct inversion (slow, should work with kernel matrix directly)
            warnings.warn(
                "Dense precision inversion invoked. Consider using the kernel matrix directly.",
                RuntimeWarning,
                stacklevel=2,
            )
            Minv = inv(M)
            return np.diag(Minv)

    def _standardize_precision(self, M):
        """Scale precision M so that covariance K has unit diagonal, without forming dense K when implicit."""
        diag_K = self._compute_inv_diag(M).copy()
        diag_K[diag_K <= 0] = 1e-12
        s = 1.0 / np.sqrt(diag_K)
        if sp.issparse(M):
            S_inv = sp.diags(1.0 / s)
            return S_inv @ M @ S_inv
        else:
            S_inv = np.diag(1.0 / s)
            return S_inv @ M @ S_inv

    def __getstate__(self):
        """
        Custom pickling behavior: exclude unpicklable SuperLU objects and locks.
        """
        state = self.__dict__.copy()
        # Remove the cached LU factorization because SuperLU objects cannot be pickled.
        # Workers will re-compute this locally.
        state["_lu"] = None
        # Locks cannot be pickled; will be recreated in __setstate__
        state.pop("_lu_lock", None)
        return state

    def __setstate__(self, state):
        """
        Restore state and ensure _lu is reset to None.
        """
        self.__dict__.update(state)
        # Ensure _lu is explicitly None upon restoration
        self._lu = None
        self._lu_lock = threading.Lock()


class MatrixKernel(Kernel):
    """
    Concrete spatial kernel built from coordinates or a precomputed matrix.

    Inherits all public attributes and methods from :class:`Kernel`
    (``n``, ``method``, ``params``, ``stores_precision``, :meth:`realization`,
    :meth:`eigenvalues`, :meth:`xtKx`, :meth:`trace`, :meth:`square_trace`).

    See Also
    --------
    MatrixKernel.from_coordinates
        Recommended entry point when working from raw sample coordinates.
    MatrixKernel.from_matrix
        Recommended entry point when a kernel or precision matrix is already
        available.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self, data: np.ndarray, mode: str = "coords", method: str = "matern", **kwargs
    ) -> None:
        """
        Construct a spatial kernel from already-prepared input data.

        This constructor is public but low-level; most users should prefer the
        factory methods :meth:`from_coordinates` or :meth:`from_matrix`, which
        dispatch to this constructor with the appropriate ``mode``.

        Parameters
        ----------
        data : np.ndarray or scipy.sparse matrix
            Input data whose interpretation is controlled by ``mode``:
            an ``(N, D)`` coordinate array when ``mode='coords'``, an ``(N, N)``
            kernel matrix when ``mode='precomputed'``, or an ``(N, N)`` precision
            matrix when ``mode='precomputed_inverse'``.
        mode : {'coords', 'precomputed', 'precomputed_inverse'}, default 'coords'
            How ``data`` should be interpreted.
        method : str, default 'matern'
            Kernel method. Must be one of ``'gaussian'``, ``'matern'``, ``'moran'``,
            ``'graph_laplacian'``, ``'car'``, or ``'precomputed'``.
        **kwargs : dict
            Kernel-specific parameters (e.g., ``bandwidth``, ``nu``, ``rho``,
            ``k_neighbors``). Unknown keys raise :class:`ValueError`.

        Raises
        ------
        ValueError
            If ``mode`` or ``method`` is unknown, or any parameter fails validation.
        """
        self._data = data
        self._mode = mode
        if mode not in ("coords", "precomputed", "precomputed_inverse"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'coords', 'precomputed', or 'precomputed_inverse'."
            )

        if method not in self._available_kernels + ["precomputed"]:
            raise ValueError(f"Unknown kernel method: {method}.")

        # Update kernel parameters from defaults
        defaults = self._get_default_params(method).copy()
        if kwargs:
            for key, value in kwargs.items():
                if key in defaults:
                    defaults[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        n = data.shape[0]
        if mode == "coords":
            self._validate_coords_params(n, method, defaults)

        super().__init__(n, method=method, **defaults)

    @staticmethod
    def _validate_coords_params(n: int, method: str, params: dict) -> None:
        """Validate parameters for coordinate-based kernel construction."""
        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}")
        if method in ("gaussian", "matern"):
            bw = params.get("bandwidth", None)
            if bw is not None and bw <= 0:
                raise ValueError(f"bandwidth must be positive, got {bw}")
        if method == "matern":
            nu = params.get("nu", None)
            if nu is not None and nu <= 0:
                raise ValueError(f"nu must be positive, got {nu}")
        if method in ("moran", "graph_laplacian", "car"):
            k = params.get("k_neighbors", None)
            if k is not None and (k < 1 or k >= n):
                raise ValueError(f"k_neighbors must be in [1, {n - 1}], got {k}")
        if method == "car":
            rho = params.get("rho", None)
            if rho is not None and rho < 0:
                raise ValueError(f"rho must be non-negative, got {rho}")

    def _get_default_params(self, method: str) -> dict[str, Any]:
        """
        Returns default parameters for specific kernel methods.

        Parameters
        ----------
        method : str
            Kernel method name. Should be one of _available_kernels.

        Returns
        -------
        dict[str, Any]
            Method defaults: bandwidth (gaussian/matern), nu (matern), k_neighbors (moran/graph_laplacian/car), rho (car).
        """
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"bandwidth": 2.0, "nu": 1.5},
            "moran": {"k_neighbors": 4},
            "graph_laplacian": {"k_neighbors": 4},
            "car": {"rho": 0.9, "k_neighbors": 4, "standardize": False},
        }
        return method_defaults.get(method, {})

    @classmethod
    def from_coordinates(cls, coords: np.ndarray, method: str = "matern", **kwargs) -> MatrixKernel:
        """
        Build kernel from spatial coordinates.

        Parameters
        ----------
        coords : np.ndarray
            Array of spatial coordinates, shape (N, D).
        method : str, default 'matern'
            Kernel method. Must be one of 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
        **kwargs : dict
            Additional kernel parameters (bandwidth, nu, rho, k_neighbors, etc.).

        Returns
        -------
        MatrixKernel
            Initialized kernel object.

        Raises
        ------
        ValueError
            If ``method`` is not one of :attr:`_available_kernels`.

        Examples
        --------
        >>> coords = np.random.randn(100, 2)
        >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian', bandwidth=1.0)
        """
        if method not in cls._available_kernels:
            raise ValueError(f"Unknown kernel method for coordinates: {method}.")

        return cls(coords, mode="coords", method=method, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray | sp.spmatrix,
        is_precision: bool = False,
        method: str = "precomputed",
        **kwargs,
    ) -> MatrixKernel:
        """
        Build kernel from a precomputed kernel matrix or its inverse.

        Parameters
        ----------
        matrix : np.ndarray or scipy.sparse matrix
            Kernel matrix (N, N) or its inverse (precision matrix).
        is_precision : bool, default False
            If True, matrix is treated as the inverse (precision) matrix K^-1.
        method : str, default 'precomputed'
            The logical kernel method (e.g., 'car' for precision matrices).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        MatrixKernel
            Initialized kernel object.

        Examples
        --------
        >>> K = np.array([[2, -1], [-1, 2]])  # kernel matrix
        >>> kernel = MatrixKernel.from_matrix(K, is_precision=False)
        """
        mode = "precomputed_inverse" if is_precision else "precomputed"
        return cls(matrix, mode=mode, method=method, **kwargs)

    def _build_kernel(self):  # noqa: C901
        method = self.method

        # ==========================================
        # 1. PREPARE RAW INPUTS (Dists or Weights)
        # ==========================================

        # Case A: Coordinates provided -> Compute Dists or W from scratch
        if self._mode == "coords":
            coords = self._data
            if method in ["gaussian", "matern"]:
                # Compute dense distance matrix
                dists = squareform(pdist(coords, metric="euclidean"))
                W = None
            elif method in ["moran", "graph_laplacian", "car"]:
                # Compute sparse adjacency graph
                k = self.params["k_neighbors"]
                nbrs = NearestNeighbors(
                    n_neighbors=k + 1, algorithm="auto", metric="euclidean"
                ).fit(coords)
                W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)

                # Mutual neighbors: keep only edges where both spots list each other
                W_mut = W + W.T
                W_mut.data = (W_mut.data > 1).astype(float)
                W_mut.setdiag(0)

                # Handle isolated nodes: add self-loop to avoid division-by-zero
                row_sums = np.asarray(W_mut.sum(axis=1)).ravel()
                isolated = row_sums == 0
                if isolated.any():
                    W_mut.setdiag(isolated.astype(float))
                W_mut.eliminate_zeros()

                W = W_mut
                dists = None
            else:
                raise ValueError(f"Unknown method for coordinates: {method}")

        # Case B: Precomputed Kernel provided
        elif self._mode == "precomputed":
            return self._data

        # Case C: Precomputed Inverse Kernel provided
        elif self._mode == "precomputed_inverse":
            M = self._data
            standardize = self.params.get("standardize", False)

            # If small, realize dense K; else keep implicit precision
            if self.n <= self._implicit_threshold:
                try:
                    M_dense = M.toarray() if sp.issparse(M) else M
                    K_dense = inv(M_dense)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "Precision matrix is singular; using pseudo-inverse.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    M_dense = M.toarray() if sp.issparse(M) else M
                    K_dense = np.linalg.pinv(M_dense)

                if standardize:
                    diag_K = np.diag(np.asarray(K_dense)).copy()
                    diag_K[diag_K <= 0] = 1e-12
                    s = 1.0 / np.sqrt(diag_K)
                    S = np.diag(s)
                    K_dense = S @ K_dense @ S

                return 0.5 * (K_dense + K_dense.T)
            else:
                self.stores_precision = True
                if standardize:
                    M = self._standardize_precision(M)
                return M

        # ==========================================
        # 2. CONSTRUCT KERNEL FROM INPUTS
        # ==========================================

        # --- Distance Based ---
        if method in ["gaussian", "matern"]:
            bw = self.params["bandwidth"]

            if method == "gaussian":
                K = np.exp(-(dists**2) / (2 * bw**2))
            elif method == "matern":
                nu = self.params["nu"]
                length_scale = bw
                # Mask zero distances; evaluate Bessel K only at non-zero distances
                mask_zero = dists == 0
                dists_safe = dists.copy()
                dists_safe[mask_zero] = 1.0  # dummy value, overwritten below
                factor = (np.sqrt(2 * nu) * dists_safe) / length_scale
                K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
                K[mask_zero] = 1.0  # correct limit: K(x, x) = 1
            return K

        # --- Graph Based ---
        elif method in ["moran", "graph_laplacian", "car"]:
            # Symmetrize and apply symmetric normalization: D^{-1/2} W D^{-1/2}
            if W is None:
                raise ValueError("Graph weights (W) required for graph kernels.")

            # Ensure float
            W = W.astype(float)

            # Symmetrize first
            W_sym = 0.5 * (W + W.T)

            # Zero out self-loops
            W_sym.setdiag(0)

            # Degree-based symmetric normalization
            row_sums = np.array(W_sym.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_D_sqrt = sp.diags(1.0 / np.sqrt(row_sums))
            W_norm = inv_D_sqrt @ W_sym @ inv_D_sqrt

            if method == "moran":
                # Already symmetric and normalized
                return W_norm

            elif method == "graph_laplacian":
                I = sp.eye(self.n, format="csr")
                return I - W_norm

            elif method == "car":
                rho = self.params["rho"]
                if rho >= 1.0:
                    warnings.warn(
                        f"rho={rho} >= 1.0 causes singularity in CAR kernel; clamping to 0.99",
                        UserWarning,
                        stacklevel=2,
                    )
                    rho = 0.99
                    self.params["rho"] = rho
                standardize = self.params["standardize"]
                I = sp.eye(self.n, format="csc")
                # M = (I - rho * W_norm) is the inverse of the CAR kernel
                M = I - rho * W_norm

                if self.n > self._implicit_threshold:
                    self.stores_precision = True
                    if standardize:
                        M = self._standardize_precision(M)
                    return M
                else:
                    try:
                        K_dense = inv(M.toarray())
                    except np.linalg.LinAlgError:
                        warnings.warn(
                            "CAR precision matrix is singular; using pseudo-inverse. "
                            "Consider reducing rho or changing k_neighbors.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        K_dense = np.linalg.pinv(M.toarray())

                    if standardize:
                        diag_K = np.diag(np.asarray(K_dense)).copy()
                        diag_K[diag_K <= 0] = 1e-12
                        s = 1.0 / np.sqrt(diag_K)
                        S = np.diag(s)
                        K_dense = S @ K_dense @ S

                    return 0.5 * (K_dense + K_dense.T)

        else:
            raise ValueError(f"Unknown kernel method: {method}")

    def __repr__(self):
        # Describe input data succinctly
        if self._mode == "coords":
            coords = self._data
            data_desc = f"coords shape={getattr(coords, 'shape', '?')}"
        elif self._mode == "precomputed":
            M = self._data
            if sp.issparse(M):
                data_desc = f"matrix shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"matrix shape={getattr(M, 'shape', '?')} dense"
        elif self._mode == "precomputed_inverse":
            M = self._data
            if sp.issparse(M):
                data_desc = f"precision shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"precision shape={getattr(M, 'shape', '?')} dense"
        else:
            data_desc = "data=?"

        return (
            f"<MatrixKernel method={self.method} mode={self._mode} n={self.n} "
            f"implicit={self.stores_precision} data={data_desc} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        # Human-friendly multi-line summary
        lines = [
            "MatrixKernel",
            f"- Method: {self.method}",
            f"- Mode: {self._mode}",
            f"- Samples: {self.n}",
            f"- Implicit: {self.stores_precision} (threshold={self._implicit_threshold})",
        ]

        # Add a brief data description
        try:
            if self._mode == "coords":
                coords = self._data
                lines.append(f"- Data: coords shape={getattr(coords, 'shape', '?')}")
            else:
                M = self._data
                if sp.issparse(M):
                    kind = "precision" if self._mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={M.shape} sparse nnz={M.nnz}")
                else:
                    kind = "precision" if self._mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={getattr(M, 'shape', '?')} dense")
        except Exception:
            lines.append("- Data: ?")

        lines.append(f"- Params: {self._format_params()}")
        return "\n".join(lines)
