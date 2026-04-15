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

__all__ = ["Kernel", "SpatialKernel"]


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
    is_implicit : bool
        If ``True``, the kernel is represented implicitly via its precision matrix and
        linear solves are used for :meth:`xtKx` and trace estimation. If ``False``,
        the realized kernel matrix is stored and used directly.

    Notes
    -----
    The internal buffer ``_K`` stores the kernel matrix when ``is_implicit=False`` and
    the precision matrix ``K^{-1}`` when ``is_implicit=True``. Public methods
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
        self.is_implicit: bool = False
        """Whether the kernel is stored in precision form (``True``) or as the realized kernel matrix (``False``)."""
        self._lu = None  # Cache for sparse LU factorization if needed
        self._lu_lock = threading.Lock()  # Thread safety for lazy LU init

        # _K stores the kernel matrix when is_implicit=False and the precision
        # matrix K^{-1} when is_implicit=True (see class Notes).
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
            f"<Kernel method={self.method} n={self.n} implicit={self.is_implicit} "
            f"threshold={self._implicit_threshold} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        return (
            "Kernel\n"
            f"- Method: {self.method}\n"
            f"- Samples: {self.n}\n"
            f"- Implicit: {self.is_implicit} (threshold={self._implicit_threshold})\n"
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
        If ``is_implicit`` is True, this forces expensive dense inversion of the
        precision matrix. Prefer :meth:`xtKx` and :meth:`trace` for implicit kernels.
        """
        if self.is_implicit:
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

        if self.is_implicit:
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

    def xtKx(self, x: np.ndarray | sp.spmatrix) -> float | np.ndarray:  # noqa: C901
        """
        Efficiently compute the quadratic form x^T K x.

        Handles both single vectors and batches. Uses implicit solvers when the
        kernel is stored in precision form (``is_implicit=True``) to avoid dense
        matrix operations. Supports sparse input matrices without densification.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            Input vector of shape (N,) or batch of vectors (N, M).
            Can be dense numpy array or sparse matrix (CSC/CSR format).

        Returns
        -------
        float or np.ndarray
            Quadratic form value(s). Scalar if input was 1D, shape (M,) if input was 2D.

        Notes
        -----
        For implicit kernels, uses sparse solve instead of matrix inversion.
        For sparse input, maintains sparsity throughout computation where possible.
        """
        # Handle sparse input
        is_sparse_input = sp.issparse(x)

        # Allow x to be a matrix (N, n_vectors) or vector (N,)
        # If vector, reshape to (N, 1) to ensure matrix math works
        if is_sparse_input:
            if x.ndim == 1 or (hasattr(x, "shape") and len(x.shape) == 1):
                x_in = x.reshape(-1, 1)
            else:
                x_in = x
            n_cols = x_in.shape[1]
        else:
            x_in = x if x.ndim > 1 else x.reshape(-1, 1)
            n_cols = x_in.shape[1]

        if self.is_implicit:
            # Case: CAR model (Large N)
            # We want x^T M^-1 x
            # Solve My = x
            if sp.issparse(self._K):
                # Cache LU factorization for efficiency (thread-safe)
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = splu(self._K.tocsc())
                if is_sparse_input:
                    # Convert to dense for solver (most efficient for sparse solvers)
                    y = self._lu.solve(x_in.toarray())
                else:
                    y = self._lu.solve(x_in)
            else:
                # Cache LU factorization for efficiency (thread-safe)
                with self._lu_lock:
                    if self._lu is None:
                        self._lu = lu_factor(self._K)
                if is_sparse_input:
                    y = lu_solve(self._lu, x_in.toarray())
                else:
                    y = lu_solve(self._lu, x_in)

            # Compute x^T @ y efficiently
            if is_sparse_input:
                # For sparse x, use multiply and sum
                result = np.asarray(x_in.multiply(y).sum(axis=0)).flatten()
            else:
                result_matrix = x_in.T.dot(y)  # (M, M)
                result = np.diag(result_matrix) if n_cols > 1 else result_matrix.item()
        else:
            # Standard Case: K is realized as a dense or sparse matrix
            if is_sparse_input:
                # Sparse @ Dense matrix multiplication
                Kx = self._K.dot(x_in.toarray() if hasattr(x_in, "toarray") else x_in)
                # x^T @ Kx with sparse x
                result = np.asarray(x_in.multiply(Kx).sum(axis=0)).flatten()
            else:
                Kx = self._K.dot(x_in)
                result_matrix = x_in.T.dot(Kx)  # (M, M)
                result = np.diag(result_matrix) if n_cols > 1 else result_matrix.item()

        # Return appropriate shape
        if n_cols > 1:
            return result if isinstance(result, np.ndarray) else np.diag(result)
        else:
            return result if np.isscalar(result) else result.item()

    def _get_rvs_trace_cache(self, n_vectors=15):
        """Generate random vectors for trace estimation caching."""
        if not self.is_implicit:
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
        if self.is_implicit:
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
        if self.is_implicit:
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


class SpatialKernel(Kernel):
    """
    Concrete spatial kernel built from coordinates or a precomputed matrix.

    Inherits all public attributes and methods from :class:`Kernel`
    (``n``, ``method``, ``params``, ``is_implicit``, :meth:`realization`,
    :meth:`eigenvalues`, :meth:`xtKx`, :meth:`trace`, :meth:`square_trace`).

    See Also
    --------
    SpatialKernel.from_coordinates
        Recommended entry point when working from raw sample coordinates.
    SpatialKernel.from_matrix
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
    def from_coordinates(
        cls, coords: np.ndarray, method: str = "matern", **kwargs
    ) -> SpatialKernel:
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
        SpatialKernel
            Initialized kernel object.

        Raises
        ------
        ValueError
            If ``method`` is not one of :attr:`_available_kernels`.

        Examples
        --------
        >>> coords = np.random.randn(100, 2)
        >>> kernel = SpatialKernel.from_coordinates(coords, method='gaussian', bandwidth=1.0)
        """
        if method not in cls._available_kernels:
            raise ValueError(f"Unknown kernel method for coordinates: {method}.")

        return cls(coords, mode="coords", method=method, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray | sp.spmatrix,
        is_inverse: bool = False,
        method: str = "precomputed",
        **kwargs,
    ) -> SpatialKernel:
        """
        Build kernel from a precomputed kernel matrix or its inverse.

        Parameters
        ----------
        matrix : np.ndarray or scipy.sparse matrix
            Kernel matrix (N, N) or its inverse (precision matrix).
        is_inverse : bool, default False
            If True, matrix is treated as the inverse (precision) matrix K^-1.
        method : str, default 'precomputed'
            The logical kernel method (e.g., 'car' for precision matrices).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        SpatialKernel
            Initialized kernel object.

        Examples
        --------
        >>> K = np.array([[2, -1], [-1, 2]])  # kernel matrix
        >>> kernel = SpatialKernel.from_matrix(K, is_inverse=False)
        """
        mode = "precomputed_inverse" if is_inverse else "precomputed"
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
                self.is_implicit = True
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
                    self.is_implicit = True
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
            f"<SpatialKernel method={self.method} mode={self._mode} n={self.n} "
            f"implicit={self.is_implicit} data={data_desc} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        # Human-friendly multi-line summary
        lines = [
            "SpatialKernel",
            f"- Method: {self.method}",
            f"- Mode: {self._mode}",
            f"- Samples: {self.n}",
            f"- Implicit: {self.is_implicit} (threshold={self._implicit_threshold})",
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
