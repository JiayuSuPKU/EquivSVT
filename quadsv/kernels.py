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


class Kernel(ABC):
    """
    Abstract base class for spatial kernels.

    Handles dense, sparse, and implicit (operator-based) kernels.
    Implements a dual-mode system that switches between explicit (dense) and
    implicit (sparse) computation based on problem size.

    Attributes
    ----------
    n : int
        Number of observations (samples).
    method : str
        The kernel method ('gaussian', 'matern', 'moran', 'graph_laplacian', 'car').
    is_implicit : bool
        If True, uses sparse solver for implicit kernels (N > 5000).
        If False, uses dense matrix operations.
    params : dict
        Additional kernel parameters (bandwidth, nu, rho, etc.).
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
            Additional kernel-specific parameters.
        """
        self.n = n
        self.method = method
        self.params = kwargs

        # Threshold for switching between dense realization and implicit solving
        self.implicit_threshold = 5000
        self.is_implicit = False
        self._lu = None  # Cache for sparse LU factorization if needed

        # _K can be the kernel matrix OR the inverse/precision matrix depending on is_implicit
        self._K = self._build_kernel()
        self.spectrum = None  # Lazy-evaluated eigenvalues cache

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
            f"threshold={self.implicit_threshold} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        return (
            "Kernel\n"
            f"- Method: {self.method}\n"
            f"- Samples: {self.n}\n"
            f"- Implicit: {self.is_implicit} (threshold={self.implicit_threshold})\n"
            f"- Params: {self._format_params()}"
        )

    def realization(self) -> np.ndarray:
        """
        Return the realized (N, N) kernel matrix.

        Returns
        -------
        np.ndarray
            Dense (N, N) kernel matrix.

        Warnings
        --------
        If is_implicit is True, this forces expensive dense inversion of the
        precision matrix. Use xtKx() and trace() for implicit kernels instead.
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

        Parameters
        ----------
        k : int, optional
            Number of largest eigenvalues to return. If None, returns all.

        Returns
        -------
        np.ndarray
            Eigenvalues sorted in descending order, shape (k,) or (n,).
        """
        if self.spectrum is not None:
            # check if we have enough cached
            if k is None and len(self.spectrum) == self.n:
                return self.spectrum
            elif k is not None and len(self.spectrum) >= k:
                return self.spectrum[-k:]

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
                if k is not None:
                    vals = np.sort(vals)[-k:]

        self.spectrum = vals
        return vals

    def xtKx(self, x: np.ndarray | sp.spmatrix) -> float | np.ndarray:
        """
        Efficiently compute the quadratic form x^T K x.

        Handles both single vectors and batches. Uses implicit solvers for
        large N (implicit_threshold) to avoid dense matrix operations.
        Supports sparse input matrices without densification.

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
                # Cache LU factorization for efficiency
                if self._lu is None:
                    self._lu = splu(self._K.tocsc())
                if is_sparse_input:
                    # Convert to dense for solver (most efficient for sparse solvers)
                    y = self._lu.solve(x_in.toarray())
                else:
                    y = self._lu.solve(x_in)
            else:
                # Cache LU factorization for efficiency
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

        For implicit kernels, uses Hutchinson's trick with random vectors
        for efficient O(N) estimation instead of O(N³) eigendecomposition.

        Returns
        -------
        float
            Trace of the kernel matrix.
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
        Custom pickling behavior: exclude unpicklable SuperLU objects.
        """
        state = self.__dict__.copy()
        # Remove the cached LU factorization because SuperLU objects cannot be pickled.
        # Workers will re-compute this locally.
        state["_lu"] = None
        return state

    def __setstate__(self, state):
        """
        Restore state and ensure _lu is reset to None.
        """
        self.__dict__.update(state)
        # Ensure _lu is explicitly None upon restoration
        self._lu = None


class SpatialKernel(Kernel):
    """
    Concrete implementation of Spatial Kernel.

    Can be built from raw coordinates OR from precomputed matrices.

    Attributes
    ----------
    data : np.ndarray
        Raw data (coordinates or matrix) used to construct the kernel.
    mode : {'coords', 'precomputed', 'precomputed_inverse'}
        How the kernel was initialized.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self, data: np.ndarray, mode: str = "coords", method: str = "matern", **kwargs
    ) -> None:
        """
        Internal constructor. Use .from_coordinates() or .from_matrix() instead.

        Parameters
        ----------
        data : np.ndarray
            Input data (coordinates or kernel matrix).
        mode : {'coords', 'precomputed', 'precomputed_inverse'}, default 'coords'
            Interpretation of the data.
        method : str, default 'matern'
            Kernel method to use.
        **kwargs : dict
            Additional kernel parameters.
        """
        self.data = data
        self.mode = mode
        assert mode in [
            "coords",
            "precomputed",
            "precomputed_inverse",
        ], "Invalid mode. Must be 'coords', 'precomputed', or 'precomputed_inverse'."

        assert method in self._available_kernels + [
            "precomputed"
        ], f"Unknown kernel method: {method}."

        # Update kernel parameters from defaults
        defaults = self._get_default_params(method).copy()
        if kwargs:
            for key, value in kwargs.items():
                if key in defaults:
                    defaults[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        n = data.shape[0]
        super().__init__(n, method=method, **defaults)

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
            Method defaults: bandwidth (gaussian/matern), nu (matern), neighbor_degree (moran/graph_laplacian/car), rho (car).
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
    ) -> "SpatialKernel":
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

        Examples
        --------
        >>> coords = np.random.randn(100, 2)
        >>> kernel = SpatialKernel.from_coordinates(coords, method='gaussian', bandwidth=1.0)
        """
        assert method in cls._available_kernels, f"Unknown kernel method for coordinates: {method}."

        return cls(coords, mode="coords", method=method, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray | sp.spmatrix,
        is_inverse: bool = False,
        method: str = "precomputed",
        **kwargs,
    ) -> "SpatialKernel":
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

    def _build_kernel(self):
        method = self.method

        # ==========================================
        # 1. PREPARE RAW INPUTS (Dists or Weights)
        # ==========================================

        # Case A: Coordinates provided -> Compute Dists or W from scratch
        if self.mode == "coords":
            coords = self.data
            if method in ["gaussian", "matern"]:
                # Compute dense distance matrix
                dists = squareform(pdist(coords, metric="euclidean"))
                W = None
            elif method in ["moran", "graph_laplacian", "car"]:
                # Compute sparse adjacency graph
                k = self.params["k_neighbors"]
                nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
                W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)
                dists = None
            else:
                raise ValueError(f"Unknown method for coordinates: {method}")

        # Case B: Precomputed Kernel provided
        elif self.mode == "precomputed":
            return self.data

        # Case C: Precomputed Inverse Kernel provided
        elif self.mode == "precomputed_inverse":
            M = self.data
            standardize = self.params.get("standardize", False)

            # If small, realize dense K; else keep implicit precision
            if self.n <= self.implicit_threshold:
                try:
                    M_dense = M.toarray() if sp.issparse(M) else M
                    K_dense = inv(M_dense)
                except np.linalg.LinAlgError:
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
                # Avoid div by zero
                dists_safe = dists.copy()
                dists_safe[dists_safe == 0] = 1e-15
                factor = (np.sqrt(2 * nu) * dists_safe) / length_scale
                K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
                np.fill_diagonal(K, 1.0)
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
                standardize = self.params["standardize"]
                I = sp.eye(self.n, format="csc")
                # M = (I - rho * W_norm) is the inverse of the CAR kernel
                M = I - rho * W_norm

                if self.n > self.implicit_threshold:
                    self.is_implicit = True
                    if standardize:
                        M = self._standardize_precision(M)
                    return M
                else:
                    try:
                        K_dense = inv(M.toarray())
                    except np.linalg.LinAlgError:
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
        if self.mode == "coords":
            coords = self.data
            data_desc = f"coords shape={getattr(coords, 'shape', '?')}"
        elif self.mode == "precomputed":
            M = self.data
            if sp.issparse(M):
                data_desc = f"matrix shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"matrix shape={getattr(M, 'shape', '?')} dense"
        elif self.mode == "precomputed_inverse":
            M = self.data
            if sp.issparse(M):
                data_desc = f"precision shape={M.shape} sparse nnz={M.nnz}"
            else:
                data_desc = f"precision shape={getattr(M, 'shape', '?')} dense"
        else:
            data_desc = "data=?"

        return (
            f"<SpatialKernel method={self.method} mode={self.mode} n={self.n} "
            f"implicit={self.is_implicit} data={data_desc} params={{ {self._format_params()} }}>"
        )

    def __str__(self):
        # Human-friendly multi-line summary
        lines = [
            "SpatialKernel",
            f"- Method: {self.method}",
            f"- Mode: {self.mode}",
            f"- Samples: {self.n}",
            f"- Implicit: {self.is_implicit} (threshold={self.implicit_threshold})",
        ]

        # Add a brief data description
        try:
            if self.mode == "coords":
                coords = self.data
                lines.append(f"- Data: coords shape={getattr(coords, 'shape', '?')}")
            else:
                M = self.data
                if sp.issparse(M):
                    kind = "precision" if self.mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={M.shape} sparse nnz={M.nnz}")
                else:
                    kind = "precision" if self.mode == "precomputed_inverse" else "matrix"
                    lines.append(f"- Data: {kind} shape={getattr(M, 'shape', '?')} dense")
        except Exception:
            lines.append("- Data: ?")

        lines.append(f"- Params: {self._format_params()}")
        return "\n".join(lines)
