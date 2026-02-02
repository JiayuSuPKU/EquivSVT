import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2, ncx2, norm
from tqdm import tqdm

from quadsv.kernels import Kernel

_DELTA = 1e-10


def liu_sf(
    t: float | np.ndarray,
    lambs: np.ndarray,
    dofs: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    kurtosis: bool = False,
) -> float | np.ndarray:
    """
    Liu approximation to linear combination of noncentral chi-squared variables.

    Approximates the tail probability Pr(Q > t) for a weighted sum of
    noncentral chi-squared random variables. This is the default p-value
    computation method when exact kernel eigenvalues are known.

    Parameters
    ----------
    t : float or np.ndarray
        Test statistic value(s). Can be scalar or array.
    lambs : np.ndarray
        Eigenvalues of the kernel matrix, shape (n_evals,).
    dofs : np.ndarray, optional
        Degrees of freedom for each eigenvalue. Default: ones (chi-squared).
    deltas : np.ndarray, optional
        Non-centrality parameters. Default: zeros (central chi-squared).
    kurtosis : bool, default False
        If True, uses kurtosis-based approximation for edge case.

    Returns
    -------
    float or np.ndarray
        Tail probability Pr(Q > t). Same shape as input `t`.

    Notes
    -----
    Uses moment-based approximation with chi-squared mixture distribution.
    Numerically stable for a wide range of eigenvalue spectra.
    """
    if dofs is None:
        dofs = np.ones_like(lambs)
    if deltas is None:
        deltas = np.zeros_like(lambs)

    t = np.asarray(t, float)
    lambs = np.asarray(lambs, float)
    dofs = np.asarray(dofs, float)
    deltas = np.asarray(deltas, float)

    # Calculate moments of weights
    lambs_pow = {i: lambs**i for i in range(1, 5)}

    c = {i: np.sum(lambs_pow[i] * dofs) + i * np.sum(lambs_pow[i] * deltas) for i in range(1, 5)}

    s1 = c[3] / (np.sqrt(c[2]) ** 3 + _DELTA)
    s2 = c[4] / (c[2] ** 2 + _DELTA)

    s12 = s1**2
    if s12 > s2:
        a = 1 / (s1 - np.sqrt(s12 - s2))
        delta_x = s1 * a**3 - a**2
        dof_x = a**2 - 2 * delta_x
    else:
        delta_x = 0
        if kurtosis:
            a = 1 / np.sqrt(s2)
            dof_x = 1 / s2
        else:
            a = 1 / (s1 + _DELTA)
            dof_x = 1 / (s12 + _DELTA)

    mu_q = c[1]
    sigma_q = np.sqrt(2 * c[2])

    mu_x = dof_x + delta_x
    sigma_x = np.sqrt(2 * (dof_x + 2 * delta_x))

    t_star = (t - mu_q) / (sigma_q + _DELTA)
    tfinal = t_star * sigma_x + mu_x

    q = ncx2.sf(tfinal, dof_x, np.maximum(delta_x, 1e-9))

    return q


def compute_null_params(
    kernel: Kernel, method: str = "welch", k_eigen: int | None = None
) -> dict[str, float | np.ndarray]:
    """
    Pre-compute null distribution parameters for spatial tests.

    Call this ONCE before running parallel tests on thousands of features.
    Caches the expensive computations (traces, eigenvalues) for reuse.

    Parameters
    ----------
    kernel : Kernel
        The spatial kernel object (SpatialKernel, FFTKernel, or compatible).
    method : {'clt', 'welch', 'liu'}, default 'welch'
        Null approximation method:
        - 'clt': Central Limit Theorem (Z-score normal approximation)
        - 'welch': Welch-Satterthwaite moment matching (fast, uses traces)
        - 'liu': Liu eigenvalue-based approximation (accurate tail, slower)
    k_eigen : int, optional
        Number of top eigenvalues to compute if method='liu' and kernel is sparse.
        If None, computes all available eigenvalues.

    Returns
    -------
    dict[str, Union[float, np.ndarray]]
        Parameters keyed by null_approx method:
        - 'method': The method used
        - For 'liu': 'eigenvalues' (np.ndarray of kernel eigenvalues)
        - For 'welch'/'clt': 'mean_Q', 'var_Q', and for 'welch' also 'scale_g', 'df_h'

    Raises
    ------
    AssertionError
        If method is not one of 'clt', 'welch', 'liu'.

    Examples
    --------
    >>> kernel = SpatialKernel.from_coordinates(coords, method='gaussian')
    >>> params = compute_null_params(kernel, method='welch')
    >>> Q, pval = spatial_q_test(data, kernel, null_params=params)
    """
    params = {"method": method}

    assert method in ["clt", "welch", "liu"], "Method must be 'clt', 'welch', or 'liu'."

    if method == "liu":
        # Requires eigenvalues
        # If kernel is implicit/sparse, we might only get top k
        vals = kernel.eigenvalues(k=k_eigen)
        # Filter numerical noise
        params["eigenvalues"] = vals[np.abs(vals) > 1e-9]
    else:
        # Compute trace and trace squared
        tr_K = kernel.trace()
        tr_K2 = kernel.square_trace()

        params["mean_Q"] = tr_K
        params["var_Q"] = 2 * tr_K2

        if method == "welch":
            # Pre-calculate Welch-Satterthwaite parameters
            if params["var_Q"] > 0:
                params["scale_g"] = params["var_Q"] / (2 * params["mean_Q"])
                params["df_h"] = (2 * params["mean_Q"] ** 2) / params["var_Q"]
            else:
                params["scale_g"] = 1.0
                params["df_h"] = 1.0

    return params

    return params


def spatial_q_test(  # noqa: C901
    Xn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
    chunk_size: int = -1,
    show_progress: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """
    Univariate spatial Q-test for detecting spatial variability.

    Tests whether a spatial variable exhibits significant clustering or dispersion
    using the specified kernel weighting scheme. Supports both single features and
    batch processing with sparse matrices.

    Parameters
    ----------
    Xn : np.ndarray or scipy.sparse matrix
        Input data array of shape (N,) for single feature or (N, M) for M features.
        Can be dense numpy array or sparse matrix (CSC/CSR format recommended).
        Should be standardized before calling unless is_standardized=True.
    kernel : Kernel
        Pre-constructed kernel object (Kernel, SpatialKernel, FFTKernel, or scipy.sparse matrix).
    null_params : dict, optional
        Pre-computed null distribution parameters from compute_null_params().
        If None, computed on-the-fly using 'welch' method (only accurate when kernel is positive semi-definite).
    return_pval : bool, default True
        If True, returns (Q, pval) tuple; if False, returns Q only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally (assumes input is N(0,1)).
    chunk_size : int, default -1
        Number of features to process in each chunk. If -1, processes all features at once.
        Useful for large feature sets to reduce memory usage. Must be <= M.
    show_progress : bool, default False
        If True, displays a progress bar during chunk processing.

    Returns
    -------
    Q : float or np.ndarray
        Test statistic value(s). Shape (M,) if input was 2D, scalar if input was 1D.
    pval : float or np.ndarray, optional
        Tail probability under null hypothesis. Only returned if return_pval=True.
        Same shape as Q.

    Raises
    ------
    ValueError
        If kernel dimensions don't match data size or if params is None and kernel is not a Kernel object.

    Notes
    -----
    Under H₀: data is spatially independent.
    Under H₁: mean-shift present.

    The test statistic Q = x^T K x where K is the kernel matrix,
    follows approximately a chi-squared mixture distribution:

    .. math::
       Q \\sim \\sum_{i=1}^{n} \\lambda_i \\chi^2_{1}

    where :math:`\\lambda_i` are the kernel eigenvalues.

    By default, we approximate the null using Welch-Satterthwaite moment matching.
    For more accurate tail probabilities, set null_params = {'method': 'liu'} or using
    null_params = compute_null_params(method = 'liu').

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = SpatialKernel.from_coordinates(coords, method='gaussian')
    >>> data = np.random.randn(100)
    >>> Q, pval = spatial_q_test(data, kernel)
    >>> # Sparse matrix example
    >>> from scipy.sparse import csr_matrix
    >>> sparse_data = csr_matrix(np.random.randn(100, 1000))
    >>> Q, pval = spatial_q_test(sparse_data, kernel, chunk_size=100, show_progress=True)
    """
    # Handle sparse matrices
    is_sparse = sp.issparse(Xn)

    if is_sparse:
        # Get shape from sparse matrix
        n, M = Xn.shape if Xn.ndim == 2 else (Xn.shape[0], 1)
        if Xn.ndim == 1 or M == 1:
            Xn = Xn.reshape(-1, 1) if hasattr(Xn, "reshape") else Xn
            M = 1
    else:
        Xn = np.asarray(Xn).astype(float)
        # Detect batch mode
        # If ndim=1 -> (N, 1). If ndim=2 -> (N, M)
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        n, M = Xn.shape

    # Determine chunking
    if chunk_size == -1 or chunk_size >= M:
        chunk_size = M

    # Process in chunks
    n_chunks = int(np.ceil(M / chunk_size))
    Q_results = []

    iterator = range(n_chunks)
    if show_progress and n_chunks > 1:
        iterator = tqdm(
            iterator,
            desc="Processing features",
            total=n_chunks,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

    for chunk_idx in iterator:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, M)

        # Extract chunk
        Xn_chunk = Xn[:, start_idx:end_idx]

        # 1. Preprocessing (Standardization)
        if is_standardized:
            z = Xn_chunk
        else:
            # Convert sparse to dense for standardization (centering destroys sparsity anyway)
            if is_sparse:
                Xn_chunk = Xn_chunk.toarray()

            # Compute stats along axis 0 (samples)
            means = np.mean(Xn_chunk, axis=0)
            stds = np.std(Xn_chunk, axis=0, ddof=1)

            # Handle constant features (std=0) to avoid NaNs
            valid_mask = stds > 1e-12
            z = np.zeros_like(Xn_chunk)

            if np.any(valid_mask):
                z[:, valid_mask] = (Xn_chunk[:, valid_mask] - means[valid_mask]) / stds[valid_mask]

        # 2. Compute Statistic Q = z^T K z
        # Use optimized class method if available
        if hasattr(kernel, "xtKx"):
            # This now supports (N, M) inputs and returns (M,)
            Q_chunk = kernel.xtKx(z)
        else:
            # Fallback for raw matrices
            if sp.issparse(kernel):
                Kz = kernel.dot(z)
            else:
                Kz = np.dot(kernel, z)

            # Efficient diagonal of z.T @ Kz without full matrix mult
            # sum(z_ij * (Kz)_ij) along axis 0
            Q_chunk = np.sum(z * Kz, axis=0)

        Q_results.append(Q_chunk)

    # Concatenate results
    Q = np.concatenate([np.atleast_1d(q) for q in Q_results])

    # Unwrap if M=1
    if M == 1 and Q.size == 1:
        Q = Q.item()

    if not return_pval:
        return Q

    # 3. Compute P-value (Vectorized)
    if null_params is None:
        if not hasattr(kernel, "trace"):
            raise ValueError("If params is None, kernel must be a Kernel object.")
        null_params = compute_null_params(kernel, method="welch")
        null_approx_method = "welch"
    else:
        null_approx_method = null_params.get("method", "welch")
        # Ensure params exist
        if len(null_params) == 1 and hasattr(kernel, "trace"):
            null_params.update(compute_null_params(kernel, method=null_approx_method))

    # P-value logic
    if null_approx_method == "clt":
        mu_Q = null_params["mean_Q"]
        var_Q = null_params["var_Q"]
        if var_Q > 0:
            z_score = (Q - mu_Q) / np.sqrt(var_Q)
            pval = chi2.sf(z_score**2, df=1)
        else:
            pval = np.ones_like(Q, dtype=float)

    elif null_approx_method == "welch":
        g = null_params["scale_g"]
        d = null_params["df_h"]
        pval = chi2.sf(Q / g, df=d)

    elif null_approx_method == "liu":
        lambs = null_params["eigenvalues"]
        # filter numerical noise
        lambs = lambs[np.abs(lambs) > _DELTA]

        # liu_sf likely needs a loop if not vectorized, or use np.vectorize
        # Assuming liu_sf handles array inputs or we map it:
        if M > 1:  # batched inputs
            pval = np.array([liu_sf(q_val, lambs) for q_val in Q])
        else:
            pval = liu_sf(Q, lambs)
    else:
        pval = np.ones_like(Q, dtype=float)

    # Unwrap if M=1
    if M == 1 and pval.size == 1:
        pval = pval.item()

    return Q, pval


def spatial_r_test(  # noqa: C901
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """
    Bivariate spatial R-test for correlation between two spatial variables.

    Computes the pairwise spatial statistic R = x^T K y, testing for spatial
    association between two variables. Supports batch processing.

    Parameters
    ----------
    Xn : np.ndarray
        First input data vector or batch. Shape (N,) or (N, M).
    Yn : np.ndarray
        Second input data vector or batch. Shape (N,) or (N, M) matching Xn.
    kernel : Kernel
        Pre-constructed kernel object compatible with xtKy() method.
    null_params : dict, optional
        Pre-computed null distribution parameters. Should include 'var_R'.
        If None, computed on-the-fly from kernel traces.
    return_pval : bool, default True
        If True, returns (R, pval) tuple; if False, returns R only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally.

    Returns
    -------
    R : float or np.ndarray
        Test statistic value(s). Shape (M,) if input was 2D, scalar if input was 1D.
    pval : float or np.ndarray, optional
        Tail probability under null hypothesis (two-tailed test). Only returned if return_pval=True.
        Based on Normal approximation.

    Raises
    ------
    ValueError
        If Xn and Yn shapes don't match or kernel dimensions are incompatible.

    Notes
    -----
    Under H₀: the two variables are spatially uncorrelated.

    The test statistic R = x^T K y is approximated as Normal under the null:

    .. math::
       R \\sim N(0, \\text{Trace}(K^2))

    P-value is computed as two-tailed: ``2 × Pr(|R| > |r_obs|)``.

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = SpatialKernel.from_coordinates(coords, method='gaussian')
    >>> x_data = np.random.randn(100)
    >>> y_data = np.random.randn(100)
    >>> R, pval = spatial_r_test(x_data, y_data, kernel)
    """
    Xn = np.asarray(Xn)
    Yn = np.asarray(Yn)
    if Xn.ndim == 1:
        Xn = Xn.reshape(-1, 1)
    if Yn.ndim == 1:
        Yn = Yn.reshape(-1, 1)

    n, M = Xn.shape
    assert Xn.shape == Yn.shape, "Xn and Yn shapes must match"
    assert n == kernel.n, "Kernel dimensions must match data size"

    # 1. Preprocessing (Standardization)
    if is_standardized:
        Zx, Zy = Xn, Yn
    else:
        # Standardize Xn
        means = np.mean(Xn, axis=0)
        stds = np.std(Xn, axis=0, ddof=1)
        valid_mask = stds > 1e-12
        Zx = np.zeros_like(Xn)

        if np.any(valid_mask):
            Zx[:, valid_mask] = (Xn[:, valid_mask] - means[valid_mask]) / stds[valid_mask]

        # Standardize Yn
        means = np.mean(Yn, axis=0)
        stds = np.std(Yn, axis=0, ddof=1)
        valid_mask = stds > 1e-12
        Zy = np.zeros_like(Yn)
        if np.any(valid_mask):
            Zy[:, valid_mask] = (Yn[:, valid_mask] - means[valid_mask]) / stds[valid_mask]

    # 2. Compute R = Zx.T @ K @ Zy
    # We use the kernel's optimized multiplication
    # K @ Zy
    if hasattr(kernel, "is_implicit") and kernel.is_implicit:
        # Implicit solve: M^-1 * Zy
        if sp.issparse(kernel._K):
            # Check for cached LU factorization
            if kernel._lu is None:
                from scipy.sparse.linalg import splu

                kernel._lu = splu(kernel._K.tocsc())
            K_Zy = kernel._lu.solve(Zy)
        else:
            # Check for cached LU factorization
            if kernel._lu is None:
                from scipy.linalg import lu_factor, lu_solve

                kernel._lu = lu_factor(kernel._K)
            K_Zy = lu_solve(kernel._lu, Zy)
    else:
        # Explicit dot
        K_Zy = kernel._K.dot(Zy)

    # Dot product: Zx.T @ (K @ Zy)
    # We only want the paired elements (diagonal of the result matrix)
    # sum(Zx_ij * (K_Zy)_ij) along axis 0
    R = np.sum(Zx * K_Zy, axis=0)

    # Unwrap if M=1
    if M == 1 and R.size == 1:
        R = R.item()

    if not return_pval:
        return R

    # 3. P-value (Normal Approximation)
    # R ~ N(0, Tr(K K^T))
    if null_params is None:
        # Compute on fly (expensive)
        tr_K2 = kernel.square_trace()
        var_R = tr_K2
    else:
        var_R = null_params.get("var_R", 1.0)

    sigma = np.sqrt(var_R)

    # Two-sided p-value for Normal distribution
    if sigma > 0:
        z_score = R / sigma
        pval = 2 * norm.sf(np.abs(z_score))
    else:
        pval = np.ones_like(R) if isinstance(R, np.ndarray) else 1.0

    return R, pval
