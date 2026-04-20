from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2, ncx2, norm
from tqdm import tqdm

from quadsv.kernels import Kernel

__all__ = ["liu_sf", "compute_null_params", "spatial_q_test", "spatial_r_test"]

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
        denom = s1 - np.sqrt(s12 - s2)
        if abs(denom) < _DELTA:
            # Catastrophic cancellation; fall back to kurtosis path
            delta_x = 0
            a = 1 / (np.sqrt(s2) + _DELTA)
            dof_x = 1 / (s2 + _DELTA)
        else:
            a = 1 / denom
            delta_x = s1 * a**3 - a**2
            dof_x = a**2 - 2 * delta_x
    else:
        delta_x = 0
        if kurtosis:
            a = 1 / (np.sqrt(s2) + _DELTA)
            dof_x = 1 / (s2 + _DELTA)
        else:
            a = 1 / (s1 + _DELTA)
            dof_x = 1 / (s12 + _DELTA)
    # Ensure valid chi-squared parameters
    dof_x = max(dof_x, _DELTA)
    delta_x = max(delta_x, 0.0)

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
    Caches the expensive computations (traces, eigenvalues) for reuse across
    both Q-tests and R-tests.

    Parameters
    ----------
    kernel : Kernel
        The spatial kernel object (MatrixKernel, FFTKernel, NUFFTKernel, or compatible).
    method : {'clt', 'welch', 'liu'}, default 'welch'
        Null approximation method for the **Q-test**. The R-test entry
        ``var_R = trace(K²)`` is always populated alongside, regardless of
        ``method`` — R-tests use a Normal approximation and only need this
        one moment.

        - 'clt': Central Limit Theorem (Z-score normal approximation)
        - 'welch': Welch-Satterthwaite moment matching (fast, uses traces)
        - 'liu': Liu eigenvalue-based approximation (accurate tail, slower)
    k_eigen : int, optional
        Number of top eigenvalues to compute if method='liu' and kernel is sparse.
        If None, computes all available eigenvalues.

    Returns
    -------
    dict[str, float or np.ndarray]
        Always populated (regardless of ``method``):

        - ``'method'`` : str — the Q-test approximation selected.
        - ``'var_R'`` : float — ``trace(K²)``, the null variance of ``R``
          (used by :func:`spatial_r_test`).

        Method-specific additions:

        - ``method='liu'`` (default for FFT / NUFFT kernels):

          * ``'eigenvalues'`` : ``np.ndarray`` of non-trivial kernel eigenvalues.

        - ``method='welch'`` (default for MatrixKernel Q-tests):

          * ``'mean_Q'`` : ``trace(K)``
          * ``'var_Q'`` : ``2 · trace(K²)``
          * ``'scale_g'`` : Welch scale parameter ``var_Q / (2 · mean_Q)``
          * ``'df_h'`` : Welch df ``2 · mean_Q² / var_Q``

        - ``method='clt'``: ``'mean_Q'``, ``'var_Q'`` only.

    Consumers (``spatial_q_test`` / ``spatial_r_test``) read only the keys
    their approximation needs; the dict is safe to reuse across calls.

    Raises
    ------
    AssertionError
        If method is not one of 'clt', 'welch', 'liu'.

    Examples
    --------
    >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian')
    >>> params = compute_null_params(kernel, method='welch')
    >>> Q, pval = spatial_q_test(data, kernel, null_params=params)
    >>> R, r_pval = spatial_r_test(x, y, kernel, null_params=params)
    """
    params = {"method": method}

    assert method in ["clt", "welch", "liu"], "Method must be 'clt', 'welch', or 'liu'."

    # `spatial_q_test` standardizes its input as Z = (X − X̄·𝟏) / σ, so the
    # realized quadratic form is Q = Zᵀ K Z = Xᵀ (H K H) X / σ² with
    # H = I − 𝟏𝟏ᵀ/n. Null moments are for HKH, NOT raw K. We obtain the
    # centered traces cheaply from two additional numbers:
    #   s1 = 𝟏ᵀ K 𝟏,   s2 = ‖K·𝟏‖² = 𝟏ᵀ K² 𝟏
    # via a single K·𝟏 application (see `Kernel._ones_stats`), giving
    #   trace(HKH)   = trace(K)  − s1/n
    #   trace((HKH)²) = trace(K²) − 2·s2/n + s1²/n²
    # The Q-test statistic is additionally a *ratio* of quadratic forms
    # (the denominator σ² is a random variable correlated with the
    # numerator), and its exact variance picks up a finite-n correction
    # derived from the Dirichlet(1/2, …, 1/2) distribution of Yᵢ²/ΣYⱼ²:
    #   Var[Q] = 2·[m · trace((HKH)²) − (trace(HKH))²] / (m+2)    m = n−1
    # The R-test is easier — independence of X, Y gives
    #   Var[R] = trace((HKH)²)   (exact, no finite-n correction).
    n = int(kernel.n)
    # ``kernel.trace()`` / ``kernel.square_trace()`` return the centered
    # moments ``trace(HKH)`` / ``trace((HKH)²)`` when ``kernel.centering``
    # is True (the default) — i.e. the moments of the operator actually
    # applied after z-scoring. The R-test variance is the squared trace
    # exactly (no finite-n correction, since X⊥Y).
    tr_HKH = float(kernel.trace())
    tr_HKH_sq = float(kernel.square_trace())
    params["var_R"] = tr_HKH_sq

    if method == "liu":
        # ``kernel.eigenvalues`` returns eigvals(HKH) when ``centering`` is
        # True (FFT / NUFFT zero out the DC mode; MatrixKernel falls back
        # to raw eigvals with a documented approximation).
        vals = kernel.eigenvalues(k=k_eigen)
        # Filter numerical noise.
        params["eigenvalues"] = vals[np.abs(vals) > 1e-9]
    else:
        # Q-test CLT / Welch moments — centered + finite-n corrected.
        m = max(n - 1, 1)
        mean_Q = tr_HKH
        var_Q = 2.0 * (m * tr_HKH_sq - tr_HKH**2) / (m + 2)
        # Numerical safety: variance must be non-negative.
        var_Q = max(var_Q, 0.0)
        params["mean_Q"] = float(mean_Q)
        params["var_Q"] = float(var_Q)

        if method == "welch":
            # Pre-calculate Welch-Satterthwaite parameters.
            if var_Q > 0 and mean_Q > 0:
                params["scale_g"] = var_Q / (2.0 * mean_Q)
                params["df_h"] = (2.0 * mean_Q**2) / var_Q
            else:
                params["scale_g"] = 1.0
                params["df_h"] = 1.0

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
        Input data array of shape (n,) for single feature or (n, M) for M features.
        Can be dense numpy array or sparse matrix (CSC/CSR format recommended).
        Should be standardized before calling unless is_standardized=True.
    kernel : Kernel
        Pre-constructed :class:`~quadsv.Kernel` (``MatrixKernel`` / ``FFTKernel`` /
        ``NUFFTKernel``) or a raw dense / sparse kernel matrix.
    null_params : dict, optional
        Pre-computed null distribution parameters from :func:`compute_null_params`.
        If None, computed on-the-fly using the ``'welch'`` method (only accurate when
        the kernel is positive semi-definite).
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

    The test statistic ``Q = xᵀ K x`` where ``K`` is the kernel matrix
    follows approximately a chi-squared mixture distribution:

    .. math::
       Q \\sim \\sum_{i=1}^{n} \\lambda_i \\chi^2_{1}

    where :math:`\\lambda_i` are the kernel eigenvalues.

    By default, the null is approximated with Welch-Satterthwaite moment matching.
    To use a different approximation, either pass
    ``null_params={'method': 'liu'}`` (this function will then call
    :func:`compute_null_params` internally with that method) or pass the fully
    pre-computed dict from ``compute_null_params(kernel, method='liu')``.

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian')
    >>> data = np.random.randn(100)
    >>> Q, pval = spatial_q_test(data, kernel)
    >>> # Sparse matrix example
    >>> from scipy.sparse import csr_matrix
    >>> sparse_data = csr_matrix(np.random.randn(100, 1000))
    >>> Q, pval = spatial_q_test(sparse_data, kernel, chunk_size=100, show_progress=True)
    """
    # Dispatch to FFT / NUFFT helpers when the kernel is one of those backends.
    # Liu's approximation (or normal for Moran) is hard-coded there — FFT/NUFFT
    # spectra are cheap, so there is no Welch option to select.
    # Lazy imports avoid a circular import with ``quadsv.fft`` / ``quadsv.nufft``.
    from quadsv.fft import FFTKernel, _q_test_fft
    from quadsv.nufft import NUFFTKernel, _q_test_nufft

    if isinstance(kernel, FFTKernel):
        return _q_test_fft(
            Xn,
            kernel,
            null_params=null_params,
            return_pval=return_pval,
            is_standardized=is_standardized,
        )
    if isinstance(kernel, NUFFTKernel):
        return _q_test_nufft(
            Xn,
            kernel,
            null_params=null_params,
            return_pval=return_pval,
            is_standardized=is_standardized,
        )

    # Matrix path (MatrixKernel or raw dense / sparse kernel matrix).
    is_sparse = sp.issparse(Xn)

    if is_sparse:
        n, M = Xn.shape if Xn.ndim == 2 else (Xn.shape[0], 1)
        if Xn.ndim == 1 or M == 1:
            Xn = Xn.reshape(-1, 1)
            M = 1
    else:
        Xn = np.asarray(Xn, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        n, M = Xn.shape

    if chunk_size == -1 or chunk_size >= M:
        chunk_size = M
    n_chunks = int(np.ceil(M / chunk_size))

    iterator = range(n_chunks)
    if show_progress and n_chunks > 1:
        iterator = tqdm(
            iterator,
            desc="Q-test chunks",
            total=n_chunks,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

    # Fast path: sparse Xn + unstandardized + kernel exposes xtKx_standardized.
    # Uses the (K·1, 1^T K 1) expansion so sparse Xn never needs densification.
    use_sparse_fastpath = is_sparse and not is_standardized and hasattr(kernel, "xtKx_standardized")

    Q_results = []
    for chunk_idx in iterator:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, M)
        Xn_chunk = Xn[:, start_idx:end_idx]

        if use_sparse_fastpath:
            # Compute means / stds from sparse directly (ddof=1 to match below).
            col_sum = np.asarray(Xn_chunk.sum(axis=0)).ravel()
            means = col_sum / n
            sq_sum = np.asarray(Xn_chunk.multiply(Xn_chunk).sum(axis=0)).ravel()
            var = (sq_sum - n * means**2) / max(n - 1, 1)
            var[var < 0] = 0.0
            stds = np.sqrt(var)
            Q_chunk = kernel.xtKx_standardized(Xn_chunk, means, stds)
        else:
            if is_standardized:
                z = Xn_chunk
            else:
                if is_sparse:
                    Xn_chunk = Xn_chunk.toarray()
                means = np.mean(Xn_chunk, axis=0)
                stds = np.std(Xn_chunk, axis=0, ddof=1)
                valid_mask = stds > 1e-12
                z = np.zeros_like(Xn_chunk)
                if np.any(valid_mask):
                    z[:, valid_mask] = (Xn_chunk[:, valid_mask] - means[valid_mask]) / stds[
                        valid_mask
                    ]

            if hasattr(kernel, "xtKx"):
                Q_chunk = kernel.xtKx(z)
            else:
                # Fallback for raw matrices.
                Kz = kernel.dot(z) if sp.issparse(kernel) else np.dot(kernel, z)
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
        if not hasattr(kernel, "square_trace"):
            raise ValueError("If params is None, kernel must be a Kernel object.")
        null_params = compute_null_params(kernel, method="welch")
        null_approx_method = "welch"
    else:
        null_approx_method = null_params.get("method", "welch")
        # Ensure params exist
        if len(null_params) == 1 and hasattr(kernel, "square_trace"):
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
    Xn: np.ndarray | sp.spmatrix,
    Yn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
    show_progress: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """
    Bivariate spatial R-test for correlation between two spatial variables.

    Computes the pairwise spatial statistic ``R = xᵀ K y``, testing for spatial
    association between two variables. Supports batch processing.

    Parameters
    ----------
    Xn : np.ndarray
        First input data vector or batch. Shape (n,) or (n, M).
    Yn : np.ndarray
        Second input data vector or batch. Shape (n,) or (n, M) matching Xn.
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

    The test statistic ``R = xᵀ K y`` is approximated as Normal under the null:

    .. math::
       R \\sim N(0, \\text{Trace}(K^2))

    P-value is computed as two-tailed: ``2 × Pr(|R| > |r_obs|)``.

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian')
    >>> x_data = np.random.randn(100)
    >>> y_data = np.random.randn(100)
    >>> R, pval = spatial_r_test(x_data, y_data, kernel)
    """
    # Dispatch to FFT / NUFFT helpers when the kernel is one of those backends.
    # Lazy imports avoid a circular import with ``quadsv.fft`` / ``quadsv.nufft``.
    from quadsv.fft import FFTKernel, _r_test_fft
    from quadsv.nufft import NUFFTKernel, _r_test_nufft

    if isinstance(kernel, FFTKernel):
        return _r_test_fft(
            Xn,
            Yn,
            kernel,
            null_params=null_params,
            return_pval=return_pval,
            is_standardized=is_standardized,
        )
    if isinstance(kernel, NUFFTKernel):
        return _r_test_nufft(
            Xn,
            Yn,
            kernel,
            null_params=null_params,
            return_pval=return_pval,
            is_standardized=is_standardized,
        )

    # Normalize shapes; preserve sparsity of inputs.
    def _prep(A):
        if sp.issparse(A):
            return A.reshape(-1, 1) if A.ndim == 1 else A
        arr = np.asarray(A, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    Xn, Yn = _prep(Xn), _prep(Yn)
    if Xn.shape != Yn.shape:
        raise ValueError(f"Xn and Yn shapes must match, got {Xn.shape} vs {Yn.shape}.")
    n, M = Xn.shape
    if n != kernel.n:
        raise ValueError(f"Kernel.n={kernel.n} does not match data rows {n}.")

    def _standardize(A):
        """Z-score A (sparse or dense) column-wise with ddof=1. Returns dense."""
        if sp.issparse(A):
            col_sum = np.asarray(A.sum(axis=0)).ravel()
            means = col_sum / n
            sq_sum = np.asarray(A.multiply(A).sum(axis=0)).ravel()
            var = (sq_sum - n * means**2) / max(n - 1, 1)
            var[var < 0] = 0.0
            stds = np.sqrt(var)
            Z = A.toarray() - means
        else:
            means = np.mean(A, axis=0)
            stds = np.std(A, axis=0, ddof=1)
            Z = A - means
        valid = stds > 1e-12
        Z[:, ~valid] = 0.0
        if np.any(valid):
            Z[:, valid] /= stds[valid]
        return Z

    if is_standardized:
        # ``is_standardized=True`` implies dense already (sparse can't be pre-z-scored).
        Zx = np.asarray(Xn.toarray() if sp.issparse(Xn) else Xn, dtype=float)
        Zy = np.asarray(Yn.toarray() if sp.issparse(Yn) else Yn, dtype=float)
    else:
        if show_progress:
            with tqdm(total=2, desc="Standardizing", leave=False) as pbar:
                Zx = _standardize(Xn)
                pbar.update(1)
                Zy = _standardize(Yn)
                pbar.update(1)
        else:
            Zx = _standardize(Xn)
            Zy = _standardize(Yn)

    # R = diag(Zx^T K Zy) via the kernel's public bilinear primitive.
    R = np.atleast_1d(np.asarray(kernel.xtKy(Zx, Zy)))

    # Unwrap if M=1
    if M == 1 and R.size == 1:
        R = R.item()

    if not return_pval:
        return R

    # 3. P-value (Normal Approximation).
    # Both X, Y are z-scored before R = Zₓᵀ K Zᵧ, so R ~ N(0, trace((HKH)²))
    # — NOT trace(K²). kernel.square_trace() returns the centered trace by
    # default (centering=True).
    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        var_R = float(kernel.square_trace())

    sigma = np.sqrt(var_R)

    # Two-sided p-value for Normal distribution
    if sigma > 0:
        z_score = R / sigma
        pval = 2 * norm.sf(np.abs(z_score))
    else:
        pval = np.ones_like(R) if isinstance(R, np.ndarray) else 1.0

    return R, pval
