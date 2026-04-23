from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2, ncx2, norm
from tqdm import tqdm

from quadsv.kernels import Kernel

__all__ = ["liu_sf", "compute_null_params", "spatial_q_test", "spatial_r_test"]

_DELTA = 1e-10


def _liu_prepare_from_cumulants(
    c: dict[int, float],
    kurtosis: bool = False,
    n: int | None = None,
) -> dict[str, float]:
    r"""Pure-math core: Liu shifted-chi² fit from cumulants ``c_1..c_4``.

    Called by both :func:`_liu_prepare` (from an explicit eigenvalue
    spectrum) and :func:`_hutchinson_cumulants` (from probe estimates),
    so the shifted-chi² algebra lives in one place. The input ``c`` is a
    mapping ``{1: c_1, 2: c_2, 3: c_3, 4: c_4}`` with
    ``c_p = trace(K^p)`` (any contributions from non-unit ``dofs`` /
    non-zero ``deltas`` must already be folded into these sums).

    Parameters
    ----------
    c : dict[int, float]
        Spectral cumulants ``{1: c_1, 2: c_2, 3: c_3, 4: c_4}``.
    kurtosis : bool, default False
        Use the kurtosis-based edge-case approximation when Liu's
        discriminant ``s_1² − s_2`` is non-positive.
    n : int, optional
        Sample size. When provided, ``sigma_Q`` is set from the
        finite-:math:`n` Dirichlet(1/2) variance
        ``Var[Q] = 2·(m·c_2 − c_1²)/(m+2)`` with ``m = n-1`` — matching
        the ``dirichlet_correction=True`` branch of
        :func:`compute_null_params`. Without ``n`` (default) the
        large-:math:`n` limit ``sigma_Q = sqrt(2·c_2)`` is used, which
        overestimates ``Var[Q]`` when the spectrum is broad
        (:math:`c_1^2 \approx m \cdot c_2`) — e.g. CAR on a dense grid —
        and collapses Liu's tail probability to zero. Passing
        ``n = kernel.n`` recovers the Welch variance.

    Returns
    --------
    dict[str, float]
        Liu coefficients for the shifted-chi² approximation, with keys:

        - ``'mu_Q'`` : float — the mean of the original Q statistic.
        - ``'sigma_Q'`` : float — the standard deviation of the original Q
          statistic (with optional finite-``n`` Dirichlet(1/2) correction).
        - ``'mu_x'`` : float — the mean of the fitted shifted-χ² variable X.
        - ``'sigma_x'`` : float — the standard deviation of X.
        - ``'dof_x'`` : float — the degrees of freedom of X.
        - ``'delta_x'`` : float — the non-centrality parameter of X.

    Consumers (e.g. :func:`spatial_q_test`) read only these coefficients
    for the final p-value calculation; the input cumulants are not
    needed beyond this point.
    """
    s1 = c[3] / (np.sqrt(c[2]) ** 3 + _DELTA)
    s2 = c[4] / (c[2] ** 2 + _DELTA)

    s12 = s1**2
    if s12 > s2:
        denom = s1 - np.sqrt(s12 - s2)
        if abs(denom) < _DELTA:
            # Catastrophic cancellation — fall back to the kurtosis path.
            delta_x = 0.0
            dof_x = 1.0 / (s2 + _DELTA)
        else:
            a = 1.0 / denom
            delta_x = s1 * a**3 - a**2
            dof_x = a**2 - 2.0 * delta_x
    else:
        delta_x = 0.0
        if kurtosis:
            dof_x = 1.0 / (s2 + _DELTA)
        else:
            dof_x = 1.0 / (s12 + _DELTA)
    dof_x = max(dof_x, _DELTA)
    delta_x = max(delta_x, 0.0)

    # sigma_Q: Dirichlet(1/2)-corrected for the z-scored ratio Q when n
    # is provided. See :func:`compute_null_params` Notes for the full
    # derivation of ``Var[Q] = 2·(m·c_2 − c_1²)/(m+2)``.
    if n is not None and n >= 2:
        m = n - 1
        var_Q = 2.0 * max(m * c[2] - c[1] ** 2, 0.0) / (m + 2)
    else:
        var_Q = 2.0 * c[2]

    return {
        "mu_Q": float(c[1]),
        "sigma_Q": float(np.sqrt(max(var_Q, 0.0))),
        "mu_x": float(dof_x + delta_x),
        "sigma_x": float(np.sqrt(2 * (dof_x + 2 * delta_x))),
        "dof_x": float(dof_x),
        "delta_x": float(delta_x),
    }


def _liu_prepare(
    lambs: np.ndarray,
    dofs: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    kurtosis: bool = False,
    n: int | None = None,
) -> dict[str, float]:
    """Precompute Liu coefficients from the kernel eigenvalue spectrum.

    Thin wrapper — builds ``c_1..c_4`` from the weighted eigenvalues and
    calls :func:`_liu_prepare_from_cumulants`.

    Parameters
    ----------
    lambs : np.ndarray
        Eigenvalues of ``K``, shape ``(n_evals,)``.
    dofs, deltas : np.ndarray, optional
        Per-eigenvalue degrees of freedom and non-centrality parameters.
        Default to central chi-squared (ones, zeros).
    kurtosis : bool, default False
        Use the kurtosis-based edge-case approximation.
    n : int, optional
        Sample size for the Dirichlet(1/2) variance correction. See
        :func:`_liu_prepare_from_cumulants` for details.
    """
    lambs = np.asarray(lambs, dtype=float)
    if dofs is None:
        dofs = np.ones_like(lambs)
    else:
        dofs = np.asarray(dofs, dtype=float)
    if deltas is None:
        deltas = np.zeros_like(lambs)
    else:
        deltas = np.asarray(deltas, dtype=float)
    lambs_pow = {i: lambs**i for i in range(1, 5)}
    c = {
        i: float(np.sum(lambs_pow[i] * dofs) + i * np.sum(lambs_pow[i] * deltas))
        for i in range(1, 5)
    }
    return _liu_prepare_from_cumulants(c, kurtosis=kurtosis, n=n)


def _hutchinson_cumulants(
    kernel: Kernel,
    n_probes: int = 60,
    rng_seed: int = 0,
    use_analytic_c12: bool = True,
) -> dict[int, float]:
    r"""Estimate ``c_p = tr(K^p)``, ``p = 1..4`` using random probes.

    General probe form. For iid ``v`` with ``E[vvᵀ] = I``:
    :math:`c_p = \mathbb{E}[\mathbf{v}^\top \mathbf{K}^p \mathbf{v}]`.
    Two matvecs per probe yield
    :math:`\mathbf{u}_s = \mathbf{K}\mathbf{v}_s` and
    :math:`\mathbf{w}_s = \mathbf{K}^2 \mathbf{v}_s`, from which
    all four cumulants fall out as inner products:

    .. math::

        \hat c_1 &= \tfrac{1}{m}\textstyle\sum_s \mathbf{v}_s^\top \mathbf{u}_s,
        &\hat c_2 &= \tfrac{1}{m}\textstyle\sum_s \|\mathbf{u}_s\|^2, \\
        \hat c_3 &= \tfrac{1}{m}\textstyle\sum_s \mathbf{u}_s^\top \mathbf{w}_s,
        &\hat c_4 &= \tfrac{1}{m}\textstyle\sum_s \|\mathbf{w}_s\|^2.

    Here we use **iid Rademacher** probes (``±1`` with equal probability),
    which has strictly smaller variance on :math:`v^\top A v` than
    :math:`\mathcal{N}(0, I)` probes at fixed probe count:

    .. math::

        \mathrm{Var}_{\text{Rad}}[v^\top A v] &= 2 \sum_{i \neq j} A_{ij}^2
        = 2\bigl(\|A\|_F^2 - \|\mathrm{diag}(A)\|^2\bigr), \\
        \mathrm{Var}_{\mathcal{N}}[v^\top A v] &= 2 \|A\|_F^2.

    ``K`` centering is inherited from :meth:`~quadsv.kernels.Kernel.Kx`
    (which applies ``H`` on both sides whenever ``centering=True``).
    Analytic substitutions listed below all read from backend-specific
    :meth:`trace` / :meth:`square_trace` methods that already embed the
    centering correction (``-s₁/n`` for ``c_1``; ``-2·s₂/n + s₁²/n²`` for ``c_2``).

    Backend-specific fast paths
    ---------------------------

    *FFTKernel* — **full spectrum always, all four cumulants analytic**
        ``c_p = Σ_k λ̃(k)^p`` is computed analytically using the ``n`` Fourier
        modes (``O(n)``) cached from :meth:`~quadsv.fft.FFTKernel.eigenvalues`.

    *MatrixKernel / NUFFTKernel — ``use_analytic_c12=True``* (default)
        ``c_1`` from :meth:`trace`, ``c_2`` from :meth:`square_trace`
        (both exact: diagonal-sum / Frobenius² on ``MatrixKernel``,
        coord-invariant ``(n/n')·Σλ`` / doubled-grid linear-conv
        ``λᵀΨλ`` on ``NUFFTKernel``). ``c_3``, ``c_4`` always come from
        the Rademacher probe estimator (closed-form formula are often
        too expensive).

    *MatrixKernel with ``stores_precision=True``* (CAR, Graph Laplacian)
        The stored object is the precision :math:`K^{-1}`, and the
        kernel's :meth:`trace` / :meth:`square_trace` are themselves
        Hutchinson estimators (``±1`` Rademacher probes through an LU
        solve on the precision). We forward the current ``n_probes`` so
        the precision-side Hutchinson budget tracks this caller's
        budget; ``c_3`` / ``c_4`` use our Rademacher probes as usual.

    *``use_analytic_c12=False``*
        All four cumulants from the same Rademacher probes — useful for
        diagnostics or when the analytic paths are known to be
        unreliable.

    Parameters
    ----------
    kernel : Kernel
        Must expose ``Kx(v)``. ``FFTKernel`` takes the fast path above.
    n_probes : int, default 60
        Probe count for the ``c_3`` / ``c_4`` estimator — also forwarded
        to :meth:`trace` / :meth:`square_trace` on precision-stored
        ``MatrixKernel``. ``m=60`` lands Liu p-values within ``~5%`` of
        the eigenvalue-exact baseline; ``m=120`` within ``~0.2%``. Cost
        scales as ``2·n_probes`` kernel matvecs (plus the same count of
        LU solves when precision-stored).
    rng_seed : int, default 0
        Seed for reproducible probe draws.
    use_analytic_c12 : bool, default True
        If ``True`` (default), substitute analytic ``c_1`` / ``c_2`` on
        ``MatrixKernel`` / ``NUFFTKernel`` as described above; on
        ``FFTKernel`` the full-spectrum fast path runs regardless.
        Set ``False`` to force the pure-probe estimator (still skips
        ``FFTKernel``'s fast path).

    Returns
    -------
    dict[int, float]
        ``{1: c_1, 2: c_2, 3: c_3, 4: c_4}`` of the input kernel ``K``.
    """
    # ------------------------------------------------------------------
    # FFTKernel fast path — full spectrum is O(n) and exact for all c_p.
    # ------------------------------------------------------------------
    from quadsv.fft import FFTKernel  # lazy to avoid circular import

    if isinstance(kernel, FFTKernel):
        # ``return_full_layout=True`` unpacks the rfft2 half-spectrum to
        # the full ``ny·nx`` layout and zeroes the DC bin when
        # ``centering=True`` (see FFTKernel.eigenvalues docstring).
        lam = np.asarray(kernel.eigenvalues(return_full_layout=True), dtype=float)
        sig = lam[np.abs(lam) > _DELTA]
        return {p: float(np.sum(sig**p)) for p in (1, 2, 3, 4)}

    # ------------------------------------------------------------------
    # General path — probe c_3, c_4; analytic / Hutchinson-via-trace
    # for c_1, c_2 depending on the backend.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(rng_seed)
    V_flat = rng.choice(np.array([-1.0, 1.0]), size=(int(kernel.n), int(n_probes)))

    def _apply(x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(kernel.Kx(x_flat))

    U = _apply(V_flat)
    W = _apply(U)
    c1_probe = float(np.mean(np.sum(V_flat * U, axis=0)))
    c2_probe = float(np.mean(np.sum(U * U, axis=0)))
    c3 = float(np.mean(np.sum(U * W, axis=0)))
    c4 = float(np.mean(np.sum(W * W, axis=0)))

    c1, c2 = c1_probe, c2_probe
    if use_analytic_c12 and hasattr(kernel, "trace") and hasattr(kernel, "square_trace"):
        is_precision_stored = bool(getattr(kernel, "stores_precision", False))
        try:
            if is_precision_stored:
                # MatrixKernel with stored precision: ``trace`` /
                # ``square_trace`` are themselves Hutchinson estimators
                # through an LU solve. Forward ``n_probes`` so the
                # internal cache uses our probe budget.
                c1 = float(kernel.trace(n_probes=n_probes))
                c2 = float(kernel.square_trace(n_probes=n_probes))
            else:
                # Analytic paths. All three backends (MatrixKernel
                # non-precision, FFTKernel, NUFFTKernel) return
                # deterministic analytic cumulants from a no-arg call.
                c1 = float(kernel.trace())
                c2 = float(kernel.square_trace())
        except (ValueError, NotImplementedError):
            # Any analytic path that refuses (e.g. indefinite-Λ
            # cancellation) silently falls back to probe estimates.
            c1, c2 = c1_probe, c2_probe

    return {1: c1, 2: c2, 3: c3, 4: c4}


def _liu_apply(t: float | np.ndarray, coef: dict[str, float]) -> np.ndarray:
    """Evaluate ``Pr(Q > t)`` from cached Liu coefficients.

    ``coef`` is the dict returned by :func:`_liu_prepare`. Broadcasts
    across array ``t`` in a single :func:`scipy.stats.ncx2.sf` call.
    """
    t = np.asarray(t, dtype=float)
    t_star = (t - coef["mu_Q"]) / (coef["sigma_Q"] + _DELTA)
    tfinal = t_star * coef["sigma_x"] + coef["mu_x"]
    return ncx2.sf(tfinal, coef["dof_x"], max(coef["delta_x"], 1e-9))


def liu_sf(
    t: float | np.ndarray,
    lambs: np.ndarray,
    dofs: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    kurtosis: bool = False,
    n: int | None = None,
) -> float | np.ndarray:
    """
    Liu approximation to a linear combination of non-central chi-squared variables.

    One-shot convenience wrapper equivalent to
    ``_liu_apply(t, _liu_prepare(lambs, ...))``. For multi-feature
    workloads, prefer the split form: call :func:`_liu_prepare` once on
    the spectrum and :func:`_liu_apply` for each Q-batch
    (:func:`compute_null_params` already caches the coefficients under
    ``null_params['liu_coef']``, so :func:`spatial_q_test` does this
    automatically).

    Parameters
    ----------
    t : float or np.ndarray
        Test statistic value(s). Array input is broadcast efficiently
        through a single :func:`scipy.stats.ncx2.sf` call.
    lambs : np.ndarray
        Eigenvalues of the kernel matrix, shape ``(n_evals,)``.
    dofs : np.ndarray, optional
        Per-eigenvalue degrees of freedom. Default: ones (chi-squared).
    deltas : np.ndarray, optional
        Non-centrality parameters. Default: zeros (central).
    kurtosis : bool, default False
        If True, use the kurtosis-based edge-case approximation.
    n : int, optional
        Sample size. When provided, applies the Dirichlet(1/2) variance
        correction ``Var[Q] = 2·(m·c_2 - c_1²)/(m+2)`` with ``m = n-1``
        for the z-scored ratio ``Q = XᵀK̃X/σ̂²``. Essential for
        broad-spectrum PSD kernels (CAR, graph_laplacian) on dense
        grids, where the large-:math:`n` limit ``2·c_2`` overestimates
        ``Var[Q]`` and collapses the tail to zero. Default ``None``
        keeps the original large-:math:`n` behavior for back-compat
        with callers supplying a raw eigenvalue mixture.

    Returns
    -------
    float or np.ndarray
        Tail probability ``Pr(Q > t)`` with the same shape as ``t``.
    """
    coef = _liu_prepare(lambs, dofs=dofs, deltas=deltas, kurtosis=kurtosis, n=n)
    return _liu_apply(t, coef)


def compute_null_params(
    kernel: Kernel,
    method: str = "welch",
    k_eigen: int | None = None,
    dirichlet_correction: bool = True,
    liu_n_probes: int | None = None,
) -> dict[str, float | np.ndarray]:
    r"""
    Pre-compute null distribution parameters for spatial tests.

    Call this ONCE before running parallel tests on thousands of features.
    Caches the expensive computations (traces, cumulants, shifted-χ² fit)
    for reuse across both Q-tests and R-tests.

    Parameters
    ----------
    kernel : Kernel
        The spatial kernel object (MatrixKernel, FFTKernel, NUFFTKernel, or compatible).
    method : {'clt', 'welch', 'liu'}, default 'welch'
        Null approximation method for the **Q-test**. The R-test entry
        ``var_R = trace(K̃²)`` is always populated alongside, regardless of
        ``method`` — R-tests use a Normal approximation and only need this
        one moment.

        - 'clt': Central Limit Theorem (Z-score normal approximation)
        - 'welch': Welch-Satterthwaite moment matching (fast, uses traces)
        - 'liu': Liu eigenvalue-based approximation (accurate tail, slower)
    k_eigen : int, optional
        Number of top eigenvalues to compute if method='liu' and kernel is sparse.
        If None, computes all available eigenvalues. Ignored when
        ``liu_n_probes`` is set (eigenvalues are bypassed entirely).
    liu_n_probes : int, optional
        If set, bypass the eigendecomposition for ``method='liu'`` and
        estimate the four spectral cumulants ``c_p = trace(K̃^p)``,
        ``p = 1..4``, directly from the kernel via Hutchinson probing
        (:func:`_hutchinson_cumulants`). Cost drops from
        :math:`O(n^3)` (dense eigensolve) or :math:`O(r^3)` (reduced
        Toeplitz-M) to :math:`2 \cdot n_\mathrm{probes}` kernel
        matvecs, at the cost of :math:`O(n_\mathrm{probes}^{-1/2})`
        Monte-Carlo error in each cumulant. Rule of thumb:
        ``n_probes = 60`` gives Liu p-values within ``~5\%`` of the
        eigenvalue-exact baseline; ``n_probes = 120`` within
        ``~0.2\%``. When ``None`` (default), the eigenvalue path is
        used.
    dirichlet_correction : bool, default True
        When ``True``: use the finite-``n`` Dirichlet(1/2) ratio
        ``Var[Q] = 2 · (m · trace((HKH)²) - trace(HKH)²) / (m+2)`` with
        ``m = n-1``. When ``False``: drop the ``mean²`` term to the
        large-``n`` limit ``Var[Q] = 2 · trace((HKH)²)``, a monotonic
        upper bound that slightly overestimates ``Var[Q]`` at finite
        ``n``. See **Notes** for the derivation.

    Returns
    -------
    dict[str, float or np.ndarray]
        Always populated (regardless of ``method``):

        - ``'method'`` : str — the Q-test approximation selected.
        - ``'var_R'`` : float — ``trace(K̃²)``, the null variance of ``R``
          (used by :func:`spatial_r_test`).

        Method-specific additions:

        - ``method='liu'`` (default for FFT / NUFFT kernels):

          * ``'cumulants'`` : ``dict {1: c_1, 2: c_2, 3: c_3, 4: c_4}``
            with ``c_p = trace(K̃^p)``. Computed from the full
            eigendecomposition when available (``liu_n_probes is
            None``) or from :math:`2m` Hutchinson probes otherwise.
          * ``'liu_coef'`` : ``dict`` with cached Liu coefficients
            ``{'mu_Q', 'sigma_Q', 'mu_x', 'sigma_x', 'dof_x',
            'delta_x'}`` derived from ``cumulants`` once; consumed by
            :func:`spatial_q_test` so per-feature p-values reduce to a
            pure :math:`t`-broadcast.

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

    Notes
    -----
    :func:`spatial_q_test` standardizes its input as
    :math:`Z = (X - \bar{X}\,\mathbf{1}) / \sigma`, so the realized
    quadratic form is

    .. math::

        Q \;=\; Z^{\top} K Z \;=\; X^{\top}\, H K H\, X / \sigma^{2},
        \qquad H = I - \mathbf{1}\mathbf{1}^{\top} / n.

    Null moments are therefore for the double-centered operator
    :math:`\tilde{K} = H K H`, not raw :math:`K`. :math:`Q` is
    additionally a *ratio* of quadratic forms — the denominator
    :math:`\sigma^{2} = X^{\top} H X / (n-1)` is a random variable
    correlated with the numerator. ``dirichlet_correction=True`` applies
    the finite-:math:`n` correction derived from the Dirichlet(1/2)
    distribution of :math:`Y_i = X_i^{2} / \sum_j X_j^{2}`:

    .. math::

        \mathrm{Var}[Q] \;=\; \frac{2 \bigl[\, m \cdot \mathrm{tr}(\tilde K^{2})
            - \mathrm{tr}(\tilde K)^{2} \,\bigr]}{m + 2},
            \qquad m = n - 1.

    With ``dirichlet_correction=False`` the finite term drops
    out and :math:`\mathrm{Var}[Q] = 2\,\mathrm{tr}(\tilde K^{2})` (large-:math:`n` limit).
    """
    params = {"method": method}

    assert method in ["clt", "welch", "liu"], "Method must be 'clt', 'welch', or 'liu'."

    # Moran's I kernel is indefinite (non-PSD) — its eigenvalues span both
    # signs, and its trace is ≈ 0 by construction. Welch / Liu are both
    # PSD-assuming moment-matching schemes (Welch needs ``mean_Q > 0``;
    # Liu fits a shifted χ² to a Σλ·χ²₁ mixture that only makes sense when
    # the λ are non-negative). Force ``'clt'`` for Moran — a direct Normal
    # approximation on the CLT-limit distribution of the standardized Q —
    # across all three backends.
    kernel_method = getattr(kernel, "method", None)
    if kernel_method == "moran" and method != "clt":
        raise ValueError(
            f"Moran's I kernel is indefinite; only null_method='clt' is "
            f"supported for the Q-test. Got method={method!r}."
        )

    # Centered traces can be computed cheaply from two additional numbers:
    #   s1 = 𝟏ᵀ K 𝟏,   s2 = ‖K·𝟏‖² = 𝟏ᵀ K² 𝟏
    # via a single K·𝟏 application (see `Kernel._ones_stats`), giving
    #   trace(HKH)   = trace(K)  − s1/n
    #   trace((HKH)²) = trace(K²) − 2·s2/n + s1²/n²
    n = int(kernel.n)
    tr_HKH = float(kernel.trace())
    tr_HKH_sq = float(kernel.square_trace())
    params["var_R"] = tr_HKH_sq

    if method == "liu":
        # Liu's method is entirely determined by the four spectral
        # cumulants c_1..c_4. We ALWAYS store them under ``cumulants``
        # and the derived shifted-χ² fit under ``liu_coef``; callers
        # consuming Liu p-values read only ``liu_coef``.
        #
        # Two paths produce the cumulants:
        #   - ``liu_n_probes is not None``: Hutchinson — 2·m matvecs.
        #   - ``liu_n_probes is None``: try the kernel's full
        #     eigendecomposition first, fall back to Hutchinson if the
        #     kernel can't produce a full spectrum (NUFFT with broad
        #     support or indefinite Λ → ``NotImplementedError``).
        if liu_n_probes is not None:
            c = _hutchinson_cumulants(kernel, n_probes=int(liu_n_probes))
        else:
            try:
                vals = kernel.eigenvalues(k=k_eigen)
                sig = vals[np.abs(vals) > 1e-9]
                c = {p: float(np.sum(sig**p)) for p in (1, 2, 3, 4)}
            except NotImplementedError:
                c = _hutchinson_cumulants(kernel, n_probes=60)
        params["cumulants"] = c
        # Pass ``n`` for the Dirichlet(1/2) variance correction — matches
        # the Welch branch below when ``dirichlet_correction=True``.
        # Broad-spectrum PSD kernels (CAR, graph_laplacian) have
        # ``c_1² ≈ m · c_2``; the large-n limit ``2·c_2`` then
        # overestimates ``Var[Q]`` by up to an order of magnitude,
        # collapsing Liu's tail probability to zero.
        liu_n = n if dirichlet_correction else None
        params["liu_coef"] = _liu_prepare_from_cumulants(c, n=liu_n)
    else:
        # Q-test CLT / Welch moments.
        mean_Q = tr_HKH
        if dirichlet_correction:
            m = max(n - 1, 1)
            var_Q = 2.0 * (m * tr_HKH_sq - tr_HKH**2) / (m + 2)
        else:
            # Large-n limit — drops the (m·sq − mean²) cancellation that
            # amplifies ``sq``-errors for broad-spectrum PSD kernels.
            var_Q = 2.0 * tr_HKH_sq
        var_Q = max(var_Q, 0.0)  # numerical safety
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
    kernel_method = getattr(kernel, "method", None)
    if null_params is None:
        if not hasattr(kernel, "square_trace"):
            raise ValueError("If params is None, kernel must be a Kernel object.")
        # Moran is indefinite — Welch/Liu degenerate, auto-pick CLT.
        default_method = "clt" if kernel_method == "moran" else "welch"
        null_params = compute_null_params(kernel, method=default_method)
        null_approx_method = default_method
    else:
        null_approx_method = null_params.get("method", "welch")
        # Ensure params exist
        if len(null_params) == 1 and hasattr(kernel, "square_trace"):
            null_params.update(compute_null_params(kernel, method=null_approx_method))
    if kernel_method == "moran" and null_approx_method != "clt":
        raise ValueError(
            f"Moran's I kernel is indefinite; only null_method='clt' is "
            f"supported for the Q-test. Got method={null_approx_method!r}."
        )

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
        # Read the cached Liu coefficients (O(1) per Q). If a caller
        # hand-built ``null_params`` with raw ``cumulants`` instead, derive
        # the coef on the fly and pass ``n=kernel.n`` so the Dirichlet(1/2)
        # variance correction fires — keeps calibration on broad-spectrum
        # PSD kernels.
        coef = null_params.get("liu_coef")
        if coef is None:
            if "cumulants" not in null_params:
                raise ValueError(
                    "null_params with method='liu' must contain either "
                    "'liu_coef' (preferred) or 'cumulants'. Build via "
                    "compute_null_params(kernel, method='liu')."
                )
            n_kernel = int(getattr(kernel, "n", 0)) or None
            coef = _liu_prepare_from_cumulants(null_params["cumulants"], n=n_kernel)
        pval = _liu_apply(Q, coef)
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
