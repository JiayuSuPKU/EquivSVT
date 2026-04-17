"""
Cross-sample spatial pattern comparison in the frequency domain.

This module implements an alignment-free, frequency-domain approach for ranking genes
by spatial-pattern difference between two groups of spatial-omics samples (e.g.,
*N* healthy vs *M* cancer slides). The key primitive is the 2D power spectrum
:math:`|\\hat{x}(k)|^2` of a rasterized gene image: power spectra are
**translation-invariant**, so samples need not be spatially registered.

Pipeline
--------

1. **Per-sample spectra** — :func:`compute_sample_spectrum` runs
   :func:`quadsv.fft.power_spectrum_2d` on each sample's ``(n_genes, ny, nx)`` array.
2. **Radial binning (default, rotation-invariant)** — :func:`radial_bin_spectrum`
   collapses the 2D spectrum onto a ``K``-dim vector indexed by normalized radial
   frequency, harmonizing samples with different ``(ny, nx)``.
3. **(Optional) 2D mode with rotation alignment** —
   :func:`align_spectra_by_rotation` rotates each sample's full 2D spectrum to
   maximize similarity to a reference, restoring comparability when directional
   anisotropy matters.
4. **Batch correction** — :func:`normalize_by_background` cancels per-slide
   gain/sensitivity differences; :func:`residualize_against_covariates` regresses out
   user-supplied covariate spectra (cell-type proportions, tissue domains, etc.).
5. **Two-group test per gene** — :func:`compare_two_groups` (or
   :func:`benchmark_statistics` for an apples-to-apples comparison of all four)
   produces a per-gene table with a permutation p-value and BH-FDR.

The :class:`SpectralComparator` class wraps the pipeline with caching of intermediate
spectra so that running :meth:`SpectralComparator.test` with several statistics does
not re-rasterize or re-bin.

Notes
-----
The default log-L2 statistic is motivated by the nonparametric two-sample test of
log-spectral densities (Bandyopadhyay & Wu, *arXiv* 2026). Permutation nulls are used
throughout because typical study sizes (3–10 slides per group) make parametric tail
approximations on multivariate statistics unreliable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.ndimage
from joblib import Parallel, delayed
from scipy.stats import ks_2samp  # noqa: F401  (exposed for downstream calibration tests)

from quadsv.fft import power_spectrum_2d
from quadsv.utils import _apply_bh_correction

__all__ = [
    "compute_sample_spectrum",
    "radial_bin_spectrum",
    "align_spectra_by_rotation",
    "normalize_by_background",
    "residualize_against_covariates",
    "shape_normalize",
    "compare_two_groups",
    "compare_two_groups_scalar",
    "benchmark_statistics",
    "SpectralComparator",
]

logger = logging.getLogger(__name__)

_AVAILABLE_STATISTICS = ("log_l2", "hotelling_lw", "mmd_rbf", "max_welch")


# ---------------------------------------------------------------------------
# Step 1 — per-sample spectra and radial binning
# ---------------------------------------------------------------------------


def compute_sample_spectrum(
    sample: np.ndarray,
    fft_solver: str = "rfft2",
    workers: int | None = None,
    center: str | None = "mean",
    return_dc: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D power spectrum of every gene in a single sample.

    By default the spatial signal is **mean-centred per gene** before the FFT so
    that the resulting power spectrum carries only the *AC* component of the
    pattern — i.e. the ``k=0`` (DC) bin is exactly zero and low-``k`` leakage from
    per-sample mean shifts is eliminated. The separated DC scalars (the per-sample
    per-gene grid means) can be returned alongside the spectrum with
    ``return_dc=True`` and are the natural target for a *classical differential
    expression* test complementary to the spectral pattern test.

    Parameters
    ----------
    sample : np.ndarray
        Rasterized expression of shape ``(n_genes, ny, nx)``.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT routine forwarded to :func:`quadsv.fft.power_spectrum_2d`.
    workers : int, optional
        Parallel workers forwarded to :mod:`scipy.fft`.
    center : {'mean', 'zscore', None}, default 'mean'
        Per-gene centering applied to the spatial signal *before* the FFT.

        - ``'mean'`` (default): subtract the grid mean. Ensures the DC bin is 0
          and the spectrum is statistically orthogonal to a DE test on the per-
          gene grid means.
        - ``'zscore'``: subtract the mean and divide by the standard deviation.
          Also removes overall magnitude, so the spectrum becomes scale-invariant
          (pure pattern shape).
        - ``None``: no centering; spectrum includes DC.
    return_dc : bool, default False
        If True, also return a ``(n_genes,)`` array of per-gene grid means (DC
        scalars of the *uncentered* signal).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Power spectra of shape ``(n_genes, ny, n_kx)``. If ``return_dc=True``,
        also returns a ``(n_genes,)`` DC array.

    Raises
    ------
    ValueError
        If ``sample`` is not 3D or ``center`` is unknown.
    """
    if sample.ndim != 3:
        raise ValueError(f"sample must be 3D (n_genes, ny, nx), got shape {sample.shape}")
    if center not in ("mean", "zscore", None):
        raise ValueError(f"center must be 'mean', 'zscore', or None, got {center!r}.")

    # DC scalars always come from the *uncentered* grid.
    dc = sample.mean(axis=(1, 2))

    if center == "mean":
        work = sample - dc[:, None, None]
    elif center == "zscore":
        sd = sample.std(axis=(1, 2), keepdims=True)
        sd = np.clip(sd, 1e-12, None)
        work = (sample - dc[:, None, None]) / sd.squeeze(axis=(1, 2))[:, None, None]
    else:
        work = sample

    # Move feature axis to last so power_spectrum_2d treats it as M.
    moved = np.moveaxis(work, 0, -1)
    p = power_spectrum_2d(moved, fft_solver=fft_solver, workers=workers)
    spec = np.moveaxis(p, -1, 0)

    if return_dc:
        return spec, dc
    return spec


def _radial_frequency_grid(
    ny: int,
    nx: int,
    fft_solver: str,
    spacing: tuple[float, float] | None = None,
) -> np.ndarray:
    """Radial frequency for each spectrum bin, shape ``(ny, n_kx)``.

    If ``spacing=(dy, dx)`` is given, frequencies are in **cycles per unit length**
    (e.g., cycles/μm if ``spacing`` is in μm). Otherwise the result is in
    cycles/pixel with both axes normalized by their grid length, i.e.
    :math:`\\sqrt{(k_x/n_x)^2 + (k_y/n_y)^2}`.
    """
    if spacing is None:
        dy = 1.0 / ny
        dx = 1.0 / nx
        # Equivalent: scale fftfreq(..., d=1) by 1/n to get "normalized" frequency.
        ky = np.fft.fftfreq(ny) * (1.0 / dy) * dy  # == np.fft.fftfreq(ny)
        kx_full = np.fft.fftfreq(nx)
        kx_rfft = np.fft.rfftfreq(nx)
    else:
        dy, dx = spacing
        ky = np.fft.fftfreq(ny, d=dy)
        kx_full = np.fft.fftfreq(nx, d=dx)
        kx_rfft = np.fft.rfftfreq(nx, d=dx)
    if fft_solver == "fft2":
        kx = kx_full
    elif fft_solver == "rfft2":
        kx = kx_rfft
    else:
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")
    Kx, Ky = np.meshgrid(kx, ky)
    return np.sqrt(Kx**2 + Ky**2)


def radial_bin_spectrum(
    spectrum: np.ndarray,
    grid_shape: tuple[int, int],
    n_bins: int = 30,
    fft_solver: str = "rfft2",
    exclude_dc: bool = True,
    spacing: tuple[float, float] | None = None,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """
    Bin a 2D power spectrum into ``n_bins`` radial frequency bins.

    By default the binning axis is the **normalized** radial frequency
    :math:`k = \\sqrt{(k_x/n_x)^2 + (k_y/n_y)^2} \\in [0,\\,\\sqrt{0.5}]`, so spectra
    from samples with different ``(ny, nx)`` map onto the same K bins. Passing
    ``spacing=(dy, dx)`` (in physical units, e.g. μm per cell) switches the binning
    axis to **cycles per unit length** (cycles/μm → multiply by 1000 for cycles/mm),
    so bins are directly comparable across samples with different physical
    resolutions. In that case, also pass ``edges`` to enforce a common bin grid
    across samples.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum of shape ``(..., ny, n_kx)``. Leading dims (e.g., genes,
        samples) are preserved.
    grid_shape : tuple[int, int]
        Original ``(ny, nx)`` of the rasterized image (needed because ``rfft2`` only
        stores half of the kx axis).
    n_bins : int, default 30
        Number of radial bins. Ignored when ``edges`` is supplied.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT solver used to produce ``spectrum``. Must match.
    exclude_dc : bool, default True
        If True, drop the zero-frequency (DC) bin from the output.
    spacing : tuple[float, float], optional
        Physical spacing ``(dy, dx)`` per grid cell (e.g., μm). If given, the
        binning axis is physical frequency in cycles per unit length.
    edges : np.ndarray, optional
        Explicit monotonically increasing bin edges (length ``n_bins + 1``) in the
        same frequency units as ``spacing`` (or normalized if ``spacing`` is None).
        When supplied, this overrides ``n_bins`` and gives every sample identical
        bin boundaries — required for cross-sample comparisons in physical units.

    Returns
    -------
    np.ndarray
        Radial spectra of shape ``(..., n_bins)`` (or ``n_bins - 1`` when
        ``exclude_dc=True``).

    Raises
    ------
    ValueError
        If ``spectrum``'s last two dims do not match the expected shape implied by
        ``grid_shape`` and ``fft_solver``.
    """
    ny, nx = grid_shape
    expected_kx = nx if fft_solver == "fft2" else nx // 2 + 1
    if spectrum.shape[-2:] != (ny, expected_kx):
        raise ValueError(
            f"spectrum last two dims {spectrum.shape[-2:]} do not match "
            f"expected ({ny}, {expected_kx}) for fft_solver='{fft_solver}'."
        )

    k = _radial_frequency_grid(ny, nx, fft_solver, spacing=spacing)
    k_max = float(k.max())

    if edges is None:
        # Edges include 0; right edge slightly past k_max so the last bin is closed.
        edges = np.linspace(0.0, k_max * (1.0 + 1e-9), n_bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be a 1D monotonically increasing array of length >= 2.")
        n_bins = len(edges) - 1
    # Bin index for each spectrum cell (0..n_bins-1).
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, n_bins - 1)

    # For rfft2 the negative-kx half is implicit but corresponds to conjugate
    # entries with identical |X|^2. To make per-bin sums match what fft2 would
    # give, double-count interior columns and single-count DC + Nyquist (if even).
    if fft_solver == "rfft2":
        col_weights = np.full(expected_kx, 2.0)
        col_weights[0] = 1.0
        if nx % 2 == 0:
            col_weights[-1] = 1.0
        weights2d = np.broadcast_to(col_weights, (ny, expected_kx)).ravel()
    else:
        weights2d = np.ones(ny * expected_kx)

    leading = spectrum.shape[:-2]
    flat = spectrum.reshape(-1, ny * expected_kx)  # (B, ny*nkx)
    out = np.zeros((flat.shape[0], n_bins))
    counts = np.zeros(n_bins)
    np.add.at(counts, idx, weights2d)
    counts[counts == 0] = 1.0  # avoid div-by-zero on empty bins
    for b in range(flat.shape[0]):
        np.add.at(out[b], idx, flat[b] * weights2d)
    out /= counts  # bin-mean power

    if exclude_dc:
        out = out[..., 1:]
    return out.reshape(*leading, out.shape[-1])


# ---------------------------------------------------------------------------
# Step 2 — optional 2D rotation alignment
# ---------------------------------------------------------------------------


def _to_full_2d(power: np.ndarray, grid_shape: tuple[int, int], fft_solver: str) -> np.ndarray:
    """Mirror an ``rfft2`` half-spectrum into a full ``(ny, nx)`` spectrum.

    Uses the Hermitian symmetry of the FFT of a real signal: ``|X[ky, kx]|² ==
    |X[(ny - ky) % ny, (nx - kx) % nx]|²``. For ``fft2`` input, returns ``power``
    unchanged.
    """
    if fft_solver == "fft2":
        return power
    ny, nx = grid_shape
    half = power.shape[-1]
    full = np.zeros(power.shape[:-1] + (nx,), dtype=power.dtype)
    full[..., :half] = power

    # Build the (-ky)-flipped version of `power` (axis -2): keep ky=0 fixed,
    # reverse the order of ky=1..ny-1.
    flipped_ky = np.empty_like(power)
    flipped_ky[..., 0, :] = power[..., 0, :]
    if ny > 1:
        flipped_ky[..., 1:, :] = power[..., :0:-1, :]

    # Mirror interior columns. Column j (1 <= j < last_interior) lives at column
    # nx - j with the ky axis reversed. Skip DC (j=0) and Nyquist (j=nx/2 when
    # nx is even) since both are self-conjugate.
    last_interior = half - 1 if nx % 2 == 0 else half
    for j in range(1, last_interior):
        full[..., nx - j] = flipped_ky[..., j]
    return full


def _polar_resample(
    spectrum_2d: np.ndarray,
    n_theta: int,
    n_radius: int,
) -> np.ndarray:
    """
    Resample a 2D spectrum (already shifted so DC is at center) onto a polar grid.

    Returns shape ``(n_theta, n_radius)``.
    """
    ny, nx = spectrum_2d.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_max = min(cy, cx)
    radii = np.linspace(1.0, r_max, n_radius)
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(radii, thetas, indexing="ij")  # (n_r, n_t)
    yy = cy + R * np.sin(T)
    xx = cx + R * np.cos(T)
    coords = np.stack([yy.ravel(), xx.ravel()], axis=0)
    sampled = scipy.ndimage.map_coordinates(spectrum_2d, coords, order=1, mode="reflect")
    return sampled.reshape(n_radius, n_theta).T  # (n_theta, n_radius)


def align_spectra_by_rotation(
    spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = 180,
    n_radius: int = 64,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Rotate each non-reference sample's 2D spectrum to maximize similarity with the
    reference's *mean* spectrum (mean across genes).

    Implementation
    --------------
    For each sample we:

    1. Reduce to a single 2D template by averaging spectra across genes.
    2. fftshift so DC is at the image center.
    3. Resample onto a polar grid with ``n_theta`` angles in :math:`[0,\\pi)` (the 180°
       symmetry of :math:`|\\hat{x}|^2` for real ``x``).
    4. Cross-correlate the reference and sample polar templates along the angular
       axis to find the best rotation (peak of the circular cross-correlation).
    5. Rotate the original 2D spectrum by that angle and return.

    Parameters
    ----------
    spectra : sequence of np.ndarray
        Per-sample power spectra of shape ``(n_genes, ny, n_kx)`` (rfft2) or
        ``(n_genes, ny, nx)`` (fft2). All samples must share the same ``n_genes``,
        but ``(ny, nx)`` may vary across samples.
    grid_shapes : sequence of tuple
        Per-sample ``(ny, nx)`` of the original rasterized image.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT routine that produced ``spectra``. ``fft2`` is recommended for rotation
        alignment because it preserves full angular content.
    reference_index : int, default 0
        Index of the sample held fixed; all others are aligned to it.
    n_theta : int, default 180
        Angular resolution of the polar resampling. The recovered rotation is
        accurate to ``180/n_theta`` degrees.
    n_radius : int, default 64
        Radial resolution of the polar resampling.

    Returns
    -------
    rotated : list of np.ndarray
        Aligned spectra, same shapes as the input.
    angles_deg : np.ndarray
        Recovered rotation angles in degrees, length ``len(spectra)``. Reference
        sample's angle is 0.

    Notes
    -----
    The rotation is applied to the **2D power spectrum**, not back to the original
    spatial image. This is enough for any downstream analysis that operates on the
    aligned spectra (radial / 2D-binned tests).
    """
    n_samples = len(spectra)
    if reference_index < 0 or reference_index >= n_samples:
        raise ValueError(f"reference_index {reference_index} out of range [0, {n_samples})")

    def _prep(sp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        full = _to_full_2d(sp, shape, fft_solver)  # (n_genes, ny, nx)
        mean_2d = full.mean(axis=0)
        return np.fft.fftshift(mean_2d)

    ref_polar = _polar_resample(
        _prep(spectra[reference_index], grid_shapes[reference_index]), n_theta, n_radius
    )
    ref_polar -= ref_polar.mean(axis=0, keepdims=True)

    rotated: list[np.ndarray] = []
    angles = np.zeros(n_samples)
    for i, (sp, shape) in enumerate(zip(spectra, grid_shapes, strict=True)):
        if i == reference_index:
            rotated.append(sp.copy())
            continue
        cur_polar = _polar_resample(_prep(sp, shape), n_theta, n_radius)
        cur_polar -= cur_polar.mean(axis=0, keepdims=True)
        # Circular cross-correlation along axis 0 (theta), summed across radii.
        # corr[k] = sum_r sum_t ref[t, r] * cur[(t-k) mod n_theta, r]
        # via FFT trick on the theta axis.
        ref_hat = np.fft.fft(ref_polar, axis=0)
        cur_hat = np.fft.fft(cur_polar, axis=0)
        corr = np.real(np.fft.ifft(ref_hat * np.conj(cur_hat), axis=0)).sum(axis=1)
        k_best = int(np.argmax(corr))
        angle_deg = k_best * 180.0 / n_theta
        angles[i] = angle_deg

        # Rotate every gene's full 2D spectrum by the recovered angle.
        full = _to_full_2d(sp, shape, fft_solver)  # (n_genes, ny, nx)
        full_shift = np.fft.fftshift(full, axes=(-2, -1))
        rot = scipy.ndimage.rotate(
            full_shift, angle=-angle_deg, axes=(-2, -1), reshape=False, order=1, mode="reflect"
        )
        rot = np.fft.ifftshift(rot, axes=(-2, -1))
        if fft_solver == "rfft2":
            ny, nx = shape
            half = nx // 2 + 1
            rot = rot[..., :half]
        rotated.append(rot)

    return rotated, angles


# ---------------------------------------------------------------------------
# Step 3 — batch-effect correction
# ---------------------------------------------------------------------------


def normalize_by_background(
    spectra: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Cancel per-sample multiplicative effects via the geometric-mean spectrum.

    For a single sample with spectra ``(n_genes, K)``, computes the geometric-mean
    spectrum across genes :math:`\\bar P(k) = \\exp(\\overline{\\log P(\\cdot, k)})`
    and returns ``spectra / bar_P``. In log-space this equals subtracting the per-bin
    mean of ``log P`` across genes — the standard recipe for cancelling sample-level
    sensitivity / depth differences.

    Parameters
    ----------
    spectra : np.ndarray
        Per-sample spectra of shape ``(..., n_genes, K)``. The geometric mean is
        computed along the ``n_genes`` axis (second-to-last).
    eps : float, default 1e-12
        Floor added before the log to avoid ``log(0)``.

    Returns
    -------
    np.ndarray
        Background-normalized spectra, same shape as the input.
    """
    log_spec = np.log(spectra + eps)
    bg = log_spec.mean(axis=-2, keepdims=True)
    return np.exp(log_spec - bg)


def residualize_against_covariates(
    gene_spectra: np.ndarray,
    covariate_spectra: np.ndarray,
    fit_intercept: bool = True,
) -> np.ndarray:
    """
    Regress each gene's spectrum on a set of covariate spectra; return the residuals.

    For one sample, fits :math:`P_g \\approx \\beta_0 + \\sum_c \\beta_c\\,C_c` per
    frequency bin (i.e., each bin treated as an observation; covariates as features),
    and subtracts the fit. Equivalent to projecting out the column space of the
    covariate matrix from the gene-spectrum matrix.

    Parameters
    ----------
    gene_spectra : np.ndarray
        Gene spectra of shape ``(n_genes, K)``.
    covariate_spectra : np.ndarray
        Covariate spectra of shape ``(n_covariates, K)``. Typical covariates: spectra
        of cell-type proportion maps, tissue-domain indicator maps, or housekeeping
        composite expression.
    fit_intercept : bool, default True
        If True, prepend a constant column to the covariate design.

    Returns
    -------
    np.ndarray
        Residual spectra of shape ``(n_genes, K)``.

    Raises
    ------
    ValueError
        If ``covariate_spectra`` has a different last-axis length than
        ``gene_spectra``.
    """
    if gene_spectra.shape[-1] != covariate_spectra.shape[-1]:
        raise ValueError(
            f"Last axis must match: gene_spectra has K={gene_spectra.shape[-1]}, "
            f"covariate_spectra has K={covariate_spectra.shape[-1]}."
        )
    K = gene_spectra.shape[-1]
    # Design matrix shape (K, n_covariates [+1]).
    X = covariate_spectra.T
    if fit_intercept:
        X = np.hstack([np.ones((K, 1)), X])
    # Solve least-squares: P_g.T = X @ beta_g -> beta_g = pinv(X) @ P_g.T per gene.
    # Closed form via pseudo-inverse (covariates K is small).
    pinv = np.linalg.pinv(X)
    fitted = (X @ pinv @ gene_spectra.T).T  # (n_genes, K)
    return gene_spectra - fitted


def shape_normalize(
    spectra: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize spectra to unit geometric mean along ``axis`` (shape-only).

    Equivalent to dividing each slice along ``axis`` by its own geometric mean:
    ``out = spectra / exp(mean(log(spectra + eps), axis))``. After this
    transform, two rows that differ only by a multiplicative rescaling —
    exactly the fingerprint of a gene that is expressed in one group but
    absent in the other — become identical; only the *shape* of the
    power-vs-frequency curve remains.

    This is the natural companion to :func:`normalize_by_background`:
    background normalization cancels per-sample gain across genes;
    :func:`shape_normalize` cancels per-(sample, gene) magnitude across
    frequencies. Composed, they leave a pure radial pattern signature.

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative radial spectra. Any leading dimensions are preserved;
        normalization acts along ``axis`` only.
    axis : int, default -1
        Axis along which to enforce unit geometric mean (typically the K /
        frequency-bin axis).
    eps : float, default 1e-12
        Floor added before the log to avoid ``log(0)``.

    Returns
    -------
    np.ndarray
        Shape-normalized spectra, same shape as the input, with unit geometric
        mean along ``axis``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    >>> out = shape_normalize(x, axis=-1)
    >>> np.allclose(out[0], out[1])  # only the shape survives
    True
    """
    log_spec = np.log(spectra + eps)
    return np.exp(log_spec - log_spec.mean(axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Step 4 — test statistics
# ---------------------------------------------------------------------------


def _stat_log_l2(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """L2 distance between mean log-spectra. Vectorized over genes."""
    eps = 1e-12
    log_a = np.log(np.maximum(group_a, eps)).mean(axis=0)  # (n_genes, K)
    log_b = np.log(np.maximum(group_b, eps)).mean(axis=0)
    return np.linalg.norm(log_a - log_b, axis=-1)


def _stat_max_welch(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Max |Welch t| across radial bins. Vectorized over genes."""
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1) + 1e-30)
    t = (mean_a - mean_b) / se
    return np.max(np.abs(t), axis=-1)


def _ledoit_wolf_shrinkage(X: np.ndarray) -> np.ndarray:
    """Simple Ledoit-Wolf shrinkage estimator for a single (n, p) sample."""
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = Xc.T @ Xc / max(n - 1, 1)
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    # Shrinkage intensity (Ledoit-Wolf 2004 simplified).
    diff = S - target
    d2 = (diff**2).sum()
    b2 = 0.0
    for i in range(n):
        z = Xc[i : i + 1].T @ Xc[i : i + 1] - S
        b2 += (z**2).sum()
    b2 = min(b2 / (n**2), d2)
    delta = b2 / max(d2, 1e-30)
    return (1 - delta) * S + delta * target


def _stat_hotelling_lw(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Regularized Hotelling T² with Ledoit-Wolf pooled covariance. Per-gene."""
    n_genes = group_a.shape[1]
    out = np.empty(n_genes)
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    for g in range(n_genes):
        Xa = group_a[:, g, :]
        Xb = group_b[:, g, :]
        Sa = _ledoit_wolf_shrinkage(Xa)
        Sb = _ledoit_wolf_shrinkage(Xb)
        S_pool = ((n_a - 1) * Sa + (n_b - 1) * Sb) / max(n_a + n_b - 2, 1)
        diff = Xa.mean(axis=0) - Xb.mean(axis=0)
        try:
            inv_S = np.linalg.pinv(S_pool)
        except np.linalg.LinAlgError:
            inv_S = np.eye(S_pool.shape[0])
        out[g] = (n_a * n_b) / (n_a + n_b) * diff @ inv_S @ diff
    return out


def _stat_mmd_rbf(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Biased MMD² with RBF kernel + median heuristic bandwidth. Per-gene."""
    n_genes = group_a.shape[1]
    out = np.empty(n_genes)
    for g in range(n_genes):
        Xa = group_a[:, g, :]
        Xb = group_b[:, g, :]
        Z = np.vstack([Xa, Xb])
        # Pairwise squared distances via broadcasting.
        diff = Z[:, None, :] - Z[None, :, :]
        d2 = (diff**2).sum(axis=-1)
        # Median heuristic on the strictly upper triangle.
        iu = np.triu_indices(Z.shape[0], k=1)
        med = np.median(d2[iu]) if iu[0].size else 1.0
        gamma = 1.0 / max(med, 1e-30)
        K = np.exp(-gamma * d2)
        n_a = Xa.shape[0]
        Kaa = K[:n_a, :n_a].mean()
        Kbb = K[n_a:, n_a:].mean()
        Kab = K[:n_a, n_a:].mean()
        out[g] = Kaa + Kbb - 2.0 * Kab
    return out


_STAT_FNS = {
    "log_l2": _stat_log_l2,
    "max_welch": _stat_max_welch,
    "hotelling_lw": _stat_hotelling_lw,
    "mmd_rbf": _stat_mmd_rbf,
}


# ---------------------------------------------------------------------------
# Step 4b — permutation engine
# ---------------------------------------------------------------------------


def _permutation_indices(
    n_samples: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``(n_perm, n_samples)`` index arrays — random permutations of 0..n-1."""
    out = np.tile(np.arange(n_samples), (n_perm, 1))
    for i in range(n_perm):
        rng.shuffle(out[i])
    return out


def _permutation_pvalue(
    observed: np.ndarray,
    null_samples: np.ndarray,
) -> np.ndarray:
    """One-sided ``Pr(null >= observed)`` with an additive ``+1`` correction."""
    n_perm = null_samples.shape[0]
    ge = (null_samples >= observed[None, :]).sum(axis=0)
    return (ge + 1.0) / (n_perm + 1.0)


def _run_statistic_with_perm(
    stat_name: str,
    spectra: np.ndarray,
    groups: np.ndarray,
    perm_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed statistic + null distribution for one statistic. Internal."""
    fn = _STAT_FNS[stat_name]
    a_mask = groups == 0
    observed = fn(spectra[a_mask], spectra[~a_mask])
    n_perm = perm_indices.shape[0]
    null = np.empty((n_perm, spectra.shape[1]))
    for p in range(n_perm):
        perm_groups = groups[perm_indices[p]]
        a = perm_groups == 0
        null[p] = fn(spectra[a], spectra[~a])
    return observed, null


# ---------------------------------------------------------------------------
# Step 4c — public test functions
# ---------------------------------------------------------------------------


def compare_two_groups(
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    n_perm: int = 1000,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Test, for every gene, whether its spatial-pattern spectrum differs between two groups.

    Parameters
    ----------
    spectra : np.ndarray
        Per-sample spectral features of shape ``(n_samples, n_genes, K)``.
    groups : np.ndarray
        Group labels of length ``n_samples`` taking exactly two distinct values
        (mapped internally to 0/1 in sorted order).
    gene_names : sequence of str, optional
        Names for the gene axis. If None, integer indices are used.
    statistic : {'log_l2', 'hotelling_lw', 'mmd_rbf', 'max_welch'}, default 'log_l2'
        Test statistic. See module docstring for trade-offs.
    n_perm : int, default 1000
        Number of label permutations for the null distribution.
    random_state : int, optional
        Seed for the permutation RNG.
    n_jobs : int, default 1
        Reserved for future parallelism over genes; currently unused (the per-stat
        implementations are already vectorized over genes).

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj`` (BH-FDR), sorted
        by descending statistic.

    Raises
    ------
    ValueError
        If ``statistic`` is unknown, ``groups`` does not contain exactly two values,
        or shapes are inconsistent.
    """
    if statistic not in _STAT_FNS:
        raise ValueError(f"Unknown statistic '{statistic}'. Options: {sorted(_STAT_FNS)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, K), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups shape {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)  # 0 = first label sorted, 1 = second

    rng = np.random.default_rng(random_state)
    perm_idx = _permutation_indices(n_samples, n_perm, rng)
    observed, null = _run_statistic_with_perm(statistic, spectra, g_int, perm_idx)
    pvals = _permutation_pvalue(observed, null)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame({"Feature": list(gene_names), "Statistic": observed, "P_value": pvals})
    _apply_bh_correction(df)
    df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
    if n_jobs != 1:  # noqa: PLR2004
        logger.debug("n_jobs ignored: per-statistic implementations are already vectorized.")
    return df


def benchmark_statistics(
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistics: Sequence[str] = _AVAILABLE_STATISTICS,
    n_perm: int = 1000,
    random_state: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run several statistics on the same data with a **shared** permutation null.

    All statistics use the same ``perm_indices``, so per-gene p-values are directly
    comparable (same Monte-Carlo noise, same exchanges).

    Parameters
    ----------
    spectra, groups, gene_names, n_perm, random_state
        Same meaning as :func:`compare_two_groups`.
    statistics : sequence of str, default ('log_l2', 'hotelling_lw', 'mmd_rbf', 'max_welch')
        Subset of the four implemented statistics to evaluate.

    Returns
    -------
    dict
        Mapping ``stat_name -> DataFrame`` (each DataFrame as in
        :func:`compare_two_groups`).

    Raises
    ------
    ValueError
        If any statistic name is unknown or input shapes are inconsistent.
    """
    for s in statistics:
        if s not in _STAT_FNS:
            raise ValueError(f"Unknown statistic '{s}'. Options: {sorted(_STAT_FNS)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, K), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)

    rng = np.random.default_rng(random_state)
    perm_idx = _permutation_indices(n_samples, n_perm, rng)
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    out: dict[str, pd.DataFrame] = {}
    for s in statistics:
        observed, null = _run_statistic_with_perm(s, spectra, g_int, perm_idx)
        pvals = _permutation_pvalue(observed, null)
        df = pd.DataFrame({"Feature": list(gene_names), "Statistic": observed, "P_value": pvals})
        _apply_bh_correction(df)
        df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
        out[s] = df
    return out


# ---------------------------------------------------------------------------
# Step 4d — scalar (DE-style) two-group test
# ---------------------------------------------------------------------------


def _welch_abs_t(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Per-feature absolute Welch t-statistic. Inputs shape ``(n, n_features)``."""
    n_a, n_b = group_a.shape[0], group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1) + 1e-30)
    return np.abs(mean_a - mean_b) / se


def compare_two_groups_scalar(
    values: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    n_perm: int = 1000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Per-gene two-sample test on scalar per-sample values (classical DE).

    The natural companion to :func:`compare_two_groups`: tested on the DC scalars
    (per-gene grid means) produced by :func:`compute_sample_spectrum` when
    ``center='mean'``, the result is statistically independent of the spectral
    pattern test because DC and AC are orthogonal by construction.

    Parameters
    ----------
    values : np.ndarray
        Per-sample per-gene scalars of shape ``(n_samples, n_genes)`` — e.g.,
        log-normalized mean expression on each slide.
    groups : np.ndarray
        Group labels of length ``n_samples`` with exactly two distinct values.
    gene_names : sequence of str, optional
        Gene names. Integer indices if None.
    n_perm : int, default 1000
        Number of sample-label permutations for the null.
    random_state : int, optional
        Seed for the permutation RNG.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic`` (``abs(Welch t)``), ``Mean_diff``
        (``mean_groupA − mean_groupB``), ``P_value``, ``P_adj`` (BH-FDR), sorted
        by descending ``Statistic``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``groups`` does not contain exactly two
        distinct values.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (n_samples, n_genes), got {values.shape}.")
    n_samples, n_genes = values.shape
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)

    rng = np.random.default_rng(random_state)
    perm_idx = _permutation_indices(n_samples, n_perm, rng)
    observed = _welch_abs_t(values[g_int == 0], values[g_int == 1])
    mean_diff = values[g_int == 0].mean(axis=0) - values[g_int == 1].mean(axis=0)

    null = np.empty((n_perm, n_genes))
    for p in range(n_perm):
        perm_groups = g_int[perm_idx[p]]
        a = perm_groups == 0
        null[p] = _welch_abs_t(values[a], values[~a])
    pvals = _permutation_pvalue(observed, null)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    df = pd.DataFrame(
        {
            "Feature": list(gene_names),
            "Statistic": observed,
            "Mean_diff": mean_diff,
            "P_value": pvals,
        }
    )
    _apply_bh_correction(df)
    return df.sort_values("Statistic", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 — high-level wrapper class
# ---------------------------------------------------------------------------


class SpectralComparator:
    """
    High-level pipeline for cross-sample spectral pattern comparison.

    Wraps :func:`compute_sample_spectrum`, :func:`radial_bin_spectrum`,
    :func:`align_spectra_by_rotation`, :func:`normalize_by_background`,
    :func:`residualize_against_covariates`, and :func:`compare_two_groups` /
    :func:`benchmark_statistics` into a fluent, caching pipeline.

    Typical use::

        cmp = SpectralComparator(samples, groups, gene_names).fit()
        cmp.normalize_background()
        results = cmp.test(statistic="log_l2", n_perm=1000)

    Parameters
    ----------
    samples : list of np.ndarray
        Per-sample rasterized arrays of shape ``(n_genes, ny_s, nx_s)``. ``n_genes``
        must be the same across samples; ``(ny_s, nx_s)`` may differ.
    groups : np.ndarray
        Group labels of length ``len(samples)`` with exactly two distinct values.
    gene_names : list of str
        Gene names of length ``n_genes``.
    feature_mode : {'radial', '2d'}, default 'radial'
        ``'radial'`` (default) reduces every spectrum to a 1D radial vector — both
        translation- and rotation-invariant. ``'2d'`` keeps the full 2D spectrum and
        rotation-aligns each sample to the first one before flattening.
    n_radial_bins : int, default 30
        Number of radial bins when ``feature_mode='radial'``.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT solver passed to :func:`compute_sample_spectrum`.
    workers : int, optional
        FFT worker count.
    center : {'mean', 'zscore', None}, default 'mean'
        Per-gene spatial-signal centering applied before the FFT. ``'mean'``
        (default) makes the spectrum exactly DC-free so the pattern test via
        :meth:`test_pattern` is statistically orthogonal to the DE test via
        :meth:`test_expression` — the two tests then provide complementary
        information. See :func:`compute_sample_spectrum` for details.
    spacings : sequence of tuple[float, float], optional
        Per-sample physical cell spacing ``(dy, dx)`` (any unit — but all samples
        must use the same unit). When supplied, radial bins are defined in
        **physical frequency** (cycles per unit length) with common edges across
        samples, so spectra from slides with different grid shapes and pixel sizes
        become directly comparable. If None, normalized cycles/pixel bins are used.
    freq_edges : np.ndarray, optional
        Explicit radial-frequency bin edges (length ``n_radial_bins + 1``) in the
        same units as ``spacings``. When None and ``spacings`` is given, edges are
        auto-generated from 0 to the minimum per-sample Nyquist frequency.

    Attributes
    ----------
    samples : list of np.ndarray
        The raw rasterized arrays passed in.
    groups : np.ndarray
        Group labels.
    gene_names : list of str
        Gene names.
    feature_mode : str
        Feature mode in use.
    spectra_ : np.ndarray or None
        Per-sample feature matrix of shape ``(n_samples, n_genes, K)``. Set by
        :meth:`fit`. Subsequent calls to :meth:`normalize_background` /
        :meth:`residualize` mutate this in place.
    dc_ : np.ndarray or None
        Per-sample per-gene DC scalars (grid means) of shape
        ``(n_samples, n_genes)``. Set by :meth:`fit`. Unaffected by the
        normalization / residualization steps (they only touch ``spectra_``).
    rotation_angles_ : np.ndarray or None
        Recovered rotation angles (degrees), set when ``feature_mode='2d'`` and
        :meth:`fit` has been called.
    """

    def __init__(  # noqa: C901
        self,
        samples: Sequence[np.ndarray],
        groups: np.ndarray,
        gene_names: Sequence[str],
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        fft_solver: str = "rfft2",
        workers: int | None = None,
        spacings: Sequence[tuple[float, float]] | None = None,
        freq_edges: np.ndarray | None = None,
        center: str | None = "mean",
    ) -> None:
        if center not in ("mean", "zscore", None):
            raise ValueError(f"center must be 'mean', 'zscore', or None, got {center!r}.")
        self.center: str | None = center
        if feature_mode not in ("radial", "2d"):
            raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
        if feature_mode == "2d" and fft_solver != "fft2":
            logger.info(
                "feature_mode='2d' works best with fft_solver='fft2'; switching automatically."
            )
            fft_solver = "fft2"

        self.samples: list[np.ndarray] = list(samples)
        self.groups: np.ndarray = np.asarray(groups)
        self.gene_names: list[str] = list(gene_names)
        self.feature_mode: str = feature_mode
        self.n_radial_bins: int = n_radial_bins
        self.fft_solver: str = fft_solver
        self.workers: int | None = workers

        n_samples = len(self.samples)
        if self.groups.shape != (n_samples,):
            raise ValueError(
                f"groups length {self.groups.shape} does not match n_samples={n_samples}."
            )
        if np.unique(self.groups).size != 2:
            raise ValueError("groups must contain exactly two distinct labels.")
        for i, s in enumerate(self.samples):
            if s.ndim != 3:
                raise ValueError(f"sample {i} must be 3D, got shape {s.shape}.")
            if s.shape[0] != len(self.gene_names):
                raise ValueError(
                    f"sample {i} has n_genes={s.shape[0]} but gene_names has "
                    f"{len(self.gene_names)}."
                )

        if spacings is not None:
            spacings_list = [tuple(float(v) for v in sp) for sp in spacings]
            if len(spacings_list) != n_samples:
                raise ValueError(
                    f"spacings length {len(spacings_list)} does not match n_samples={n_samples}."
                )
            if any(len(sp) != 2 for sp in spacings_list):
                raise ValueError("each entry in spacings must be a (dy, dx) pair.")
            self.spacings: list[tuple[float, float]] | None = spacings_list
        else:
            self.spacings = None
        self.freq_edges: np.ndarray | None = (
            None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        )

        self.spectra_: np.ndarray | None = None
        self.dc_: np.ndarray | None = None
        self.rotation_angles_: np.ndarray | None = None
        self._raw_2d_spectra: list[np.ndarray] | None = None
        self._grid_shapes: list[tuple[int, int]] = [(s.shape[1], s.shape[2]) for s in self.samples]

        # Non-uniform FFT mode: set by :meth:`from_coords`, never here.
        self.mode: str = "fft"
        self._coords: list[np.ndarray] | None = None
        self._values: list[np.ndarray] | None = None
        self._unit_scales: list[float] | None = None
        self._nufft_grid_shape: tuple[int, int] | None = None
        self._nufft_spacing: tuple[float, float] | None = None
        self._nufft_eps: float = 1e-6

    # ------------------------------------------------------------------
    @classmethod
    def from_coords(  # noqa: C901
        cls,
        coords: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        groups: np.ndarray,
        gene_names: Sequence[str],
        grid_shape: tuple[int, int],
        spacing: tuple[float, float],
        unit_scales: Sequence[float] | None = None,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        center: str | None = "mean",
        freq_edges: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> SpectralComparator:
        """
        Build a comparator over **non-uniform** spatial samples via NUFFT.

        Unlike the default constructor — which expects per-sample *rasterized*
        arrays — this alternative entry point takes the raw per-spot
        coordinates and expression matrices and uses a 2D type-1 non-uniform
        FFT (:func:`quadsv.nufft.power_spectrum_2d_nufft`) to evaluate
        :math:`|\\hat c(k)|^2` on a *common* uniform k-space grid. All
        downstream steps (radial binning, background normalization,
        :meth:`test_pattern`, :meth:`test_expression`) work identically.

        Parameters
        ----------
        coords : sequence of np.ndarray
            Per-sample spot coordinates of shape ``(N_s, 2)`` in order
            ``(y, x)``. Units are per-sample (see ``unit_scales``).
        values : sequence of np.ndarray
            Per-sample expression matrices of shape ``(N_s, n_genes)``. The
            gene axis must be aligned across samples (second axis, length
            equal to ``len(gene_names)``).
        groups : np.ndarray
            Group labels of length ``n_samples`` with exactly two distinct values.
        gene_names : sequence of str
            Gene names; must match the second axis of every entry of ``values``.
        grid_shape : tuple[int, int]
            ``(ny, nx)`` of the common k-space grid.
        spacing : tuple[float, float]
            ``(dy, dx)`` per-cell spacing of the common grid, in whatever
            **common** physical unit all samples should be compared in
            (typically μm). This is the same ``spacing`` used by
            :func:`radial_bin_spectrum` so radial bins come out in cycles per
            that unit.
        unit_scales : sequence of float, optional
            Per-sample multiplier that converts each sample's raw ``coords``
            into the common unit of ``spacing``. For example, if sample A's
            coords are already in μm and sample B's coords are in Visium
            full-res pixels at 0.35 μm/pixel, pass ``unit_scales=[1.0, 0.35]``
            with ``spacing`` in μm. Default: ``[1.0] * n_samples``.
        feature_mode, n_radial_bins, center, freq_edges
            Same meaning as in the default constructor.
        eps : float, default 1e-6
            NUFFT tolerance.

        Returns
        -------
        SpectralComparator
            A fresh instance in ``mode='nufft'``. Call :meth:`fit` as usual
            to populate :attr:`spectra_` / :attr:`dc_`.

        Raises
        ------
        ValueError
            If inputs have inconsistent shapes, or ``unit_scales`` has a
            wrong length.
        """
        if len(coords) != len(values):
            raise ValueError(
                f"coords and values must have the same length; got {len(coords)} vs {len(values)}."
            )
        n_samples = len(coords)
        groups = np.asarray(groups)
        if groups.shape != (n_samples,):
            raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
        if np.unique(groups).size != 2:
            raise ValueError("groups must contain exactly two distinct labels.")
        n_genes = len(gene_names)
        for i, (c_i, v_i) in enumerate(zip(coords, values, strict=True)):
            if c_i.ndim != 2 or c_i.shape[1] != 2:
                raise ValueError(f"coords[{i}] must be shape (N, 2), got {c_i.shape}.")
            if v_i.ndim != 2 or v_i.shape != (c_i.shape[0], n_genes):
                raise ValueError(
                    f"values[{i}] must be shape (N={c_i.shape[0]}, n_genes={n_genes}), "
                    f"got {v_i.shape}."
                )
        if unit_scales is None:
            unit_scales = [1.0] * n_samples
        if len(unit_scales) != n_samples:
            raise ValueError(
                f"unit_scales length {len(unit_scales)} does not match n_samples={n_samples}."
            )

        # Construct the instance WITHOUT running __init__'s sample-shape checks.
        # We need the same attribute layout, populated with NUFFT metadata.
        self = cls.__new__(cls)
        if feature_mode not in ("radial", "2d"):
            raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
        if center not in ("mean", "zscore", None):
            raise ValueError(f"center must be 'mean', 'zscore', or None, got {center!r}.")

        self.samples: list[np.ndarray] = []  # not used in nufft mode
        self.groups = groups
        self.gene_names = list(gene_names)
        self.feature_mode = feature_mode
        self.n_radial_bins = int(n_radial_bins)
        self.fft_solver = "fft2"  # NUFFT output is full-spectrum; match that layout
        self.workers = None
        self.center = center
        self.spacings = [tuple(float(v) for v in spacing)] * n_samples
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        self.spectra_ = None
        self.dc_ = None
        self.rotation_angles_ = None
        self._raw_2d_spectra = None
        self._grid_shapes = [tuple(int(v) for v in grid_shape)] * n_samples

        self.mode = "nufft"
        self._coords = [np.asarray(c, dtype=np.float64) for c in coords]
        self._values = [np.asarray(v, dtype=np.float64) for v in values]
        self._unit_scales = [float(s) for s in unit_scales]
        self._nufft_grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self._nufft_spacing = (float(spacing[0]), float(spacing[1]))
        self._nufft_eps = float(eps)
        return self

    # ------------------------------------------------------------------
    def _compute_nufft_spectra(self, n_jobs: int = -1) -> tuple[list[np.ndarray], np.ndarray]:
        """NUFFT equivalent of the FFT-mode per-sample spectrum pass. Returns
        ``(raw_2d_spectra, dc)`` with the same layout as the FFT path."""
        from quadsv.nufft import power_spectrum_2d_nufft

        def _one(i: int) -> tuple[np.ndarray, np.ndarray]:
            pts = self._coords[i]
            vals = self._values[i]  # (N, n_genes)
            scale = self._unit_scales[i]

            # DC = mean expression per gene at the real (non-uniform) spots.
            dc = vals.mean(axis=0)

            centered = vals - dc if self.center == "mean" else vals
            if self.center == "zscore":
                sd = vals.std(axis=0)
                sd = np.clip(sd, 1e-12, None)
                centered = (vals - dc) / sd

            spec = power_spectrum_2d_nufft(
                pts,
                centered,
                grid_shape=self._nufft_grid_shape,
                spacing=self._nufft_spacing,
                unit_scale=scale,
                eps=self._nufft_eps,
                center_coords=True,
            )  # shape (ny, nx, n_genes)
            # Conform to (n_genes, ny, nx) like compute_sample_spectrum.
            return np.moveaxis(spec, -1, 0), dc

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one)(i) for i in range(len(self._coords))
        )
        raw_2d = [r[0] for r in results]
        dc = np.stack([r[1] for r in results], axis=0)
        return raw_2d, dc

    # ------------------------------------------------------------------
    def fit(self, n_jobs: int = -1) -> SpectralComparator:
        """
        Compute per-sample power spectra and (if ``feature_mode='2d'``) rotation-align.

        Parameters
        ----------
        n_jobs : int, default -1
            Parallelism over samples for the per-sample FFT.

        Returns
        -------
        SpectralComparator
            ``self``, for chaining.
        """
        logger.info(
            "Computing per-sample spectra (mode=%s, n_samples=%d, center=%s)...",
            self.mode,
            len(self._grid_shapes),
            self.center,
        )
        if self.mode == "nufft":
            self._raw_2d_spectra, self.dc_ = self._compute_nufft_spectra(n_jobs=n_jobs)
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(compute_sample_spectrum)(
                    s,
                    fft_solver=self.fft_solver,
                    workers=self.workers,
                    center=self.center,
                    return_dc=True,
                )
                for s in self.samples
            )
            self._raw_2d_spectra = [r[0] for r in results]
            self.dc_ = np.stack([r[1] for r in results], axis=0)  # (n_samples, n_genes)

        if self.feature_mode == "2d":
            aligned, angles = align_spectra_by_rotation(
                self._raw_2d_spectra,
                grid_shapes=self._grid_shapes,
                fft_solver=self.fft_solver,
            )
            self._raw_2d_spectra = aligned
            self.rotation_angles_ = angles

        # Build a common bin-edge grid for physical-frequency mode.
        if self.feature_mode == "radial" and self.spacings is not None and self.freq_edges is None:
            # Min per-sample Nyquist: f_Nyquist = 1 / (2 * max(dy, dx)) along the most
            # coarsely sampled axis. Use the minimum across samples so every sample
            # has support up to that frequency.
            nyquists = [1.0 / (2.0 * max(dy, dx)) for (dy, dx) in self.spacings]
            f_max = float(min(nyquists))
            self.freq_edges = np.linspace(0.0, f_max * (1.0 + 1e-9), self.n_radial_bins + 1)
            logger.info(
                "Auto-generated %d radial bins on [0, %.4g] cycles per unit length.",
                self.n_radial_bins,
                f_max,
            )

        # Reduce to per-sample feature matrices of shape (n_genes, K) and stack.
        feats = []
        for i, (sp, shape) in enumerate(zip(self._raw_2d_spectra, self._grid_shapes, strict=True)):
            if self.feature_mode == "radial":
                spacing = self.spacings[i] if self.spacings is not None else None
                f = radial_bin_spectrum(
                    sp,
                    grid_shape=shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self.fft_solver,
                    spacing=spacing,
                    edges=self.freq_edges,
                )
            else:
                # 2D mode: flatten the (ny, nx) spectrum but optionally truncate to a
                # low-frequency square block of side ``n_radial_bins`` (re-using the
                # parameter as a low-pass cutoff). This keeps K manageable.
                ny, nx = shape
                k = min(self.n_radial_bins, ny // 2, nx // 2)
                low = sp[:, :k, :k] if sp.shape[-1] > k else sp[:, :k, :]
                f = low.reshape(low.shape[0], -1)
            feats.append(f)
        # Resample to common K (samples may differ slightly due to truncation).
        K = min(f.shape[-1] for f in feats)
        feats = [f[..., :K] for f in feats]
        self.spectra_ = np.stack(feats, axis=0)  # (n_samples, n_genes, K)
        return self

    # ------------------------------------------------------------------
    def normalize_background(self) -> SpectralComparator:
        """Apply per-sample geometric-mean background normalization. In-place on ``spectra_``."""
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .normalize_background().")
        for i in range(self.spectra_.shape[0]):
            self.spectra_[i] = normalize_by_background(self.spectra_[i])
        return self

    # ------------------------------------------------------------------
    def shape_normalize(self) -> SpectralComparator:
        """
        Make every (sample, gene) spectrum unit-geometric-mean along frequency.

        Calls :func:`shape_normalize` along the last axis of :attr:`spectra_`
        (the K / frequency-bin axis) and mutates :attr:`spectra_` in place.
        :attr:`dc_` is untouched. Typical usage for magnitude-invariant
        downstream clustering::

            cmp.fit().normalize_background().shape_normalize()

        Returns
        -------
        SpectralComparator
            ``self``, for chaining.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .shape_normalize().")
        self.spectra_ = shape_normalize(self.spectra_, axis=-1)
        return self

    # ------------------------------------------------------------------
    def residualize(self, covariates: Sequence[np.ndarray]) -> SpectralComparator:
        """
        Regress out covariate spectra per sample.

        Parameters
        ----------
        covariates : sequence of np.ndarray
            Per-sample covariate arrays of shape ``(n_covariates, ny_s, nx_s)``,
            using the same grid shapes as the corresponding sample.

        Returns
        -------
        SpectralComparator
            ``self``, for chaining.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .residualize().")
        if len(covariates) != len(self.samples):
            raise ValueError(
                f"covariates length {len(covariates)} != n_samples {len(self.samples)}."
            )
        for i, cov in enumerate(covariates):
            if cov.ndim != 3:
                raise ValueError(f"covariate sample {i} must be 3D, got {cov.shape}.")
            cov_2d = compute_sample_spectrum(cov, fft_solver=self.fft_solver, workers=self.workers)
            shape = self._grid_shapes[i]
            spacing = self.spacings[i] if self.spacings is not None else None
            if self.feature_mode == "radial":
                cov_feat = radial_bin_spectrum(
                    cov_2d,
                    grid_shape=shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self.fft_solver,
                    spacing=spacing,
                    edges=self.freq_edges,
                )
            else:
                ny, nx = shape
                k = min(self.n_radial_bins, ny // 2, nx // 2)
                low = cov_2d[:, :k, :k] if cov_2d.shape[-1] > k else cov_2d[:, :k, :]
                cov_feat = low.reshape(low.shape[0], -1)
            cov_feat = cov_feat[..., : self.spectra_.shape[-1]]
            self.spectra_[i] = residualize_against_covariates(self.spectra_[i], cov_feat)
        return self

    # ------------------------------------------------------------------
    def test_pattern(
        self,
        statistic: str = "log_l2",
        n_perm: int = 1000,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """
        Two-group spectral-pattern test on the cached :attr:`spectra_`.

        With ``center='mean'`` (the default), the spectrum is DC-free and this
        test is statistically orthogonal to :meth:`test_expression`. See
        :func:`compare_two_groups` for parameters and return format.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .test_pattern().")
        return compare_two_groups(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistic=statistic,
            n_perm=n_perm,
            random_state=random_state,
        )

    # Back-compat alias — `test()` still runs the pattern test.
    test = test_pattern

    # ------------------------------------------------------------------
    def test_expression(
        self,
        n_perm: int = 1000,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """
        Classical DE test on the DC component (per-sample per-gene grid mean).

        Complementary to :meth:`test_pattern`: this asks *"does total
        gene expression on the slide differ between groups?"* while the pattern
        test asks *"does the shape of the spatial pattern differ?"*. When
        ``center='mean'`` the two tests are orthogonal by construction.

        Parameters
        ----------
        n_perm : int, default 1000
            Number of label permutations for the null.
        random_state : int, optional
            Seed for the permutation RNG.

        Returns
        -------
        pd.DataFrame
            Columns ``Feature``, ``Statistic`` (``abs(Welch t)``), ``Mean_diff``,
            ``P_value``, ``P_adj`` — sorted by ``Statistic`` descending.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.dc_ is None:
            raise RuntimeError("Call .fit() before .test_expression().")
        return compare_two_groups_scalar(
            self.dc_,
            self.groups,
            gene_names=self.gene_names,
            n_perm=n_perm,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    def benchmark(
        self,
        statistics: Sequence[str] = _AVAILABLE_STATISTICS,
        n_perm: int = 1000,
        random_state: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run :func:`benchmark_statistics` on the cached :attr:`spectra_`."""
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .benchmark().")
        return benchmark_statistics(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistics=statistics,
            n_perm=n_perm,
            random_state=random_state,
        )
