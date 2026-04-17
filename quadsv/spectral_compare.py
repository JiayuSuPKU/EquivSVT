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

import itertools
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import ks_2samp  # noqa: F401  (exposed for downstream calibration tests)
from scipy.stats import t as _t_dist
from tqdm.auto import tqdm

from quadsv.fft import power_spectrum_2d
from quadsv.utils import _apply_bh_correction

__all__ = [
    "compute_sample_spectrum",
    "radial_bin_spectrum",
    "align_spectra_by_rotation",
    "estimate_rotations_from_landmarks",
    "apply_rotations_to_spectra",
    "normalize_by_background",
    "residualize_against_covariates",
    "shape_normalize",
    "compare_two_groups",
    "compare_two_groups_masked",
    "compare_two_groups_scalar",
    "benchmark_statistics",
    "SpectralComparator",
]

logger = logging.getLogger(__name__)

_AVAILABLE_STATISTICS = ("log_l2", "hotelling_lw", "mmd_rbf", "cauchy_welch")

# Absolute clipping threshold for ``center='zscore'`` (both the FFT per-sample
# helper :func:`compute_sample_spectrum` and the NUFFT :class:`SpectralComparator`
# loop use this). Guards against sparse genes producing arbitrarily large
# standardized values that dominate the pattern test. ±6σ covers > 99.99% of
# the Normal distribution so this is a gentle cap, not a heavy truncation.
_ZSCORE_CLIP = 6.0


# ---------------------------------------------------------------------------
# SpatialData helpers (Phase D)
# ---------------------------------------------------------------------------


def _sd_get_table(sd_obj: Any, table_key: str | None):
    """Fetch the expression table from a :class:`spatialdata.SpatialData` object.

    When ``table_key`` is None, fall back to the single table present (error if
    there are multiple). Returns an :class:`anndata.AnnData` table.
    """
    tables = sd_obj.tables
    if table_key is None:
        keys = list(tables.keys())
        if len(keys) == 0:
            raise ValueError("SpatialData has no tables; supply `table_key` explicitly.")
        if len(keys) > 1:
            raise ValueError(
                f"SpatialData has multiple tables {keys}; supply `table_key` explicitly."
            )
        return tables[keys[0]]
    if table_key not in tables:
        raise KeyError(f"SpatialData has no table '{table_key}'; available: {list(tables.keys())}.")
    return tables[table_key]


def _sd_infer_grid_and_spacing(
    sd_obj: Any,
    *,
    table_key: str | None,
    override_shape: tuple[int, int] | None = None,
    override_spacing: tuple[float, float] | None = None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Infer ``(grid_shape, spacing)`` for a SpatialData sample.

    Preferred order:
      1. explicit user overrides, when both ``override_shape`` and
         ``override_spacing`` are supplied;
      2. the SpatialData object's first rasterized image's
         ``(ny, nx)`` dims and pixel pitch;
      3. fall back to the table's attached coordinates via
         :func:`quadsv.nufft._infer_grid_from_coords`.
    """
    if override_shape is not None and override_spacing is not None:
        return (
            (int(override_shape[0]), int(override_shape[1])),
            (float(override_spacing[0]), float(override_spacing[1])),
        )

    # Try image layers (they encode both shape and spacing).
    images = getattr(sd_obj, "images", {}) or {}
    if images:
        first_key = next(iter(images))
        img = images[first_key]
        # DataArray-like: look at the last two dims.
        try:
            ny, nx = int(img.shape[-2]), int(img.shape[-1])
            grid = (ny, nx)
            spacing = (1.0, 1.0)
            return grid, spacing
        except Exception:  # pragma: no cover
            pass

    # Fall back: coords from the table's obsm.
    table = _sd_get_table(sd_obj, table_key)
    if "spatial" in table.obsm:
        from quadsv.nufft import _infer_grid_from_coords

        coords = np.asarray(table.obsm["spatial"], dtype=np.float64)
        return _infer_grid_from_coords(coords, oversample=2.0)

    raise ValueError(
        "Could not infer (grid_shape, spacing) from SpatialData — no images and no "
        "obsm['spatial'] on the requested table. Supply `grid_shape` and `spacing` "
        "explicitly."
    )


def _rasterize_spatialdata_lazy(
    sd_obj: Any,
    *,
    grid_shape: tuple[int, int],
    table_key: str | None,
    gene_names: Sequence[str],
) -> np.ndarray:
    """Project a SpatialData table onto a regular ``(ny, nx)`` grid, one gene
    at a time if the source is sparse. Returns ``(n_genes, ny, nx)``.

    This is a minimal implementation: spots / cells are nearest-neighbor-
    binned onto the target grid by rescaling their ``obsm['spatial']``
    coordinates. It is designed so we never materialize the dense
    ``.X`` slab — only one gene column is ever dense in memory.
    """
    table = _sd_get_table(sd_obj, table_key)
    if "spatial" not in table.obsm:
        raise ValueError("SpatialData table has no obsm['spatial']; cannot rasterize.")
    coords = np.asarray(table.obsm["spatial"], dtype=np.float64)
    ny, nx = grid_shape

    # Map coordinates to integer bin indices.
    y = coords[:, 0]
    x = coords[:, 1]
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    row = np.clip((y_norm * (ny - 1)).round().astype(int), 0, ny - 1)
    col = np.clip((x_norm * (nx - 1)).round().astype(int), 0, nx - 1)

    X = table.X
    is_sparse = sp.issparse(X)
    if is_sparse:
        X_csc = X.tocsc()

    out = np.zeros((len(gene_names), ny, nx), dtype=np.float64)
    name_to_col = {g: j for j, g in enumerate(table.var_names)}
    for gi, gname in enumerate(gene_names):
        if gname not in name_to_col:
            raise KeyError(f"gene '{gname}' not found in SpatialData table.")
        j = name_to_col[gname]
        if is_sparse:
            vals = np.asarray(X_csc[:, j].toarray(), dtype=np.float64).ravel()
        else:
            vals = np.asarray(X[:, j], dtype=np.float64).ravel()
        # Accumulate into the grid cell (sum); duplicates at the same bin are added.
        np.add.at(out[gi], (row, col), vals)
    return out


# ---------------------------------------------------------------------------
# Step 1 — per-sample spectra and radial binning
# ---------------------------------------------------------------------------


def compute_sample_spectrum(
    sample: np.ndarray,
    fft_solver: str = "rfft2",
    workers: int | None = None,
    center: str | None = "mean",
    return_dc: bool = False,
    zscore_clip: float | None = 6.0,
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
          (pure pattern shape). Sparse genes are **heavily down-weighted**
          by the guard described below.
        - ``None``: no centering; spectrum includes DC.
    return_dc : bool, default False
        If True, also return a ``(n_genes,)`` array of per-gene grid means (DC
        scalars of the *uncentered* signal).
    zscore_clip : float or None, default 6.0
        Only active when ``center='zscore'``. Two-part guard against
        sparse / near-constant genes producing extreme-outlier z-scores
        that dominate the pattern test:

        1. The per-gene std is floored at the median non-zero std across
           genes (robust, data-driven floor); genes with all-zero or
           numerically-zero std are effectively zeroed out. This prevents
           sparse genes with a single non-zero pixel from blowing up to
           arbitrarily large z-scores.
        2. After standardization, values are clipped to
           ``[-zscore_clip, +zscore_clip]``.

        Pass ``None`` to disable both guards (reproduces the pre-fix
        behavior). Integer values < 3 are not recommended — heavy
        clipping biases the spectrum towards low frequencies.

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
        sd = sample.std(axis=(1, 2))  # (n_genes,)
        if zscore_clip is not None:
            # Robust floor: median of the positive per-gene stds. Sparse / near
            # constant genes get floored at this typical scale, so the ratio
            # (value - mean) / std does not explode.
            positive = sd[sd > 0]
            floor = float(np.median(positive)) if positive.size else 1.0
            sd_safe = np.maximum(sd, floor * 0.1)
        else:
            sd_safe = np.clip(sd, 1e-12, None)
        work = (sample - dc[:, None, None]) / sd_safe[:, None, None]
        if zscore_clip is not None:
            np.clip(work, -float(zscore_clip), float(zscore_clip), out=work)
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


def _build_landmark_polar_stack(
    spectra: np.ndarray,
    grid_shape: tuple[int, int],
    fft_solver: str,
    n_theta: int,
    n_radius: int,
) -> np.ndarray:
    """Build a ``(n_landmarks, n_theta, n_radius)`` polar stack for one sample.

    Each landmark's 2D spectrum is fftshifted (DC at centre), resampled onto
    the polar grid, and zero-meaned along theta so the DC angular component
    doesn't dominate the cross-correlation.
    """
    full = _to_full_2d(spectra, grid_shape, fft_solver)  # (n_landmarks, ny, nx)
    shifted = np.fft.fftshift(full, axes=(-2, -1))
    out = np.empty((shifted.shape[0], n_theta, n_radius), dtype=float)
    for j in range(shifted.shape[0]):
        polar = _polar_resample(shifted[j], n_theta, n_radius)
        out[j] = polar - polar.mean(axis=0, keepdims=True)
    return out


def estimate_rotations_from_landmarks(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = 180,
    n_radius: int = 64,
    progress: bool = False,
) -> np.ndarray:
    """
    Estimate the per-sample rotation that best aligns every landmark
    spectrum to the reference sample's corresponding landmark.

    For each non-reference sample the routine picks a single rotation angle
    that maximises the **sum over landmarks** of the per-landmark circular
    cross-correlation along the polar-angle axis — i.e. each landmark
    aligns to its same-index counterpart in the reference (not to a mean
    template). This is strictly stronger than mean-template alignment
    because it ignores cross-landmark noise (the off-diagonal ``i ≠ j``
    terms that mean-of-means picks up) and picks up anisotropy shared
    across every landmark at a common orientation.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra. Shape ``(n_landmarks, ny, n_kx)`` with
        ``(ny, n_kx)`` following ``fft_solver``. The first dimension
        (``n_landmarks``) must match across samples — landmark ``j`` in
        sample A is compared to landmark ``j`` in sample B.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``landmark_spectra`` — rfft2 spectra are expanded
        to full 2D before resampling to preserve angular content.
    reference_index : int, default 0
        Which sample's landmarks act as the rotation reference (its angle
        is fixed at 0).
    n_theta : int, default 180
        Angular resolution of the polar resampling. Recovered angles are
        accurate to ``180 / n_theta`` degrees.
    n_radius : int, default 64
        Radial resolution of the polar resampling.
    progress : bool, default False
        If True, show a tqdm bar over non-reference samples.

    Returns
    -------
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is exactly 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range or any two samples have
        inconsistent ``n_landmarks``.
    """
    n_samples = len(landmark_spectra)
    if reference_index < 0 or reference_index >= n_samples:
        raise ValueError(f"reference_index {reference_index} out of range [0, {n_samples})")

    n_landmarks = landmark_spectra[reference_index].shape[0]
    for i, s in enumerate(landmark_spectra):
        if s.shape[0] != n_landmarks:
            raise ValueError(
                f"landmark_spectra[{i}] has n_landmarks={s.shape[0]}, "
                f"expected {n_landmarks} (must match across samples)."
            )

    ref_polar = _build_landmark_polar_stack(
        landmark_spectra[reference_index],
        grid_shapes[reference_index],
        fft_solver,
        n_theta,
        n_radius,
    )
    ref_hat = np.fft.fft(ref_polar, axis=1)  # (n_landmarks, n_theta, n_radius)

    angles = np.zeros(n_samples)
    iter_samples: Any = range(n_samples)
    if progress:
        iter_samples = tqdm(iter_samples, total=n_samples, desc="Rotation estimation")
    for i in iter_samples:
        if i == reference_index:
            continue
        cur_polar = _build_landmark_polar_stack(
            landmark_spectra[i], grid_shapes[i], fft_solver, n_theta, n_radius
        )
        cur_hat = np.fft.fft(cur_polar, axis=1)
        # Per-landmark circular cross-correlation along theta; sum across
        # landmarks AND radii → best rotation common to every landmark.
        corr = np.real(np.fft.ifft(ref_hat * np.conj(cur_hat), axis=1))
        total = corr.sum(axis=(0, 2))  # (n_theta,)
        k_best = int(np.argmax(total))
        angles[i] = k_best * 180.0 / n_theta
    return angles


def apply_rotations_to_spectra(
    spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    angles_deg: np.ndarray,
    *,
    fft_solver: str = "fft2",
    progress: bool = False,
) -> list[np.ndarray]:
    """
    Rotate each sample's 2D power spectra by a per-sample angle.

    Parameters
    ----------
    spectra : sequence of np.ndarray
        Per-sample 2D power spectra — any first-axis dimension (e.g. full
        ``n_genes``). Shape ``(n, ny, n_kx)`` with ``(ny, n_kx)`` matching
        ``fft_solver``.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    angles_deg : np.ndarray
        Per-sample rotation angles in degrees (e.g. produced by
        :func:`estimate_rotations_from_landmarks`). Length must equal
        ``len(spectra)``.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of ``spectra``.
    progress : bool, default False
        Show a tqdm bar across samples.

    Returns
    -------
    rotated : list of np.ndarray
        Per-sample rotated spectra with the same shape as the input.

    Notes
    -----
    Rotation is done on the **2D power spectrum** (fftshifted so DC sits at
    the centre), not back on the spatial image. That is enough for any
    downstream analysis that operates on aligned spectra (radial or 2D-bin
    tests). Samples whose angle is exactly 0 are passed through as-is.
    """
    if len(angles_deg) != len(spectra):
        raise ValueError(
            f"angles_deg length {len(angles_deg)} does not match spectra length {len(spectra)}."
        )
    if len(grid_shapes) != len(spectra):
        raise ValueError(
            f"grid_shapes length {len(grid_shapes)} does not match spectra length {len(spectra)}."
        )
    out: list[np.ndarray] = []
    # strict=False: lengths are already verified above.
    iter_samples: Any = enumerate(zip(spectra, grid_shapes, strict=False))
    if progress:
        iter_samples = tqdm(iter_samples, total=len(spectra), desc="Rotation application")
    for i, (spec_i, shape) in iter_samples:
        angle_deg = float(angles_deg[i])
        if angle_deg == 0.0:
            out.append(np.asarray(spec_i).copy())
            continue
        full = _to_full_2d(spec_i, shape, fft_solver)  # (n, ny, nx)
        full_shift = np.fft.fftshift(full, axes=(-2, -1))
        rot = scipy.ndimage.rotate(
            full_shift, angle=-angle_deg, axes=(-2, -1), reshape=False, order=1, mode="reflect"
        )
        rot = np.fft.ifftshift(rot, axes=(-2, -1))
        if fft_solver == "rfft2":
            ny, nx = shape
            half = nx // 2 + 1
            rot = rot[..., :half]
        out.append(rot)
    return out


def align_spectra_by_rotation(
    landmark_spectra: Sequence[np.ndarray],
    grid_shapes: Sequence[tuple[int, int]],
    *,
    target_spectra: Sequence[np.ndarray] | None = None,
    fft_solver: str = "fft2",
    reference_index: int = 0,
    n_theta: int = 180,
    n_radius: int = 64,
    progress: bool = False,
) -> tuple[list[np.ndarray] | None, np.ndarray]:
    """
    Two-step rotation alignment: estimate per-sample rotations from
    **landmark** spectra (whose first dimension must match across samples),
    then apply those rotations to a separate set of **target** spectra (the
    full gene panel for each sample, typically a superset of the
    landmarks).

    This is a convenience wrapper around
    :func:`estimate_rotations_from_landmarks` and
    :func:`apply_rotations_to_spectra`. Calling those directly is the
    right pattern when you want to inspect / cache the per-sample angles
    before applying them.

    Implementation
    --------------
    For every non-reference sample:

    1. Expand each landmark's 2D power spectrum to full-fft2 layout,
       fftshift so DC sits at the centre, and resample onto a polar
       ``(n_theta, n_radius)`` grid.
    2. Compute per-landmark circular cross-correlation along the
       polar-angle axis against the reference sample's same-index
       landmark. **Every landmark contributes its own cross-correlation**
       and the per-sample rotation is the angle that maximises the sum
       across landmarks (and across radii). Mean-template alignment —
       what the previous implementation did — was strictly weaker
       because the off-diagonal ``i ≠ j`` pair terms in
       ``corr(mean(a), mean(b))`` are pure noise.
    3. Rotate every entry of ``target_spectra[i]`` (if supplied) by the
       recovered angle.

    Parameters
    ----------
    landmark_spectra : sequence of np.ndarray
        Per-sample landmark spectra, shape ``(n_landmarks, ny, n_kx)``
        per sample. ``n_landmarks`` must match across samples.
    grid_shapes : sequence of tuple[int, int]
        Per-sample ``(ny, nx)`` of the original rasterized image.
    target_spectra : sequence of np.ndarray, optional
        Per-sample spectra to which the recovered rotations are applied.
        Any first-axis dimension (e.g. full gene panel). If ``None``, only
        the angles are returned.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT layout of both inputs. ``fft2`` is recommended so the full
        angular content is present.
    reference_index : int, default 0
    n_theta : int, default 180
    n_radius : int, default 64
    progress : bool, default False

    Returns
    -------
    rotated : list of np.ndarray or None
        Per-sample rotated target spectra (or ``None`` when
        ``target_spectra`` is omitted).
    angles_deg : np.ndarray
        ``(n_samples,)`` recovered rotation angles in degrees. Reference
        angle is 0.

    Raises
    ------
    ValueError
        If ``reference_index`` is out of range, if ``landmark_spectra``
        samples disagree on ``n_landmarks``, or if
        ``target_spectra`` length does not match.
    """
    angles = estimate_rotations_from_landmarks(
        landmark_spectra,
        grid_shapes,
        fft_solver=fft_solver,
        reference_index=reference_index,
        n_theta=n_theta,
        n_radius=n_radius,
        progress=progress,
    )
    if target_spectra is None:
        return None, angles
    if len(target_spectra) != len(landmark_spectra):
        raise ValueError(
            f"target_spectra length {len(target_spectra)} does not match "
            f"landmark_spectra length {len(landmark_spectra)}."
        )
    rotated = apply_rotations_to_spectra(
        target_spectra,
        grid_shapes,
        angles,
        fft_solver=fft_solver,
        progress=progress,
    )
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

    Notes
    -----
    **Equivalence with a per-sample one-hot covariate.** When stacking all
    ``(sample, gene)`` log-spectra as rows and regressing each frequency bin
    against a one-hot *sample-ID* indicator (i.e., a fixed sample effect
    per bin), the residuals are, by construction, the per-sample demeaned
    log-spectra — which is exactly what this function computes in log-space.
    That is: running this function sample-by-sample is mathematically
    identical to fitting a one-hot-sample covariate in log-space and
    residualizing. This is why a separate "residualize against
    per-sample one-hot" step is unnecessary — it is already covered here.

    :func:`residualize_against_covariates` is the complementary step for
    non-trivial covariates whose shape across frequency bins is not
    constant (cell-type proportion spectra, tissue-domain maps, etc.).
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
    Normalize spectra to sum-1 along ``axis`` (probability-vector shape).

    Divides each slice along ``axis`` by its own L1 norm so the resulting
    slice is a proper probability distribution over frequency bins:
    ``out = spectra / spectra.sum(axis)``. Rows that differ only by a
    positive scalar (the fingerprint of a gene expressed in one group and
    absent in another) become identical — only the **shape** of the
    power-vs-frequency curve survives.

    This is the natural companion to :func:`normalize_by_background`:
    background normalization cancels per-sample gain across genes;
    :func:`shape_normalize` cancels per-(sample, gene) magnitude across
    frequencies. Composed, they leave a pure, unit-sum radial pattern
    signature that is directly comparable as a distribution (so e.g.
    Jensen-Shannon / total-variation distances are well-defined).

    Parameters
    ----------
    spectra : np.ndarray
        Non-negative radial spectra. Any leading dimensions are preserved;
        normalization acts along ``axis`` only.
    axis : int, default -1
        Axis along which to enforce the sum-1 constraint (typically the K /
        frequency-bin axis).
    eps : float, default 1e-12
        Floor added to the denominator to avoid division-by-zero when an
        entire slice is numerically zero.

    Returns
    -------
    np.ndarray
        Shape-normalized spectra, same shape as the input, summing to 1 along
        ``axis``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    >>> out = shape_normalize(x, axis=-1)
    >>> np.allclose(out.sum(axis=-1), 1.0)
    True
    >>> np.allclose(out[0], out[1])  # only the shape survives
    True
    """
    total = spectra.sum(axis=axis, keepdims=True)
    return spectra / (total + eps)


# ---------------------------------------------------------------------------
# Step 4 — test statistics
# ---------------------------------------------------------------------------


def _resolve_freq_weights(freq_weights: np.ndarray | None, K: int) -> np.ndarray:
    """Validate / normalize frequency-bin weights; return a length-``K`` array summing to 1.

    Passing None yields uniform weights ``1/K`` — recovering the unweighted
    statistic. Any other input is cast to ``float``, required to be
    non-negative and not all-zero, and rescaled to sum-1. Non-uniform
    weights are how users express a kernel-like frequency preference (e.g.,
    low-pass polynomial vs exponential decay) inside the spectral distance.
    """
    if freq_weights is None:
        return np.full(K, 1.0 / K)
    w = np.asarray(freq_weights, dtype=float).ravel()
    if w.shape != (K,):
        raise ValueError(f"freq_weights must have length K={K}, got shape {w.shape}.")
    if np.any(w < 0):
        raise ValueError("freq_weights must be non-negative.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("freq_weights must not sum to zero.")
    return w / total


def _stat_log_l2(
    group_a: np.ndarray,
    group_b: np.ndarray,
    freq_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted L2 distance between mean log-spectra. Vectorized over genes.

    The (default) uniform-weight case reduces to the plain L2 distance on
    ``K`` frequency bins — up to an overall ``1/sqrt(K)`` scale that is
    irrelevant under a permutation null. Non-uniform weights (which must be
    non-negative and sum to 1) let the user emphasize low or high
    frequencies the same way a kernel spectrum does (polynomial vs
    exponential decay, etc.).
    """
    eps = 1e-12
    log_a = np.log(np.maximum(group_a, eps)).mean(axis=0)  # (n_genes, K)
    log_b = np.log(np.maximum(group_b, eps)).mean(axis=0)
    diff = log_a - log_b  # (n_genes, K)
    K = diff.shape[-1]
    w = _resolve_freq_weights(freq_weights, K)
    return np.sqrt(np.sum(w * diff**2, axis=-1))


def _welch_t_per_bin(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Per-bin Welch t-statistic (signed). Shape ``(n_genes, K)``."""
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1) + 1e-30)
    return (mean_a - mean_b) / se


def _stat_welch_abs_per_bin(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Per-bin ``|Welch t|``. Shape ``(n_genes, K)`` — the raw statistic for
    the Cauchy-combined pattern test."""
    return np.abs(_welch_t_per_bin(group_a, group_b))


def _stat_welch_p_per_bin(
    group_a: np.ndarray, group_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic two-sided Welch t-test per bin. Returns ``(|t|, p)`` each
    of shape ``(n_genes, K)``.

    Using the t-distribution instead of a permutation null is what makes the
    Cauchy-combined pattern test actually powerful — permutation p-values
    are floored at ``1/(n_perm + 1)`` per bin, which would also floor the
    Cauchy-combined gene-level p-value (e.g. ``n_perm=500`` caps it at
    ~1e-3, so no gene can survive BH correction across thousands of tests).
    The t-distribution tail delivers arbitrarily small p-values for strong
    per-bin signal, which is exactly what the Cauchy combination was
    designed for.
    """
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    mean_a = group_a.mean(axis=0)
    mean_b = group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros_like(mean_a)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros_like(mean_b)
    se2_a = var_a / max(n_a, 1)
    se2_b = var_b / max(n_b, 1)
    se2 = se2_a + se2_b + 1e-30
    t_stat = (mean_a - mean_b) / np.sqrt(se2)
    # Welch-Satterthwaite degrees of freedom. Fall back to equal-variance df
    # only when one group has n<2 (so the Welch formula is undefined).
    if n_a > 1 and n_b > 1:
        df_num = se2**2
        df_den = (se2_a**2) / max(n_a - 1, 1) + (se2_b**2) / max(n_b - 1, 1) + 1e-30
        df = df_num / df_den
    else:
        df = np.full_like(mean_a, float(max(n_a + n_b - 2, 1)))
    df = np.maximum(df, 1.0)
    pvals = 2.0 * _t_dist.sf(np.abs(t_stat), df)
    # Clip the absolute floor to the smallest representable positive float so
    # Cauchy's tan(pi(0.5 - p)) stays finite.
    return np.abs(t_stat), np.clip(pvals, np.finfo(float).tiny, 1.0)


def _cauchy_combine(pvals: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Cauchy combination test (Liu & Xie 2020).

    For p-values :math:`p_1, \\dots, p_K`, forms
    :math:`T = \\frac{1}{K}\\sum_k \\tan(\\pi\\,(0.5 - p_k))` and returns
    the analytic tail probability under the standard Cauchy null,
    :math:`p = 0.5 - \\arctan(T) / \\pi`. Robust to arbitrary dependence
    between the input p-values — that is the whole point of Cauchy
    combination — so it is safe to apply over correlated frequency bins
    without decorrelating them first.

    Parameters
    ----------
    pvals : np.ndarray
        Input p-values in ``[0, 1]``. Values at the exact endpoints are
        clipped away from them to keep :math:`\\tan` finite.
    axis : int, default -1
        Axis along which to combine.

    Returns
    -------
    np.ndarray
        Combined p-value(s); one less axis than ``pvals``.
    """
    eps = np.finfo(float).eps
    clipped = np.clip(pvals, eps, 1.0 - eps)
    T = np.mean(np.tan(np.pi * (0.5 - clipped)), axis=axis)
    return 0.5 - np.arctan(T) / np.pi


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
    "hotelling_lw": _stat_hotelling_lw,
    "mmd_rbf": _stat_mmd_rbf,
}

# `cauchy_welch` lives outside _STAT_FNS because it returns a ``(n_genes, K)``
# per-bin array (not a per-gene scalar) and needs a bespoke runner that turns
# per-bin permutation p-values into a Cauchy-combined gene-level p-value.


# ---------------------------------------------------------------------------
# Step 4b — permutation engine
# ---------------------------------------------------------------------------


def _permutation_indices(
    n_samples: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``(n_perm, n_samples)`` index arrays — random permutations of 0..n-1.

    Retained for back-compatibility; new code should prefer
    :func:`_exchangeable_group_labels`, which returns group-label matrices
    directly and supports the exact-enumeration path for small samples.
    """
    out = np.tile(np.arange(n_samples), (n_perm, 1))
    for i in range(n_perm):
        rng.shuffle(out[i])
    return out


def _exchangeable_group_labels(
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    n_perm_max: int = 10000,
) -> tuple[np.ndarray, bool]:
    """Build a null-distribution set of group relabellings.

    For small samples the total number of distinct two-group label
    assignments (``C(n, n_a)``) can be tiny compared to the user's
    requested ``n_perm``, which means the permutation p-value is floored
    at ``1/(C(n, n_a) + 1)``. In that regime an **exact** enumeration
    of every possible relabelling is both cheaper and strictly more
    accurate (zero Monte-Carlo noise, sharp p-values).

    Parameters
    ----------
    groups : np.ndarray
        Observed group labels, length ``n_samples`` with exactly two
        unique values.
    n_perm : int
        Number of random shuffles to produce when exact enumeration is
        infeasible. Ignored on the exact path.
    rng : np.random.Generator
        RNG for the sampling fallback.
    n_perm_max : int, default 10000
        If ``C(n_samples, n_a)`` is at most this, every distinct relabelling
        is enumerated (``is_exact=True``) and ``n_perm`` is overridden to
        the enumeration count. Otherwise ``n_perm`` random shuffles of
        ``groups`` are returned (``is_exact=False``).

    Returns
    -------
    perm_labels : np.ndarray
        ``(n_used, n_samples)`` int array; each row is a valid relabelling
        (same ``n_a`` / ``n_b`` marginals as ``groups``).
    is_exact : bool
        True if every row is a distinct relabelling and together they
        span every possible partition; False if the rows are independent
        random shuffles.
    """
    groups = np.asarray(groups)
    n = len(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if uniq.size != 2:
        raise ValueError(f"groups must have exactly two unique values, got {uniq}.")
    n_a = int(counts[0])
    total = int(math.comb(n, n_a))
    if total <= n_perm_max:
        perm_labels = np.empty((total, n), dtype=groups.dtype)
        a_val, b_val = uniq[0], uniq[1]
        for i, subset in enumerate(itertools.combinations(range(n), n_a)):
            perm_labels[i] = b_val
            perm_labels[i, list(subset)] = a_val
        return perm_labels, True
    perm_labels = np.empty((n_perm, n), dtype=groups.dtype)
    base = groups.copy()
    for i in range(n_perm):
        rng.shuffle(base)
        perm_labels[i] = base
    return perm_labels, False


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
    perm_labels: np.ndarray,
    *,
    freq_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed statistic + null distribution for one statistic. Internal.

    ``perm_labels`` is a ``(n_perm_used, n_samples)`` matrix of group
    relabellings (as produced by :func:`_exchangeable_group_labels`).

    ``freq_weights`` is forwarded only to statistics that accept it (currently
    ``log_l2``); other statistics ignore it.
    """
    fn = _STAT_FNS[stat_name]
    uniq = np.unique(groups)
    a_val = uniq[0]
    a_mask = groups == a_val

    def _call(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if stat_name == "log_l2":
            return fn(a, b, freq_weights=freq_weights)
        return fn(a, b)

    observed = _call(spectra[a_mask], spectra[~a_mask])
    n_perm = perm_labels.shape[0]
    null = np.empty((n_perm, spectra.shape[1]))
    for p in range(n_perm):
        a = perm_labels[p] == a_val
        null[p] = _call(spectra[a], spectra[~a])
    return observed, null


def _run_cauchy_welch_analytic(
    spectra: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin Welch t test + Cauchy-combined gene-level p-value.

    Both the per-bin significance and the gene-level combination are
    **analytic**: per-bin p-values come from the Welch t-distribution (not
    a permutation null) and the gene-level p comes from the Cauchy
    combination (Liu & Xie 2020), which is valid under arbitrary
    dependence between bins. This is what gives the Cauchy-Welch test
    real power versus the other (permutation-based) statistics in this
    module — permutation p-values are floored at ``1/(n_perm + 1)`` per
    bin, which would cap the combined gene-level p at ~1e-3 for typical
    ``n_perm=500`` and wipe out BH-FDR significance across thousands of
    genes.

    Returns
    -------
    observed_abs_t : np.ndarray
        ``(n_genes, K)`` observed per-bin ``|t|`` — used as the reported
        statistic summary (the max across bins sorts the output table
        sensibly, same convention as before).
    combined_pvals : np.ndarray
        ``(n_genes,)`` Cauchy-combined gene-level p-values built from per-bin
        analytic Welch p-values.
    per_bin_pvals : np.ndarray
        ``(n_genes, K)`` per-bin analytic Welch two-sided p-values.
    """
    a_mask = groups == 0
    abs_t, per_bin_pvals = _stat_welch_p_per_bin(spectra[a_mask], spectra[~a_mask])
    combined = _cauchy_combine(per_bin_pvals, axis=-1)
    return abs_t, combined, per_bin_pvals


# ---------------------------------------------------------------------------
# Step 4c — public test functions
# ---------------------------------------------------------------------------


def compare_two_groups(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    n_perm: int = 1000,
    random_state: int | None = None,
    n_jobs: int = 1,
    freq_weights: np.ndarray | None = None,
    n_perm_max: int = 10000,
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
    statistic : {'log_l2', 'hotelling_lw', 'mmd_rbf', 'cauchy_welch'}, default 'log_l2'
        Test statistic:

        - ``'log_l2'`` — (optionally weighted) L2 distance between mean
          log-spectra. Global / summary statistic.
        - ``'hotelling_lw'`` — regularized Hotelling :math:`T^2`.
        - ``'mmd_rbf'`` — RBF-kernel maximum mean discrepancy.
        - ``'cauchy_welch'`` — per-bin Welch two-sided t-test with
          **analytic** (t-distribution) p-values combined across bins via
          Cauchy's combination (Liu & Xie 2020). Analytic is the whole
          point: permutation p-values would floor at ``1/(n_perm + 1)``
          per bin, which would also floor the gene-level combined
          p-value and destroy BH-FDR power across thousands of genes.
          Yields an extra ``P_value_per_bin`` column.
    n_perm : int, default 1000
        Number of label permutations for the null distribution. **Ignored**
        when ``statistic='cauchy_welch'`` (that path is fully analytic).
    random_state : int, optional
        Seed for the permutation RNG (ignored for ``'cauchy_welch'``).
    n_jobs : int, default 1
        Reserved for future parallelism over genes; currently unused (the per-stat
        implementations are already vectorized over genes).
    freq_weights : np.ndarray, optional
        Only used by ``statistic='log_l2'``. Non-negative weights of length
        ``K`` (the number of frequency bins); internally renormalized to
        sum-1. Lets the user emphasize specific frequencies — e.g., a
        polynomial low-pass shape to mirror a CAR kernel, or an exponential
        high-pass shape to mirror a Gaussian kernel. ``None`` (default)
        means uniform weights.
    n_perm_max : int, default 10000
        If the total number of distinct two-group relabellings
        ``C(n_samples, n_a)`` is at most this, every possible relabelling
        is enumerated (**exact permutation test**) and ``n_perm`` is
        overridden to the enumeration count. This is both faster and
        strictly more accurate than sampling in the small-sample regime
        (e.g. 6-vs-6 → 924 partitions, 5-vs-5 → 252). Above the threshold
        the test falls back to ``n_perm`` random shuffles.

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``
        (BH-FDR), sorted by descending statistic. When
        ``statistic='cauchy_welch'``, the frame also carries a
        ``P_value_per_bin`` object column — each entry is an
        ``(K,)`` numpy array of per-bin permutation p-values for that gene.

    Raises
    ------
    ValueError
        If ``statistic`` is unknown, ``groups`` does not contain exactly two values,
        or shapes are inconsistent.
    """
    _available = set(_STAT_FNS) | {"cauchy_welch"}
    if statistic not in _available:
        raise ValueError(f"Unknown statistic '{statistic}'. Options: {sorted(_available)}.")
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

    if statistic == "cauchy_welch":
        if freq_weights is not None:
            logger.debug("freq_weights is ignored by statistic='cauchy_welch'.")
        observed, combined_p, per_bin_p = _run_cauchy_welch_analytic(spectra, g_int)
        summary_stat = observed.max(axis=-1)  # reportable scalar per gene
        if gene_names is None:
            gene_names = [str(i) for i in range(n_genes)]
        df = pd.DataFrame(
            {
                "Feature": list(gene_names),
                "Statistic": summary_stat,
                "P_value": combined_p,
                "P_value_per_bin": list(per_bin_p),
            }
        )
        _apply_bh_correction(df)
        df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
        if n_jobs != 1:  # noqa: PLR2004
            logger.debug("n_jobs ignored: per-statistic implementations are already vectorized.")
        return df

    perm_labels, is_exact = _exchangeable_group_labels(g_int, n_perm, rng, n_perm_max=n_perm_max)
    if is_exact:
        logger.info(
            "Exact permutation test: enumerated %d distinct relabellings " "(C(%d, %d)).",
            perm_labels.shape[0],
            n_samples,
            int((g_int == 0).sum()),
        )
    observed, null = _run_statistic_with_perm(
        statistic, spectra, g_int, perm_labels, freq_weights=freq_weights
    )
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
    n_perm_max: int = 10000,
) -> dict[str, pd.DataFrame]:
    """
    Run several statistics on the same data with a **shared** permutation null.

    All statistics use the same ``perm_indices``, so per-gene p-values are directly
    comparable (same Monte-Carlo noise, same exchanges).

    Parameters
    ----------
    spectra, groups, gene_names, n_perm, random_state
        Same meaning as :func:`compare_two_groups`.
    statistics : sequence of str, default ``_AVAILABLE_STATISTICS``
        Subset of the implemented statistics (``'log_l2'``,
        ``'hotelling_lw'``, ``'mmd_rbf'``, ``'cauchy_welch'``).

    Returns
    -------
    dict
        Mapping ``stat_name -> DataFrame`` (each DataFrame as in
        :func:`compare_two_groups`; ``'cauchy_welch'`` carries the extra
        ``P_value_per_bin`` column).

    Raises
    ------
    ValueError
        If any statistic name is unknown or input shapes are inconsistent.
    """
    _available = set(_STAT_FNS) | {"cauchy_welch"}
    for s in statistics:
        if s not in _available:
            raise ValueError(f"Unknown statistic '{s}'. Options: {sorted(_available)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D (n_samples, n_genes, K), got {spectra.shape}.")
    n_samples, n_genes, _ = spectra.shape
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError(f"groups must contain exactly two distinct values, got {uniq}.")
    g_int = (groups == uniq[1]).astype(int)

    rng = np.random.default_rng(random_state)
    perm_labels, is_exact = _exchangeable_group_labels(g_int, n_perm, rng, n_perm_max=n_perm_max)
    if is_exact:
        logger.info(
            "Exact permutation test: enumerated %d distinct relabellings.",
            perm_labels.shape[0],
        )
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    out: dict[str, pd.DataFrame] = {}
    for s in statistics:
        if s == "cauchy_welch":
            observed, combined_p, per_bin_p = _run_cauchy_welch_analytic(spectra, g_int)
            summary = observed.max(axis=-1)
            df = pd.DataFrame(
                {
                    "Feature": list(gene_names),
                    "Statistic": summary,
                    "P_value": combined_p,
                    "P_value_per_bin": list(per_bin_p),
                }
            )
        else:
            observed, null = _run_statistic_with_perm(s, spectra, g_int, perm_labels)
            pvals = _permutation_pvalue(observed, null)
            df = pd.DataFrame(
                {"Feature": list(gene_names), "Statistic": observed, "P_value": pvals}
            )
        _apply_bh_correction(df)
        df = df.sort_values("Statistic", ascending=False).reset_index(drop=True)
        out[s] = df
    return out


# ---------------------------------------------------------------------------
# Step 4d — scalar (DE-style) two-group test
# ---------------------------------------------------------------------------


def compare_two_groups_masked(  # noqa: C901
    spectra: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    gene_names: Sequence[str] | None = None,
    statistic: str = "log_l2",
    n_perm: int = 1000,
    random_state: int | None = None,
    min_samples_per_group: int = 2,
    freq_weights: np.ndarray | None = None,
    n_perm_max: int = 10000,
) -> pd.DataFrame:
    """
    Per-gene two-group pattern test with **incomplete data** across samples.

    For each gene, only the subset of samples with ``presence[:, g] == True``
    contributes to the observed statistic and to the label-permutation null.
    Genes that fail to reach ``min_samples_per_group`` observations in at
    least one group are reported with ``NaN`` p-values and the number of
    observed samples per group, so the user sees why they were skipped.

    Parameters
    ----------
    spectra : np.ndarray
        ``(n_samples, n_genes, K)``.
    groups : np.ndarray
        ``(n_samples,)``, exactly two distinct labels.
    presence : np.ndarray
        ``(n_samples, n_genes)`` boolean mask. ``True`` = gene is observed
        in that sample (contributes); ``False`` = gene is absent (ignored).
    gene_names : sequence of str, optional
    statistic : {'log_l2', 'hotelling_lw', 'mmd_rbf', 'cauchy_welch'}, default 'log_l2'
    n_perm : int, default 1000
    random_state : int, optional
    min_samples_per_group : int, default 2
        Minimum observed samples in each group for the gene to be tested.
    freq_weights : np.ndarray, optional
        Only consumed by ``statistic='log_l2'`` (same semantics as
        :func:`compare_two_groups`).

    Returns
    -------
    pd.DataFrame
        Columns ``Feature``, ``Statistic``, ``P_value``, ``P_adj``,
        ``n_obs_A``, ``n_obs_B``. For ``'cauchy_welch'`` a
        ``P_value_per_bin`` column is also included (``None`` for skipped
        genes). BH-FDR is computed only over tested genes.
    """
    _available = set(_STAT_FNS) | {"cauchy_welch"}
    if statistic not in _available:
        raise ValueError(f"Unknown statistic '{statistic}'. Options: {sorted(_available)}.")
    if spectra.ndim != 3:
        raise ValueError(f"spectra must be 3D, got {spectra.shape}.")
    n_samples, n_genes, K = spectra.shape
    if presence.shape != (n_samples, n_genes):
        raise ValueError(
            f"presence shape {presence.shape} != (n_samples, n_genes) = "
            f"({n_samples}, {n_genes})."
        )
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if uniq.size != 2:
        raise ValueError("groups must contain exactly two distinct values.")
    g_int = (groups == uniq[1]).astype(int)
    rng = np.random.default_rng(random_state)

    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]

    rows: list[dict[str, Any]] = []
    for g in range(n_genes):
        mask = presence[:, g]
        ga = g_int[mask] == 0
        gb = g_int[mask] == 1
        n_a, n_b = int(ga.sum()), int(gb.sum())
        row: dict[str, Any] = {
            "Feature": gene_names[g],
            "n_obs_A": n_a,
            "n_obs_B": n_b,
            "Statistic": np.nan,
            "P_value": np.nan,
        }
        if statistic == "cauchy_welch":
            row["P_value_per_bin"] = None

        if n_a < min_samples_per_group or n_b < min_samples_per_group:
            rows.append(row)
            continue

        sub = spectra[mask, g : g + 1, :]  # (n_obs, 1, K)
        sub_groups = g_int[mask]

        if statistic == "cauchy_welch":
            observed, combined_p, per_bin_p = _run_cauchy_welch_analytic(sub, sub_groups)
            row["Statistic"] = float(observed.max())
            row["P_value"] = float(combined_p[0])
            row["P_value_per_bin"] = per_bin_p[0]
        else:
            # Per-gene exchange set — enumerate exactly when C(n_obs, n_a_obs)
            # is small, otherwise sample. Subsets are typically smaller than
            # the global one so the exact path kicks in more often here.
            perm_labels, _ = _exchangeable_group_labels(
                sub_groups, n_perm, rng, n_perm_max=n_perm_max
            )
            observed, null = _run_statistic_with_perm(
                statistic, sub, sub_groups, perm_labels, freq_weights=freq_weights
            )
            pval = _permutation_pvalue(observed, null)
            row["Statistic"] = float(observed[0])
            row["P_value"] = float(pval[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    # BH-correction over tested (non-NaN) genes only.
    tested = df["P_value"].notna()
    df["P_adj"] = np.nan
    if tested.any():
        sub_df = df.loc[tested, ["Feature", "P_value"]].copy()
        _apply_bh_correction(sub_df)
        df.loc[tested, "P_adj"] = sub_df["P_adj"].to_numpy()
    return df.sort_values("Statistic", ascending=False, na_position="last").reset_index(drop=True)


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
    n_perm_max: int = 10000,
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
    perm_labels, is_exact = _exchangeable_group_labels(g_int, n_perm, rng, n_perm_max=n_perm_max)
    if is_exact:
        logger.info(
            "Exact permutation test (DE): enumerated %d distinct relabellings.",
            perm_labels.shape[0],
        )
    observed = _welch_abs_t(values[g_int == 0], values[g_int == 1])
    mean_diff = values[g_int == 0].mean(axis=0) - values[g_int == 1].mean(axis=0)

    null = np.empty((perm_labels.shape[0], n_genes))
    for p in range(perm_labels.shape[0]):
        a = perm_labels[p] == 0
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

    Accepts a list of :class:`anndata.AnnData` (→ NUFFT backend, irregular
    spots) or a list of :class:`spatialdata.SpatialData` (→ FFT backend,
    rasterized grids) and chains per-sample spectra, radial binning,
    background normalization, optional residualization, and a
    permutation-based two-group test. Both backends honor the same
    public API — :meth:`fit`, :meth:`normalize_background`,
    :meth:`shape_normalize`, :meth:`residualize`, :meth:`test_pattern`,
    :meth:`test_expression`.

    Sparse ``adata.X`` / SpatialData-table expression matrices are
    **never fully densified**; the per-sample spectrum loop converts
    one gene column at a time to dense.

    Parameters
    ----------
    samples : list of :class:`anndata.AnnData` or list of :class:`spatialdata.SpatialData`
        All entries must be of the same concrete type (mixed lists raise
        :class:`TypeError`).
    groups : np.ndarray
        Group labels of length ``len(samples)`` with exactly two distinct values.
    gene_names : sequence of str, optional
        If None, inferred from the first sample; every other sample must then
        share the same ``var_names``.
    feature_mode : {'radial', '2d'}, default 'radial'
        Radial binning (rotation-invariant) vs flattened 2D spectrum with
        rotation alignment.
    n_radial_bins : int, default 30
        Number of radial bins in ``'radial'`` mode.
    fft_solver : {'fft2', 'rfft2'}, default 'rfft2'
        FFT solver used by the SpatialData / FFT backend.
    workers : int, optional
        FFT worker count.
    coordinates_key : str, default 'spatial'
        AnnData ``obsm`` key holding ``(n_obs, 2)`` coordinates (NUFFT backend).
    layer : str, optional
        AnnData layer to read instead of ``.X`` (NUFFT backend).
    table_key : str, optional
        Table name inside each :class:`~spatialdata.SpatialData` object.
        Required when a sample has more than one table.
    unit_scales : sequence of float, optional
        Per-sample multiplier converting raw coords into the common unit used
        by ``spacing`` (NUFFT backend). Default ``[1.0] * n_samples``.
    grid_shape, spacing : optional
        When supplied, used for every sample; otherwise each sample's grid is
        auto-inferred. Per-sample grids are fine — cross-sample alignment
        lives in physical-frequency space via radial binning.
    freq_edges : np.ndarray, optional
        Explicit radial-frequency bin edges.
    center : {'mean', 'zscore', None}, default 'mean'
        Pre-FFT centering. ``'mean'`` makes DC exactly zero so the pattern
        test and :meth:`test_expression` are orthogonal.
    eps : float, default 1e-6
        NUFFT tolerance (AnnData path only).

    Attributes
    ----------
    samples : list
        The input list, stored by reference.
    groups : np.ndarray
        Group labels.
    gene_names : list of str
        Resolved gene names (either passed in or inferred from sample 0).
    mode : {'nufft', 'fft'}
        Which backend this comparator is using — set at construction time
        based on the sample type.
    spectra_ : np.ndarray or None
        Per-sample feature matrix of shape ``(n_samples, n_genes, K)``. Set
        by :meth:`fit`; mutated in place by :meth:`normalize_background` /
        :meth:`residualize` / :meth:`shape_normalize`.
    dc_ : np.ndarray or None
        Per-sample per-gene DC scalars (grid means) of shape
        ``(n_samples, n_genes)``. Unaffected by the post-fit transforms on
        ``spectra_``.
    rotation_angles_ : np.ndarray or None
        Recovered rotation angles (degrees), set when
        ``feature_mode='2d'`` and :meth:`fit` has been called.

    Examples
    --------
    Irregular spots → NUFFT backend:

    >>> import anndata as ad
    >>> import numpy as np
    >>> from quadsv import SpectralComparator
    >>> rng = np.random.default_rng(0)
    >>> def mk(seed):
    ...     a = ad.AnnData(X=rng.standard_normal((200, 4)))
    ...     a.var_names = [f"g{i}" for i in range(4)]
    ...     a.obsm["spatial"] = rng.uniform(0, 20, size=(200, 2))
    ...     return a
    >>> samples = [mk(i) for i in range(4)]
    >>> groups = np.array([0, 0, 1, 1])
    >>> cmp = SpectralComparator(samples, groups).fit().normalize_background()
    >>> cmp.mode
    'nufft'
    """

    def __init__(  # noqa: C901
        self,
        samples: Sequence[Any],
        groups: np.ndarray,
        gene_names: Sequence[str] | None = None,
        *,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        fft_solver: str = "rfft2",
        workers: int | None = None,
        coordinates_key: str = "spatial",
        layer: str | None = None,
        table_key: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        center: str | None = "mean",
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        min_samples_per_group: int = 2,
        nufft_chunk_size: int = 64,
    ) -> None:
        """
        Build a comparator over a list of spatial-omics samples.

        Parameters
        ----------
        samples : list of :class:`anndata.AnnData` or list of :class:`spatialdata.SpatialData`
            Per-sample objects. All entries must be of the same concrete type:

            - list of ``AnnData`` → irregular-grid samples; the NUFFT backend
              is used. Each sample's ``obsm[coordinates_key]`` is treated as
              ``(y, x)`` spot coordinates; its ``.X`` (or ``.layers[layer]``
              if ``layer`` is given) provides the expression matrix. Sparse
              matrices are converted to dense **one gene column at a time**
              inside the per-sample spectrum loop — ``adata.X`` is never
              fully densified.
            - list of ``SpatialData`` → regular-grid samples; the FFT
              backend is used. Each sample is rasterized to
              ``(n_genes, ny, nx)`` via the :class:`SpatialData` table at
              ``table_key``; same per-gene densification rule applies.
        groups : np.ndarray
            Group labels of length ``n_samples`` with exactly two distinct values.
        gene_names : sequence of str, optional
            Gene names. If ``None``, inferred from the first sample's
            ``.var_names``; all samples must then share the same ``.var_names``.
        feature_mode, n_radial_bins, fft_solver, workers, freq_edges, center
            Same meaning as before (see class docstring).
        coordinates_key : str, default ``'spatial'``
            ``obsm`` key holding ``(n_obs, 2)`` coordinates (AnnData inputs only).
        layer : str, optional
            Layer key to read instead of ``.X`` (AnnData inputs only).
        table_key : str, optional
            Table name inside the :class:`SpatialData` object to use as the
            expression matrix. If None, the default table is used.
        unit_scales : sequence of float, optional
            Per-sample multiplier for AnnData coords (e.g., pixel → μm). Default
            ``[1.0] * n_samples``.
        grid_shape, spacing : optional
            When supplied, used for every sample; otherwise auto-inferred
            per sample (NUFFT auto-inference via
            :func:`quadsv.nufft._infer_grid_from_coords`). Different samples
            keep their own grid sizes; cross-sample alignment happens in
            physical-frequency space via radial binning with per-sample
            ``spacing``.
        eps : float, default 1e-6
            NUFFT tolerance (AnnData path only).
        presence_threshold : float, default 0.0
            Per-(sample, gene) minimum *fraction of non-zero spots* for the
            gene to count as "observed" in that sample. Genes that do not
            clear the threshold in a sample are marked absent in
            :attr:`presence_` and their spectrum rows are excluded from the
            pattern test on a per-gene basis. ``0.0`` (default) means every
            (sample, gene) pair counts as observed — the classical
            complete-data behavior. Typical relaxed values are
            ``0.01`` – ``0.05``.
        min_samples_per_group : int, default 2
            Minimum number of observed samples in *each* group for a gene
            to be tested by :meth:`test_pattern`. Genes below the threshold
            receive ``NaN`` p-values and a ``None`` ``P_value_per_bin``
            (when applicable); they remain listed so the user sees why
            they were skipped. :meth:`test_expression` always uses every
            sample regardless of ``presence_``, matching the user
            expectation that absence itself is informative for DE.
        nufft_chunk_size : int, default 64
            Number of genes to feed into a single batched ``finufft.nufft2d1``
            call on the NUFFT backend. The NUFFT per-sample loop densifies
            exactly this many columns of ``adata.X`` at a time, runs the
            batched transform, discards, and moves on — so RAM footprint
            peaks at ``O(N_spots · chunk · 16 B)`` per sample rather than
            the full ``(n_spots, n_genes)`` slab. Values in the 32–128 range
            balance finufft's per-call overhead against the memory cap.
        """
        if center not in ("mean", "zscore", None):
            raise ValueError(f"center must be 'mean', 'zscore', or None, got {center!r}.")
        if feature_mode not in ("radial", "2d"):
            raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
        if feature_mode == "2d" and fft_solver != "fft2":
            logger.info(
                "feature_mode='2d' works best with fft_solver='fft2'; switching automatically."
            )
            fft_solver = "fft2"

        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        n_samples = len(samples_list)

        groups = np.asarray(groups)
        if groups.shape != (n_samples,):
            raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
        if np.unique(groups).size != 2:
            raise ValueError("groups must contain exactly two distinct labels.")

        mode, resolved_gene_names = self._detect_mode_and_genes(
            samples_list, gene_names, layer=layer, table_key=table_key
        )

        self.center: str | None = center
        self.groups: np.ndarray = groups
        self.gene_names: list[str] = list(resolved_gene_names)
        self.feature_mode: str = feature_mode
        self.n_radial_bins: int = int(n_radial_bins)
        self.fft_solver: str = fft_solver
        self.workers: int | None = workers
        self.freq_edges: np.ndarray | None = (
            None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        )
        self.mode: str = mode
        # NUFFT output is always full-2D (fft2 layout, DC at [0,0]); only the
        # SpatialData / FFT path honours the user-chosen solver ('fft2' or
        # 'rfft2'). This internal field is what downstream helpers
        # (radial_bin_spectrum, align_spectra_by_rotation, covariate
        # rasterization) see — keeps the public `fft_solver` attribute as a
        # pure user knob.
        self._spectrum_fft_solver: str = "fft2" if mode == "nufft" else fft_solver
        self.samples: list[Any] = samples_list
        self._layer = layer
        self._table_key = table_key
        self._coordinates_key = coordinates_key
        self._nufft_eps = float(eps)
        self.presence_threshold: float = float(presence_threshold)
        self.min_samples_per_group: int = int(min_samples_per_group)
        self.nufft_chunk_size: int = int(nufft_chunk_size)
        if self.nufft_chunk_size < 1:
            raise ValueError(f"nufft_chunk_size must be >= 1, got {self.nufft_chunk_size}.")

        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError(
                f"presence_threshold must be in [0, 1], got {self.presence_threshold}."
            )
        if self.min_samples_per_group < 2:
            raise ValueError(
                f"min_samples_per_group must be >= 2, got {self.min_samples_per_group}."
            )

        # Populated in _prepare_* below.
        self.spectra_: np.ndarray | None = None
        self.dc_: np.ndarray | None = None
        self.presence_: np.ndarray | None = None
        """Per-sample, per-gene presence mask of shape ``(n_samples, n_genes)``.
        ``True`` = the gene clears ``presence_threshold`` in that sample and
        contributes to the pattern test; ``False`` = the gene is treated as
        incomplete for that sample (excluded from pattern test, still counted
        by :meth:`test_expression`). Populated by :meth:`fit`."""
        self.rotation_angles_: np.ndarray | None = None
        self._raw_2d_spectra: list[np.ndarray] | None = None
        self._grid_shapes: list[tuple[int, int]] = []
        self.spacings: list[tuple[float, float]] | None = None

        # NUFFT-only fields; unused in FFT mode.
        self._coords: list[np.ndarray] | None = None
        self._unit_scales: list[float] | None = None

        if self.mode == "nufft":
            self._prepare_nufft_inputs(
                unit_scales=unit_scales,
                grid_shape=grid_shape,
                spacing=spacing,
            )
        else:  # fft / SpatialData
            self._prepare_fft_inputs(grid_shape=grid_shape, spacing=spacing)

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_mode_and_genes(
        samples_list: list[Any],
        gene_names: Sequence[str] | None,
        *,
        layer: str | None,
        table_key: str | None,
    ) -> tuple[str, list[str]]:
        """Determine which backend to use and resolve gene_names from the first sample."""
        import anndata as _ad
        import spatialdata as _sd

        is_anndata = all(isinstance(s, _ad.AnnData) for s in samples_list)
        is_spatialdata = all(isinstance(s, _sd.SpatialData) for s in samples_list)
        if not (is_anndata or is_spatialdata):
            raise TypeError(
                "samples must be a list of all AnnData (→ NUFFT backend) or "
                "all SpatialData (→ FFT backend). Mixed lists and other "
                "types are not supported."
            )

        if is_anndata:
            first = samples_list[0]
            if gene_names is None:
                gene_names = list(first.var_names)
            # Validate consistent gene axis (names must all match).
            for i, s in enumerate(samples_list):
                if list(s.var_names) != list(gene_names):
                    raise ValueError(
                        f"sample {i} has var_names that do not match the reference "
                        "(all AnnData samples must share the same gene axis)."
                    )
                if layer is not None and layer not in s.layers:
                    raise KeyError(f"sample {i} is missing layer '{layer}'.")
            return "nufft", list(gene_names)

        # SpatialData path — gene names live on the requested table.
        first = samples_list[0]
        table = _sd_get_table(first, table_key)
        if gene_names is None:
            gene_names = list(table.var_names)
        for i, s in enumerate(samples_list):
            tbl = _sd_get_table(s, table_key)
            if list(tbl.var_names) != list(gene_names):
                raise ValueError(
                    f"sample {i}'s table has var_names that do not match the reference."
                )
        return "fft", list(gene_names)

    # ------------------------------------------------------------------
    def _prepare_nufft_inputs(
        self,
        *,
        unit_scales: Sequence[float] | None,
        grid_shape: tuple[int, int] | None,
        spacing: tuple[float, float] | None,
    ) -> None:
        """AnnData path: pull coords per sample, auto-infer per-sample
        (grid_shape, spacing) when unset, keep .X references for per-gene
        lazy densification inside the spectrum loop."""
        from quadsv.nufft import _infer_grid_from_coords

        n_samples = len(self.samples)
        if unit_scales is None:
            unit_scales = [1.0] * n_samples
        if len(unit_scales) != n_samples:
            raise ValueError(
                f"unit_scales length {len(unit_scales)} does not match n_samples={n_samples}."
            )
        self._unit_scales = [float(s) for s in unit_scales]

        coords_list: list[np.ndarray] = []
        grids: list[tuple[int, int]] = []
        spacings: list[tuple[float, float]] = []
        for i, ad_s in enumerate(self.samples):
            if self._coordinates_key not in ad_s.obsm:
                raise KeyError(
                    f"sample {i} has no obsm['{self._coordinates_key}']; "
                    f"available: {list(ad_s.obsm.keys())}."
                )
            c = np.asarray(ad_s.obsm[self._coordinates_key], dtype=np.float64)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(
                    f"sample {i} obsm['{self._coordinates_key}'] must be (N, 2), got {c.shape}."
                )
            coords_list.append(c)
            if grid_shape is not None and spacing is not None:
                gs_i = (int(grid_shape[0]), int(grid_shape[1]))
                sp_i = (float(spacing[0]), float(spacing[1]))
            else:
                gs_i, sp_i = _infer_grid_from_coords(c * self._unit_scales[i], oversample=2.0)
            grids.append(gs_i)
            spacings.append(sp_i)

        self._coords = coords_list
        self._grid_shapes = grids
        self.spacings = spacings

    # ------------------------------------------------------------------
    def _prepare_fft_inputs(
        self,
        *,
        grid_shape: tuple[int, int] | None,
        spacing: tuple[float, float] | None,
    ) -> None:
        """SpatialData path: lock in per-sample grid_shape / spacing. Actual
        rasterization happens lazily in :meth:`fit` to keep construction
        cheap and avoid holding dense arrays before necessary."""
        grids: list[tuple[int, int]] = []
        spacings: list[tuple[float, float]] = []
        for sd_s in self.samples:
            gs_i, sp_i = _sd_infer_grid_and_spacing(
                sd_s,
                table_key=self._table_key,
                override_shape=grid_shape,
                override_spacing=spacing,
            )
            grids.append(gs_i)
            spacings.append(sp_i)
        self._grid_shapes = grids
        self.spacings = spacings

    # ------------------------------------------------------------------
    def _compute_nufft_spectra(  # noqa: C901
        self,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """NUFFT per-sample spectrum pass. Densifies ``adata.X`` in chunks of
        :attr:`nufft_chunk_size` columns and runs one batched
        ``finufft.nufft2d1`` per chunk — the full ``.X`` slab is never
        materialized.

        When ``progress=True`` a two-level tqdm bar is shown (outer over
        samples, inner over gene chunks).

        Parameters
        ----------
        n_jobs : int, default -1
            If ``n_jobs == 1`` (or ``progress=True``) the per-sample loop runs
            sequentially so the progress bar is accurate. Otherwise a
            joblib thread pool distributes samples across workers; per-sample
            progress is still reported when available.
        progress : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        raw_2d : list of np.ndarray
            Per-sample ``(n_genes, ny, nx)`` 2D power spectra (float64).
        dc : np.ndarray
            ``(n_samples, n_genes)`` DC scalars (per-gene grid means).
        presence : np.ndarray
            ``(n_samples, n_genes)`` boolean mask: True = gene cleared the
            presence threshold in that sample.
        """
        from quadsv.nufft import power_spectrum_2d_nufft

        chunk_size = max(1, int(self.nufft_chunk_size))
        n_samples_total = len(self.samples)

        def _one(i: int, pbar: tqdm | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            adata = self.samples[i]
            pts = self._coords[i]
            scale = self._unit_scales[i]
            grid_i = self._grid_shapes[i]
            spacing_i = self.spacings[i]

            X_src = adata.X if self._layer is None else adata.layers[self._layer]
            n_genes = len(self.gene_names)
            n_spots = X_src.shape[0]

            if sp.issparse(X_src):
                dc = np.asarray(X_src.mean(axis=0)).ravel()
                nnz_per = np.asarray((X_src != 0).sum(axis=0)).ravel()
                if self.center == "zscore":
                    sq = X_src.multiply(X_src)
                    sq_mean = np.asarray(sq.mean(axis=0)).ravel()
                    sd = np.sqrt(np.maximum(sq_mean - dc**2, 0.0))
                else:
                    sd = None
                X_csc = X_src.tocsc()
                X_dense = None
            else:
                X_dense = np.asarray(X_src, dtype=np.float64)
                dc = X_dense.mean(axis=0)
                nnz_per = (X_dense != 0).sum(axis=0)
                sd = X_dense.std(axis=0) if self.center == "zscore" else None
                X_csc = None

            presence_i = (nnz_per / max(n_spots, 1)) >= self.presence_threshold

            if self.center == "zscore":
                positive = sd[sd > 0] if sd is not None else np.empty(0)
                sd_floor = float(np.median(positive)) * 0.1 if positive.size else 1.0
                sd_safe = np.maximum(sd, sd_floor) if sd is not None else None
            else:
                sd_safe = None

            ny, nx = grid_i
            spec_stack = np.empty((n_genes, ny, nx), dtype=np.float64)

            # Batched NUFFT: pull `chunk_size` columns at a time, standardize
            # per gene, run one batched type-1 NUFFT, write into spec_stack.
            for start in range(0, n_genes, chunk_size):
                stop = min(start + chunk_size, n_genes)
                cols = slice(start, stop)

                if X_csc is not None:
                    block = np.asarray(X_csc[:, cols].toarray(), dtype=np.float64)
                else:
                    block = X_dense[:, cols].astype(np.float64, copy=True)

                # block: (n_spots, chunk_width)
                if self.center == "mean":
                    block -= dc[None, cols]
                elif self.center == "zscore":
                    block = (block - dc[None, cols]) / sd_safe[None, cols]
                    np.clip(block, -_ZSCORE_CLIP, _ZSCORE_CLIP, out=block)

                p_chunk = power_spectrum_2d_nufft(
                    pts,
                    block,
                    grid_shape=grid_i,
                    spacing=spacing_i,
                    unit_scale=scale,
                    eps=self._nufft_eps,
                    center_coords=True,
                )
                # p_chunk shape: (ny, nx, chunk_width); move feature axis first.
                spec_stack[start:stop] = np.moveaxis(p_chunk, -1, 0)
                if pbar is not None:
                    pbar.update(1)

            return spec_stack, dc, presence_i

        raw_2d: list[np.ndarray | None] = [None] * n_samples_total
        dc_list: list[np.ndarray | None] = [None] * n_samples_total
        pres_list: list[np.ndarray | None] = [None] * n_samples_total

        run_sequential = progress or n_jobs == 1
        if run_sequential:
            n_chunks_total = sum(
                int(np.ceil(len(self.gene_names) / chunk_size)) for _ in self.samples
            )
            pbar: tqdm | None = (
                tqdm(total=n_chunks_total, desc="NUFFT spectra (per-gene chunks)")
                if progress
                else None
            )
            for i in range(n_samples_total):
                if pbar is not None:
                    pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
                r0, r1, r2 = _one(i, pbar=pbar)
                raw_2d[i] = r0
                dc_list[i] = r1
                pres_list[i] = r2
            if pbar is not None:
                pbar.close()
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(i) for i in range(n_samples_total)
            )
            for i, r in enumerate(results):
                raw_2d[i], dc_list[i], pres_list[i] = r

        dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
        presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
        return [np.asarray(x) for x in raw_2d], dc, presence

    # ------------------------------------------------------------------
    def _compute_fft_spectra(
        self,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """FFT per-sample spectrum pass for SpatialData inputs. Rasterizes
        each sample's table to a ``(n_genes, ny, nx)`` block **one gene at a
        time** when the source table is sparse, then computes the 2D FFT
        power spectrum in a single batched ``scipy.fft`` call. Returns
        ``(raw_2d, dc, presence)`` matching :meth:`_compute_nufft_spectra`.

        A tqdm progress bar over samples is shown when ``progress=True``.
        """

        def _one(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            sd_obj = self.samples[i]
            shape = self._grid_shapes[i]
            raster = _rasterize_spatialdata_lazy(
                sd_obj,
                grid_shape=shape,
                table_key=self._table_key,
                gene_names=self.gene_names,
            )
            frac_nonzero = (raster != 0).reshape(raster.shape[0], -1).mean(axis=1)
            presence_i = frac_nonzero >= self.presence_threshold
            spec, dc = compute_sample_spectrum(
                raster,
                fft_solver=self.fft_solver,
                workers=self.workers,
                center=self.center,
                return_dc=True,
            )
            return spec, dc, presence_i

        n_samples_total = len(self.samples)
        raw_2d: list[np.ndarray | None] = [None] * n_samples_total
        dc_list: list[np.ndarray | None] = [None] * n_samples_total
        pres_list: list[np.ndarray | None] = [None] * n_samples_total

        run_sequential = progress or n_jobs == 1
        if run_sequential:
            it = range(n_samples_total)
            if progress:
                it = tqdm(it, desc="FFT spectra (per sample)", total=n_samples_total)
            for i in it:
                r0, r1, r2 = _one(i)
                raw_2d[i] = r0
                dc_list[i] = r1
                pres_list[i] = r2
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(i) for i in range(n_samples_total)
            )
            for i, r in enumerate(results):
                raw_2d[i], dc_list[i], pres_list[i] = r

        dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
        presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
        return [np.asarray(x) for x in raw_2d], dc, presence

    # ------------------------------------------------------------------
    def fit(
        self,
        n_jobs: int = -1,
        landmark_genes: Sequence[str] | None = None,
        progress: bool = True,
    ) -> SpectralComparator:
        """
        Compute per-sample power spectra and (if ``feature_mode='2d'``) rotation-align.

        Parameters
        ----------
        n_jobs : int, default -1
            Parallelism over samples for the per-sample FFT/NUFFT. Ignored
            when ``progress=True`` (the inner progress bar requires a
            sequential driver loop — finufft is already multi-threaded via
            OpenMP so this rarely matters in practice).
        landmark_genes : sequence of str, optional
            Only used in ``feature_mode='2d'``. Names of genes (matched
            against :attr:`gene_names`) whose spectra define the
            rotation-alignment template. The recovered rotation is still
            applied to every gene; landmarks only choose what the per-sample
            template "looks like". If None (default), the global mean
            spectrum across all genes is used — the robust default for
            unsupervised data.
        progress : bool, default True
            If True, show tqdm progress bars over the three phases:
            per-sample spectrum compute, optional rotation alignment, and
            per-sample radial binning. Set to False to silence output.

        Returns
        -------
        SpectralComparator
            ``self``, for chaining.

        Raises
        ------
        KeyError
            If any ``landmark_genes`` entry is missing from
            :attr:`gene_names`.

        Notes
        -----
        **Runtime profile (dominant costs, in order)**

        1. Per-sample spectrum pass (``_compute_{nufft,fft}_spectra``). For
           AnnData inputs this is one batched ``finufft.nufft2d1`` per gene
           chunk, scaling roughly like
           ``O(n_samples · n_genes · (N log N + K log K) / chunk_size)``.
           finufft's per-call overhead is ~1 ms, so batching across genes
           (the :attr:`nufft_chunk_size` kwarg) gives the 10–50× speedup
           versus the previous one-gene-at-a-time loop. For SpatialData
           inputs this is one batched 2D FFT per sample on the rasterized
           ``(n_genes, ny, nx)`` block.
        2. Column densification from sparse ``adata.X`` / SpatialData
           tables — keeps memory small (``~n_spots · chunk_size · 8 B``)
           but costs one CSC slice per chunk.
        3. Optional rotation alignment (``feature_mode='2d'``) — cross-
           correlation per sample; cheap relative to (1).
        4. Radial binning into the common ``freq_edges`` grid —
           ``O(n_samples · n_genes · K)``; negligible.

        **Peak RAM**

        - ``self._raw_2d_spectra`` is the biggest steady-state footprint,
          ``O(n_samples · n_genes · ny · nx · 8 B)``. For 2 000 genes on
          a 128 × 128 grid across 20 samples this is ~5 GB; dropping to
          float32 halves it at the cost of a minor precision loss in the
          downstream tests (acceptable for everything except Liu's SF
          eigenvalue path, which already runs on :class:`FFTKernel` /
          :class:`NUFFTKernel` spectra, not ``spectra_``).
        - Per-chunk transient: ``(n_spots, chunk_size)`` dense block
          plus a ``(n_spots, chunk_size)`` complex array for finufft's
          RHS — both discarded after each call.

        **Explored improvements already in place**

        - Gene-chunk batching of ``finufft.nufft2d1`` (the single biggest
          win vs the old one-gene-at-a-time loop).
        - Sparse-aware presence / DC / std computation — no full ``.X``
          densification at any point.
        - Radial binning reuses precomputed frequency grids across
          samples, so it's one pass through a small ``(ny, n_kx)`` array
          per sample regardless of ``n_genes``.

        **Further improvements to consider**

        - Swap the internal ``spectra_`` stack to ``float32``; halves
          RAM. Would need a targeted test sweep against the permutation
          p-values first.
        - GPU finufft (``cufinufft``) — currently not a dependency but
          is a drop-in replacement for the ``finufft`` calls here.
        - Parallelise the per-sample outer loop with joblib when
          ``progress=False`` — wiring exists, it's just off under the
          progress-bar driver.
        """
        logger.info(
            "Computing per-sample spectra (mode=%s, n_samples=%d, center=%s)...",
            self.mode,
            len(self._grid_shapes),
            self.center,
        )
        if self.mode == "nufft":
            self._raw_2d_spectra, self.dc_, self.presence_ = self._compute_nufft_spectra(
                n_jobs=n_jobs, progress=progress
            )
        else:
            self._raw_2d_spectra, self.dc_, self.presence_ = self._compute_fft_spectra(
                n_jobs=n_jobs, progress=progress
            )

        if self.feature_mode == "2d":
            # Slice out the landmark subset from each sample's spectra.
            # When landmark_genes is None we use every gene as a landmark,
            # which gives the strongest signal-to-noise for rotation
            # estimation (cross-gene noise cancels because each gene is
            # cross-correlated against its own same-index counterpart).
            if landmark_genes is not None:
                name_to_idx = {g: i for i, g in enumerate(self.gene_names)}
                missing = [g for g in landmark_genes if g not in name_to_idx]
                if missing:
                    raise KeyError(f"landmark_genes not in gene_names: {missing}")
                landmark_idx_arr = np.asarray([name_to_idx[g] for g in landmark_genes], dtype=int)
                landmark_spectra = [s[landmark_idx_arr] for s in self._raw_2d_spectra]
            else:
                landmark_spectra = self._raw_2d_spectra
            aligned, angles = align_spectra_by_rotation(
                landmark_spectra,
                grid_shapes=self._grid_shapes,
                target_spectra=self._raw_2d_spectra,
                fft_solver=self._spectrum_fft_solver,
                progress=progress,
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
        iter_binning = zip(self._raw_2d_spectra, self._grid_shapes, strict=True)
        if progress:
            iter_binning = tqdm(
                enumerate(iter_binning),
                total=len(self._raw_2d_spectra),
                desc="Radial binning",
            )
        else:
            iter_binning = enumerate(iter_binning)
        for i, (spec_i, shape) in iter_binning:
            if self.feature_mode == "radial":
                spacing_i = self.spacings[i] if self.spacings is not None else None
                f = radial_bin_spectrum(
                    spec_i,
                    grid_shape=shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=spacing_i,
                    edges=self.freq_edges,
                )
            else:
                # 2D mode: flatten the (ny, nx) spectrum but optionally truncate to a
                # low-frequency square block of side ``n_radial_bins`` (re-using the
                # parameter as a low-pass cutoff). This keeps K manageable.
                ny, nx = shape
                k = min(self.n_radial_bins, ny // 2, nx // 2)
                low = spec_i[:, :k, :k] if spec_i.shape[-1] > k else spec_i[:, :k, :]
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
            cov_2d = compute_sample_spectrum(
                cov, fft_solver=self._spectrum_fft_solver, workers=self.workers
            )
            shape = self._grid_shapes[i]
            spacing = self.spacings[i] if self.spacings is not None else None
            if self.feature_mode == "radial":
                cov_feat = radial_bin_spectrum(
                    cov_2d,
                    grid_shape=shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
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
        freq_weights: np.ndarray | None = None,
        n_perm_max: int = 10000,
    ) -> pd.DataFrame:
        """
        Two-group spectral-pattern test on the cached :attr:`spectra_`.

        With ``center='mean'`` (the default), the spectrum is DC-free and this
        test is statistically orthogonal to :meth:`test_expression`. See
        :func:`compare_two_groups` for parameters and return format.

        When the comparator was constructed with ``presence_threshold > 0``
        and at least one ``(sample, gene)`` pair is absent, the test
        automatically switches to :func:`compare_two_groups_masked` so that
        genes are evaluated only over samples where they were observed
        (returned frame then carries ``n_obs_A`` / ``n_obs_B`` columns and
        ``NaN`` p-values for genes with too few observations in either
        group).

        ``freq_weights`` (optional, currently consumed by ``statistic='log_l2'``)
        is a non-negative length-``K`` array that is internally renormalized
        to sum-1 and used to bias the L2 distance toward specific frequency
        bands — e.g., a polynomial low-pass profile to mirror a CAR kernel
        or an exponential tail to mirror a Gaussian kernel. Equal weights
        (default) recover the plain ``log_l2`` statistic.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .test_pattern().")
        use_masked = self.presence_ is not None and not self.presence_.all()
        if use_masked:
            return compare_two_groups_masked(
                self.spectra_,
                self.groups,
                self.presence_,
                gene_names=self.gene_names,
                statistic=statistic,
                n_perm=n_perm,
                random_state=random_state,
                min_samples_per_group=self.min_samples_per_group,
                freq_weights=freq_weights,
                n_perm_max=n_perm_max,
            )
        return compare_two_groups(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistic=statistic,
            n_perm=n_perm,
            random_state=random_state,
            freq_weights=freq_weights,
            n_perm_max=n_perm_max,
        )

    # Back-compat alias — `test()` still runs the pattern test.
    test = test_pattern

    # ------------------------------------------------------------------
    def test_expression(
        self,
        n_perm: int = 1000,
        random_state: int | None = None,
        n_perm_max: int = 10000,
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
            n_perm_max=n_perm_max,
        )

    # ------------------------------------------------------------------
    def benchmark(
        self,
        statistics: Sequence[str] = _AVAILABLE_STATISTICS,
        n_perm: int = 1000,
        random_state: int | None = None,
        n_perm_max: int = 10000,
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
            n_perm_max=n_perm_max,
        )
