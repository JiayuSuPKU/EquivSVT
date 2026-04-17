"""
Non-uniform FFT (NUFFT) power spectra for spatial omics on irregular grids.

When data sit on a regular grid (e.g., a rasterized Visium slide),
:func:`quadsv.power_spectrum_2d` computes :math:`|\\hat{x}(k)|^2` with a plain
2D FFT. For data whose spatial coordinates are **irregular** — e.g., Slide-seq,
Stereo-seq, or a Visium slide read straight from
``adata.obsm['spatial']`` without rasterization — we need a non-uniform FFT
instead. :func:`power_spectrum_2d_nufft` evaluates

.. math::

   \\hat c(k_y, k_x) = \\sum_{j=1}^N c_j \\,
       \\exp\\!\\bigl[-i(k_y\\,y_j + k_x\\,x_j)\\bigr]

at the same uniform ``(ny, nx)`` k-space grid that :func:`power_spectrum_2d`
would produce for a rasterized input of the same physical extent, then returns
:math:`|\\hat c|^2` with the standard scipy FFT layout (DC at ``[0, 0]``).
Anything downstream — :func:`quadsv.spectral_compare.radial_bin_spectrum`,
:class:`quadsv.SpectralComparator` with physical ``spacings`` — works
identically.

Per-sample unit handling
------------------------

Samples in different studies may ship coordinates in different units (μm,
Visium full-resolution pixels, etc.). :func:`power_spectrum_2d_nufft` accepts a
``unit_scale`` multiplier that converts each sample's coordinate axis to the
*common* physical unit chosen for ``spacing``. The common frequency grid is
``1 / (ny·dy)`` by ``1 / (nx·dx)`` cycles per unit length, so spectra from
slides with different shapes and pixel conventions become directly comparable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["power_spectrum_2d_nufft"]


def _check_finufft() -> None:
    """Import finufft lazily; raise a helpful error if it is not installed."""
    try:
        import finufft  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "power_spectrum_2d_nufft requires the finufft package. "
            "Install with `pip install 'quadsv[nufft]'` or `pip install finufft`."
        ) from e


def power_spectrum_2d_nufft(
    coords: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
    unit_scale: float = 1.0,
    eps: float = 1e-6,
    center_coords: bool = True,
) -> np.ndarray:
    """
    Compute the 2D power spectrum :math:`|\\hat{c}(k)|^2` of one or more non-uniform
    spatial signals via a type-1 NUFFT.

    The output has the same ``(ny, n_kx)`` layout as
    :func:`quadsv.power_spectrum_2d` with ``fft_solver='fft2'`` — DC at
    ``[0, 0]``, Nyquist at ``[ny/2, nx/2]`` (when dimensions are even) — so the
    result is a drop-in substitute for the rasterized spectrum in the rest of
    the pipeline.

    Parameters
    ----------
    coords : np.ndarray
        Non-uniform spatial coordinates, shape ``(N, 2)`` in the order
        ``(y, x)``. Values outside the physical domain implied by
        ``grid_shape`` and ``spacing`` are folded into ``[-π, π)`` by finufft.
    values : np.ndarray
        Signal strengths at each coordinate. Shape ``(N,)`` for a single
        feature, or ``(N, M)`` for ``M`` stacked features (e.g., genes) on the
        same coordinates. Real-valued; promoted to complex internally.
    grid_shape : tuple[int, int]
        ``(ny, nx)`` of the target uniform k-space grid. Match whatever grid
        you use for rasterized samples so the two paths produce comparable
        spectra.
    spacing : tuple[float, float]
        ``(dy, dx)`` physical spacing per cell of the target grid, in the same
        unit as ``unit_scale * coords``. Together with ``grid_shape`` this
        defines the physical domain extent ``(ny · dy, nx · dx)``.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` before scaling into ``[-π, π)``. Use
        this to convert per-sample coordinate units into the common unit of
        ``spacing`` (e.g., 0.35 if ``coords`` are in Visium full-res pixels at
        0.35 μm/pixel and ``spacing`` is in μm).
    eps : float, default 1e-6
        NUFFT tolerance forwarded to finufft.
    center_coords : bool, default True
        If True, subtract the mean of ``coords`` before scaling — avoids
        wrapping artefacts when coordinates are stored with an arbitrary origin
        offset (e.g., Visium pixel coordinates start at a few thousand). Power
        spectra are translation-invariant so recentering does not change the
        result.

    Returns
    -------
    np.ndarray
        Power spectrum. Shape ``(ny, nx)`` for 1D ``values`` or
        ``(ny, nx, M)`` for 2D ``values``, with DC at index ``[0, 0]``.

    Raises
    ------
    ImportError
        If :mod:`finufft` is not installed.
    ValueError
        If input shapes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).uniform(0, 100, size=(500, 2))
    >>> vals = np.random.default_rng(1).standard_normal(500)
    >>> P = power_spectrum_2d_nufft(coords, vals, grid_shape=(32, 32), spacing=(4.0, 4.0))
    >>> P.shape
    (32, 32)
    """
    _check_finufft()
    import finufft

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"values first dim {values.shape[0]} must match coords N={coords.shape[0]}."
        )

    ny, nx = grid_shape
    dy, dx = spacing
    if ny <= 0 or nx <= 0 or dy <= 0 or dx <= 0:
        raise ValueError(f"grid_shape and spacing must be positive, got {grid_shape}, {spacing}.")

    y = coords[:, 0].astype(np.float64) * unit_scale
    x = coords[:, 1].astype(np.float64) * unit_scale
    if center_coords:
        y = y - y.mean()
        x = x - x.mean()

    # Physical domain extents implied by the target uniform grid.
    Ly = ny * dy
    Lx = nx * dx

    # Scale into finufft's [-π, π) window so that mode index k (centred
    # at zero, range [-n/2, (n-1)/2]) corresponds to physical frequency
    # k / L cycles per unit length — matching np.fft.fftfreq(n, d).
    y_scaled = y * (2.0 * np.pi / Ly)
    x_scaled = x * (2.0 * np.pi / Lx)

    # Batched transforms: finufft accepts shape (n_tr, M) for c.
    squeeze = values.ndim == 1
    if squeeze:
        c = values.astype(np.complex128, copy=False)
    else:
        # finufft expects (n_tr, N_points).
        c = np.ascontiguousarray(values.T.astype(np.complex128, copy=False))

    # type-1 NUFFT: nonuniform points -> uniform k-space grid.
    # Output shape: (ny, nx) or (n_tr, ny, nx). DC at CENTRE ([ny//2, nx//2]).
    f_hat = finufft.nufft2d1(y_scaled, x_scaled, c, n_modes=(ny, nx), eps=eps, isign=-1)

    # Power spectrum.
    power = (f_hat.real**2 + f_hat.imag**2).astype(np.float64)

    # Move DC from the centre to [0, 0] so the layout matches scipy.fft.fft2.
    power = np.fft.ifftshift(power, axes=(-2, -1))

    if squeeze:
        return power
    # Put the feature axis back at the end to match power_spectrum_2d(x=(ny, nx, M)).
    return np.moveaxis(power, 0, -1)
