"""
Visium I/O and hex-to-regular-grid rasterization for the spectral-comparison pipeline.

10x Visium slides place ~5,000 spots on a hexagonal grid. The ``array_row`` and
``array_col`` indices use an "orange-crate" convention: for a 6.5 mm capture area,
rows span ``0..77`` and columns span ``0..126``, where **even rows use even column
indices and odd rows use odd column indices** (i.e., adjacent rows are horizontally
offset by one array_col step = 50 μm physical).

This module produces a regular rectangular array of shape ``(n_genes, ny, nx)``
ready for :func:`quadsv.power_spectrum_2d`, together with the physical grid spacing
``(dy, dx)`` in micrometres. Two rasterization modes are supported:

- ``grid='dense'`` (default) — ``(78, 128)`` array with 50% of cells filled from real
  spots; empty cells imputed from their two nearest hex neighbours. Physical spacing
  ``(dy, dx) = (100·√3/2, 50)`` μm per cell. Preserves exact hex geometry.
- ``grid='collapsed'`` — ``(78, 64)`` array using ``array_col // 2`` as the column
  index. Physical spacing ``(dy, dx) = (100·√3/2, 100)`` μm. Faster, but the 50 μm
  horizontal offset between alternating rows is dropped (≤5 % geometric distortion).

Typical workflow::

    from quadsv.io_visium import load_visium_sample, visium_to_grid
    adata = load_visium_sample("/path/to/spaceranger_out/sample_A")
    grid, spacing_um = visium_to_grid(adata, genes=["CAMK2A", "GFAP", "VIM"])
    # grid.shape = (3, 78, 128); spacing_um = (86.603, 50.0)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "VISIUM_V1_SPOT_SPACING_UM",
    "load_visium_sample",
    "visium_to_grid",
    "visium_hex_spacing_um",
]

logger = logging.getLogger(__name__)

#: Physical center-to-center distance between Visium v1 (6.5 mm capture area) spots, in μm.
VISIUM_V1_SPOT_SPACING_UM: float = 100.0


def visium_hex_spacing_um(
    spot_spacing_um: float = VISIUM_V1_SPOT_SPACING_UM,
    grid: str = "dense",
) -> tuple[float, float]:
    """
    Physical ``(dy, dx)`` per grid cell for a Visium hex raster.

    Parameters
    ----------
    spot_spacing_um : float, default 100.0
        Center-to-center distance between adjacent spots (100 μm for Visium v1).
    grid : {'dense', 'collapsed'}, default 'dense'
        Rasterization mode. See module docstring.

    Returns
    -------
    tuple[float, float]
        ``(dy, dx)`` in micrometres per grid cell.

    Raises
    ------
    ValueError
        If ``grid`` is unknown.
    """
    dy = spot_spacing_um * np.sqrt(3.0) / 2.0
    if grid == "dense":
        dx = spot_spacing_um / 2.0
    elif grid == "collapsed":
        dx = spot_spacing_um
    else:
        raise ValueError(f"grid must be 'dense' or 'collapsed', got '{grid}'.")
    return float(dy), float(dx)


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    """Read ``tissue_positions_list.csv`` (Space Ranger v1) or ``tissue_positions.csv`` (v2)."""
    candidates = [
        spatial_dir / "tissue_positions_list.csv",  # Space Ranger < 2.0, no header
        spatial_dir / "tissue_positions.csv",  # Space Ranger >= 2.0, with header
    ]
    for path in candidates:
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No tissue_positions[_list].csv found in {spatial_dir}.")

    if path.name == "tissue_positions_list.csv":
        df = pd.read_csv(
            path,
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )
    else:
        df = pd.read_csv(path)
    return df


def load_visium_sample(  # noqa: C901
    path: str | Path,
    matrix_path: str | Path | None = None,
    in_tissue_only: bool = True,
) -> "anndata.AnnData":  # noqa: F821, UP037  (forward ref avoids a hard anndata import)
    """
    Load a Visium Space Ranger output directory as an :class:`anndata.AnnData`.

    The loader accepts either the conventional flat layout
    (``<path>/<sample>_filtered_feature_bc_matrix.h5`` + ``<path>/spatial/``) or the
    canonical Space Ranger ``outs/`` layout (``<path>/filtered_feature_bc_matrix.h5``
    + ``<path>/spatial/``), and auto-detects which one is present.

    Parameters
    ----------
    path : str or Path
        Directory containing the filtered matrix and ``spatial/`` subfolder
        (typically the Space Ranger ``outs/`` or a sibling directory).
    matrix_path : str or Path, optional
        Explicit path to the filtered ``.h5`` matrix. If None, the function searches
        ``<path>/filtered_feature_bc_matrix.h5`` and ``<path>/*_filtered_feature_bc_matrix.h5``.
    in_tissue_only : bool, default True
        If True, restrict to spots with ``in_tissue == 1``.

    Returns
    -------
    anndata.AnnData
        ``adata.obs`` contains ``in_tissue``, ``array_row``, ``array_col``,
        ``pxl_row_in_fullres``, ``pxl_col_in_fullres``. ``adata.obsm['spatial']``
        holds ``(pxl_col, pxl_row)`` in full-resolution pixel units, and
        ``adata.uns['spatial']`` stores the raw ``scalefactors_json.json`` and the
        directory path.

    Raises
    ------
    FileNotFoundError
        If the matrix or spatial folder cannot be located.
    ImportError
        If :mod:`anndata` / :mod:`scanpy` are not installed.
    """
    try:
        import anndata  # noqa: F401
        import scanpy as sc
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "load_visium_sample requires scanpy + anndata. "
            "Install with `pip install 'quadsv[spatial]'` or `pip install scanpy`."
        ) from e

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory.")

    # Find spatial folder (same dir, or under outs/).
    spatial_dir = path / "spatial"
    if not spatial_dir.is_dir():
        alt = path / "outs" / "spatial"
        if alt.is_dir():
            spatial_dir = alt
            path = path / "outs"
        else:
            raise FileNotFoundError(f"No spatial/ subfolder found under {path}.")

    if matrix_path is None:
        mp = path / "filtered_feature_bc_matrix.h5"
        if not mp.exists():
            matches = sorted(path.glob("*_filtered_feature_bc_matrix.h5"))
            if not matches:
                raise FileNotFoundError(f"No filtered_feature_bc_matrix.h5 found under {path}.")
            mp = matches[0]
    else:
        mp = Path(matrix_path)

    logger.info("Reading Visium matrix: %s", mp)
    adata = sc.read_10x_h5(mp)
    adata.var_names_make_unique()

    tp = _read_tissue_positions(spatial_dir)
    tp = tp.set_index("barcode")

    # Align obs to the spots present in the matrix.
    missing = set(adata.obs_names) - set(tp.index)
    if missing:
        warnings.warn(
            f"{len(missing)} barcodes in matrix lack entries in tissue_positions; "
            "they will be dropped.",
            UserWarning,
            stacklevel=2,
        )
        adata = adata[adata.obs_names.isin(tp.index)].copy()
    tp = tp.reindex(adata.obs_names)
    for col in ("in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"):
        adata.obs[col] = tp[col].astype(int if col == "in_tissue" else float).values

    adata.obsm["spatial"] = np.column_stack(
        [tp["pxl_col_in_fullres"].to_numpy(), tp["pxl_row_in_fullres"].to_numpy()]
    )

    scalefactors_path = spatial_dir / "scalefactors_json.json"
    if scalefactors_path.exists():
        with scalefactors_path.open() as f:
            scalefactors = json.load(f)
    else:
        scalefactors = {}
    adata.uns["spatial"] = {"scalefactors": scalefactors, "path": str(path)}

    if in_tissue_only:
        mask = adata.obs["in_tissue"].to_numpy().astype(bool)
        adata = adata[mask].copy()

    logger.info(
        "Loaded Visium sample: %d spots (%d in tissue), %d genes.",
        adata.n_obs,
        int(adata.obs["in_tissue"].sum()) if "in_tissue" in adata.obs else adata.n_obs,
        adata.n_vars,
    )
    return adata


def _fill_nearest_hex_neighbor(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill ``grid`` (shape ``(..., ny, nx)``) at locations where ``mask==False``
    with the mean of each cell's two nearest hex neighbours.

    For a Visium hex raster with alternating-parity columns, the empty cells lie at
    ``(r, c)`` with ``c % 2 != r % 2``. Their two nearest spots on the same row
    are at ``(r, c-1)`` and ``(r, c+1)``; both are real spots. This function uses
    that pair (handling edge columns by one-sided averaging).
    """
    out = grid.copy()
    ny, nx = mask.shape
    empty = np.argwhere(~mask)
    if empty.size == 0:
        return out
    for r, c in empty:
        left = out[..., r, c - 1] if c - 1 >= 0 and mask[r, c - 1] else None
        right = out[..., r, c + 1] if c + 1 < nx and mask[r, c + 1] else None
        if left is not None and right is not None:
            out[..., r, c] = 0.5 * (left + right)
        elif left is not None:
            out[..., r, c] = left
        elif right is not None:
            out[..., r, c] = right
        # else: leave as zero (isolated empty — can't happen on a Visium hex).
    return out


def visium_to_grid(  # noqa: C901
    adata: "anndata.AnnData",  # noqa: F821, UP037
    genes: list[str] | None = None,
    layer: str | None = None,
    grid: str = "dense",
    fill: str = "nearest",
    spot_spacing_um: float = VISIUM_V1_SPOT_SPACING_UM,
    max_row: int | None = None,
    max_col: int | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Rasterize a Visium ``adata`` onto a regular rectangular grid.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``adata.obs['array_row']`` and ``adata.obs['array_col']`` set
        (e.g. from :func:`load_visium_sample`).
    genes : list of str, optional
        Gene names to include. If None, all genes in ``adata.var_names`` are used.
    layer : str, optional
        Name of ``adata.layers`` to pull expression from. If None, uses ``adata.X``.
    grid : {'dense', 'collapsed'}, default 'dense'
        Rasterization mode. See module docstring.
    fill : {'nearest', 'zero'}, default 'nearest'
        How to fill grid cells with no spot. ``'nearest'`` averages the two nearest
        hex-neighbour spots in the same row (recommended; avoids FFT aliasing).
        ``'zero'`` leaves missing cells at 0, which is fast but introduces
        half-grid-frequency artefacts. Ignored when ``grid='collapsed'``.
    spot_spacing_um : float, default 100.0
        Center-to-center spacing between adjacent spots in μm.
    max_row, max_col : int, optional
        Explicit output grid size. Defaults to the maximum ``array_row + 1`` and
        ``array_col + 1`` observed in ``adata``; pass explicit values to pad all
        samples to a common shape.

    Returns
    -------
    grid_arr : np.ndarray
        Array of shape ``(n_genes, ny, nx)`` ready for
        :func:`quadsv.power_spectrum_2d`. Dtype ``float64``.
    spacing_um : tuple[float, float]
        Physical spacing ``(dy, dx)`` in μm per grid cell.

    Raises
    ------
    KeyError
        If ``array_row`` / ``array_col`` are absent from ``adata.obs``.
    ValueError
        If ``grid`` or ``fill`` has an unknown value, or ``layer`` is absent.
    """
    if "array_row" not in adata.obs or "array_col" not in adata.obs:
        raise KeyError(
            "adata.obs must contain 'array_row' and 'array_col'. "
            "Load the sample with quadsv.io_visium.load_visium_sample first."
        )
    if grid not in ("dense", "collapsed"):
        raise ValueError(f"grid must be 'dense' or 'collapsed', got '{grid}'.")
    if fill not in ("nearest", "zero"):
        raise ValueError(f"fill must be 'nearest' or 'zero', got '{fill}'.")

    rows = adata.obs["array_row"].to_numpy().astype(int)
    cols = adata.obs["array_col"].to_numpy().astype(int)

    # Decide grid shape.
    ny = int(rows.max() + 1) if max_row is None else max_row
    if grid == "dense":
        nx = int(cols.max() + 1) if max_col is None else max_col
    else:  # collapsed: use array_col // 2 as column index (range [0, nx_even))
        nx = int(cols.max() // 2 + 1) if max_col is None else max_col

    # Pull expression matrix.
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    if genes is not None:
        gene_idx = [adata.var_names.get_loc(g) for g in genes]
        X = X[:, gene_idx]
        gene_names = list(genes)
    else:
        gene_names = list(adata.var_names)

    n_genes = X.shape[1]
    out = np.zeros((n_genes, ny, nx), dtype=np.float64)
    filled = np.zeros((ny, nx), dtype=bool)

    if grid == "dense":
        for i in range(X.shape[0]):
            r, c = rows[i], cols[i]
            if 0 <= r < ny and 0 <= c < nx:
                out[:, r, c] = X[i]
                filled[r, c] = True
        if fill == "nearest":
            out = _fill_nearest_hex_neighbor(out, filled)
    else:  # collapsed
        for i in range(X.shape[0]):
            r, c = rows[i], cols[i] // 2
            if 0 <= r < ny and 0 <= c < nx:
                out[:, r, c] = X[i]

    spacing = visium_hex_spacing_um(spot_spacing_um=spot_spacing_um, grid=grid)
    logger.info(
        "Rasterized %d genes x %d spots onto %s grid of shape (%d, %d), spacing=%s μm.",
        n_genes,
        adata.n_obs,
        grid,
        ny,
        nx,
        spacing,
    )
    del gene_names  # currently informational only; returned via the caller's genes list
    return out, spacing
