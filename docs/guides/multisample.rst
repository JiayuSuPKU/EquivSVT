Cross-sample spectral comparison
================================

When you have **two groups of spatial-omics samples** — e.g. a set of healthy slides
and a set of cancer slides — and want to ask *"which genes show the largest
difference in spatial pattern between the two groups?"*, the
:mod:`quadsv.multisample` module gives you an alignment-free, frequency-domain
pipeline with valid permutation-based hypothesis testing.

Why frequency domain?
---------------------

The 2D power spectrum :math:`|\hat x(k)|^2` of a rasterized gene image is
**translation-invariant**: shifting the image leaves the power spectrum unchanged.
Radial averaging (the default) additionally makes the representation
**rotation-invariant**. Both properties together mean samples never need to be
spatially registered onto each other — a hard requirement when, for example,
healthy and cancer slides have no shared anatomy.

DC/AC decomposition: expression vs pattern
------------------------------------------

By default the pipeline **mean-centres each gene's spatial signal before the
FFT** (``center='mean'`` on :class:`~quadsv.ComparatorIrregular`). This cleanly
splits the information into two orthogonal pieces:

- **DC scalar** per ``(gene, sample)`` — the grid mean, i.e. total normalized
  expression on the slide. Tested across groups via
  :meth:`~quadsv.ComparatorIrregular.test_expression` (Welch *t* with a
  permutation null, BH-FDR); this is a classical differential-expression test.
- **AC spectrum** — the pattern shape, with DC exactly zero. Tested across
  groups via :meth:`~quadsv.ComparatorIrregular.test_pattern` (the log-L2 and
  other statistics described below).

Because DC and AC live in orthogonal subspaces, the two tests carry
**complementary information**: a gene may be "only DE" (same pattern, different
magnitude), "only pattern" (same total expression, different spatial
organisation), or "both". Run them side by side and inspect where the hits
overlap vs separate.

If you want the pattern test to also be *scale-invariant* (ignoring overall
amplitude changes), pass ``center='zscore'``; disable the split with
``center=None`` (legacy behaviour — DC and nearby bins can leak between the
tests).

Pipeline
--------

The :class:`~quadsv.ComparatorIrregular` class chains five steps:

1. Per-sample 2D power spectra (:func:`~quadsv.power_spectrum_2d`).
2. Reduction to a low-dimensional feature vector — radial 1D bins by default,
   or 2D with rotation alignment if you opt in.
3. **Background normalization** that cancels per-slide gain/sensitivity differences
   (geometric-mean spectrum across all genes per sample).
4. **Optional covariate-spectrum residualization** to regress out the spatial
   patterns of "uninteresting" features (cell-type proportion maps, tissue-domain
   indicators, housekeeping composites).
5. A per-gene two-group test with a sample-label permutation null and BH-FDR.

Four test statistics ship out of the box and share one permutation null so they
are directly comparable:

================  ========================================================
``log_l2``        L2 distance between mean log-spectra (default)
``hotelling_lw``  Regularized Hotelling T² with Ledoit-Wolf covariance
``mmd_rbf``       RBF-kernel maximum mean discrepancy
``max_welch``     Max per-bin Welch t-statistic (omnibus + interpretation)
================  ========================================================

Use :func:`~quadsv.benchmark_statistics` to evaluate all four on the same data.

Input types
-----------

:class:`~quadsv.ComparatorIrregular` accepts two — and only two — kinds of
per-sample container:

- a list of :class:`anndata.AnnData` → **NUFFT backend** (irregular spots,
  common across Visium, Slide-seq, Stereo-seq, MERFISH). Each sample keeps
  its **own** ``(grid_shape, spacing)``; cross-sample comparability is
  obtained in physical-frequency space via radial binning.
- a list of :class:`spatialdata.SpatialData` → **FFT backend** (regular
  rasterized grids, e.g., Visium HD).

Sparse ``adata.X`` / layer matrices are **not densified up front** — the
spectrum loop converts exactly one gene column at a time to dense.

Toy walkthrough (AnnData / NUFFT)
---------------------------------

The minimal end-to-end example below builds eight synthetic samples (4 per
group) of 10 genes. Gene ``g0`` carries a low-frequency stripe pattern in
group 1 only.

.. code-block:: python

   import anndata as ad
   import numpy as np
   from quadsv import ComparatorIrregular

   rng = np.random.default_rng(3)
   ny = nx = 32
   n_genes = 10
   gene_names = [f"g{i}" for i in range(n_genes)]

   def make_adata(group: int) -> ad.AnnData:
       # Spot layout: regular 32x32 grid; the NUFFT backend does not care,
       # it auto-infers grid_shape and spacing from the coords.
       yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
       coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)

       X = rng.standard_normal((ny * nx, n_genes)) * 0.1
       if group == 1:
           stripes = np.sin(2 * np.pi * yy / 16.0).ravel()
           X[:, 0] += 1.5 * stripes

       a = ad.AnnData(X=X)
       a.var_names = gene_names
       a.obsm["spatial"] = coords
       return a

   samples = [make_adata(0) for _ in range(4)] + [make_adata(1) for _ in range(4)]
   groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

   cmp = (
       ComparatorIrregular(samples, groups, gene_names)
       .fit()
       .normalize_background()
   )
   results = cmp.test(statistic="log_l2", n_perm=300, random_state=0)
   print(results.head())

The implanted gene ``g0`` ranks first in the resulting table.

FFT walkthrough (SpatialData)
-----------------------------

For rasterized-grid samples, use :class:`~quadsv.ComparatorGrid` with a
sequence of :class:`spatialdata.SpatialData` objects. Rasterization is done
per sample via :func:`spatialdata.rasterize_bins` — same recipe as
:class:`~quadsv.DetectorGrid`, so you'll recognize the kwargs:

.. code-block:: python

   import spatialdata as sd
   from quadsv import ComparatorGrid

   samples_sd = [sd.read_zarr(p) for p in paths_by_group]
   cmp = ComparatorGrid(
       samples_sd,
       groups,
       bins="bin_shapes",            # SpatialElement name shared by every sdata
       table_name="counts",          # table inside each sdata
       col_key="array_col",          # obs column with bin-column indices
       row_key="array_row",          # obs column with bin-row indices
       value_key=None,               # None → rasterizes expression off .X
       fft_chunk_size=256,           # genes per batched scipy.fft call
   ).fit().normalize_background()
   cmp.test(statistic="log_l2", n_perm=300)

Magnitude-robust gene clustering via ``shape_normalize``
--------------------------------------------------------

When clustering genes by "how their spatial pattern changes between groups" —
as in the contrastive-programs analysis in the example notebooks — the raw
log-spectrum contrast ``log P_B - log P_A`` is easily dominated by genes that
are simply expressed in one group and absent in the other: a multiplicative
magnitude difference produces a constant offset at every frequency bin, which
overwhelms genuine shape differences.

:func:`quadsv.shape_normalize` fixes this by dividing each
``(sample, gene)`` spectrum by its own geometric mean (equivalently, subtracting
the per-row mean of ``log P``). Two rows that differ only by a positive scalar
become identical after the transform; only the shape of the power-vs-frequency
curve survives. Chain it after background normalization:

.. code-block:: python

   cmp = (
       ComparatorIrregular(samples, groups, gene_names)
       .fit()
       .normalize_background()   # cancels per-sample gain across genes
       .shape_normalize()        # cancels per-(sample, gene) magnitude across freq
   )
   # cmp.spectra_ now has unit geometric mean along the K axis; use it for
   # downstream KMeans / hierarchical clustering of gene-level contrast vectors.

The per-gene DE test via :meth:`ComparatorIrregular.test_expression` is
unaffected — it reads from ``cmp.dc_`` directly.

Cross-sample unit conversion (NUFFT path)
-----------------------------------------

When different samples ship coordinates in different physical units — e.g.,
one slide in μm and another in Visium full-resolution pixels at
0.35 μm/pixel — pass a per-sample ``unit_scales`` list that converts each
sample's raw coords into the common unit. Radial bins then come out in
cycles per that unit on all samples:

.. code-block:: python

   cmp = ComparatorIrregular(
       samples,                 # list[AnnData]
       groups,
       gene_names=gene_names,
       unit_scales=[1.0, 0.35, 1.0, 0.35],   # per-sample multiplier
       spacing=(50.0, 50.0),                  # common physical spacing, μm
       n_radial_bins=30,
   ).fit().normalize_background()

If ``grid_shape`` / ``spacing`` are left unset, each sample's k-grid is
auto-inferred from its coords via
:func:`quadsv.nufft._infer_grid_from_coords`.

:func:`quadsv.nufft.power_spectrum_2d_nufft` is the lower-level primitive
(one sample at a time). Correctness is validated against the rasterized FFT
on real Visium data: ``FFT(zero-filled raster)`` equals
``NUFFT(raw coords)`` to ~10⁻⁶ relative tolerance (see
``tests/test_nufft.py``).

Visium hex grids
----------------

For 10x Visium slides, :func:`quadsv.io_visium.load_visium_sample` (from the
submodule) reads a Space Ranger output directory into an
:class:`anndata.AnnData`. You can feed that :class:`~anndata.AnnData`
directly to :class:`~quadsv.ComparatorIrregular` — the NUFFT backend handles
the irregular hex layout without any manual rasterization step. If you do
want the explicit hex-to-grid rasterization for other purposes,
:func:`quadsv.io_visium.visium_to_grid` returns the ``(n_genes, 78, 128)``
array and the physical spacing ``(dy, dx) = (100·√3/2, 50)`` μm per cell
for v1 Visium.

The minimum resolvable pattern of a v1 Visium slide is roughly
``2 · 86.6 μm ≈ 173 μm`` (Nyquist along the coarser axis).

Choosing covariate maps for residualization
-------------------------------------------

Pass a list of per-sample covariate arrays (each of shape
``(n_covariates, ny_s, nx_s)``) to :meth:`~quadsv.ComparatorIrregular.residualize`.
Useful candidates:

- **Cell-type proportion maps** from your favorite deconvolution tool (Cell2location,
  CARD, RCTD). One channel per cell type.
- **Tissue-domain indicator maps** from spatial clustering (BayesSpace, GraphST, etc).
- **A composite "housekeeping" expression image** to absorb depth gradients.

Residualization is applied **after** background normalization and **before** testing.

When to use 2D mode
-------------------

The default radial mode is rotation-invariant. Switch to ``feature_mode="2d"`` when
your samples have a meaningful, biologically conserved orientation
(e.g., DV/AP axes preserved across slides). The pipeline will rotation-align each
sample's full 2D spectrum to a chosen reference before flattening, giving you back
directional anisotropy without requiring per-pixel registration.

Reference
---------

The default ``log_l2`` statistic follows the nonparametric two-sample test of
log-spectral densities by Bandyopadhyay & Wu
(`arXiv:2602.10774 <https://arxiv.org/html/2602.10774>`_). The pipeline is
inspired by the alignment-free philosophy of
`SpaGFT <https://www.nature.com/articles/s41467-024-51590-5>`_ but replaces the
graph Fourier basis with the standard 2D FFT, which is faster on regular grids and
exposes a translation-invariance property that graph Fourier does not.

API reference
-------------

:class:`quadsv.ComparatorIrregular` is the main public entry point.
Lower-level primitives live in the :mod:`quadsv.multisample` and
:mod:`quadsv.nufft` submodules:

- :func:`quadsv.multisample.compare_two_groups`
- :func:`quadsv.multisample.benchmark_statistics`
- :func:`quadsv.fft.power_spectrum_2d`
- :func:`quadsv.nufft.power_spectrum_2d_nufft`
