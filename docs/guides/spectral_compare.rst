Cross-sample spectral comparison
================================

When you have **two groups of spatial-omics samples** — e.g. a set of healthy slides
and a set of cancer slides — and want to ask *"which genes show the largest
difference in spatial pattern between the two groups?"*, the
:mod:`quadsv.spectral_compare` module gives you an alignment-free, frequency-domain
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
FFT** (``center='mean'`` on :class:`~quadsv.SpectralComparator`). This cleanly
splits the information into two orthogonal pieces:

- **DC scalar** per ``(gene, sample)`` — the grid mean, i.e. total normalized
  expression on the slide. Tested across groups via
  :meth:`~quadsv.SpectralComparator.test_expression` (Welch *t* with a
  permutation null, BH-FDR); this is a classical differential-expression test.
- **AC spectrum** — the pattern shape, with DC exactly zero. Tested across
  groups via :meth:`~quadsv.SpectralComparator.test_pattern` (the log-L2 and
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

The :class:`~quadsv.SpectralComparator` class chains five steps:

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

Toy walkthrough
---------------

The minimal end-to-end example below builds eight synthetic samples (4 per group)
of 10 genes on a 32x32 grid; gene ``g0`` carries a low-frequency stripe pattern
in group 1 only.

.. code-block:: python

   import numpy as np
   from quadsv import SpectralComparator

   rng = np.random.default_rng(3)
   ny = nx = 32
   n_genes = 10
   gene_names = [f"g{i}" for i in range(n_genes)]

   def make_sample(group: int) -> np.ndarray:
       x = rng.standard_normal((n_genes, ny, nx)) * 0.1
       if group == 1:
           y = np.arange(ny)[:, None]
           stripes = np.broadcast_to(np.sin(2 * np.pi * y / 16.0), (ny, nx))
           x[0] += stripes * 1.5
       return x

   samples = [make_sample(0) for _ in range(4)] + [make_sample(1) for _ in range(4)]
   groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

   cmp = (
       SpectralComparator(samples, groups, gene_names)
       .fit()
       .normalize_background()
   )
   results = cmp.test(statistic="log_l2", n_perm=300, random_state=0)
   print(results.head())

The implanted gene ``g0`` ranks first in the resulting table.

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
       SpectralComparator(samples, groups, gene_names, spacings=spacings)
       .fit()
       .normalize_background()   # cancels per-sample gain across genes
       .shape_normalize()        # cancels per-(sample, gene) magnitude across freq
   )
   # cmp.spectra_ now has unit geometric mean along the K axis; use it for
   # downstream KMeans / hierarchical clustering of gene-level contrast vectors.

The per-gene DE test via :meth:`SpectralComparator.test_expression` is
unaffected — it reads from ``cmp.dc_`` directly.

Visium hex grids → physical frequency
-------------------------------------

For 10x Visium slides, :func:`quadsv.load_visium_sample` reads a Space Ranger
output directory into an :class:`anndata.AnnData`, and
:func:`quadsv.visium_to_grid` rasterizes the hex-arranged spots onto a regular
``(78, 128)`` grid (``'dense'`` mode) with the hex offset preserved and empty
cells filled from their two nearest in-row hex neighbours. The function returns
both the ``(n_genes, 78, 128)`` array and the physical spacing
``(dy, dx) = (100·√3/2, 50)`` μm per cell.

Pass that spacing (the same for every v1 Visium slide) to
:class:`~quadsv.SpectralComparator` via ``spacings=`` so that radial bins are
defined in **cycles per μm** (physical frequency) with a single common edge grid
across samples — slides with slightly different in-tissue shapes become directly
comparable. The Nyquist limit along the coarser axis is
``1 / (2 · 86.6 μm) ≈ 5.77 cycles/mm`` (equivalent to a minimum resolvable
pattern of ~170 μm).

A complete DLPFC-vs-GBM example is in
``scripts/dlpfc_vs_gbm_spectral_comparison.ipynb`` and
``scripts/idhm_vs_gbm_spectral_comparison.ipynb``.

Choosing covariate maps for residualization
-------------------------------------------

Pass a list of per-sample covariate arrays (each of shape
``(n_covariates, ny_s, nx_s)``) to :meth:`~quadsv.SpectralComparator.residualize`.
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

See :class:`quadsv.SpectralComparator`, :func:`quadsv.compare_two_groups`,
:func:`quadsv.benchmark_statistics`, and :func:`quadsv.power_spectrum_2d` for full
parameter documentation.
