Cross-sample Comparison
=======================

Suppose you have two groups of spatial-omics samples (for example a
set of healthy controls and a set of cancer sections) and want to
ask which genes show the biggest spatial-pattern difference between
the groups. The :class:`~quadsv.ComparatorIrregular` and
:class:`~quadsv.ComparatorGrid` classes give you a frequency-domain
pipeline that does this without spatial registration, with a
permutation-based p-value per gene.

.. note::

   This API is under active development. Signatures may shift
   between minor releases.


Why frequency domain?
---------------------

The 2-D power spectrum :math:`|\hat x(k)|^2` of a rasterised gene
image is translation-invariant: shifting the image leaves the
spectrum unchanged. Radial averaging additionally makes the
representation rotation-invariant. Together these mean samples never
need to be spatially registered onto each other, which would
otherwise be a hard requirement when (for example) healthy and
cancer slides have no shared anatomy.


Five-step pipeline
------------------

:class:`~quadsv.ComparatorIrregular` chains five stages:

1. Per-sample 2-D power spectra.
2. Reduction to a low-dimensional feature vector. The default is
   radial 1-D bins.
3. Background normalisation that cancels per-slide differences in
   gain and sensitivity (geometric-mean spectrum across all genes
   per sample).
4. Optional residualisation against covariate spectra (cell-type
   maps, tissue-domain indicators, ...).
5. Per-gene two-group test with a sample-label permutation null and
   BH-FDR correction.

.. dropdown:: DC vs AC: separating expression level from pattern shape

   By default the pipeline mean-centres each gene's spatial signal
   before the FFT (``center="mean"``). This splits the information
   cleanly into two orthogonal pieces.

   The **DC scalar** is the per-sample grid mean, i.e. total
   normalised expression. It is tested across groups with
   :meth:`~quadsv.ComparatorIrregular.test_expression`, which runs
   a Welch t-test under a permutation null with BH-FDR. This is a
   spatially-aware differential-expression test.

   The **AC spectrum** is the pattern shape, with DC exactly zero.
   It is tested with
   :meth:`~quadsv.ComparatorIrregular.test_pattern` using one of
   the four statistics listed below.

   The two tests carry complementary information. A gene may be
   "only DE" (same pattern, different magnitude), "only pattern"
   (same total expression, different spatial layout), or both. Run
   them side by side and inspect where the hits overlap and where
   they separate.

   ``center="zscore"`` makes the pattern test scale-invariant
   (it ignores overall amplitude). ``center=None`` disables the
   split. This is legacy behaviour and lets DC and nearby bins
   leak between the two tests.

The four pattern-test statistics ship out of the box and share a
single permutation null, so they are directly comparable:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Statistic
     - What it measures
   * - ``log_l2`` (default)
     - L2 distance between the two groups' mean log-spectra.
   * - ``hotelling_lw``
     - Regularised Hotelling :math:`T^2` with Ledoit-Wolf
       covariance.
   * - ``mmd_rbf``
     - Maximum mean discrepancy with an RBF kernel.
   * - ``max_welch``
     - Max per-bin Welch t-statistic. Useful as an interpretable
       omnibus.

Use :func:`quadsv.comparators.multisample.benchmark_statistics` to
score all four on the same data.


Picking a class
---------------

Two backends, mirroring the detector layer:

- :class:`~quadsv.ComparatorIrregular` takes a list of
  :class:`anndata.AnnData` (irregular spots, common across Visium,
  Slide-seq, Stereo-seq, MERFISH). Spectra are computed with a
  batched type-1 NUFFT. Each sample keeps its own grid shape and
  spacing. Cross-sample comparability comes from radial binning in
  physical-frequency space.
- :class:`~quadsv.ComparatorGrid` takes a list of
  :class:`spatialdata.SpatialData` (regular rasterised bins, e.g.
  Visium HD). Spectra are computed with a single batched 2-D FFT
  per sample.

Sparse ``adata.X`` and layer matrices are not densified up front.
The spectrum loop converts exactly one gene column at a time. The
:func:`~quadsv.Comparator` factory dispatches between the two
classes based on the input list type. Mixed lists raise
``TypeError``.


Toy walkthrough (AnnData / NUFFT)
---------------------------------

Eight synthetic samples (4 per group) of 10 genes. Gene ``g0``
carries a low-frequency stripe pattern in group 1 only.

.. code-block:: python

   import anndata as ad
   import numpy as np
   from quadsv import ComparatorIrregular

   rng = np.random.default_rng(3)
   ny = nx = 32
   gene_names = [f"g{i}" for i in range(10)]

   def make_adata(group: int) -> ad.AnnData:
       yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
       coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
       X = rng.standard_normal((ny * nx, len(gene_names))) * 0.1
       if group == 1:
           X[:, 0] += 1.5 * np.sin(2 * np.pi * yy / 16.0).ravel()
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


Walkthrough (SpatialData / FFT)
-------------------------------

For rasterised-grid samples, swap in :class:`~quadsv.ComparatorGrid`
and pass the same bin / table / coord keys you would pass to
:class:`~quadsv.DetectorGrid`:

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
       value_key=None,               # None means rasterise expression off .X
       fft_chunk_size=256,           # genes per batched scipy.fft call
   ).fit().normalize_background()
   cmp.test(statistic="log_l2", n_perm=300)


Mixed coordinate units (NUFFT path)
-----------------------------------

.. dropdown:: When samples ship coordinates in different physical units

   Some pipelines store coordinates in mixed units. For example one
   slide may be in μm and another in Visium full-resolution pixels
   at 0.35 μm/pixel. Pass ``unit_scales`` to convert each sample's
   raw coords into the common unit. Radial bins then come out in
   cycles per that unit on every sample:

   .. code-block:: python

      cmp = ComparatorIrregular(
          samples,
          groups,
          gene_names=gene_names,
          unit_scales=[1.0, 0.35, 1.0, 0.35],
          spacing=(50.0, 50.0),       # common physical spacing, μm
          n_radial_bins=30,
      ).fit().normalize_background()

   If ``grid_shape`` and ``spacing`` are left unset, each sample's
   k-grid is auto-inferred from its coords via
   :func:`quadsv.kernels.nufft._infer_grid_from_coords`.
   :func:`quadsv.kernels.nufft.power_spectrum_2d_nufft` is the
   lower-level primitive that runs one sample at a time.


Visium hex grids
----------------

For 10x Visium slides,
:func:`quadsv.utils.load_visium_sample` reads a Space Ranger output
directory into an :class:`anndata.AnnData`. You can feed that
:class:`~anndata.AnnData` directly to
:class:`~quadsv.ComparatorIrregular`. The NUFFT backend handles the
hex layout natively, no manual rasterisation needed. If you do want
the explicit hex-to-grid rasterisation,
:func:`quadsv.utils.visium_to_grid` returns a ``(n_genes, 78, 128)``
array and the physical spacing ``(dy, dx) = (100·√3/2, 50)`` μm per
cell for v1 Visium. The smallest resolvable pattern is roughly
``2 · 86.6 μm ≈ 173 μm`` along the coarser axis (the Nyquist
limit).


Choosing covariate maps for residualisation
-------------------------------------------

Pass a list of per-sample covariate arrays of shape
``(n_covariates, ny_s, nx_s)`` to
:meth:`~quadsv.ComparatorIrregular.residualize`. Useful candidates:

- Cell-type proportion maps from a deconvolution tool such as
  Cell2location, CARD, or RCTD. One channel per cell type.
- Tissue-domain indicator maps from a spatial clustering method
  such as BayesSpace or GraphST.
- A composite "housekeeping" expression image to absorb depth
  gradients.

Residualisation is applied after background normalisation and
before testing.


See also
--------

- :doc:`/guides/quickstart` for the single-sample workflow.
- :doc:`/guides/scaling` for how the FFT and NUFFT routines scale.
- :class:`quadsv.ComparatorIrregular` and
  :class:`quadsv.ComparatorGrid` for the class reference.
- :func:`quadsv.comparators.multisample.compare_two_groups` and
  :func:`quadsv.comparators.multisample.benchmark_statistics` for
  the array-level primitives.
- :func:`quadsv.kernels.fft.power_spectrum_2d` and
  :func:`quadsv.kernels.nufft.power_spectrum_2d_nufft` for the
  spectrum primitives.
