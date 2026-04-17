Quick Start
===========

Get started with ``quadsv`` in 5 minutes.

The library exposes **four conceptual layers** and nothing else:

1. **Kernels** — :class:`~quadsv.SpatialKernel` (dense/sparse),
   :class:`~quadsv.FFTKernel` (regular grid), :class:`~quadsv.NUFFTKernel`
   (irregular 2D coordinates).
2. **Statistical tests** — :func:`~quadsv.spatial_q_test` /
   :func:`~quadsv.spatial_r_test` plus their FFT and NUFFT variants. All six
   share the same signature shape.
3. **PatternDetector** — :class:`~quadsv.PatternDetector` for
   :class:`anndata.AnnData` with a ``backend={'matrix', 'nufft'}`` switch;
   :class:`~quadsv.PatternDetectorFFT` for :class:`spatialdata.SpatialData`.
4. **Pattern comparators** — cross-sample pattern comparison across a list of
   AnnData (→ NUFFT) or SpatialData (→ FFT) objects.


Q-test basics for spatial variability
-------------------------------------

Test whether a single gene exhibits significant spatial clustering.

.. code-block:: python

   import numpy as np
   from quadsv import SpatialKernel, spatial_q_test

   rng = np.random.default_rng(0)
   coords = rng.standard_normal((500, 2))
   gene_expression = rng.standard_normal(500)

   # CAR kernel (recommended: strictly positive definite)
   kernel = SpatialKernel.from_coordinates(
       coords,
       method="car",
       k_neighbors=15,
       rho=0.9,
   )

   Q, pval = spatial_q_test(gene_expression, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

**Interpretation:**

- High Q + low p-value → gene is spatially clustered/dispersed
- Low Q + high p-value → gene is spatially random

All six test variants share one signature — ``(x[, y], kernel,
null_params=None, return_pval=True, is_standardized=False)`` — so you can swap
backends without changing the call:

.. code-block:: python

   from quadsv import spatial_q_test_fft, spatial_q_test_nufft

   # spatial_q_test_fft(img_2d, fft_kernel)
   # spatial_q_test_nufft(values_1d, nufft_kernel)


R-test basics for spatial co-expression
---------------------------------------

Test whether two genes are spatially co-expressed.

.. code-block:: python

   from quadsv import spatial_r_test

   gene1 = rng.standard_normal(500)
   gene2 = rng.standard_normal(500)

   R, pval = spatial_r_test(gene1, gene2, kernel)
   print(f"R = {R:.4f}, p-value = {pval:.4e}")


Testing SVG for all genes with AnnData
--------------------------------------

Detect all spatially variable genes (SVGs) in a tissue sample using
:class:`~quadsv.PatternDetector`, the wrapper around :class:`anndata.AnnData`.
The single :meth:`~quadsv.PatternDetector.build_kernel` entry point picks the
right kernel representation via ``backend``:

- ``backend="matrix"`` (default) — :class:`~quadsv.SpatialKernel`. The class
  internally switches between a materialized dense kernel and a sparse
  precision-matrix + LU-solve representation based on ``n_obs``; you never
  choose the representation directly.
- ``backend="nufft"`` — :class:`~quadsv.NUFFTKernel`. Ideal for ≥ 10⁴
  irregular spots; O(N log N) per feature via finufft.

.. code-block:: python

   import anndata as ad
   from quadsv import PatternDetector

   adata = ad.read_h5ad("spatial_tissue.h5ad")
   print(f"Data: {adata.n_obs} spots x {adata.n_vars} genes")

   detector = (
       PatternDetector(adata, min_cells_frac=0.05)
       .build_kernel(
           backend="matrix",
           method="car",
           coordinates_key="spatial",
           rho=0.9,
           k_neighbors=15,
       )
   )

   q_results = detector.compute_qstat(source="var", n_jobs=4, return_pval=True)

   svgs = q_results[q_results["P_adj"] < 0.05]
   print(f"Found {len(svgs)} SVGs at FDR < 5%")
   print(q_results.head())


Scaling to ≥ 10⁴ irregular spots via NUFFT
------------------------------------------

Switching to the NUFFT backend needs one argument change. The k-grid and
per-axis spacing are auto-inferred from the coordinates:

.. code-block:: python

   detector = PatternDetector(adata).build_kernel(
       backend="nufft",
       method="matern",
       coordinates_key="spatial",
       bandwidth=25.0,   # in the same units as adata.obsm["spatial"]
       nu=1.5,
   )
   q_results = detector.compute_qstat(n_jobs=4)


Pairwise spatial co-expression
------------------------------

After identifying SVGs, test top genes for spatial co-expression. The API is
identical for both backends:

.. code-block:: python

   top_genes = q_results.nlargest(100, "Q").index.tolist()

   r_results = detector.compute_rstat(
       source="var",
       features_x=top_genes,
       features_y=None,    # None = all pairs within features_x
       n_jobs=4,
       return_pval=True,
   )

   coexp_pairs = r_results[r_results["P_adj"] < 0.05]
   print(f"{len(coexp_pairs)} spatially co-expressed pairs")


Large regular grids via FFT + SpatialData
-----------------------------------------

For regular grids (e.g., Visium HD with 1M+ spots), use FFT acceleration
through :class:`~quadsv.PatternDetectorFFT`, which consumes
:class:`spatialdata.SpatialData`:

.. code-block:: python

   import spatialdata as sd
   from quadsv import PatternDetectorFFT

   sdata = sd.read_zarr("visium_hd.zarr")

   detector_fft = PatternDetectorFFT(
       sdata,
       min_count=10,
       kernel_method="car",
       rho=0.9,
       neighbor_degree=1,
       topology="square",
       fft_solver="rfft2",
   )

   results = detector_fft.compute_qstat(
       bins=["Visium_HD_bin_name"],
       table_name=["table_name"],
       col_key="array_col",
       row_key="array_row",
       n_jobs=4,
       workers=2,
       chunk_size=256,
       return_pval=True,
   )
   svgs_fft = results[results["P_adj"] < 0.05]


Next steps
----------

- **Theory**: :doc:`/guides/theory`
- **Kernel design**: :doc:`/guides/kernels`
- **Cross-sample comparison**: :doc:`/guides/spectral_compare`
- **API reference**: :doc:`/autoapi/quadsv/index`
