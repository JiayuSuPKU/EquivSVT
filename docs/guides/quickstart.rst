Quick Start
===========

Get started with ``quadsv`` in 5 minutes.

The library exposes *four conceptual layers*:

.. list-table::
   :header-rows: 1
   :widths: 14 32 54

   * - Layer
     - What it is for
     - Implementations
   * - **Kernels**
     - Describe the spatial pattern to capture and which frequencies to prioritize.
     - :class:`~quadsv.MatrixKernel` on arbitrary coords or adjacency
       (dense / sparse / sparse-precision auto-switched);
       :class:`~quadsv.FFTKernel` on a regular grid with periodic
       boundaries; :class:`~quadsv.NUFFTKernel` on irregular 2-D coords.
   * - **Statistical tests**
     - Evaluate the Q / R quadratic form and its null p-value on a
       given data vector (or batch).
     - :func:`~quadsv.spatial_q_test` for univariate spatial variability
       and :func:`~quadsv.spatial_r_test` for bivariate co-expression.
   * - **Detectors**
     - Transcriptome-wide SVG / co-expression screening over a single sample.
     - :class:`~quadsv.DetectorIrregular` for
       :class:`anndata.AnnData` (``backend={'matrix', 'nufft'}``
       switch); :class:`~quadsv.DetectorGrid` for
       :class:`spatialdata.SpatialData` (FFT-accelerated on rasterised
       bins).
   * - **Comparators**
     - Alignment-free cross-sample pattern comparison between groups of slides 
       in the frequency domain.
     - :class:`~quadsv.ComparatorIrregular` for a list of
       :class:`anndata.AnnData` (NUFFT backend);
       :class:`~quadsv.ComparatorGrid` for a list of
       :class:`spatialdata.SpatialData` (FFT backend).


Q-test basics for spatial variability
-------------------------------------

Test whether a single gene exhibits significant spatial clustering. The
default recommendation on irregular coords is :class:`~quadsv.NUFFTKernel`
with the Matérn kernel — positive definite, ``O(n log n)`` per feature via
``finufft``, and the grid shape / spacing are auto-inferred from the
coords (no kernel matrix ever materialised).

.. code-block:: python

   import numpy as np
   from quadsv import NUFFTKernel, spatial_q_test

   rng = np.random.default_rng(0)
   coords = rng.uniform(0, 20, size=(500, 2))
   gene_expression = rng.standard_normal(500)

   # Matérn kernel (positive definite, smooth low-pass)
   kernel = NUFFTKernel(
       coords,
       method="matern",
       bandwidth=2.0,
       nu=1.5,
   )

   Q, pval = spatial_q_test(gene_expression, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

**Interpretation:**

- High Q + low p-value → gene is spatially clustered/dispersed
- Low Q + high p-value → gene is spatially random

The same ``spatial_q_test`` / ``spatial_r_test`` calls dispatch on the
kernel type — a :class:`~quadsv.MatrixKernel` takes a flat ``(n,)`` or
``(n, M)`` array, an :class:`~quadsv.FFTKernel` takes a 2-D grid
``(ny, nx)`` or ``(ny, nx, M)``, and an :class:`~quadsv.NUFFTKernel` takes
a flat ``(n,)`` / ``(n, M)`` just like MatrixKernel. No per-backend
functions to import:

.. code-block:: python

   from quadsv import spatial_q_test

   # spatial_q_test(values_1d, matrix_kernel)  # Matrix path
   # spatial_q_test(img_2d, fft_kernel)        # FFT path
   # spatial_q_test(values_1d, nufft_kernel)   # NUFFT path


R-test basics for spatial co-expression
---------------------------------------

Test whether two genes are spatially co-expressed (reuses the same
kernel built above):

.. code-block:: python

   from quadsv import spatial_r_test

   gene1 = rng.standard_normal(500)
   gene2 = rng.standard_normal(500)

   R, pval = spatial_r_test(gene1, gene2, kernel)
   print(f"R = {R:.4f}, p-value = {pval:.4e}")


Testing SVG for all genes with AnnData
--------------------------------------

Detect all spatially variable genes (SVGs) in a tissue sample using
:class:`~quadsv.DetectorIrregular`, the wrapper around
:class:`anndata.AnnData`. Kernel type and parameters are picked in
``__init__``; the data is attached via :meth:`setup_data`:

- ``backend="nufft"`` (recommended) — :class:`~quadsv.NUFFTKernel`.
  ``O(n log n)`` per feature via ``finufft``, never forms an ``(n, n)``
  matrix. Grid shape / spacing auto-inferred from the coords. Pairs
  naturally with Matérn / Gaussian kernels.
- ``backend="matrix"`` — :class:`~quadsv.MatrixKernel`. Picks between a
  materialized dense kernel and a sparse precision + LU-solve
  representation based on ``n_obs``; use this for graph kernels (``car``,
  ``moran``) on smaller ``n`` or for precomputed ``adata.obsp``
  adjacencies.

.. code-block:: python

   import anndata as ad
   from quadsv import DetectorIrregular

   adata = ad.read_h5ad("spatial_tissue.h5ad")
   print(f"Data: {adata.n_obs} spots x {adata.n_vars} genes")

   detector = (
       DetectorIrregular(
           kernel_method="matern",
           backend="nufft",
           bandwidth=25.0,   # in the same units as adata.obsm["spatial"]
           nu=1.5,
       )
       .setup_data(adata, obsm_key="spatial", min_cells_frac=0.05)
   )

   q_results = detector.compute_qstat(source="var", n_jobs=4, return_pval=True)

   svgs = q_results[q_results["P_adj"] < 0.05]
   print(f"Found {len(svgs)} SVGs at FDR < 5%")
   print(q_results.head())


Graph-kernel alternative via the Matrix backend
-----------------------------------------------

If you need a graph-flavored kernel (CAR, Moran, graph Laplacian) on a
smaller ``n`` — or if your ``adata.obsp`` already carries a custom
adjacency — switch to ``backend="matrix"``:

.. code-block:: python

   detector = DetectorIrregular(
       kernel_method="car",
       backend="matrix",
       rho=0.9,
       k_neighbors=15,
   ).setup_data(adata, obsm_key="spatial", min_cells_frac=0.05)
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
through :class:`~quadsv.DetectorGrid`, which consumes
:class:`spatialdata.SpatialData`. Kernel hyperparameters go to
``__init__``; the SpatialData and its bin / table / coord layout go to
:meth:`setup_data`:

.. code-block:: python

   import spatialdata as sd
   from quadsv import DetectorGrid

   sdata = sd.read_zarr("visium_hd.zarr")

   detector = DetectorGrid(
       kernel_method="car",
       rho=0.9,
       neighbor_degree=1,
       topology="square",
       fft_solver="rfft2",
   ).setup_data(
       sdata,
       bins="Visium_HD_bin_name",
       table_name="table_name",
       col_key="array_col",
       row_key="array_row",
       min_count=10,
   )

   results = detector.compute_qstat(
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
- **Cross-sample comparison**: :doc:`/guides/multisample`
- **API reference**: :doc:`/autoapi/quadsv/index`
