Quick Start
===========

Get started with ``quadsv`` in 5 minutes.

Q-test basics for spatial variability
------------------------------------------

Test whether a single gene exhibits significant spatial clustering.

.. code-block:: python

   import numpy as np
   from quadsv.kernels import SpatialKernel
   from quadsv.statistics import spatial_q_test

   # Load spatial coordinates (N x 2)
   coords = np.random.randn(500, 2)
   
   # Load gene expression (N,)
   gene_expression = np.random.randn(500)

   # Build CAR kernel (recommended)
   kernel = SpatialKernel.from_coordinates(
       coords, 
       method='car',        # CAR kernel: strictly positive definite
       k_neighbors=15,      # 15-NN graph
       rho=0.9              # Spatial dependence parameter
   )

   # Compute Q-statistic and p-value
   Q, pval = spatial_q_test(gene_expression, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

**Interpretation:**

- High Q-statistic + low p-value → gene is spatially clustered/dispersed
- Low Q-statistic + high p-value → gene is spatially random


R-test basics for spatial co-expression
---------------------------------------

Test whether two genes are spatially co-expressed.

.. code-block:: python

   from quadsv.statistics import spatial_r_test

   gene1 = np.random.randn(500)
   gene2 = np.random.randn(500)

   R, pval = spatial_r_test(gene1, gene2, kernel)
   print(f"R = {R:.4f}, p-value = {pval:.4e}")


Testing SVG for all genes with AnnData
-----------------------------------------------

Detect all spatially variable genes (SVGs) in a tissue sample using ``PatternDetector``, 
a convenient wrapper class to run Q-tests and R-tests on ``AnnData`` objects.

.. code-block:: python

   import anndata as ad
   from quadsv.detector import PatternDetector

   # Load data
   adata = ad.read_h5ad("spatial_tissue.h5ad")
   print(f"Data: {adata.n_obs} spots x {adata.n_vars} genes")

   # Initialize detector
   detector = PatternDetector(adata, min_cells_frac=0.05)

   # Build kernel from spatial coordinates in adata.obsm
   detector.build_kernel_from_coordinates(
       adata.obsm['spatial'],
       method='car',
       rho=0.9,
       k_neighbors=15
   )

   # Test all genes in parallel
   results = detector.compute_qstat(
       source='var',           # Test genes (not observations)
       n_jobs=4,               # Use 4 CPU cores
       return_pval=True        # Include p-values
   )

   # Filter results
   svg_genes = results[results['P_adj'] < 0.05]
   print(f"Found {len(svg_genes)} SVGs at FDR < 5%")
   print(results.head())

Pairwise spatial co-expression with AnnData
-----------------------------------------------

After identifying SVGs, test for spatial co-expression among top genes.

.. code-block:: python

   # Select top 100 SVGs by Q-statistic
   top_genes = results.nlargest(100, 'Q').index.tolist()

   # Test all pairwise combinations
   r_results = detector.compute_rstat(
       source='var',
       features_x=top_genes,
       features_y=None,           # None = all pairs within features_x
       n_jobs=4,
       return_pval=True
   )

   # Filter co-expressed pairs
   coexp_pairs = r_results[r_results['P_adj'] < 0.05]
   print(f"Found {len(coexp_pairs)} spatially co-expressed pairs")



Large-scale analysis using FFT with SpatialData
-----------------------------------------------

For regular grids (e.g., Visium HD with 1M+ spots), use FFT acceleration. 
The ``PatternDetectorFFT`` class works with ``spatialdata.SpatialData`` objects and implicitly
calls ``sdata.rasterize_bins()`` to create dense inputs on a fixed-resolution grid.

.. code-block:: python

   import spatialdata as sd
   from quadsv.detector_fft import PatternDetectorFFT

   # Load SpatialData object
   sdata = sd.read_zarr("visium_hd.zarr")

   # Initialize FFT detector
   detector_fft = PatternDetectorFFT(
       sdata,
       min_count=10,              # Minimum gene counts
       kernel_method='car',       # CAR kernel
       rho=0.9,
       neighbor_degree=1,         # 1-hop neighborhood
       topology='square',         # Square grid
       fft_solver='rfft2'         # Real FFT for memory efficiency
   )

   # Compute Q-statistics (O(N log N) complexity!)
   results = detector_fft.compute_qstat(
       bins=['Visium_HD_bin_name'],
       table_name=['table_name'],
       col_key='array_col',
       row_key='array_row',
       n_jobs=4,
       workers=2,                 # Parallel FFT workers
       chunk_size=256,            # Process 256 genes at a time
       return_pval=True
   )

   svg_genes = results[results['P_adj'] < 0.05]
   print(f"Analyzed 1M+ spots in minutes: {len(svg_genes)} SVGs")

  # Compute R-statistics for top genes
   top_genes_fft = results.nlargest(100, 'Q').index.tolist()
   r_results_fft = detector_fft.compute_rstat(
       bins=['Visium_HD_bin_name'],
       table_name=['table_name'],
       col_key='array_col',
       row_key='array_row',
       features_x=top_genes_fft,
       features_y=None,
       n_jobs=4,
       workers=2,
       chunk_size=128,
       return_pval=True
   )

   coexp_pairs_fft = r_results_fft[r_results_fft['P_adj'] < 0.05]
   print(f"Found {len(coexp_pairs_fft)} spatially co-expressed pairs in large grid")


Next Steps
----------

- **Theory**: Read :doc:`theory` for mathematical background
- **Kernel design**: See :doc:`kernels` for practical tips on kernel design
- **API Reference**: Browse :doc:`api/index`
