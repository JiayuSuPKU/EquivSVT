quadsv.detector
===============

High-level pattern detection for AnnData objects.

.. automodule:: quadsv.detector
   :members:
   :member-order: bysource
   :show-inheritance:

Usage example
-------------

.. code-block:: python

   import anndata as ad
   from quadsv.detector import PatternDetector

   # Load spatial transcriptomics data
   adata = ad.read_h5ad("spatial_data.h5ad")

   # Initialize detector
   detector = PatternDetector(adata, min_cells_frac=0.05)

   # Build kernel from coordinates
   detector.build_kernel_from_coordinates(
       adata.obsm['spatial'],
       method='car',
       k_neighbors=15,
       rho=0.9
   )

   # Compute Q-statistics (SVG detection)
   results = detector.compute_qstat(
       source='var',
       n_jobs=4,
       return_pval=True
   )

   # Filter significant genes
   svg_genes = results[results['P_adj'] < 0.05]
   print(f"Found {len(svg_genes)} SVGs")
