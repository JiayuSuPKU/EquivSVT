quadsv.detector_fft
===================

FFT-accelerated pattern detection for large grids (Visium HD).

.. automodule:: quadsv.detector_fft
   :members:
   :member-order: bysource
   :show-inheritance:

Usage example
-------------

.. code-block:: python

   import spatialdata as sd
   from quadsv.detector_fft import PatternDetectorFFT

   # Load SpatialData object (e.g., Visium HD)
   sdata = sd.read_zarr("visium_hd.zarr")

   # Initialize FFT detector
   detector = PatternDetectorFFT(
       sdata,
       min_count=10,
       kernel_method='car',
       rho=0.9,
       topology='square'
   )

   # Compute Q-statistics (O(N log N) complexity!)
   results = detector.compute_qstat(
       bins=['Visium_HD_bin'],
       table_name=['table'],
       n_jobs=4,
       workers=2,
       return_pval=True
   )

   svg_genes = results[results['P_adj'] < 0.05]
   print(f"Found {len(svg_genes)} SVGs in large grid")
