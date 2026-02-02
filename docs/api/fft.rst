quadsv.fft
==========

FFT-based spatial testing for regular grid data.

.. automodule:: quadsv.fft
   :members:
   :member-order: bysource
   :show-inheritance:

Usage example
-------------

.. code-block:: python

   import numpy as np
   from quadsv.fft import FFTKernel
   from quadsv.statistics import spatial_q_test

   # Create FFT kernel for 1000x1000 grid
   kernel_fft = FFTKernel(
       shape=(1000, 1000),
       method='car',
       rho=0.9,
       topology='square'
   )

   # Generate synthetic grid data
   data_grid = np.random.randn(1000, 1000)

   # Compute Q-test efficiently (O(N log N))
   Q, pval = spatial_q_test(data_grid, kernel_fft, return_pval=True)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")

Supported Topologies
~~~~~~~~~~~~~~~~~~~~~

- **square**: 4-neighbor connectivity (default)
- **hexagonal**: 6-neighbor connectivity

Performance
~~~~~~~~~~~

For N = 1,000,000 spots (1000×1000 grid):
- Dense eigendecomposition: ~10 hours
- FFT eigendecomposition: ~1 minute
- Q-test: O(N log N) ≈ 20 seconds
