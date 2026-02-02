quadsv.statistics
=================

Statistical testing framework (Q-tests, R-tests, null approximations).

.. automodule:: quadsv.statistics
   :members:
   :member-order: bysource
   :show-inheritance:

Null approximation methods
--------------------------

Three strategies available via ``null_approx`` parameter:

- **clt**: O(N) via Hutchinson trace. For indefinite kernels and Z-scores.
- **welch**: O(N) via Hutchinson trace. Gamma moment matching. Default for large N.
- **liu**: O(N³) eigendecomposition. 4-moment weighted chi-square. Most accurate for N < 5000 or FFT grids.

See :func:`compute_null_params` for details.
