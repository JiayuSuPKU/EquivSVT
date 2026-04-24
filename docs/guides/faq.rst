FAQ
===

Frequently Asked Questions (FAQ)

**Q: What does quadsv mean?**

*quadsv* stands for *quadratic-form spatial variability*. It unifies spatial pattern detection methods under a single quadratic-form framework,

.. math::

   Q_n = \mathbf{z}^\top \tilde{\mathbf{K}} \mathbf{z},

where the (double-centered) kernel matrix :math:`\tilde{\mathbf{K}}=\mathbf{HKH}` encodes spatial structure.


**Q: Why is Moran's I potentially problematic for SVG detection?**

Moran's I uses the adjacency matrix :math:`\mathbf{W}` as a kernel, which is *indefinite* (has both positive and negative eigenvalues). This causes *spectral cancellation*—signals aligned with positive and negative eigenspaces cancel out, leading to false negatives and theoretical inconsistency.

The **CAR kernel** :math:`\mathbf{K} = (\mathbf{I} - \rho \mathbf{\tilde{W}})^{-1}` resolves this by ensuring *strict positive definiteness*. All eigenvalues > 0, guaranteeing consistency.

.. code-block:: python

   # Use CAR instead of Moran's I
   kernel = MatrixKernel.from_coordinates(
       coords, method='car', k_neighbors=4, rho=0.9
   )

See :doc:`/guides/theory` for mathematical details (Theorem 2).


**Q: What's the difference between Q-test and R-test?**

- **Q-test** (univariate): :math:`Q = \mathbf{z}^\top \mathbf{K} \mathbf{z}` — Tests if a single feature is spatially variable
- **R-test** (bivariate): :math:`R = \mathbf{x}^\top \mathbf{K} \mathbf{y}` — Tests if two features are spatially co-expressed

Use Q-test for identifying spatially variable genes (SVGs). Use R-test for finding spatially co-expressed gene pairs.


**Q: Which backend should I pick?**

``quadsv`` has three implementations for spatial kernels and one test layer built on top of them. 
You almost always pick via the detector's ``backend`` keyword rather than hand-building a kernel:

``backend="matrix"`` (:class:`~quadsv.MatrixKernel`)

- Works on any spatial coordinates and graphs (e.g., expression-based k-NN graphs).
- The kernel class auto-decides internally whether to materialize the dense
  ``(N, N)`` matrix or keep just the sparse precision matrix and solve with
  an LU factorization on demand.
- Example: in situ data like MERFISH, or single-cell lineage trees.

``backend="nufft"`` (:class:`~quadsv.NUFFTKernel`)

- Works on any 2D spatial coordinates.
- Kernel action is evaluated via type-1/type-2 non-uniform FFTs in
  ``O(N log N)`` per feature.
- Ideal for ≥ 1M spots where even the sparse-precision matrix becomes too slow.
- Example: in situ data like MERFISH, or Visium HD (segmented).

``DetectorGrid`` (:class:`~quadsv.FFTKernel`)

- Specialized for regular rasterized grids and consumes :class:`spatialdata.SpatialData` directly.
- O(N log N) via FFT spectral decomposition with no k-NN graph needed.
- Example: Visium HD on a fixed grid (16/8/2um).

.. code-block:: python

   from quadsv import DetectorIrregular, DetectorGrid

   # Irregular layout, small-to-moderate N
   det = DetectorIrregular(kernel_method="car", backend="matrix").setup_data(adata)

   # Irregular layout, large N
   det = DetectorIrregular(kernel_method="matern", backend="nufft").setup_data(adata)

   # Regular grid, SpatialData
   det = DetectorGrid(kernel_method="car", rho=0.9, topology="square").setup_data(
       sdata, bins="...", table_name="...", col_key="array_col", row_key="array_row"
   )

**Q: Can I use quadsv with single-cell data (not spatially resolved)?**

Yes, if you can define spatial relationships (e.g., k-NN graph in PCA
space, pseudotime ordering, lineage trees). Pass the coordinates to
:meth:`quadsv.MatrixKernel.from_coordinates`, or a precomputed kernel /
precision matrix directly to :meth:`quadsv.MatrixKernel.from_matrix`. If
you have an :class:`~anndata.AnnData` and want to build the kernel from
``.obsp``, call :meth:`DetectorIrregular.setup_data` with ``obsp_key=...``
(optionally ``is_distance=True`` if the matrix stores distances rather than
affinities).

**Q: Does quadsv support 3D spatial coordinates?**

Yes, but currently only for matrix-based kernels. Pass 3D coordinates to :meth:`quadsv.kernels.MatrixKernel.from_coordinates` as usual.
If you would like to see NUFFT and FFT backends support 3D coordinates, please open a feature request at `https://github.com/JiayuSuPKU/EquivSVT/issues`.


Further help
~~~~~~~~~~~~

- **Documentation**: See :doc:`/guides/quickstart` and :doc:`/guides/theory`
- **API Reference**: Browse :doc:`/autoapi/quadsv/index`
- **GitHub Issues**: Open a ticket at `https://github.com/JiayuSuPKU/EquivSVT/issues <https://github.com/JiayuSuPKU/EquivSVT/issues>`_
