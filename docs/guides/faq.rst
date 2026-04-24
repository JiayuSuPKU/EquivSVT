FAQ
===

Frequently Asked Questions (FAQ)

**Q: What does quadsv mean?**

*quadsv* stands for *quadratic-form spatial variability*. It unifies spatial pattern detection methods under a single quadratic-form framework,

.. math::

   Q_n = \mathbf{z}^\top \mathbf{K} \mathbf{z},

where the kernel matrix :math:`\mathbf{K}` encodes spatial structure.


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

``quadsv`` has three kernel representations and one test layer built on top
of them. You almost always pick via the detector's ``backend`` keyword
rather than hand-building a kernel:

**``backend="matrix"`` (default,** :class:`~quadsv.MatrixKernel`\ **)**

- Works on any spatial coordinates or graphs (irregular layouts).
- The kernel class auto-decides internally whether to materialize the dense
  ``(N, N)`` matrix or keep just the sparse precision matrix and solve with
  an LU factorization on demand — **you never see the dense-vs-sparse
  switch**, it is memory-driven.
- Example: standard Visium, MERFISH, single-cell lineage trees.

**``backend="nufft"`` (**\ :class:`~quadsv.NUFFTKernel`\ **)**

- Also handles irregular spots but never forms an ``(N, N)`` matrix —
  kernel action is evaluated via type-1/type-2 non-uniform FFTs in
  ``O(N log N)`` per feature.
- Ideal for ≥ 10⁴ spots where even the sparse-precision matrix becomes
  prohibitive.
- Grid resolution and spacing are auto-inferred from the coordinates.

**``DetectorGrid`` (**\ :class:`~quadsv.FFTKernel`\ **)**

- Specialized for regular rasterized grids with periodic boundary
  conditions. Consumes :class:`spatialdata.SpatialData` directly.
- O(N log N) via FFT spectral decomposition with no k-NN graph needed.
- Example: Visium HD at millions of spots on a fixed grid.

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


Further help
~~~~~~~~~~~~

- **Documentation**: See :doc:`/guides/quickstart` and :doc:`/guides/theory`
- **API Reference**: Browse :doc:`/autoapi/quadsv/index`
- **GitHub Issues**: Open a ticket at `https://github.com/JiayuSuPKU/EquivSVT/issues <https://github.com/JiayuSuPKU/EquivSVT/issues>`_
