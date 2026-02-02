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
   kernel = SpatialKernel.from_coordinates(
       coords, method='car', k_neighbors=4, rho=0.9
   )

See :doc:`theory` for mathematical details (Theorem 2).


**Q: What's the difference between Q-test and R-test?**

- **Q-test** (univariate): :math:`Q = \mathbf{z}^\top \mathbf{K} \mathbf{z}` — Tests if a single feature is spatially variable
- **R-test** (bivariate): :math:`R = \mathbf{x}^\top \mathbf{K} \mathbf{y}` — Tests if two features are spatially co-expressed

Use Q-test for identifying spatially variable genes (SVGs). Use R-test for finding spatially co-expressed gene pairs.


**Q: What's the difference between matrix and FFT versions?**

**Matrix-based** (``SpatialKernel``, ``PatternDetector``):

- Works on any spatial coordinates or graphs (irregular layouts)
- Uses implicit sparse solvers for scalability (CAR kernel stores precision matrix only)
- Null distribution via Welch approximation (no eigendecomposition needed)
- Example: Visium, single-cell segmented spatial data, single-cell lineage trees

**FFT-based** (``FFTKernel``, ``PatternDetectorFFT``):

- Specialized for regular grids with even spacing
- O(N log N) complexity via FFT spectral decomposition
- Requires periodic boundary conditions
- Example: VisiumHD, SpatialData rasterized to a fixed resolution (e.g., 1000x1000 grid)

.. code-block:: python

   # Matrix version (general)
   from quadsv.kernels import SpatialKernel
   kernel = SpatialKernel.from_coordinates(coords, method='car')
   
   # FFT version (grids)
   from quadsv.fft import FFTKernel
   kernel_fft = FFTKernel(shape=(1000, 1000), method='car')

For irregular data → use matrix version. For large regular grids → use FFT version.

**Q: Can I use quadsv with single-cell data (not spatially resolved)?**

Yes, if you can define spatial relationships (e.g., k-NN graph in PCA space, pseudotime ordering, lineage trees). 
Pass the coordinates to :meth:`quadsv.kernels.SpatialKernel.from_coordinates` or a precomputed kernel matrix directly to :meth:`quadsv.kernels.SpatialKernel.from_matrix`.
If using Anndata and ``PatternDetector``, try :meth:`quadsv.detector.PatternDetector.build_kernel_from_coordinates` and :meth:`quadsv.detector.PatternDetector.build_kernel_from_obsp` for matrix input.

**Q: Does quadsv support 3D spatial coordinates?**

Yes, but currently only for matrix-based kernels. Pass 3D coordinates to :meth:`quadsv.kernels.SpatialKernel.from_coordinates` as usual.


Further help
~~~~~~~~~~~~

- **Documentation**: See :doc:`quickstart` and :doc:`theory`
- **API Reference**: Browse :doc:`api/index`
- **GitHub Issues**: Open a ticket at `https://github.com/JiayuSuPKU/EquivSVT/issues <https://github.com/JiayuSuPKU/EquivSVT/issues>`_
