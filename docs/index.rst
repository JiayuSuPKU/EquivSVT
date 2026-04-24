Welcome
=======

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   guides/installation
   guides/quickstart
   guides/theory
   guides/scaling
   guides/kernels
   guides/multisample
   guides/faq

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   autoapi/quadsv/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   changelog

`quadsv <https://github.com/JiayuSuPKU/EquivSVT>`_ is a Python library for
consistent and scalable spatial pattern detection in omics data.
It implements kernel-based hypothesis tests (Q-tests and R-tests) that unify
major spatial variability detection methods under a single quadratic-form framework.

In the associated paper (`Su et al. 2026 <https://arxiv.org/pdf/2602.02825>`_),
we show that virtually all spatial variable gene (SVG) detection methods --- Moran's I,
parametric models, and dependence tests --- are mathematically equivalent instances
of the Q-test, differing primarily in kernel choice.
We reveal that several widely used methods, including Moran's I, are *inconsistent*,
and propose scalable corrections via the CAR kernel.


Key features
------------

- **Reliable**: Positive definite kernels eliminate false negatives from Moran's I spectral cancellation
- **Scalable**: Sparse solvers and FFT/NUFFT acceleration handle millions of spots and cells
- **Universal**: Works with Visium, Visium HD, MERFISH, lineage trees, any spatial/graph data
- **Integrated**: Native AnnData and SpatialData support


Quick example
-------------

.. code-block:: python

   import numpy as np
   from quadsv import NUFFTKernel, spatial_q_test

   # Spatial coordinates and gene expression
   coords = np.random.default_rng(0).uniform(0, 20, size=(500, 2))
   gene = np.random.default_rng(1).standard_normal(500)

   # Build Matérn kernel via NUFFT (recommended: PD, O(n log n))
   kernel = NUFFTKernel(coords, method='matern', bandwidth=2.0, nu=1.5)

   # Test for spatial variability
   Q, pval = spatial_q_test(gene, kernel)
   print(f"Q = {Q:.4f}, p-value = {pval:.4e}")


Getting started
---------------

- **Installation**: See :doc:`guides/installation`
- **Quick start**: See :doc:`guides/quickstart` for a 5-minute tutorial
- **Theory**: Read :doc:`guides/theory` for mathematical background
- **Kernel design**: See :doc:`guides/kernels` for practical kernel selection tips


Reference
---------

Su, Jiayu, et al. "On the consistent and scalable detection of spatial patterns." arXiv (2026): 2602.02825. `link to paper <https://arxiv.org/pdf/2602.02825>`_


Reporting issues
----------------

If you encounter any issues, please report them on the `GitHub Issues page <https://github.com/JiayuSuPKU/EquivSVT/issues>`_.
