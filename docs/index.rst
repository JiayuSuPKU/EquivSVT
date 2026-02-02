Consistent and scalable detection of spatial patterns
==============================================================

Welcome to ``quadsv``, a Python library for detecting patterns in spatial omics data and beyond.

In our accompanied paper, we unify spatial pattern detection methods through a single quadratic-form Q-statistic framework. 
Major spatial variable gene (SVG) detection methods (Moran's I, parametric models, dependence tests) are mathematically 
equivalent instances of the Q-test, differing primarily in kernel choice. 
We reveal that several widely used methods, including Moran's I, are inconsistent, and propose scalable corrections. 

This package implements the resulting test, enabling robust pattern detection across millions of spatial locations and single-cell lineage-tracing datasets.

**Key features**:

- **Unified framework**: Major SVG detection methods reduce to the quadratic-form Q-test
- **Theoretically consistent**: CAR kernel ensures robust detection of mean shifts (Theorems 1 and 2)
- **Scalable**: Sparse and implicit kernel operations for general graphs, FFT acceleration for grids

Quick links
-----------

- **New to quadsv?** Start with :doc:`quickstart`
- **Understanding theory?** Read :doc:`theory`
- **Choosing kernels?** See :doc:`kernels`
- **API Reference?** Browse :doc:`api/index`

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quickstart
   theory
   kernels
   faq
   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
