Installation
============

Install from PyPI (recommended)
--------------------------------

.. code-block:: bash

   pip install quadsv

``quadsv`` ships with one set of runtime dependencies — everything needed for
the full four-layer API (Kernels, statistical tests, PatternDetector /
PatternDetectorFFT, SpectralComparator). There are no "optional" runtime extras
you need to remember to add.

The ``[dev]`` and ``[docs]`` extras are development-time only:

.. code-block:: bash

   pip install 'quadsv[dev]'    # testing + linting + packaging + jupyter/matplotlib
   pip install 'quadsv[docs]'   # Sphinx + theme + autodoc + autobuild


Development installation
------------------------

Latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/JiayuSuPKU/EquivSVT.git#egg=quadsv

Alternatively, clone and install in editable mode with the dev tools:

.. code-block:: bash

   git clone https://github.com/JiayuSuPKU/EquivSVT.git
   cd EquivSVT
   pip install -e '.[dev,docs]'


Requirements
------------

- **Python**: 3.10+
- **Runtime dependencies** (all required):

  * ``numpy``, ``scipy``, ``scikit-learn``, ``pandas``
  * ``scanpy`` (which transitively pulls ``anndata``)
  * ``spatialdata`` — used by :class:`~quadsv.PatternDetectorFFT` and the
    :class:`~quadsv.PatternComparatorNUFFT` FFT backend
  * ``finufft`` — used by :class:`~quadsv.NUFFTKernel`, the ``*_nufft`` tests,
    :class:`~quadsv.PatternDetector` with ``backend='nufft'``, and the
    :class:`~quadsv.PatternComparatorNUFFT` NUFFT backend
  * ``joblib``, ``tqdm``

Verify installation
-------------------

.. code-block:: python

   import quadsv

   print("quadsv version:", quadsv.__version__)
   print("Public API:", sorted(quadsv.__all__))
   # ['FFTKernel', 'NUFFTKernel', 'PatternDetector', 'PatternDetectorFFT',
   #  'SpatialKernel', 'SpectralComparator', 'spatial_q_test',
   #  'spatial_q_test_fft', 'spatial_q_test_nufft', 'spatial_r_test',
   #  'spatial_r_test_fft', 'spatial_r_test_nufft']
