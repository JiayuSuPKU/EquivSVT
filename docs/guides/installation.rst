Installation
============

Install from PyPI (recommended)
--------------------------------

.. code-block:: bash

   pip install quadsv

``quadsv`` ships with one set of runtime dependencies. The ``[dev]`` and ``[docs]`` extras are development-time only:

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

  * ``scanpy`` (which transitively pulls ``anndata``, ``numpy``, ``scipy``, ``scikit-learn``, ``pandas``)
  * ``spatialdata`` — used by :class:`~quadsv.DetectorGrid` and the
    :class:`~quadsv.ComparatorIrregular` FFT backend
  * ``finufft`` — used by :class:`~quadsv.NUFFTKernel`,
    :class:`~quadsv.DetectorIrregular` with ``backend='nufft'``, and the
    :class:`~quadsv.ComparatorIrregular` NUFFT backend
  * ``joblib``, ``tqdm``

Verify installation
-------------------

.. code-block:: python

   import quadsv

   print("quadsv version:", quadsv.__version__)
   print("Public API:", sorted(quadsv.__all__))
   # ['ComparatorGrid', 'ComparatorIrregular', 'DetectorGrid',
   #  'DetectorIrregular', 'FFTKernel', 'MatrixKernel', 'MatrixKernelBase',
   #  'NUFFTKernel', 'spatial_q_test', 'spatial_r_test']
