Installation
============

Install from PyPI (recommended)
--------------------------------

.. code-block:: bash

   pip install quadsv

Or install with optional dependencies:

.. code-block:: bash

   pip install quadsv[spatial]   # For AnnData/SpatialData support
   pip install quadsv[dev]       # For development tools
   pip install quadsv[docs]      # For building documentation


Development installation
------------------------

Clone and install in editable mode:

.. code-block:: bash

   git clone https://github.com/JiayuSuPKU/EquivSVT.git
   cd EquivSVT
   pip install -e ".[dev,spatial]"


Requirements
------------

- **Python**: 3.10+ (3.14+ users: see note below)
- **Core dependencies**: numpy, scipy, scikit-learn, pandas, scanpy, joblib, tqdm
- **Optional**: anndata (PatternDetector), spatialdata (PatternDetectorFFT)

**Note for Python 3.14+**: Some third-party pytest plugins (napari, npe2) have pydantic v1 incompatibilities. Tests run via ``python -m unittest`` or with plugins disabled in ``pyproject.toml``.


Verify installation
-------------------

.. code-block:: python

   import quadsv
   from quadsv.kernels import SpatialKernel
   from quadsv.statistics import spatial_q_test
   
   print(f"quadsv version: {quadsv.__version__}")
   print("✓ Installation successful!")


