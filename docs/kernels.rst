Kernel Design
=============

Introduction
------------

The spatial Q-test framework unifies pattern detection under the quadratic form,

.. math::

   Q_n = \mathbf{z}^\top \mathbf{K} \mathbf{z}.

The kernel matrix :math:`\mathbf{K}` encodes spatial structure. **Test consistency requires positive definiteness**; indefinite kernels suffer spectral cancellation and produce false negatives.

**See theory:** :doc:`theory` for Theorems 1-2, CAR kernel solution, null distributions, and FFT acceleration.


Kernel spectrum and power
--------------------------

The Q-test power for a spatial pattern :math:`f` is governed by the spectrum

.. math::

   Q(f) = \sum_{\omega} |\hat{f}_\omega|^2 \lambda(\omega).

**Key principles:**

- **Spectrum is a frequency filter**: :math:`\lambda(\omega)` determines which spatial frequencies the test amplifies
- **No universally optimal kernel**: with fixed trace, boosting some frequencies suppresses others
- **Data-driven kernels are risky**: gene-wise tuning inflates Type I error; global tuning biases toward dominant patterns

**Kernel comparison:**

- **Gaussian** (distance): Soft low-pass. The exponential spectral decay may cause loss of mid/high-frequency signals.
- **Matérn** (distance): Polynomial decay with tunable smoothness.
- **Moran's I** (graph): **Indefinite spectrum** → spectral cancellation. **Do not recommend** for SVG detection.
- **Graph Laplacian** (graph): High-pass filter. Detects roughness/textures, weak for clustering.
- **CAR / Inverse Laplacian** (graph, default): Polynomial low-pass with heavy tail. **Best choice** for clustering with better mid/high-frequency sensitivity.
- **FFT-CAR** (grid, default): Same as CAR but O(N log N) via FFT. **Use for Visium HD** and regular grids.

**Practical takeaway:** Use **CAR** (graphs) or **FFT-CAR** (grids) as robust defaults. Use **Matérn** for distance-based kernels.

Quick Start examples
--------------------

**Spatial transcriptomics (Visium, MERFISH)**

.. code-block:: python

   from quadsv.kernels import SpatialKernel
   kernel = SpatialKernel.from_coordinates(
       coords, method='car', k_neighbors=4, rho=0.9
   )

**Large-scale grids (Visium HD)**

.. code-block:: python

   from quadsv.fft import FFTKernel
   # the spatial data is presented as a 1000x1000 rasterized grid (e.g., via SpatialData)
   kernel = FFTKernel(shape=(1000, 1000), method='car', rho=0.9, neighbor_degree=1)

**Small datasets or distance-based**

.. code-block:: python

   kernel = SpatialKernel.from_coordinates(
       coords, method='matern', bandwidth=1.0, nu=1.5
   )


Parameter tuning
----------------

**Kernel bandwidth** (Gaussian, Matérn)

Use median pairwise distance as starting point:

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import pdist
   
   dist = pdist(coords)
   bandwidth_init = np.median(dist)
   kernel = SpatialKernel.from_coordinates(
       coords, method='gaussian', bandwidth=bandwidth_init
   )

**CAR spatial dependence** (rho)

- rho = 0.5: Moderate dependence
- rho = 0.9: Strong dependence (recommended)
- rho → 1: Maximum smoothing

.. code-block:: python

   rhos = [0.5, 0.7, 0.9, 0.95]
   for rho in rhos:
       kernel = SpatialKernel.from_coordinates(
           coords, method='car', rho=rho
       )
       Q, pval = spatial_q_test(data, kernel)
       print(f"rho={rho}: Q={Q:.4f}, pval={pval:.4e}")

**k-Neighbors** (graph kernels)

- k = 5–10: Sparse, local
- k = 30+: Dense, global

.. code-block:: python

   ks = [5, 10, 15, 20, 30]
   for k in ks:
       kernel = SpatialKernel.from_coordinates(
           coords, method='car', k_neighbors=k, rho=0.9
       )
       trace = kernel.trace()
       print(f"k={k}: trace={trace:.4f}")

Custom kernel design
--------------------

**Option 1: Build from custom adjacency or distance matrix**

Use ``SpatialKernel.from_matrix()``:

.. code-block:: python

   import numpy as np
   from quadsv.kernels import SpatialKernel
   
   # From adjacency matrix
   W = ...  # (N, N) adjacency, e.g., from graph structure
   L = np.eye(W.shape[0]) - 0.9 * W  # CAR precision (K^-1)
   kernel = SpatialKernel.from_matrix(L, is_inverse=True, method='precomputed')


**Option 2: Subclass ``Kernel`` for custom implementations**

Implement required methods:

- ``_build_kernel()``: Return kernel matrix (explicit) or precision (implicit)
- ``realization()``: Return dense (N, N) kernel matrix
- ``xtKx(x)``: Compute :math:`x^\top K x`
- ``trace()``: Return trace of kernel
- ``eigenvalues(k)``: Return top k eigenvalues
- ``square_trace()``: Return :math:`\text{tr}(K^2)`

.. code-block:: python

   from quadsv.kernels import Kernel
   import numpy as np
   
   class MyKernel(Kernel):
       def _build_kernel(self):
           # Return your custom (N, N) SPD matrix
           K = self.params["K"]
           return K
       
       def realization(self):
           return self._K
       
       def xtKx(self, x):
           return x.T @ self._K @ x
       
       def trace(self):
           return np.trace(self._K)
       
       def eigenvalues(self, k=None):
           evals = np.linalg.eigvalsh(self._K)
           return evals[-k:] if k else evals
       
       def square_trace(self):
           return np.trace(self._K @ self._K)
   
   # Usage
   K_custom = np.eye(100)  # Replace with your SPD kernel
   kernel = MyKernel(n=100, method="custom", K=K_custom)

**Option 3: Subclass ``FFTKernel`` for grid data**

Override ``_compute_eigenvalues()`` to define custom spectrum. Store result in ``self.spectrum``:

.. code-block:: python

   import numpy as np
   import scipy.fft
   from quadsv.fft import FFTKernel
   
   class CustomFFTKernel(FFTKernel):
       def _compute_eigenvalues(self):
           # Define custom spectral filter
           fy = scipy.fft.fftfreq(self.ny, d=self.dy)
           fx = scipy.fft.fftfreq(self.nx, d=self.dx)
           FY, FX = np.meshgrid(fy, fx, indexing="ij")
           r2 = FX**2 + FY**2
           lam = 1.0 / (1.0 + 10.0 * r2)  # Polynomial decay
           if self.fft_solver == "rfft2":
               lam = lam[:, :(self.nx // 2 + 1)]
           return lam.ravel()  # Must return flattened spectrum
   
   kernel_fft = CustomFFTKernel(shape=(256, 256), method="custom")

**Tip:** Keep :math:`\lambda(\omega) > 0` for all frequencies to ensure consistency.


Spectrum debugging
------------------

Verify kernel properties by inspecting eigenvalues:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Get eigenvalues
   evals = kernel.eigenvalues(k=100)
   
   # Check positivity
   print(f"Min eigenvalue: {evals[-1]:.6e}")
   print(f"All positive: {np.all(evals > 1e-9)}")
   
   # Plot spectrum
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
   
   ax1.semilogy(evals, 'o-')
   ax1.set_ylabel('Eigenvalue magnitude')
   ax1.set_xlabel('Rank')
   ax1.set_title('Spectrum (log scale)')
   
   ax2.plot(evals, 'o-')
   ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
   ax2.set_ylabel('Eigenvalue')
   ax2.set_xlabel('Rank')
   ax2.set_title('Spectrum (linear scale)')
   
   plt.tight_layout()
   plt.show()

**Interpretation:**

- All evals > 0: Strictly positive definite (consistent Q-test)
- Some evals ≤ 0: Indefinite (spectral cancellation risk)
- Monotonic decay: Low-frequency bias
- Power-law decay: Scale-free spectrum


See also
--------

- :doc:`quickstart` for practical examples
- :doc:`theory` for mathematical background
- :doc:`api/kernels` and :doc:`api/fft` for detailed kernel API documentation
