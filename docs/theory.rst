Theoretical Results
====================================

In our accompanied paper, we demonstrate that virtually all major spatial variable gene (SVG) detection methods, 
including graph-based ones like Moran's I, parametric models, and non-parametric dependence tests, reduce to
a single quadratic-form statistic (**Q-statistic**),

.. math::
   Q_n = \mathbf{z}^\top \mathbf{K} \mathbf{z},

where :math:`\mathbf{z}` is the standardized gene expression vector and :math:`\mathbf{K}` is a kernel matrix encoding spatial structure.
Under the null hypothesis of spatial independence, :math:`Q_n` follows a weighted chi-square distribution whose weights are the eigenvalues of :math:`\mathbf{K}`. 
We can approximate the null distribution using moment-matching methods to compute p-values efficiently, yielding a **Q-test**.
The choice of kernel :math:`\mathbf{K}` critically affects the consistency and power of the resulting Q-test.

Here we summarize key theoretical results underpinning the quadratic form.

Theorems
~~~~~~~~~~~~~~~~~~~~

**Theorem 1: Q-tests detect mean shifts only**

All spatial Q-tests detect mean-shift patterns (:math:`\mathbb{E}[\mathbf{x}|S=\mathbf{s}] \neq \mathbb{E}[\mathbf{x}]`).

This is the direct results of using a linear kernel :math:`l(x_i, x_j) = x_i x_j` in the quadratic form that reduces the conditional :math:`X|S=s_i` to its mean. 
If investigating higher-order spatial moments (variance, distributional changes), use non-linear kernels such as Gaussian or polynomial kernels
(e.g., :math:`Q_n = (\mathbf{z}^2)^\top\mathbf{K} \mathbf{z}^2`).

However, in applications such as spatial transcriptomics, this distributional information is *absent* because we observe only a single realization
:math:`(x_i, s_i)` drawn from :math:`X \mid S=s_i` at each location. This constraint blurs the line between mean independence and statistical independence.
Motivated by this observation, we adopt a functional perspective, treating the "signal" as a deterministic element of a Hilbert space :math:`f \in L^2(\mathcal{S})`. 
Using the spectrum theory of kernel operators, we derive the following condition for test consistency regarding mean dependence.

**Theorem 2: Consistency requires positive definiteness**

A spatial Q-test is universally consistent (power → 1 as :math:`N \to \infty`) to all non-constant (deterministic) patterns if and only if the kernel :math:`\mathbf{K}` is strictly positive definite.

Under :math:`H_0`, :math:`Q_n` approximates a weighted chi-square: :math:`Q_n \sim \sum_i \lambda_i \chi^2_1`. If :math:`\lambda_i < 0` (indefinite kernel), negative eigenspace signals cancel positive signals (spectral cancellation), reducing test power.

*Implication*: Choose kernels with positive eigenvalues:

.. list-table::
   :widths: 20 20 40

   * - Kernel
     - Spectrum
     - Consistency
   * - Gaussian
     - Positive definite
     - ✓ Guaranteed
   * - Matérn
     - Positive definite
     - ✓ Guaranteed
   * - Moran's I
     - Indefinite
     - ✗ Spectral cancellation
   * - Laplacian
     - Semi-definite
     - ✓ Guaranteed (high-frequency Moran's I)
   * - CAR (inverse Laplacian)
     - Positive definite
     - ✓ Guaranteed (Low-frequency Moran's I)

CAR is a scalable correction to Moran's I
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Conditional Autoregressive (CAR) kernel provides strict positive definiteness

.. math::
   \mathbf{K} = (\mathbf{I} - \rho \tilde{\mathbf{W}})^{-1}

where:

- :math:`\tilde{\mathbf{W}}` is the row-normalized adjacency matrix
- :math:`0 < \rho < 1` is the autoregressive parameter (default: 0.9)
- :math:`(\mathbf{I} - \rho \tilde{\mathbf{W}})` is the CAR precision matrix

**Key properties:**

- Strictly positive definite for all :math:`0 < \rho < 1`
- Theoretically consistent (Theorem 2)
- Scalable with a sparse precision matrix using implicit kernel operations
- Polynomial spectral decay that emphasizes smooth, large-scale patterns while maintaining a heavy tail for mid/high frequencies

*Recommendation*: Use CAR for all spatial pattern detection tasks.



Null distribution approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under the null hypothesis (spatial independence), :math:`Q_n` follows a weighted chi-square:

.. math::
   Q_n \sim \sum_{i=1}^m \lambda_i \chi^2_1

where :math:`\lambda_i` are eigenvalues of :math:`\mathbf{K}` and :math:`m = \text{rank}(\mathbf{K})`.

Three approximation methods are provided, balancing accuracy and speed:

.. list-table::
   :widths: 15 15 15 40

   * - Method
     - Complexity
     - Applicability
     - Use case
   * - CLT
     - O(N)
     - Works for all kernels
     - Large N, indefinite kernels
   * - Welch
     - O(N)
     - Positive semi-definite kernels only
     - Default, large N
   * - Liu
     - O(N³)
     - Positive semi-definite kernels only
     - N ≤ 5000 or FFT grids

**CLT**: Approximates :math:`Q_n` as normal with mean :math:`\mu = \text{tr}(\mathbf{K})` and variance :math:`\sigma^2 = 2\text{tr}(\mathbf{K}^2)`.

**Welch/Satterthwaite**: Matches first two moments to a scaled chi-square distribution using Hutchinson trace estimation. Recommended default.

**Liu**: Exact eigendecomposition followed by polynomial moment-matching. Most accurate but requires O(N³) computation.

R-test: bivariate spatial co-expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extends Q-tests to test spatial correlation between two features:

.. math::
   R_{xy} = \mathbf{x}^\top \mathbf{K} \mathbf{y}

where :math:`\mathbf{x}, \mathbf{y}` are standardized features.

**Null distribution**: :math:`R_{xy} \sim \mathcal{N}(0, \sigma^2)` with :math:`\sigma^2 = \text{tr}(\mathbf{K}^2)` under spatial independence.

**Typical workflow:**

1. Identify spatially variable genes (SVGs) via univariate Q-test
2. Test pairwise R-statistics among top SVGs
3. Control false discovery rate (FDR) across comparisons

FFT acceleration for regular grids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Block-Toeplitz kernels on regular grids (e.g., Visium HD, imaging), eigenvalues decouple via FFT:

.. math::
   \mathbf{K} = \mathbf{U} \Lambda \mathbf{U}^\top
   \quad \Rightarrow \quad
   Q_n = \sum_i \lambda_i |u_i^\top \mathbf{z}|^2

where :math:`\mathbf{U}` is the FFT basis and :math:`\Lambda` is diagonal.

**Complexity reduction:**

- Explicit eigendecomposition: O(N³)
- FFT eigenvalues: O(N log N)
- Q-test computation: O(N log N)

**Example: 1000×1000 grid**

- Explicit kernel: ~10 hours
- FFT kernel: ~1 minute

**Supported topologies:** Square (4-neighbor, default) and hexagonal (6-neighbor).

Practical summary
~~~~~~~~~~~~~~~~~

============== ======================== ================================
Method         Test consistency         Use case
============== ======================== ================================
Moran's I      ✗ Spectral cancellation  Autocorrelation
Graph Lap.     ✓ Guaranteed             High-frequency, local variation
CAR            ✓ Guaranteed             Low-frequency, smoothed patterns
============== ======================== ================================

**Best practice**: Use CAR kernel for consistent, high-power detection across functional patterns. Use FFT-accelerated CAR on regular grids.

See also
--------

- :doc:`quickstart` — Practical usage examples
- :doc:`kernels` — Kernel selection and design
- :doc:`api/statistics` — Statistical API reference