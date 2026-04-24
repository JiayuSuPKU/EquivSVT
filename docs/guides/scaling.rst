Scalable Computation
======================

Evaluating :math:`Q_n = \mathbf{z}^\top \tilde{\mathbf{K}} \mathbf{z}` and
its *p*-value naively requires the dense :math:`n \times n` matrix
:math:`\tilde{\mathbf{K}}` and its full spectrum, costing
:math:`\mathcal{O}(n^3)` time and :math:`\mathcal{O}(n^2)` memory.
Modern spatial-omics datasets (:math:`n \sim 10^3\!-\!10^7` spots,
:math:`m \sim 10^3\!-\!10^4` features) make that intractable.

``quadsv`` replaces the dense baseline with scalable operators that never
materialise :math:`\tilde{\mathbf{K}}` and never run a full
eigendecomposition. This page summarises the three orthogonal axes of the
design — **null-distribution approximations**, **structured or sparse
kernel operators**, and **Fourier-accelerated** primitives — and their
combined complexity. Notation: :math:`c_p := \operatorname{tr}(\tilde{\mathbf{K}}^p)`
is the *p*-th spectral power sum; under the null
:math:`\mathbb{E}[Q_n] = c_1`,
:math:`\operatorname{Var}[Q_n] = 2c_2 + \epsilon_{\text{kurtosis}}`.

Null-distribution approximations
--------------------------------

Under :math:`H_0`, :math:`Q_n \xrightarrow{d} \sum_{i=1}^n \lambda_i^{(n)}
\chi_1^2`. An exact *p*-value can be obtained by Davies' inversion of

.. math::
   \varphi_{Q_n}(t)
   \;=\; \prod_{i=1}^n \bigl(1 - 2it\lambda_i^{(n)}\bigr)^{-1/2},

but only after paying :math:`\mathcal{O}(n^3)` for the full spectrum and
:math:`\mathcal{O}(Kn)` for each observed :math:`Q_n` (where *K* is the
number of quadrature nodes). To avoid the eigendecomposition entirely,
``quadsv`` ships two moment-matching fits evaluable in :math:`\mathcal{O}(1)`
per feature once the cumulants are cached:

- **Welch-Satterthwaite** and the **CLT** match :math:`c_1, c_2` to a
  scaled central :math:`\chi^2` (Welch; PSD kernels only) or a Normal
  (CLT; works for indefinite :math:`\tilde{\mathbf{K}}` too). Both need
  only :math:`\operatorname{tr}(\tilde{\mathbf{K}})` and
  :math:`\operatorname{tr}(\tilde{\mathbf{K}}^2)`.

- **Liu's approximation** matches the first four cumulants
  :math:`c_1\!..\!c_4` to a shifted non-central :math:`\chi^2`. The
  higher cumulants are estimated via **Hutchinson's trace estimator**:

  .. math::
     c_p \;=\; \mathbb{E}\bigl[\mathbf{v}^\top \tilde{\mathbf{K}}^p \mathbf{v}\bigr]
     \;\approx\;
     \tfrac{1}{V}\,\textstyle\sum_{s=1}^{V} \mathbf{v}_s^\top \tilde{\mathbf{K}}^p \mathbf{v}_s,

  with :math:`\{\mathbf{v}_s\} \stackrel{\text{iid}}{\sim} \{\pm 1\}^n`
  (Rademacher). Two matvecs per probe :math:`\mathbf{v}_s` deliver all
  four cumulants from the same probe pool.

**Finite-n Dirichlet correction.** Because
:math:`\mathbf{z} = \mathbf{H}\mathbf{x}/\hat{\sigma}` uses the *sample*
variance :math:`\hat{\sigma}^2 = \mathbf{x}^\top\mathbf{H}\mathbf{x}/(n-1)`
in the denominator — a random quantity correlated with the numerator —
the exact null variance follows a Dirichlet(1/2) ratio:

.. math::
   \operatorname{Var}[Q_n]
   \;=\; \frac{2\,[\,(n-1)\,c_2 - c_1^{2}\,]}{n+1}
   \;<\; 2 c_2.

Welch and Liu both use this corrected variance. Without it, Liu's right
tail collapses to zero on broad-spectrum kernels where
:math:`c_1^{2} \approx (n-1)\,c_2` (CAR on a dense grid is the prototypical
failure mode).

.. note::

   Moment-matching's right tail decays faster than the true null on
   heavy-tailed spectra, which can over-reject in the extreme tails. Liu
   is the most tail-accurate of the three; Welch is tighter in the bulk;
   CLT is the coarsest. When in doubt, run Liu.

Structured and sparse kernel operators
--------------------------------------

For an arbitrary spatial coordinate cloud :math:`\{\mathbf{s}_i\}`, three
strategies let matvecs scale sub-quadratically:

- **Sparse kernels** (:math:`k`-NN, thresholded distance): :math:`\mathbf{K}`
  has :math:`\mathcal{O}(nk)` non-zeros. Matvecs and Hutchinson cumulants
  cost :math:`\mathcal{O}(nk)`. *Caveat*: many ad-hoc sparse kernels
  (notably Moran's I) are **indefinite**, which induces the
  *cancellation* pathology and costs power on composite spatial patterns.

- **Low-rank kernels** :math:`\mathbf{K} \approx \mathbf{U}\mathbf{U}^\top`
  with :math:`\mathbf{U} \in \mathbb{R}^{n \times r}`,
  :math:`r \ll n`. Matvecs cost :math:`\mathcal{O}(nr)`; spectra fall
  out of an :math:`\mathcal{O}(r^3)` eigen-problem on
  :math:`\mathbf{U}^\top\mathbf{U}`. Factorization cost is
  :math:`\mathcal{O}(nr^2)` via Nyström or Random Fourier Features.
  *Caveat*: finite rank creates an infinite-dimensional null space
  (*blind spots*), so power vanishes on high-frequency patterns.

- **Sparse precision (CAR)**: the *precision* matrix
  :math:`\mathbf{\Omega} = \mathbf{I} - \rho\widetilde{\mathbf{W}}` is
  sparse with :math:`\mathcal{O}(nk)` non-zeros even though
  :math:`\mathbf{K} = \mathbf{\Omega}^{-1}` is dense. On typical 2-D
  graphs a sparse LU (or Cholesky) factorisation of :math:`\mathbf{\Omega}`
  costs :math:`\mathcal{O}(n^{3/2})` time with :math:`\mathcal{O}(n\log n)`
  fill-in; each subsequent :math:`\mathbf{K}\mathbf{v}` is an
  :math:`\mathcal{O}(n\log n)` triangular solve. Preserves full rank
  and positive-definiteness.

Fourier-accelerated methods
---------------------------

Under periodic boundary conditions, any translation-invariant kernel on a
regular 2-D or 3-D grid is diagonalised by the Fourier basis:

.. math::
   \mathbf{K}
   \;=\; \mathbf{F}^{\mathsf H}\,\operatorname{diag}(\boldsymbol{\mu})\,\mathbf{F}.

The grid spectrum :math:`\boldsymbol{\mu}` is analytic in the kernel
hyperparameters (bandwidth, :math:`\nu`, :math:`\rho`, neighbour
structure) — no eigendecomposition needed. Matvecs become element-wise
products at the cost of one :math:`\mathcal{O}(n\log n)` FFT:

.. math::
   \mathbf{K}\mathbf{v}
   \;=\; \tfrac{1}{n}\,\operatorname{vec}\bigl(\boldsymbol{\mu} \odot (\mathcal{F}\mathbf{v})\bigr).

**Irregular coordinates (NUFFT).** For non-uniform
:math:`\{\mathbf{s}_i\}`, we approximate :math:`\mathbf{K}` with an
oversampled :math:`n'`-grid (:math:`n' > n`):

.. math::
   \mathbf{K}
   \;\approx\; \tfrac{1}{n'}\, \mathbf{U}\,\operatorname{diag}(\boldsymbol{\mu})\,\mathbf{U}^{\mathsf H},
   \qquad \mathbf{U} \in \mathbb{C}^{n \times n'}.

Matvecs are a type-1 / type-2 NUFFT round-trip at cost
:math:`\mathcal{O}(n' \log n')`:

.. math::
   \mathbf{K}\mathbf{v}
   \;=\; \frac{1}{n'}\,
   \overbrace{\mathbf{U}\bigl(\boldsymbol{\mu} \odot \underbrace{\mathbf{U}^{\mathsf H}\mathbf{v}}_{\text{type-1}}\bigr)}^{\text{type-2}}.

Because :math:`\mathbf{U}` is non-unitary, eigenvalues of
:math:`\mathbf{K}` do not strictly follow :math:`\boldsymbol{\mu}`. The
first two cumulants still have closed form via Toeplitz FFT convolutions.
Defining :math:`\mathbf{G} := \mathbf{U}^{\mathsf H}\mathbf{U}`
with :math:`G_{k,k'} = n\,\phi(k'-k)`, :math:`\phi(\boldsymbol{\Delta}) =
\tfrac{1}{n}\sum_i e^{\mathrm{i}\boldsymbol{\Delta}\cdot\mathbf{s}_i}`:

.. math::
   c_1 &= \frac{n}{n'} \sum_{k} \mu_k \quad \text{(exact — } \operatorname{diag}(\mathbf{G}) = n\mathbf{I}\text{),} \\
   c_2 &= \frac{n^2}{(n')^2}\, \boldsymbol{\mu}^{\mathsf{H}}\, \boldsymbol{\Psi}\, \boldsymbol{\mu},
   \qquad \Psi_{k,k'} = |\phi(k'-k)|^{2}.

:math:`c_2` is evaluated by two 2-D FFTs — one on :math:`|\phi|^2`, one
on :math:`\boldsymbol{\mu}` — at :math:`\mathcal{O}(n'\log n')`.

.. note::

   **Graph kernels under NUFFT define a** *different* **test.** NUFFT
   implicitly builds a *grid-stencil* kernel on the auxiliary grid,
   which is topologically different from an exact :math:`k`-NN graph
   on :math:`\{\mathbf{s}_i\}` with open boundaries.
   :math:`\mathbf{K}^{\text{NUFFT}}` is therefore **not an
   approximation** of :math:`\mathbf{K}^{\text{sparse}}`; their
   :math:`Q_n` statistics are not expected to match numerically.

   This does **not** invalidate the NUFFT Q-test — it is simply a
   different Q-test, evaluated on the grid-stencil operator rather
   than the :math:`k`-NN graph. Specifically, it is:

   * **well-defined** (the NUFFT operator is real, symmetric,
     translation-invariant by construction);
   * **consistent** whenever its effective spectrum is non-negative
     (e.g. ``'car'``, ``'graph_laplacian'``, and ``'moran'`` under CLT
     via the null-distribution approximations above); and
   * **powerful** — ``O((n + n') log n')`` per feature with no
     :math:`k`-NN construction step, so it scales to ``n`` where the
     sparse-graph path cannot go.

   Distance-based kernels (Gaussian, Matérn) are unaffected: their
   NUFFT and Matrix forms converge to the same translation-invariant
   operator as the oversampling factor grows.

Complexity summary
------------------

.. list-table:: Computational complexity of :math:`Q_n` scaling strategies.
   :header-rows: 1
   :widths: 28 18 12 22 22

   * - Backend
     - Precompute
     - Memory
     - Time per :math:`Q`
     - Key limitation
   * - Dense + Davies (baseline)
     - :math:`\mathcal{O}(n^3)`
     - :math:`\mathcal{O}(n^2)`
     - :math:`\mathcal{O}(n^2) + \mathcal{O}(Kn)`
     - Intractable for large :math:`n`
   * - Dense + Liu / Welch / CLT\ :sup:`†`
     - :math:`\mathcal{O}(n^2)`
     - :math:`\mathcal{O}(n^2)`
     - :math:`\mathcal{O}(n^2)`
     - Inaccurate right tail
   * - **Sparse kernel** (:math:`k`-NN)
     - :math:`\mathcal{O}(nk)`
     - :math:`\mathcal{O}(nk)`
     - :math:`\mathcal{O}(nk)`
     - Power loss (*cancellation*)
   * - **Low-rank kernel** (rank-:math:`r`)
     - :math:`\mathcal{O}(nr + r^3)`\ :sup:`‡`
     - :math:`\mathcal{O}(nr)`
     - :math:`\mathcal{O}(nr)`
     - Power loss (*blind spots*)
   * - **Sparse precision** (CAR)
     - :math:`\mathcal{O}(n^{3/2})`
     - :math:`\mathcal{O}(n\log n)`
     - :math:`\mathcal{O}(n\log n)`
     - Triangular solves hard to parallelise
   * - **FFT** (full spectrum)
     - :math:`\mathcal{O}(n\log n)`
     - :math:`\mathcal{O}(n)`
     - :math:`\mathcal{O}(n\log n)`
     - Requires periodic grid
   * - **NUFFT**
     - :math:`\mathcal{O}((n+n')\log n')`
     - :math:`\mathcal{O}(n')`
     - :math:`\mathcal{O}((n+n')\log n')`
     - Mismatch with graph kernels\ :sup:`§`

Per-\ :math:`Q` cost is dominated by the kernel matvec; the null
evaluation on top is either Davies inversion
(:math:`\mathcal{O}(Kn)` per feature) or constant-time Liu / Welch once
the cumulants are cached.

:sup:`†`\  **Liu vs Welch** share the same complexity but not the same
constant. By default Liu uses :math:`V = 60` Rademacher probes (so
:math:`2V` matvecs); Welch needs only :math:`c_1, c_2` which are often
available analytically (FFT spectrum sum, NUFFT Toeplitz conv, or the
exact :math:`\operatorname{tr}(\mathbf{W}) = 0` identity on row-normalised
graph kernels).

:sup:`‡`\  **Low-rank cost** depends on the factorisation. A top-:math:`r`
eigendecomposition costs :math:`\mathcal{O}(n^2 r)` but is often preferred
— it returns the most global (lowest-frequency) modes.

:sup:`§`\  **NUFFT for graph kernels.** See the NUFFT note above; the
grid-stencil substitution breaks the original :math:`k`-NN adjacency so
:math:`Q_n^{\text{NUFFT}}` is not a direct approximation of
:math:`Q_n^{\text{sparse}}`.

Default recommendation
----------------------

- **Continuous kernels on arbitrary coordinates (Gaussian, Matérn)** —
  use :class:`~quadsv.NUFFTKernel` with Liu. :math:`\mathcal{O}(n)` memory,
  :math:`\mathcal{O}(n\log n)` time per feature, no power loss.
- **Regular rasterised grids (Visium HD, imaging)** — use
  :class:`~quadsv.FFTKernel` with Liu. Same asymptotic complexity as
  NUFFT, smaller constant.
- **General graph kernels (phylogenetic trees, single-cell k-NN
  graphs)** — use :class:`~quadsv.MatrixKernel` with ``method='car'``
  (sparse precision). :math:`\mathcal{O}(n\log n)` per feature after the
  :math:`\mathcal{O}(n^{3/2})` one-time factorisation.
- **Small** :math:`n \lesssim 5\,000` — any backend will do; the dense
  Matrix path is the simplest.

See also
--------

- :doc:`/guides/theory` — derivation of the null distribution and the
  Dirichlet correction.
- :doc:`/guides/kernels` — kernel-selection practicalities.
- :doc:`/guides/quickstart` — end-to-end usage recipes.
