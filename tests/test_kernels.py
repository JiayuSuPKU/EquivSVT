"""
Unit tests for kernel classes and methods.
"""

import unittest

import numpy as np
import scipy.sparse as sp

from quadsv.kernels import MatrixKernel


class TestMatrixKernel(unittest.TestCase):
    """Test cases for MatrixKernel class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a small 5x5 grid
        self.n = 25
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

    def test_from_coordinates_docstring_example(self):
        """Test from_coordinates classmethod as shown in docstring."""
        # Example from docstring
        coords = np.random.randn(100, 2)
        kernel = MatrixKernel.from_coordinates(coords, method="gaussian", bandwidth=1.5)

        self.assertEqual(kernel.n, 100)
        self.assertEqual(kernel.method, "gaussian")
        self.assertEqual(kernel.params["bandwidth"], 1.5)

    def test_gaussian_kernel(self):
        """Test Gaussian RBF kernel construction."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)

        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

        # Kernel should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))

    def test_matern_kernel(self):
        """Test Matérn kernel construction."""
        kernel = MatrixKernel(self.coords, mode="coords", method="matern", bandwidth=1.0, nu=1.5)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)

        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

        # Kernel should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))

    def test_moran_kernel(self):
        """Test Moran's I adjacency kernel construction."""
        kernel = MatrixKernel(self.coords, mode="coords", method="moran", k_neighbors=4)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)

        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

        # Should be sparse-like (many zeros)
        # Convert to dense if sparse to avoid efficiency warning
        K_dense = K.toarray() if hasattr(K, "toarray") else K
        self.assertGreater(np.sum(K_dense == 0), self.n)

    def test_graph_laplacian_kernel(self):
        """Test Graph Laplacian kernel construction."""
        kernel = MatrixKernel(self.coords, mode="coords", method="graph_laplacian", k_neighbors=4)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

    def test_car_explicit(self):
        """Test CAR kernel in explicit mode (small N)."""
        kernel = MatrixKernel(self.coords, mode="coords", method="car", k_neighbors=4, rho=0.9)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.stores_precision)  # Small N, should be explicit

        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))

        # Should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_xtKx_computation(self):
        """Test quadratic form computation x^T K x."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)
        x = np.random.randn(self.n)

        # Compute via method
        result = kernel.xtKx(x)

        # Compute directly
        K = kernel.realization()
        expected = x.T @ K @ x

        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_xtKx_sparse_input(self):
        """Test quadratic form computation with sparse input."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)

        # Create sparse and dense versions
        x_dense = np.random.randn(self.n)
        x_sparse = sp.csr_matrix(x_dense.reshape(-1, 1))

        # Compute with both
        result_dense = kernel.xtKx(x_dense)
        result_sparse = kernel.xtKx(x_sparse)

        # Should be identical
        np.testing.assert_almost_equal(result_sparse, result_dense, decimal=10)

    def test_xtKx_sparse_batch(self):
        """Test quadratic form computation with sparse batch input."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)

        # Create sparse and dense batches
        X_dense = np.random.randn(self.n, 10)
        X_sparse = sp.csr_matrix(X_dense)

        # Compute with both
        result_dense = kernel.xtKx(X_dense)
        result_sparse = kernel.xtKx(X_sparse)

        # Should be identical
        np.testing.assert_allclose(result_sparse, result_dense, rtol=1e-10)

        # Verify shape
        self.assertEqual(len(result_sparse), 10)
        self.assertEqual(len(result_dense), 10)

    def test_trace_computation(self):
        """Test trace computation."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)

        # Compute via method
        trace_result = kernel.trace()

        # Compute directly
        K = kernel.realization()
        expected_trace = np.trace(K)

        np.testing.assert_almost_equal(trace_result, expected_trace, decimal=10)

    def test_square_trace_computation(self):
        """Test trace(K^2) computation."""
        kernel = MatrixKernel(self.coords, mode="coords", method="gaussian", bandwidth=1.0)

        # Compute via method
        sq_trace = kernel.square_trace()

        # Compute directly
        K = kernel.realization()
        expected = np.sum(K**2)

        np.testing.assert_almost_equal(sq_trace, expected, decimal=10)

    def test_car_implicit(self):
        """Test CAR kernel in implicit mode (larger N)."""
        # Create larger dataset
        n_large = 6400
        x = np.linspace(0, 10, int(np.sqrt(n_large)))
        y = np.linspace(0, 10, int(np.sqrt(n_large)))
        xx, yy = np.meshgrid(x, y)
        coords_large = np.column_stack((xx.ravel(), yy.ravel()))

        # Construct the kernel
        kernel = MatrixKernel(coords_large, mode="coords", method="car", k_neighbors=4, rho=0.9)
        self.assertEqual(kernel.n, n_large)
        self.assertTrue(kernel.stores_precision)  # Should be implicit due to size

        true_K = kernel.realization()
        self.assertEqual(true_K.shape, (n_large, n_large))

        # check if xtKx gives the same result as direct computation
        x = np.random.randn(n_large)
        result = kernel.xtKx(x)
        expected = x.T @ true_K @ x
        self.assertAlmostEqual(result, expected, places=5)

        # check trace computation
        trace_result = kernel.trace() / (n_large**2)
        expected_trace = np.trace(true_K) / (n_large**2)
        self.assertAlmostEqual(trace_result, expected_trace, places=5)

        # check square trace computation
        sq_trace_result = kernel.square_trace() / (n_large**2)
        expected_sq_trace = np.sum(true_K**2) / (n_large**2)
        self.assertAlmostEqual(sq_trace_result, expected_sq_trace, places=4)


class TestMatrixKernelStandardization(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Small grid for explicit inversion path
        n_side = 6
        x = np.linspace(0, 5, n_side)
        y = np.linspace(0, 5, n_side)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        self.n = self.coords.shape[0]

    def test_standardize_explicit_diagonal(self):
        """Diagonal of realized K should be ~1 when standardize is True."""
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
            standardize=True,
        )
        self.assertFalse(kernel.stores_precision)  # small N explicit path
        K = kernel.realization()
        diag = np.diag(K)
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5, atol=1e-3)

    def test_no_standardize_diagonal_not_unity(self):
        """Without standardize, diagonal need not be exactly 1."""
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
            standardize=False,
        )
        K = kernel.realization()
        diag = np.diag(K)
        # Check that there's at least some deviation from 1
        self.assertGreater(np.max(np.abs(diag - 1.0)), 1e-3)


class TestMatrixKernelImplicitTrace(unittest.TestCase):
    """Test trace and square_trace for implicit kernel matrices."""

    def setUp(self):
        """Set up a large dataset for implicit kernel testing."""
        np.random.seed(42)
        # Create large grid that exceeds implicit_threshold (5000)
        n_side = 72  # 72x72 = 5184 > 5000
        x = np.linspace(0, 10, n_side)
        y = np.linspace(0, 10, n_side)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        self.n = self.coords.shape[0]

    def test_implicit_trace_accuracy(self):
        """Test trace() estimation for implicit CAR kernel."""
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
        )
        self.assertTrue(kernel.stores_precision)
        self.assertEqual(kernel.n, self.n)

        # Compute trace via implicit method (Hutchinson's trick)
        implicit_trace = kernel.trace()

        # Compute trace directly from realization (ground truth)
        K_realized = kernel.realization()
        true_trace = np.trace(K_realized)

        # 0.5% budget: Hutchinson with n_probes=15 on a 72×72 CAR whose
        # spectrum has a heavy large-eigenvalue tail — Hutchinson averages
        # that tail efficiently, so the probe-noise band tightens well below
        # the naive 1/√15 ≈ 26% heuristic for flatter spectra.
        rel_error = np.abs(implicit_trace - true_trace) / (np.abs(true_trace) + 1e-10)
        self.assertLess(
            rel_error, 0.005, msg=f"Trace estimation relative error {rel_error:.4f} exceeds 0.5%"
        )

    def test_implicit_square_trace_accuracy(self):
        """Test square_trace() estimation for implicit CAR kernel."""
        kernel = MatrixKernel.from_coordinates(
            self.coords,
            method="car",
            k_neighbors=4,
            rho=0.85,
        )
        self.assertTrue(kernel.stores_precision)

        # Compute square trace via implicit method
        implicit_sq_trace = kernel.square_trace()

        # Compute square trace directly from realization
        K_realized = kernel.realization()
        true_sq_trace = np.sum(K_realized**2)

        # Relative error check
        rel_error = np.abs(implicit_sq_trace - true_sq_trace) / (np.abs(true_sq_trace) + 1e-10)
        self.assertLess(
            rel_error,
            0.005,
            msg=f"Square trace estimation relative error {rel_error:.4f} exceeds 0.5%",
        )


class TestKernelUtilities(unittest.TestCase):
    """Additional tests to cover utility helpers and error paths."""

    def setUp(self):
        np.random.seed(7)
        x = np.linspace(0, 3, 4)
        y = np.linspace(0, 3, 4)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))

    def test_format_params_repr_str(self):
        """_format_params, __repr__, and __str__ should handle arrays and sparse matrices."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.2)
        kernel.params["arr"] = np.ones((2, 2))
        kernel.params["sparse"] = sp.csr_matrix(np.eye(2))

        params_str = kernel._format_params()
        self.assertIn("bandwidth=1.2", params_str)
        self.assertIn("array(shape=(2, 2)", params_str)
        self.assertIn("sparse(shape=(2, 2)", params_str)

        repr_str = repr(kernel)
        self.assertIn("MatrixKernel", repr_str)
        self.assertIn("method=gaussian", repr_str)

        str_out = str(kernel)
        self.assertIn("MatrixKernel", str_out)
        self.assertIn("Method: gaussian", str_out)

    def test_from_matrix_precomputed_and_inverse(self):
        """from_matrix should support precomputed kernels and precision matrices."""
        K = np.array([[2.0, -1.0], [-1.0, 2.0]])
        kernel_pre = MatrixKernel.from_matrix(K, is_precision=False)
        np.testing.assert_allclose(kernel_pre.realization(), K, rtol=1e-12)

        # Singular precision triggers pseudo-inverse path
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        kernel_inv = MatrixKernel.from_matrix(M, is_precision=True, method="car", standardize=True)
        K_inv = kernel_inv.realization()
        self.assertEqual(K_inv.shape, (2, 2))
        self.assertTrue(np.allclose(K_inv, K_inv.T, rtol=1e-12))

    def test_invalid_kernel_param_raises(self):
        """Unknown kernel parameter should raise ValueError."""
        with self.assertRaises(ValueError):
            MatrixKernel.from_coordinates(self.coords, method="gaussian", bad_param=1)

    def test_eigenvalues_cache(self):
        """eigenvalues should reuse cached spectrum for smaller k."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="moran", k_neighbors=2)
        vals_full = kernel.eigenvalues(k=4)
        vals_subset = kernel.eigenvalues(k=2)
        # Spectrum is sorted descending, so top-2 are the first 2 elements
        np.testing.assert_allclose(vals_subset, vals_full[:2], rtol=1e-12)

    def test_getstate_setstate_resets_lu(self):
        """__getstate__ and __setstate__ should reset cached LU factorization."""
        kernel = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.0)
        kernel._lu = object()
        state = kernel.__getstate__()
        self.assertIsNone(state.get("_lu"))

        kernel2 = MatrixKernel.from_coordinates(self.coords, method="gaussian", bandwidth=1.0)
        kernel2.__setstate__(state)
        self.assertIsNone(kernel2._lu)


class TestXtKxStandardized(unittest.TestCase):
    """``MatrixKernel.xtKx_standardized`` must match the manual
    ``z = (X - μ)/σ; kernel.xtKx(z)`` pipeline to float-precision — it's an
    algebraic identity, so the tolerance is ~1e-10, not an approximation band.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.coords = rng.standard_normal((100, 2))
        self.kernel = MatrixKernel.from_coordinates(
            self.coords, method="matern", bandwidth=1.0, nu=1.5
        )

    def test_matches_manual_zscore_dense(self):
        """Dense ``X``: ``xtKx_standardized(X, μ, σ)`` equals
        ``xtKx((X - μ)/σ)`` to float-precision."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 5))
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)
        Z = (X - means) / stds

        q_fast = self.kernel.xtKx_standardized(X, means, stds)
        q_manual = np.asarray(self.kernel.xtKx(Z))
        np.testing.assert_allclose(q_fast, q_manual, rtol=1e-10, atol=1e-12)

    def test_matches_manual_zscore_sparse(self):
        """Sparse ``X``: sparse fast-path agrees with the dense reference to
        float-precision — verifies the ``(K·1, 1ᵀK1)`` expansion."""
        import scipy.sparse as sp

        rng = np.random.default_rng(2)
        X_dense = rng.standard_normal((100, 4))
        # Sparsify by zeroing ~60% of entries; keep sparsity realistic.
        X_dense[rng.random(X_dense.shape) < 0.6] = 0.0
        X_sparse = sp.csc_matrix(X_dense)
        # Means/stds must be computed from the same underlying data the fast
        # path will see (sparse's column stats).
        means = np.asarray(X_sparse.mean(axis=0)).ravel()
        sq = np.asarray(X_sparse.multiply(X_sparse).sum(axis=0)).ravel()
        n = X_sparse.shape[0]
        var = (sq - n * means**2) / max(n - 1, 1)
        var[var < 0] = 0.0
        stds = np.sqrt(var)
        Z = np.zeros_like(X_dense)
        valid = stds > 1e-12
        Z[:, valid] = (X_dense[:, valid] - means[valid]) / stds[valid]

        q_fast = self.kernel.xtKx_standardized(X_sparse, means, stds)
        q_manual = np.asarray(self.kernel.xtKx(Z))
        np.testing.assert_allclose(q_fast, q_manual, rtol=1e-10, atol=1e-12)

    def test_zero_variance_column_returns_zero(self):
        """Constant columns (std=0) must return 0 per the documented contract."""
        X = np.zeros((100, 3))
        X[:, 0] = 1.0  # constant column → std=0
        X[:, 1] = np.linspace(-1, 1, 100)
        X[:, 2] = 7.5  # another constant column
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)

        q = self.kernel.xtKx_standardized(X, means, stds)
        self.assertAlmostEqual(q[0], 0.0, places=12)
        self.assertAlmostEqual(q[2], 0.0, places=12)
        # Middle column has nonzero std → nonzero Q.
        self.assertGreater(q[1], 0.0)


if __name__ == "__main__":
    unittest.main()
