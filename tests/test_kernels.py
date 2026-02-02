"""
Unit tests for kernel classes and methods.
"""
import unittest
import numpy as np
import time
import scipy.sparse as sp
from quadsv.kernels import SpatialKernel


class TestSpatialKernel(unittest.TestCase):
    """Test cases for SpatialKernel class."""
    
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
        kernel = SpatialKernel.from_coordinates(coords, method='gaussian', bandwidth=1.5)
        
        self.assertEqual(kernel.n, 100)
        self.assertEqual(kernel.method, 'gaussian')
        self.assertEqual(kernel.params['bandwidth'], 1.5)
    
    def test_gaussian_kernel(self):
        """Test Gaussian RBF kernel construction."""
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.is_implicit)
        
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        
        # Kernel should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)
        
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))

    def test_matern_kernel(self):
        """Test Matérn kernel construction."""
        kernel = SpatialKernel(self.coords, mode='coords', method='matern', bandwidth=1.0, nu=1.5)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.is_implicit)
        
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        
        # Kernel should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)
        
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(self.n))
        
    def test_moran_kernel(self):
        """Test Moran's I adjacency kernel construction."""
        kernel = SpatialKernel(self.coords, mode='coords', method='moran', k_neighbors=4)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.is_implicit)
        
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        
        # Should be sparse-like (many zeros)
        # Convert to dense if sparse to avoid efficiency warning
        K_dense = K.toarray() if hasattr(K, 'toarray') else K
        self.assertGreater(np.sum(K_dense == 0), self.n)
        
    def test_graph_laplacian_kernel(self):
        """Test Graph Laplacian kernel construction."""
        kernel = SpatialKernel(self.coords, mode='coords', method='graph_laplacian', k_neighbors=4)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.is_implicit)
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        
    def test_car_explicit(self):
        """Test CAR kernel in explicit mode (small N)."""
        kernel = SpatialKernel(self.coords, mode='coords', method='car', 
                               k_neighbors=4, rho=0.9)
        self.assertEqual(kernel.n, self.n)
        self.assertFalse(kernel.is_implicit)  # Small N, should be explicit
        
        K = kernel.realization()
        self.assertEqual(K.shape, (self.n, self.n))
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_xtKx_computation(self):
        """Test quadratic form computation x^T K x."""
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        x = np.random.randn(self.n)
        
        # Compute via method
        result = kernel.xtKx(x)
        
        # Compute directly
        K = kernel.realization()
        expected = x.T @ K @ x
        
        np.testing.assert_almost_equal(result, expected, decimal=10)
    
    def test_xtKx_sparse_input(self):
        """Test quadratic form computation with sparse input."""
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        
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
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        
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
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        
        # Compute via method
        trace_result = kernel.trace()
        
        # Compute directly
        K = kernel.realization()
        expected_trace = np.trace(K)
        
        np.testing.assert_almost_equal(trace_result, expected_trace, decimal=10)
        
    def test_square_trace_computation(self):
        """Test trace(K^2) computation."""
        kernel = SpatialKernel(self.coords, mode='coords', method='gaussian', bandwidth=1.0)
        
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
        kernel = SpatialKernel(coords_large, mode='coords', method='car', 
                               k_neighbors=4, rho=0.9)
        self.assertEqual(kernel.n, n_large)
        self.assertTrue(kernel.is_implicit)  # Should be implicit due to size
        
        true_K = kernel.realization()
        self.assertEqual(true_K.shape, (n_large, n_large))
        
        # check if xtKx gives the same result as direct computation
        x = np.random.randn(n_large)
        result = kernel.xtKx(x)
        expected = x.T @ true_K @ x
        self.assertAlmostEqual(result, expected, places=5)
        
        # check trace computation
        trace_result = kernel.trace() / (n_large ** 2)
        expected_trace = np.trace(true_K) / (n_large ** 2)
        self.assertAlmostEqual(trace_result, expected_trace, places=5)
        
        # check square trace computation
        sq_trace_result = kernel.square_trace() / (n_large ** 2)
        expected_sq_trace = np.sum(true_K**2) / (n_large ** 2)
        self.assertAlmostEqual(sq_trace_result, expected_sq_trace, places=4)


class TestCARStandardize(unittest.TestCase):
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
        kernel = SpatialKernel.from_coordinates(
            self.coords,
            method='car',
            k_neighbors=4,
            rho=0.85,
            standardize=True,
        )
        self.assertFalse(kernel.is_implicit)  # small N explicit path
        K = kernel.realization()
        diag = np.diag(K)
        self.assertEqual(K.shape, (self.n, self.n))
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5, atol=1e-3)

    def test_no_standardize_diagonal_not_unity(self):
        """Without standardize, diagonal need not be exactly 1."""
        kernel = SpatialKernel.from_coordinates(
            self.coords,
            method='car',
            k_neighbors=4,
            rho=0.85,
            standardize=False,
        )
        K = kernel.realization()
        diag = np.diag(K)
        # Check that there's at least some deviation from 1
        self.assertGreater(np.max(np.abs(diag - 1.0)), 1e-3)


class TestImplicitTraceOperations(unittest.TestCase):
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
        kernel = SpatialKernel.from_coordinates(
            self.coords,
            method='car',
            k_neighbors=4,
            rho=0.85,
        )
        self.assertTrue(kernel.is_implicit)
        self.assertEqual(kernel.n, self.n)

        # Compute trace via implicit method (Hutchinson's trick)
        implicit_trace = kernel.trace()

        # Compute trace directly from realization (ground truth)
        K_realized = kernel.realization()
        true_trace = np.trace(K_realized)

        # They should be close (Hutchinson's estimator has variance but should converge)
        # Relative error should be small
        rel_error = np.abs(implicit_trace - true_trace) / (np.abs(true_trace) + 1e-10)
        self.assertLess(rel_error, 0.001, 
                        msg=f"Trace estimation relative error {rel_error:.4f} exceeds 0.1%")

    def test_implicit_square_trace_accuracy(self):
        """Test square_trace() estimation for implicit CAR kernel."""
        kernel = SpatialKernel.from_coordinates(
            self.coords,
            method='car',
            k_neighbors=4,
            rho=0.85,
        )
        self.assertTrue(kernel.is_implicit)

        # Compute square trace via implicit method
        implicit_sq_trace = kernel.square_trace()

        # Compute square trace directly from realization
        K_realized = kernel.realization()
        true_sq_trace = np.sum(K_realized**2)

        # Relative error check
        rel_error = np.abs(implicit_sq_trace - true_sq_trace) / (np.abs(true_sq_trace) + 1e-10)
        self.assertLess(rel_error, 0.001,
                        msg=f"Square trace estimation relative error {rel_error:.4f} exceeds 0.1%")


class TestStandardizationPerformance(unittest.TestCase):
    """Test standardization of inverse kernel matrices with sparse vs dense performance comparison."""

    def setUp(self):
        """Set up test fixtures with various sizes."""
        np.random.seed(123)
        # Moderate size for testing standardization
        self.n_dense = 4900 # below implicit threshold
        self.n_sparse = 5184 # just above implicit threshold
        
    def _create_sparse_precision(self, n, k_neighbors=4):
        """Helper to create a sparse precision matrix from coordinates."""
        # Create grid
        side = int(np.sqrt(n))
        x = np.linspace(0, 10, side)
        y = np.linspace(0, 10, side)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack((xx.ravel()[:n], yy.ravel()[:n]))
        
        # Build adjacency
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
        W = nbrs.kneighbors_graph(coords, mode='connectivity').astype(float)
        
        # Symmetrize and normalize
        W_sym = 0.5 * (W + W.T)
        row_sums = np.array(W_sym.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_D_sqrt = sp.diags(1.0 / np.sqrt(row_sums))
        W_norm = inv_D_sqrt @ W_sym @ inv_D_sqrt
        
        # Create precision M = I - rho * W_norm
        rho = 0.85
        I = sp.eye(side * side, format='csc')
        M = I - rho * W_norm
        return M.tocsc()
        
    def _standardize_sparse_vs_dense_runtime(self, n):
        """Compare runtime of sparse batched solve vs dense inversion for standardization."""
        side = int(np.sqrt(n))
        n = side * side  # ensure perfect square for grid
        M_sparse = self._create_sparse_precision(n)
        M_dense = M_sparse.toarray()
        
        print(f"\n{'='*60}")
        print(f"Standardization Runtime Comparison (n={n})")
        print(f"{'='*60}")
        
        # Test sparse approach (batched solves)
        if n > 5000:
            print("Testing sparse batched approach...")
        else:
            print("Testing sparse approach (converted to dense for small n)...")

        start_sparse = time.time()
        kernel_sparse = SpatialKernel.from_matrix(
            M_sparse,
            is_inverse=True,
            method='car',
            standardize=True,
        )
        K_sparse = kernel_sparse.realization()
        time_sparse = time.time() - start_sparse
        diag_sparse = np.diag(K_sparse)
        
        print(f"  Sparse time: {time_sparse:.3f}s")
        print(f"  Diagonal check: min={diag_sparse.min():.6f}, max={diag_sparse.max():.6f}, mean={diag_sparse.mean():.6f}")
        
        # Test dense approach (full inversion)
        print("\nTesting dense inversion approach...")
        start_dense = time.time()
        kernel_dense = SpatialKernel.from_matrix(
            M_dense,
            is_inverse=True,
            method='car',
            standardize=True,
        )
        K_dense = kernel_dense.realization()
        time_dense = time.time() - start_dense
        diag_dense = np.diag(K_dense)
        
        print(f"  Dense time: {time_dense:.3f}s")
        print(f"  Diagonal check: min={diag_dense.min():.6f}, max={diag_dense.max():.6f}, mean={diag_dense.mean():.6f}")
        
        # Compare results
        speedup = time_dense / time_sparse
        print(f"\n  Speedup (sparse/dense): {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        # Verify both give similar results
        np.testing.assert_allclose(diag_sparse, np.ones(n), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(diag_dense, np.ones(n), rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(diag_sparse, diag_dense, rtol=1e-3, atol=1e-3)
        
        # Sparse should generally be faster (though not guaranteed on small n)
        self.assertGreater(
            speedup, 0.8, msg="Sparse approach should be competitive with dense"
        )

    def test_standardize_runtime(self):
        """Test standardization runtime for small and large n."""
        self._standardize_sparse_vs_dense_runtime(self.n_dense)

        with self.assertWarns(RuntimeWarning):
            self._standardize_sparse_vs_dense_runtime(self.n_sparse)

if __name__ == '__main__':
    unittest.main()
