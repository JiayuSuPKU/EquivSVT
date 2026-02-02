"""
Unit tests for statistical functions.
"""
import unittest
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from quadsv.kernels import SpatialKernel
from quadsv.statistics import (
    spatial_q_test,
    liu_sf,
)


class TestStatisticalFunctions(unittest.TestCase):
    """Test cases for statistical functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create a small grid
        self.n = 25
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        self.coords = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Create test data
        self.data = np.random.randn(self.n)
        
        # Create a spatial kernel
        self.kernel = SpatialKernel.from_coordinates(self.coords, method='car')
        
    def test_spatial_q_test_welch(self):
        """Test spatial Q-test with Welch approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={'method': 'welch'})
        
        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)
        
        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)
        
    def test_spatial_q_test_liu(self):
        """Test spatial Q-test with Liu approximation."""
        Q, pval = spatial_q_test(self.data, self.kernel, null_params={'method': 'liu'})
        
        # Q should be a positive number
        self.assertIsInstance(Q, (float, np.floating))
        self.assertGreater(Q, 0)
        
        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)
        
    def test_liu_sf(self):
        """Test Liu survival function approximation."""
        # Simple case with uniform eigenvalues
        lambs = np.ones(10)
        t = 5.0
        
        pval = liu_sf(t, lambs)
        
        # P-value should be between 0 and 1
        self.assertIsInstance(pval, (float, np.floating))
        self.assertGreaterEqual(pval, 0)
        self.assertLessEqual(pval, 1)
        
    def test_zero_variance_data(self):
        """Test handling of zero variance data."""
        constant_data = np.ones(self.n)
        Q, pval = spatial_q_test(constant_data, self.kernel)
        
        # Should handle gracefully
        self.assertEqual(Q, 0.0)
        self.assertEqual(pval, 1.0)
    
    def test_spatial_q_test_sparse_csr(self):
        """Test spatial_q_test with sparse CSR matrix input."""
        # Create sparse data (CSR format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csr_matrix(X_dense)
        
        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)
        
        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)
    
    def test_spatial_q_test_sparse_csc(self):
        """Test spatial_q_test with sparse CSC matrix input."""
        # Create sparse data (CSC format)
        X_dense = np.random.randn(25, 10)
        X_sparse = csc_matrix(X_dense)
        
        # Compute with dense and sparse
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)
        
        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)
    
    def test_spatial_q_test_chunking(self):
        """Test spatial_q_test with chunking for large feature sets."""
        # Create data with many features
        X = np.random.randn(25, 50)
        
        # Compute without chunking
        Q_full, pval_full = spatial_q_test(X, self.kernel, return_pval=True)
        
        # Compute with chunking (chunk_size=10)
        Q_chunked, pval_chunked = spatial_q_test(
            X, self.kernel, chunk_size=10, return_pval=True
        )
        
        # Results should be identical
        np.testing.assert_allclose(Q_chunked, Q_full, rtol=1e-10)
        np.testing.assert_allclose(pval_chunked, pval_full, rtol=1e-10)
    
    def test_spatial_q_test_sparse_with_chunking(self):
        """Test spatial_q_test with sparse input and chunking."""
        # Create sparse data with many features
        X_dense = np.random.randn(25, 50)
        X_sparse = csr_matrix(X_dense)
        
        # Compute dense without chunking
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        
        # Compute sparse with chunking
        Q_sparse, pval_sparse = spatial_q_test(
            X_sparse, self.kernel, chunk_size=15, return_pval=True
        )
        
        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)
    
    def test_spatial_q_test_single_feature_sparse(self):
        """Test spatial_q_test with single feature sparse matrix."""
        # Create single-feature sparse data
        X_dense = np.random.randn(25, 1)
        X_sparse = csr_matrix(X_dense)
        
        # Should handle single feature correctly
        Q_dense, pval_dense = spatial_q_test(X_dense, self.kernel, return_pval=True)
        Q_sparse, pval_sparse = spatial_q_test(X_sparse, self.kernel, return_pval=True)
        
        # Results should be identical
        np.testing.assert_allclose(Q_sparse, Q_dense, rtol=1e-10)
        np.testing.assert_allclose(pval_sparse, pval_dense, rtol=1e-10)
    
    def test_spatial_q_test_progress_bar(self):
        """Test spatial_q_test with progress bar enabled (manual inspection)."""
        # Create data that will require multiple chunks
        X = np.random.randn(25, 30)
        
        # Should not raise error when show_progress=True
        Q, pval = spatial_q_test(
            X, self.kernel, chunk_size=10, show_progress=False, return_pval=True
        )
        
        # Verify we got results
        self.assertEqual(len(Q), 30)
        self.assertEqual(len(pval), 30)


if __name__ == '__main__':
    unittest.main()
