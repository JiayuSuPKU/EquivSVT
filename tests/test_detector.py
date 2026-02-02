"""
Unit tests for PatternDetector class.
"""

import unittest
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from quadsv.detector import PatternDetector


class TestPatternDetector(unittest.TestCase):
    """Test cases for PatternDetector class."""

    def setUp(self):
        """Set up test fixtures with AnnData."""
        np.random.seed(42)

        # Create AnnData object
        self.n_obs = 100
        self.n_vars = 50

        # Create spatial coordinates
        x = np.linspace(0, 4, int(np.sqrt(self.n_obs)))
        y = np.linspace(0, 4, int(np.sqrt(self.n_obs)))
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack((xx.ravel(), yy.ravel()))[: self.n_obs]

        # Create expression matrix
        X = np.random.randn(self.n_obs, self.n_vars)
        X[X < 0] = 0  # Make it non-negative like counts

        # Create obs dataframe with some categorical and numeric features
        obs = pd.DataFrame(
            {
                "cell_type": pd.Categorical(
                    ["A", "B", "C"] * (self.n_obs // 3) + ["A"] * (self.n_obs % 3)
                ),
                "numeric_feature": np.random.randn(self.n_obs),
                "batch": pd.Categorical(
                    ["batch1", "batch2"] * (self.n_obs // 2) + ["batch1"] * (self.n_obs % 2)
                ),
            },
            index=[f"Cell_{i}" for i in range(self.n_obs)],
        )

        # Create var dataframe
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(self.n_vars)])

        # Create obsp with connectivity matrix
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(coords)
        W = nbrs.kneighbors_graph(coords, mode="connectivity").astype(float)

        # Create layers
        layers = {"normalized": X + np.random.randn(self.n_obs, self.n_vars) * 0.1}

        # Create real AnnData object
        self.adata = anndata.AnnData(
            X=sp.csr_matrix(X),
            obs=obs,
            var=var,
            obsm={"spatial": coords},
            obsp={"connectivities": W},
            layers=layers,
        )

        # Store coords for later use in tests
        self.coords = coords

    def test_docstring_example_workflow(self):
        """Test the workflow shown in class docstring."""
        # Example workflow from docstring
        detector = PatternDetector(self.adata, min_cells=10)
        detector.build_kernel_from_coordinates(
            self.adata.obsm["spatial"], method="gaussian", bandwidth=2.0
        )

        # Should be able to compute qstat
        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 2.0)

    def test_class_initialization(self):
        """Test PatternDetector initialization."""
        detector = PatternDetector(self.adata, min_cells=5)

        self.assertEqual(detector.n, self.n_obs)
        self.assertIsNone(detector.kernel_)
        self.assertIsNone(detector.kernel_params_)

    def test_available_kernels(self):
        """Test that available kernels are properly defined."""
        detector = PatternDetector(self.adata, min_cells=5)

        expected_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]
        self.assertEqual(detector._available_kernels, expected_kernels)

    def test_build_kernel_from_coordinates_docstring_example(self):
        """Test example from build_kernel_from_coordinates docstring."""
        detector = PatternDetector(self.adata, min_cells=5)
        coords = np.random.randn(self.n_obs, 2)

        # Example from docstring
        detector.build_kernel_from_coordinates(coords, method="gaussian", bandwidth=1.5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 1.5)

    def test_build_kernel_from_coordinates_gaussian(self):
        """Test building Gaussian kernel from coordinates."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_coordinates(self.coords, method="gaussian", bandwidth=1.0)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "gaussian")
        self.assertEqual(detector.kernel_params_["bandwidth"], 1.0)

    def test_build_kernel_from_coordinates_matern(self):
        """Test building Matérn kernel from coordinates."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_coordinates(self.coords, method="matern", bandwidth=1.0, nu=1.5)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "matern")
        self.assertEqual(detector.kernel_params_["nu"], 1.5)

    def test_build_kernel_from_coordinates_moran(self):
        """Test building Moran kernel from coordinates."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_coordinates(self.coords, method="moran", k_neighbors=8)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "moran")
        self.assertEqual(detector.kernel_params_["k_neighbors"], 8)

    def test_build_kernel_from_coordinates_car(self):
        """Test building CAR kernel from coordinates."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_coordinates(self.coords, method="car", k_neighbors=8, rho=0.9)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "car")
        self.assertEqual(detector.kernel_params_["rho"], 0.9)

    def test_build_kernel_invalid_method(self):
        """Test that invalid kernel method raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector.build_kernel_from_coordinates(self.coords, method="invalid_method")

        self.assertIn("not recognized", str(context.exception))

    def test_build_kernel_coordinate_shape_mismatch(self):
        """Test that coordinate shape mismatch raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        wrong_coords = np.random.randn(10, 2)  # Wrong number of cells

        with self.assertRaises(ValueError) as context:
            detector.build_kernel_from_coordinates(wrong_coords, method="gaussian")

        self.assertIn("does not match", str(context.exception))

    def test_build_kernel_from_obsp_precomputed(self):
        """Test building kernel from precomputed obsp matrix."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_obsp("connectivities", method="precomputed")

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "precomputed")

    def test_build_kernel_from_obsp_moran(self):
        """Test building Moran kernel from connectivity matrix."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_obsp("connectivities", is_distance=False, method="moran")

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "moran")

    def test_build_kernel_from_obsp_graph_laplacian(self):
        """Test building Laplacian kernel from connectivity matrix."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_obsp(
            "connectivities", is_distance=False, method="graph_laplacian"
        )

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "graph_laplacian")

    def test_build_kernel_from_obsp_car(self):
        """Test building CAR kernel from connectivity matrix."""
        detector = PatternDetector(self.adata, min_cells=5)

        detector.build_kernel_from_obsp("connectivities", is_distance=False, method="car", rho=0.8)

        self.assertIsNotNone(detector.kernel_)
        self.assertEqual(detector.kernel_method_, "car")

    def test_build_kernel_from_obsp_car_standardize(self):
        """Precomputed CAR path with standardize should yield unit diagonal when realized (small n)."""
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_obsp(
            "connectivities", is_distance=False, method="car", rho=0.9, standardize=True
        )
        # Realize and check diagonal ~ 1
        K = detector.kernel_.realization()
        diag = np.diag(K)
        np.testing.assert_allclose(diag, np.ones_like(diag), rtol=1e-5, atol=1e-3)

    def test_build_kernel_from_obsp_missing_key(self):
        """Test that missing obsp key raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(KeyError) as context:
            detector.build_kernel_from_obsp("nonexistent_key", method="moran")

        self.assertIn("not found", str(context.exception))

    def test_build_kernel_from_obsp_invalid_method(self):
        """Test that invalid method for obsp raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector.build_kernel_from_obsp("connectivities", method="invalid_method")

        self.assertIn("not recognized", str(context.exception))

    def test_prepare_data_var_source(self):
        """Test data preparation from var (genes)."""
        detector = PatternDetector(self.adata, min_cells=5)

        # Prepare data for a subset of genes
        gene_subset = self.adata.var_names[:5].tolist()
        X_csc, names, means, stds = detector._prepare_data(
            source="var", keys=gene_subset, min_cells=1
        )

        # Check shapes
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertLessEqual(len(names), 5)

        # Check that means and stds are properly computed
        self.assertEqual(len(means), len(names))
        self.assertEqual(len(stds), len(names))

        # Check all stds are positive (constant features filtered)
        self.assertTrue(np.all(stds > 0))

    def test_prepare_data_obs_numeric(self):
        """Test data preparation from obs with numeric features."""
        detector = PatternDetector(self.adata, min_cells=5)

        X_csc, names, means, stds = detector._prepare_data(
            source="obs", keys=["numeric_feature"], min_cells=1
        )

        # Check shapes
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], "numeric_feature")

    def test_prepare_data_obs_categorical(self):
        """Test data preparation from obs with categorical features (one-hot encoding)."""
        detector = PatternDetector(self.adata, min_cells=5)

        X_csc, names, means, stds = detector._prepare_data(
            source="obs", keys=["cell_type"], min_cells=1
        )

        # Check that one-hot encoding occurred
        # cell_type has 3 categories, so we expect 3 columns
        self.assertEqual(X_csc.shape[0], self.n_obs)
        self.assertEqual(len(names), 3)  # One for each category

        # Check that names contain category values
        self.assertTrue(any("A" in name or "B" in name or "C" in name for name in names))

    def test_prepare_data_no_keys_obs_raises(self):
        """Test that not providing keys for obs source raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector._prepare_data(source="obs", keys=None, min_cells=1)

        self.assertIn("Keys must be provided", str(context.exception))

    def test_prepare_data_invalid_source(self):
        """Test that invalid source raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector._prepare_data(source="invalid", keys=None, min_cells=1)

        self.assertIn("Source must be", str(context.exception))

    def test_run_without_kernel_raises(self):
        """Test that running PatternDetector without kernel raises error."""
        detector = PatternDetector(self.adata, min_cells=5)

        with self.assertRaises(ValueError) as context:
            detector.compute_qstat(source="var", features=None)

        self.assertIn("Kernel not initialized", str(context.exception))

    def test_compute_qstat_docstring_example(self):
        """Test compute_qstat example from docstring."""
        # Example from docstring: build kernel then compute qstat
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.adata.obsm["spatial"])

        # Select two genes as in docstring example
        gene_subset = self.adata.var_names[:2].tolist()
        results = detector.compute_qstat(
            source="var",
            features=gene_subset,
            n_jobs=1,  # Use 1 job for deterministic testing
        )

        # Check structure matches docstring
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("Q", results.columns)
        self.assertIn("Z_score", results.columns)
        self.assertIn("P_value", results.columns)
        self.assertIn("P_adj", results.columns)

        # Results should be sorted by Q descending
        self.assertTrue(results["Q"].is_monotonic_decreasing or len(results) <= 1)

        # Check we can get top genes
        top_genes = results.iloc[: min(10, len(results))]
        self.assertLessEqual(len(top_genes), 10)

    def test_run_var_source(self):
        """Test running PatternDetector on var (genes) source without mocks."""
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.coords, method="gaussian", bandwidth=1.0)

        # Run PatternDetector on a subset of genes
        # We select enough genes to likely trigger batch processing if batch size is small
        gene_subset = self.adata.var_names[:10].tolist()

        # Use n_jobs=1 to keep debugging simple, but this will exercise the full math stack
        results = detector.compute_qstat(source="var", features=gene_subset, n_jobs=1)

        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue("Q" in results.columns)
        self.assertTrue("P_value" in results.columns)
        self.assertTrue("Z_score" in results.columns)

        # Check value ranges for real computation
        self.assertTrue(np.all(results["Q"] >= 0))
        self.assertTrue(np.all(results["P_value"] >= 0))
        self.assertTrue(np.all(results["P_value"] <= 1.0))

        # Check that results are sorted by p-value
        self.assertTrue(results["P_value"].is_monotonic_increasing)

    def test_run_obs_source(self):
        """Test running PatternDetector on obs (cell metadata) source without mocks."""
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.coords, method="gaussian", bandwidth=1.0)

        # Run PatternDetector on obs features
        results = detector.compute_qstat(source="obs", features=["numeric_feature"], n_jobs=1)

        # Check results
        self.assertIsInstance(results, pd.DataFrame)
        self.assertTrue(len(results) > 0)
        self.assertTrue(np.all(results["P_value"] >= 0))
        self.assertTrue(np.all(results["P_value"] <= 1.0))

    def test_run_with_layer(self):
        """Test running PatternDetector with specific layer."""
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.coords, method="gaussian", bandwidth=1.0)

        # Mock the compute_null_params to avoid actual computation
        with patch("quadsv.statistics.compute_null_params") as mock_compute_null:
            with patch("quadsv.statistics.spatial_q_test") as mock_spatial_q_test:
                mock_compute_null.return_value = {"mean_Q": 1.0, "var_Q": 0.5}
                mock_spatial_q_test.return_value = (2.5, 0.05)

                gene_subset = self.adata.var_names[:3].tolist()
                results = detector.compute_qstat(
                    source="var", features=gene_subset, layer="normalized", n_jobs=1
                )

                self.assertIsInstance(results, pd.DataFrame)

    def test_compute_rstat_docstring_symmetric_example(self):
        """Test compute_rstat symmetric mode example from docstring."""
        # Example from docstring: all pairwise correlations within gene set
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.adata.obsm["spatial"])

        # Select 3 genes for symmetric pairwise
        gene_subset = self.adata.var_names[:3].tolist()
        results = detector.compute_rstat(features_x=gene_subset, n_jobs=1)

        # Check structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("Feature_1", results.columns)
        self.assertIn("Feature_2", results.columns)
        self.assertIn("R", results.columns)
        self.assertIn("Z_score", results.columns)
        self.assertIn("P_value", results.columns)
        self.assertIn("P_adj", results.columns)

        # Results should be sorted by |Z_score| descending
        abs_z = results["Z_score"].abs()
        self.assertTrue(abs_z.is_monotonic_decreasing or len(results) <= 1)

        # Symmetric mode now returns all pairs including (A,A), (A,B), (B,A), etc.
        # For 3 genes, we expect up to 9 pairs
        self.assertLessEqual(len(results), 9)

    def test_compute_rstat_docstring_bipartite_example(self):
        """Test compute_rstat bipartite mode example from docstring."""
        # Example from docstring: cross-correlation between two gene sets
        detector = PatternDetector(self.adata, min_cells=5)
        detector.build_kernel_from_coordinates(self.adata.obsm["spatial"])

        # Select genes as in docstring: 2 x 2 = 4 pairs
        features_x = self.adata.var_names[:2].tolist()
        features_y = self.adata.var_names[2:4].tolist()

        results = detector.compute_rstat(features_x=features_x, features_y=features_y, n_jobs=1)

        # Check structure
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("R", results.columns)
        self.assertIn("Z_score", results.columns)

        # Bipartite mode: all X vs Y pairs
        # 2 x 2 = 4 pairs maximum
        self.assertLessEqual(len(results), 4)


class TestPatternDetectorEdgeCases(unittest.TestCase):
    """Test edge cases for PatternDetector class."""

    def test_compute_with_constant_features(self):
        """Test that constant features are filtered out."""
        # Create mock adata with some constant features
        n_obs = 50
        n_vars = 10

        X = np.random.randn(n_obs, n_vars)
        X[:, 0] = 1.0  # Constant feature
        X[:, 1] = 0.0  # Zero feature
        coords = np.random.randn(n_obs, 2)

        adata = anndata.AnnData(
            X=sp.csr_matrix(X),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]),
            obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_obs)]),
            obsm={"spatial": coords},
        )

        detector = PatternDetector(adata)
        X_csc, names, means, stds = detector._prepare_data(source="var", keys=None, min_cells=1)

        # Constant features should be filtered
        self.assertLess(len(names), n_vars)
        self.assertTrue(np.all(stds > 0))

    def test_compute_with_sparse_features(self):
        """Test PatternDetector with very sparse features."""
        n_obs = 100
        n_vars = 20

        # Create very sparse matrix with numeric dtype
        X = sp.random(n_obs, n_vars, density=0.05, format="csr", dtype=np.float64)

        adata = anndata.AnnData(X=X, var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_vars)]))

        detector = PatternDetector(adata, min_cells=5)

        # Should handle sparse data gracefully
        X_csc, names, means, stds = detector._prepare_data(
            source="var",
            keys=None,
            min_cells=10,  # Require at least 10 non-zero cells
        )

        self.assertIsInstance(X_csc, sp.csc_matrix)
        self.assertTrue(len(names) <= n_vars)


if __name__ == "__main__":
    unittest.main()
