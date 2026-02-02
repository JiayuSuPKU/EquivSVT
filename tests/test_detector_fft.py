"""
Unit tests for PatternDetectorFFT class.
Uses lightweight mocks for SpatialData and rasterization to avoid heavy dependencies.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quadsv.detector_fft import PatternDetectorFFT


class MockCoord:
    def __init__(self, values):
        self.values = values


class MockDataArray:
    """Minimal xarray-like container for tests."""

    def __init__(self, data, features):
        self.data = np.asarray(data)
        self._features = np.asarray(features)
        self.coords = {"c": MockCoord(self._features)}
        self.shape = self.data.shape

    def sel(self, c):
        # Select along feature axis (axis 0)
        idx = [np.where(self._features == f)[0][0] for f in c]
        return MockDataArray(self.data[idx], self._features[idx])

    @property
    def values(self):
        return self.data


class MockVarNames:
    """Mimics AnnData's var_names interface."""

    def __init__(self, names):
        self._names = list(names)

    def to_list(self):
        return self._names


class MockTable:
    """Minimal table with AnnData-like X and feature subset support."""

    def __init__(self, X, features):
        self.X = np.asarray(X)
        self.features = list(features)
        self.var_names = MockVarNames(features)

    def __getitem__(self, keys):
        # Handle 2D indexing: table[:, features] or table[features]
        if isinstance(keys, tuple):
            # 2D indexing: keys = (row_slice, feature_names)
            row_slice, feature_keys = keys
            idx = [self.features.index(k) for k in feature_keys]
            return MockTable(self.X[row_slice, :][:, idx], feature_keys)
        else:
            # 1D indexing: keys = feature_names
            idx = [self.features.index(k) for k in keys]
            return MockTable(self.X[:, idx], keys)


class MockSpatialData:
    """Container that mimics SpatialData for the detector."""

    def __init__(self, table_name, table):
        self.tables = {table_name: table}
        self._store = {}

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value


class TestPatternDetectorFFT(unittest.TestCase):
    def setUp(self):
        # Create small synthetic raster data: features x ny x nx
        self.features = np.array(["f1", "f2", "f3"])
        self.ny, self.nx = 2, 2
        self.raster_data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # f1
                [[0.0, 1.0], [0.0, 1.0]],  # f2
                [[2.0, 0.0], [0.0, 1.0]],  # f3
            ]
        )
        self.mock_da = MockDataArray(self.raster_data, self.features)

        # Table counts for min_count filtering (cells x features)
        self.table_X = np.array(
            [
                [10, 0, 10],
                [5, 1, 5],
                [0, 0, 0],
                [3, 0, 2],
            ]
        )
        self.table = MockTable(self.table_X, list(self.features))
        self.sdata = MockSpatialData("cells", self.table)

    def test_compute_qstat_returns_dataframe(self):
        """PatternDetectorFFT.compute_qstat produces a DataFrame with expected columns."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)
            df = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=True,
                chunk_size=2,
            )
        # Expect all features retained
        self.assertEqual(set(df.index), set(self.features))
        # Required columns
        self.assertTrue({"Q", "P_value", "Z_score"}.issubset(df.columns))
        # Adjusted p-value column added when return_pval=True
        self.assertIn("P_adj", df.columns)

    def test_min_count_filtering(self):
        """Features below min_count are filtered out."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(
                self.sdata,
                min_count=5,  # f2 has total count 1, will be filtered out
                kernel_method="gaussian",
                bandwidth=1.0,
            )
            df = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=False,
                chunk_size=2,
            )
        # f2 should be removed due to low count
        self.assertNotIn("f2", df.index)
        self.assertIn("f1", df.index)
        self.assertIn("f3", df.index)

    def test_kernel_reuse_same_shape(self):
        """Kernel is reused when grid shape stays the same across calls."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)
            df1 = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                n_jobs=1,
                return_pval=False,
            )
            # Keep a reference to the kernel
            kernel_ref = detector.kernel_
            # Second call with same shape should reuse kernel (same object)
            df2 = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                n_jobs=1,
                return_pval=False,
            )
            self.assertIs(detector.kernel_, kernel_ref)
            # DataFrames should have matching indices
            self.assertEqual(set(df1.index), set(df2.index))

    def test_docstring_example_init_and_workflow(self):
        """Test docstring example: detector initialization and workflow setup."""
        # >>> detector = PatternDetectorFFT(sdata, kernel_method='car', rho=0.8)
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="car", rho=0.8)
            # Verify initialization
            self.assertEqual(detector.kernel_method_, "car")
            self.assertEqual(detector.kernel_params_["rho"], 0.8)
            self.assertIsNone(detector.kernel_)  # Lazy initialization

    def test_docstring_example_compute_qstat_workflow(self):
        """Test docstring example: full compute_qstat workflow."""
        # Create mock data with features matching the test table
        features = np.array(["f1", "f2", "f3"])
        ny, nx = 4, 4
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="car", rho=0.8)
            # >>> q_results = detector.compute_qstat('grid', 'table', 'col_idx', 'row_idx',
            # ...                                    features=['f1', 'f2', 'f3'])
            q_results = detector.compute_qstat(
                bins="grid",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
                features=list(features),
                n_jobs=1,
                return_pval=True,
                chunk_size=10,
            )

            # Verify result structure from docstring
            self.assertIsInstance(q_results, pd.DataFrame)
            self.assertEqual(set(q_results.index), set(features))
            required_cols = {"Q", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(q_results.columns))
            # Q values should be sorted descending
            q_vals = q_results["Q"].values
            self.assertTrue(np.all(q_vals[:-1] >= q_vals[1:]))

    def test_docstring_example_compute_rstat_symmetric(self):
        """Test docstring example: compute_rstat in symmetric (pairwise) mode."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="car", rho=0.8)
            # >>> r_results = detector.compute_rstat('grid_bins', 'counts', 'col_idx', 'row_idx')
            r_results = detector.compute_rstat(
                bins="grid_bins",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
                features_x=None,  # Use all features in symmetric mode
                features_y=None,
                workers=4,
                return_pval=True,
                chunk_size=10,
            )
            self.assertEqual(detector.kernel_params_["workers"], 4)

            # Verify result structure
            self.assertIsInstance(r_results, pd.DataFrame)
            required_cols = {"Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(r_results.columns))

            # In symmetric mode, should have upper triangular + diagonal pairs
            # For 3 features: (0,1), (0,2), (1,2) = 3 pairs
            # Plus diagonal pairs (0,0), (1,1), (2,2) = 3 pairs
            # Total = 6 pairs
            self.assertEqual(len(r_results), 6)

            # R values should be sorted by absolute value descending
            r_vals = np.abs(r_results["R"].values)
            self.assertTrue(np.all(r_vals[:-1] >= r_vals[1:]))

    def test_docstring_example_compute_rstat_bipartite(self):
        """Test docstring example: compute_rstat in bipartite (X vs Y) mode."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=2.0)
            # Bipartite mode: compute correlations between two gene sets
            features_x = ["f1", "f2"]
            features_y = ["f3"]

            r_results = detector.compute_rstat(
                bins="grid_bins",
                table_name="cells",
                col_key="col_idx",
                row_key="row_idx",
                features_x=features_x,
                features_y=features_y,
                workers=4,
                return_pval=True,
                chunk_size=10,
            )
            self.assertEqual(detector.kernel_params_["workers"], 4)

            # Verify result structure
            self.assertIsInstance(r_results, pd.DataFrame)
            required_cols = {"Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"}
            self.assertTrue(required_cols.issubset(r_results.columns))

            # In bipartite mode: 2 x 1 = 2 pairs
            self.assertEqual(len(r_results), 2)

            # All Feature_1 should be from features_x, Feature_2 from features_y
            self.assertTrue(all(f1 in features_x for f1 in r_results["Feature_1"]))
            self.assertTrue(all(f2 in features_y for f2 in r_results["Feature_2"]))

    def test_compute_qstat_no_pval(self):
        """compute_qstat returns only Q and Z_score when return_pval=False."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)
            df = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=False,
                chunk_size=2,
            )
            # Should have Q and Z_score but not P_value or P_adj
            self.assertIn("Q", df.columns)
            self.assertIn("Z_score", df.columns)
            self.assertNotIn("P_value", df.columns)
            self.assertNotIn("P_adj", df.columns)

    def test_compute_rstat_symmetric_upper_triangle(self):
        """compute_rstat symmetric mode returns only upper triangular pairs."""
        features = np.array(["f1", "f2", "f3", "f4"])
        ny, nx = 3, 3
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=mock_da):
            detector = PatternDetectorFFT(sdata, kernel_method="gaussian", bandwidth=1.0)
            r_results = detector.compute_rstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features_x=None,
                features_y=None,
                workers=1,
                return_pval=True,
                chunk_size=10,
            )

            # For 4 features in symmetric mode: C(4,2) = 6 pairs + 4 diagonals
            self.assertEqual(len(r_results), 10)

            # Verify pairs are upper triangular (no duplicates, Feature_1 <= Feature_2 lexicographically)
            for _, row in r_results.iterrows():
                # In this case, lexicographic ordering ensures upper triangle
                self.assertTrue(row["Feature_1"] <= row["Feature_2"])

    def test_parallel_qstat_single_job(self):
        """compute_qstat with n_jobs=1 produces consistent results."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)
            df = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=True,
                chunk_size=2,
            )
            # Should produce valid results
            self.assertEqual(len(df), len(self.features))
            self.assertTrue(all(np.isfinite(df["Q"])))
            self.assertTrue(all(np.isfinite(df["Z_score"])))

    def test_parallel_qstat_multiple_jobs(self):
        """compute_qstat with n_jobs>1 produces identical results to n_jobs=1."""
        # Create larger dataset for meaningful parallel test
        features = np.array([f"gene_{i}" for i in range(20)])
        ny, nx = 8, 8
        np.random.seed(42)
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        # Update table to match features
        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=mock_da):
            detector = PatternDetectorFFT(sdata, kernel_method="gaussian", bandwidth=1.0)

            # Serial execution
            df_serial = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=True,
                chunk_size=5,
            )

            # Reset kernel to force rebuild
            detector.kernel_ = None

            # Parallel execution
            df_parallel = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=2,
                return_pval=True,
                chunk_size=5,
            )

            # Results should be identical (order may differ)
            self.assertEqual(set(df_serial.index), set(df_parallel.index))

            # Sort both by index for comparison
            df_serial = df_serial.sort_index()
            df_parallel = df_parallel.sort_index()

            # Q values should match
            np.testing.assert_allclose(df_serial["Q"].values, df_parallel["Q"].values, rtol=1e-10)
            np.testing.assert_allclose(
                df_serial["Z_score"].values, df_parallel["Z_score"].values, rtol=1e-10
            )
            np.testing.assert_allclose(
                df_serial["P_value"].values, df_parallel["P_value"].values, rtol=1e-10
            )

    def test_parallel_qstat_workers_parameter(self):
        """compute_qstat workers parameter is passed to FFTKernel."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)

            # Test with explicit workers parameter
            df = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                workers=2,  # FFT-level parallelization
                return_pval=True,
                chunk_size=2,
            )

            # Verify kernel was created with workers parameter
            self.assertIsNotNone(detector.kernel_)
            # Workers should be stored in kernel params
            self.assertEqual(detector.kernel_params_["workers"], 2)

            # Results should still be valid
            self.assertEqual(len(df), len(self.features))
            self.assertTrue(all(np.isfinite(df["Q"])))

    def test_parallel_qstat_chunk_size_effect(self):
        """Different chunk_size values produce identical results."""
        features = np.array([f"gene_{i}" for i in range(12)])
        ny, nx = 6, 6
        np.random.seed(123)
        raster_data = np.random.randn(len(features), ny, nx)
        mock_da = MockDataArray(raster_data, features)

        table_X = np.random.randint(0, 100, size=(ny * nx, len(features)))
        table = MockTable(table_X, list(features))
        sdata = MockSpatialData("cells", table)

        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=mock_da):
            detector = PatternDetectorFFT(sdata, kernel_method="car", rho=0.8)

            # Small chunks
            df_small = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=True,
                chunk_size=3,
            )

            # Reset kernel
            detector.kernel_ = None

            # Large chunks
            df_large = detector.compute_qstat(
                bins="bins",
                table_name="cells",
                col_key="col",
                row_key="row",
                features=None,
                n_jobs=1,
                return_pval=True,
                chunk_size=10,
            )

            # Sort for comparison
            df_small = df_small.sort_index()
            df_large = df_large.sort_index()

            # Results should be identical
            np.testing.assert_allclose(df_small["Q"].values, df_large["Q"].values, rtol=1e-10)
            np.testing.assert_allclose(
                df_small["P_value"].values, df_large["P_value"].values, rtol=1e-10
            )

    def test_parallel_qstat_n_jobs_auto(self):
        """compute_qstat with n_jobs=-1 uses all available cores."""
        with patch("quadsv.detector_fft.sd.rasterize_bins", return_value=self.mock_da):
            with patch("os.cpu_count", return_value=4):
                detector = PatternDetectorFFT(self.sdata, kernel_method="gaussian", bandwidth=1.0)

                # n_jobs=-1 should detect and use cpu_count
                df = detector.compute_qstat(
                    bins="bins",
                    table_name="cells",
                    col_key="col",
                    row_key="row",
                    features=None,
                    n_jobs=-1,  # Auto-detect
                    return_pval=True,
                    chunk_size=2,
                )

                # Should still produce valid results
                self.assertEqual(len(df), len(self.features))
                self.assertTrue(all(np.isfinite(df["Q"])))


if __name__ == "__main__":
    unittest.main()
