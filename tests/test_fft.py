"""
Unit tests for FFT-based kernel and statistical tests.
Compares FFTKernel results with SpatialKernel for grid data.
"""

import unittest

import numpy as np

from quadsv.fft import FFTKernel, spatial_q_test_fft, spatial_r_test_fft
from quadsv.kernels import SpatialKernel
from quadsv.statistics import spatial_q_test as spatial_q_test_standard
from quadsv.statistics import spatial_r_test as spatial_r_test_standard
from quadsv.utils import compute_torus_distance_matrix


class TestFFTKernelBasics(unittest.TestCase):
    """Test FFTKernel initialization and basic methods."""

    def test_fft_kernel_init_square_fft2(self):
        """Test FFTKernel initialization with square grid."""
        shape = (10, 10)
        kernel = FFTKernel(
            shape, topology="square", method="gaussian", bandwidth=1.0, fft_solver="fft2"
        )
        assert kernel.ny == 10
        assert kernel.nx == 10
        assert kernel.spectrum is not None
        assert len(kernel.spectrum) == 100  # 10 * 10

    def test_fft_kernel_init_square_rfft2(self):
        """Test FFTKernel initialization with square grid."""
        shape = (10, 10)
        kernel = FFTKernel(
            shape, topology="square", method="gaussian", bandwidth=1.0, fft_solver="rfft2"
        )
        assert kernel.ny == 10
        assert kernel.nx == 10
        assert kernel.spectrum is not None
        assert len(kernel.spectrum) == 60  # 10 * (10//2 +1)

    def test_fft_kernel_init_hex(self):
        """Test FFTKernel initialization with hexagonal grid."""
        shape = (8, 8)
        kernel = FFTKernel(
            shape, topology="hex", method="matern", bandwidth=1.0, nu=1.5, fft_solver="fft2"
        )
        assert kernel.ny == 8
        assert kernel.nx == 8
        assert len(kernel.spectrum) == 64

    def test_fft_matern_eigenvalues(self):
        """Test eigenvalues method."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5, fft_solver="rfft2")
        evals = kernel.eigenvalues()
        assert len(evals) == 15  # 5 * (5//2 +1)
        assert np.all(np.isfinite(evals))
        assert np.all(evals >= 0)

    def test_fft_moran_kernel(self):
        """Test Moran kernel initialization and eigenvalues."""
        shape = (6, 6)
        kernel = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="rfft2")
        evals = kernel.eigenvalues()
        assert len(evals) == 24  # 6 * (6//2 +1)
        assert np.all(np.isreal(evals))
        assert np.all(evals >= -1.1) and np.all(evals <= 1.1)

    def test_fft_car_kernel(self):
        """Test CAR (Conditional Autoregressive) kernel initialization and eigenvalues."""
        shape = (6, 6)
        kernel = FFTKernel(shape, method="car", neighbor_degree=1, rho=0.9, fft_solver="fft2")
        evals = kernel.eigenvalues()
        assert len(evals) == 36  # 6 * 6
        assert np.all(np.isreal(evals))
        assert np.all(evals > 0)

    def test_fft_solver_eigenvalues(self):
        """Test that changing neighbor degree changes eigenvalue spectrum."""
        shape = (8, 8)
        k1 = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="rfft2")
        k2 = FFTKernel(shape, method="moran", neighbor_degree=1, fft_solver="fft2")

        evals1 = k1.eigenvalues(return_full=True)
        evals2 = k2.eigenvalues()

        # Spectra should be different
        assert np.allclose(evals1, evals2)

    def test_xtKx_2d_input(self):
        """Test xtKx with 2D input (single pattern)."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5)
        x = np.random.randn(5, 5)
        Q = kernel.xtKx(x)
        assert np.isfinite(Q)
        assert np.isscalar(Q)

    def test_xtKx_3d_input(self):
        """Test xtKx with 3D input (batched patterns)."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(5, 5, 3)
        Q = kernel.xtKx(x)
        assert Q.shape == (3,)
        assert np.all(np.isfinite(Q))

    def test_xtKx_zero_vector(self):
        """Test xtKx with zero vector should give 0."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.zeros((5, 5))
        Q = kernel.xtKx(x)
        assert np.isclose(Q, 0.0, atol=1e-10)

    def test_xtKx_scaling(self):
        """Test xtKx scales correctly with input magnitude."""
        shape = (5, 5)
        kernel = FFTKernel(shape, method="matern", bandwidth=1.0, nu=1.5)
        x = np.random.randn(5, 5)
        Q1 = kernel.xtKx(x)
        Q2 = kernel.xtKx(2.0 * x)
        # Q(2x) should be 4*Q(x)
        assert np.isclose(Q2, 4.0 * Q1, rtol=1e-9)


class TestSpatialQTest(unittest.TestCase):
    """Test FFT-based spatial Q test."""

    def test_spatial_q_test_2d(self):
        """Test spatial_q_test with 2D input."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        Q = spatial_q_test_fft(x, kernel, return_pval=False)
        assert np.isfinite(Q)
        assert np.isscalar(Q)

    def test_spatial_q_test_3d(self):
        """Test spatial_q_test with 3D batched input."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8, 5)
        Q = spatial_q_test_fft(x, kernel, return_pval=False)
        assert Q.shape == (5,)

    def test_spatial_q_test_with_pval(self):
        """Test spatial_q_test returns both Q and p-value."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        Q, pval = spatial_q_test_fft(x, kernel, return_pval=True)
        assert np.isfinite(Q)
        assert 0 <= pval <= 1.0

    def test_spatial_q_test_standardization(self):
        """Test that standardization works correctly."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)

        # Create data with non-zero mean and std != 1
        x = np.random.randn(8, 8) * 2.0 + 5.0

        Q1 = spatial_q_test_fft(x, kernel, is_standardized=False, return_pval=False)

        # Manually standardize
        x_std = (x - x.mean()) / x.std(ddof=1)
        Q2 = spatial_q_test_fft(x_std, kernel, is_standardized=True, return_pval=False)

        # Should be the same
        assert np.isclose(Q1, Q2, rtol=1e-9)


class TestSpatialRTest(unittest.TestCase):
    """Test FFT-based spatial R test (bivariate)."""

    def test_spatial_r_test_2d(self):
        """Test spatial_r_test with 2D inputs."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R = spatial_r_test_fft(x, y, kernel, return_pval=False)
        assert np.isfinite(R)
        assert np.isscalar(R)

    def test_spatial_r_test_3d(self):
        """Test spatial_r_test with 3D batched inputs."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8, 3)
        y = np.random.randn(8, 8, 3)
        R = spatial_r_test_fft(x, y, kernel, return_pval=False)
        assert R.shape == (3,)

    def test_spatial_r_test_with_pval(self):
        """Test spatial_r_test returns both R and p-value."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R, pval = spatial_r_test_fft(x, y, kernel, return_pval=True)
        assert np.isfinite(R)
        assert 0 <= pval <= 1.0

    def test_spatial_r_test_symmetry(self):
        """Test that R(x, y) == R(y, x) (symmetry of kernel)."""
        shape = (8, 8)
        kernel = FFTKernel(shape, method="gaussian", bandwidth=1.0)
        x = np.random.randn(8, 8)
        y = np.random.randn(8, 8)
        R_xy = spatial_r_test_fft(x, y, kernel, return_pval=False)
        R_yx = spatial_r_test_fft(y, x, kernel, return_pval=False)
        assert np.isclose(R_xy, R_yx, rtol=1e-9)


class TestFFTVsSpatialKernelComparison(unittest.TestCase):
    """Compare FFT-based kernel with SpatialKernel on grid data."""

    @staticmethod
    def create_grid_and_kernels(nx, ny, method="gaussian", **kwargs):
        """Helper to create grid and corresponding kernels."""
        # Create regular grid coordinates
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        # FFT kernel
        # filter out additional parameters for graph-based methods
        fft_kwargs = kwargs.copy()
        if fft_kwargs:
            for key in kwargs.keys():
                if key in ["neighbor_degree", "rho", "bandwidth", "nu"]:
                    continue
                else:
                    fft_kwargs.pop(key, None)  # Remove unsupported keys

        fft_kernel = FFTKernel((ny, nx), topology="square", method=method, **fft_kwargs)

        # Standard spatial kernel (no periodic boundaries)
        d_torus = compute_torus_distance_matrix(coords, domain_dims=(ny, nx))
        if method in ["moran", "car"]:
            k_neighbors = kwargs.get("k_neighbors", 4)
            # Build Graph (Standard KNN logic)
            np.fill_diagonal(d_torus, np.inf)
            n = len(coords)
            W = np.zeros((n, n))
            for i in range(n):
                neighbors = np.argsort(d_torus[i, :])[:k_neighbors]
                W[i, neighbors] = 1.0

            # Symmetrize and normalize the graph
            W = 0.5 * (W + W.T)
            W[W > 0] = 1.0
            np.fill_diagonal(W, 0)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
            W = d_inv_sqrt @ W @ d_inv_sqrt  # Normalized adjacency

            if method == "car":
                rho = kwargs.get("rho", 0.9)
                W = np.eye(n) - rho * W
                spatial_kernel = SpatialKernel.from_matrix(W, method=method, is_inverse=True)
            else:  # Moran
                spatial_kernel = SpatialKernel.from_matrix(W, method=method)

        elif method == "gaussian":
            k_torus = np.exp(-(d_torus**2) / (2 * kwargs.get("bandwidth", 1.0) ** 2))
            spatial_kernel = SpatialKernel.from_matrix(k_torus, method=method)
        elif method == "matern":
            from scipy.special import gamma, kv

            nu = kwargs.get("nu", 1.5)
            bw = kwargs.get("bandwidth", 1.0)
            # Matern kernel formula
            d_torus[d_torus == 0] = 1e-15
            fac = (np.sqrt(2 * nu) * d_torus) / bw
            k_torus = (2 ** (1 - nu) / gamma(nu)) * (fac**nu) * kv(nu, fac)
            np.fill_diagonal(k_torus, 1.0)
            spatial_kernel = SpatialKernel.from_matrix(k_torus, method=method)

        return coords, fft_kernel, spatial_kernel

    def test_gaussian_kernel_eigenvalues_close(self):
        """Compare Gaussian kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="gaussian", bandwidth=1.0
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Gaussian kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_matern_kernel_eigenvalues_close(self):
        """Compare Matern kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="matern", bandwidth=1.0, nu=1.5
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Matern kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_moran_eigenvalues_close(self):
        """Compare Moran kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="moran", neighbor_degree=1, k_neighbors=4
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert (
            np.mean(rel_error) < 1e-5
        ), f"Moran kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_car_kernel_eigenvalues_close(self):
        """Compare CAR kernel eigenvalues."""
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            30, 30, method="car", neighbor_degree=1, k_neighbors=4, rho=0.9
        )

        fft_evals = np.sort(fft_k.eigenvalues(return_full=True))[::-1]  # Descending
        spatial_evals = np.sort(spatial_k.eigenvalues())[::-1]  # Descending

        # Should be extremely close
        assert len(fft_evals) == len(spatial_evals)
        rel_error = np.abs(fft_evals - spatial_evals) / (np.abs(spatial_evals) + 1e-10)
        assert np.mean(rel_error) < 1e-5, f"CAR kernel eigenvalues rel error: {np.mean(rel_error)}"

    def test_q_statistic_close_on_smooth_data(self):
        """Compare Q statistics on smooth (low-frequency) data."""
        nx, ny = 20, 20
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            nx, ny, method="gaussian", bandwidth=2.0
        )

        # Create smooth data (low frequency = interior points dominate, boundary effects less)
        np.random.seed(42)
        # Simulate low-frequency pattern
        x_data = np.zeros((ny, nx))
        for _i in range(3):
            freq_x = np.random.randint(1, 4)
            freq_y = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.randn() * 5.0
            x_data += amplitude * np.sin(
                2 * np.pi * (freq_x * np.arange(nx) / nx + freq_y * np.arange(ny)[:, None] / ny)
                + phase
            )

        # FFT test
        Q_fft = spatial_q_test_fft(x_data, fft_k, return_pval=False)

        # Standard test on original data
        Q_std = spatial_q_test_standard(x_data.ravel(), spatial_k, return_pval=False)

        # Should be extremely close
        rel_error = np.abs(Q_fft - Q_std) / (np.abs(Q_std) + 1e-10)
        assert rel_error < 1e-5, f"Q statistic {Q_fft:.2f}/{Q_std:.2f}, rel error: {rel_error: .2f}"

    def test_r_statistic_close_on_smooth_data(self):
        """Compare R statistics (bivariate) on smooth data."""
        nx, ny = 20, 20
        coords, fft_k, spatial_k = self.create_grid_and_kernels(
            nx, ny, method="gaussian", bandwidth=2.0
        )

        # Create two independent smooth datasets
        np.random.seed(42)
        a = np.zeros((ny, nx))
        for _i in range(3):
            freq_x = np.random.randint(1, 4)
            freq_y = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.randn() * 5.0
            a += amplitude * np.sin(
                2 * np.pi * (freq_x * np.arange(nx) / nx + freq_y * np.arange(ny)[:, None] / ny)
                + phase
            )

        b = np.random.randn(ny, nx)
        x_data = a + b
        y_data = a - b

        x_flat = x_data.ravel()
        y_flat = y_data.ravel()

        # FFT test
        R_fft = spatial_r_test_fft(x_data, y_data, fft_k, return_pval=False)

        # Standard test on original data
        R_std = spatial_r_test_standard(x_flat, y_flat, spatial_k, return_pval=False)

        # Results should be extremely close
        rel_error = np.abs(R_fft - R_std) / (np.abs(R_std) + 1e-10)
        assert rel_error < 1e-5, f"R statistic {R_fft:.2f}/{R_std:.2f}, rel error: {rel_error: .2f}"

    def test_batched_vs_sequential_q(self):
        """Test that batched Q computation matches sequential."""
        nx, ny = 20, 20
        fft_k = FFTKernel((ny, nx), method="gaussian", bandwidth=1.0)

        np.random.seed(42)
        batch_size = 5
        x_batch = np.random.randn(ny, nx, batch_size)

        # Batched
        Q_batch = spatial_q_test_fft(x_batch, fft_k, return_pval=False)

        # Sequential
        Q_seq = np.array(
            [
                spatial_q_test_fft(x_batch[..., i], fft_k, return_pval=False)
                for i in range(batch_size)
            ]
        )

        assert np.allclose(Q_batch, Q_seq, rtol=1e-10)

    def test_batched_vs_sequential_r(self):
        """Test that batched R computation matches sequential."""
        nx, ny = 20, 20
        fft_k = FFTKernel((ny, nx), method="gaussian", bandwidth=1.0)

        np.random.seed(42)
        batch_size = 5
        x_batch = np.random.randn(ny, nx, batch_size)
        y_batch = np.random.randn(ny, nx, batch_size)

        # Batched
        R_batch = spatial_r_test_fft(x_batch, y_batch, fft_k, return_pval=False)

        # Sequential
        R_seq = np.array(
            [
                spatial_r_test_fft(x_batch[..., i], y_batch[..., i], fft_k, return_pval=False)
                for i in range(batch_size)
            ]
        )

        assert np.allclose(R_batch, R_seq, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
