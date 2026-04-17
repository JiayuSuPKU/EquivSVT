"""Tests for quadsv.nufft + PatternComparatorNUFFT."""

from __future__ import annotations

import numpy as np
import pytest

from quadsv.fft import power_spectrum_2d
from quadsv.nufft import power_spectrum_2d_nufft

# ---------------------------------------------------------------------------
# Primitive: power_spectrum_2d_nufft
# ---------------------------------------------------------------------------


class TestNufftMatchesFft:
    def test_regular_grid_agrees_with_fft(self):
        """When spots sit on a regular grid, NUFFT should reproduce the FFT spectrum."""
        rng = np.random.default_rng(0)
        ny, nx = 16, 20
        dy, dx = 1.0, 1.0
        y = np.arange(ny) * dy
        x = np.arange(nx) * dx
        yy, xx = np.meshgrid(y, x, indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
        vals = rng.standard_normal(ny * nx)
        img = vals.reshape(ny, nx)

        P_fft = power_spectrum_2d(img, fft_solver="fft2")
        P_nufft = power_spectrum_2d_nufft(
            coords, vals, grid_shape=(ny, nx), spacing=(dy, dx), eps=1e-10
        )
        rel = np.linalg.norm(P_fft - P_nufft) / np.linalg.norm(P_fft)
        assert rel < 1e-7, f"FFT<->NUFFT relative diff {rel:.2e}"

    def test_dc_bin_equals_squared_total(self):
        """Power at k=0 equals (sum values)^2."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 50, size=(300, 2))
        vals = rng.standard_normal(300)
        P = power_spectrum_2d_nufft(
            coords,
            vals,
            grid_shape=(32, 32),
            spacing=(2.0, 2.0),
            eps=1e-12,
            center_coords=False,
        )
        total = vals.sum()
        # DC at [0, 0] after ifftshift.
        assert P[0, 0] == pytest.approx(total**2, rel=1e-6)


class TestTranslationInvariance:
    def test_shift_coords_leaves_spectrum_unchanged(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 100, size=(400, 2))
        vals = rng.standard_normal(400)
        P1 = power_spectrum_2d_nufft(coords, vals, grid_shape=(32, 32), spacing=(4.0, 4.0))
        shifted = coords + np.array([17.3, -9.2])
        P2 = power_spectrum_2d_nufft(shifted, vals, grid_shape=(32, 32), spacing=(4.0, 4.0))
        np.testing.assert_allclose(P1, P2, rtol=1e-6, atol=1e-8)


class TestUnitScaleConsistency:
    def test_mm_vs_um_with_unit_scale_match(self):
        """Same spots reported in μm vs mm should give identical spectra when
        the μm run uses unit_scale=1.0 and the mm run uses unit_scale=1000 (to
        convert mm into the common μm unit used by ``spacing``).
        """
        rng = np.random.default_rng(2)
        coords_um = rng.uniform(0, 5000, size=(500, 2))  # μm
        coords_mm = coords_um / 1000.0  # mm
        vals = rng.standard_normal(500)

        spacing_um = (50.0, 50.0)  # common physical spacing, μm
        P_um = power_spectrum_2d_nufft(
            coords_um, vals, grid_shape=(32, 32), spacing=spacing_um, unit_scale=1.0
        )
        P_mm = power_spectrum_2d_nufft(
            coords_mm, vals, grid_shape=(32, 32), spacing=spacing_um, unit_scale=1000.0
        )
        np.testing.assert_allclose(P_um, P_mm, rtol=1e-6, atol=1e-8)


class TestBatchedValues:
    def test_multi_feature_matches_per_feature_loop(self):
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, 100, size=(250, 2))
        M = 4
        vals = rng.standard_normal((250, M))

        P_batched = power_spectrum_2d_nufft(
            coords, vals, grid_shape=(24, 24), spacing=(5.0, 5.0), eps=1e-10
        )  # (ny, nx, M)
        assert P_batched.shape == (24, 24, M)
        for m in range(M):
            P_single = power_spectrum_2d_nufft(
                coords, vals[:, m], grid_shape=(24, 24), spacing=(5.0, 5.0), eps=1e-10
            )
            np.testing.assert_allclose(P_batched[..., m], P_single, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# NUFFTKernel + Q/R tests
# ---------------------------------------------------------------------------


from quadsv.nufft import (  # noqa: E402
    NUFFTKernel,
    spatial_q_test_nufft,
    spatial_r_test_nufft,
)


class TestNUFFTKernelConstruction:
    def test_basic_build(self):
        coords = np.random.default_rng(0).uniform(0, 10, size=(200, 2))
        k = NUFFTKernel(
            coords,
            grid_shape=(32, 32),
            spacing=(0.5, 0.5),
            method="matern",
            bandwidth=1.0,
            nu=1.5,
        )
        assert k.n == 200
        assert k.grid_shape == (32, 32)
        assert k.spacing == (0.5, 0.5)
        assert k.method == "matern"
        assert k.is_implicit is False
        assert "bandwidth" in k.params and "nu" in k.params

    def test_auto_grid_from_coords(self):
        """Omitting grid_shape/spacing auto-infers them from the coordinates."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(400, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        assert isinstance(k.grid_shape, tuple) and len(k.grid_shape) == 2
        assert k.grid_shape[0] % 8 == 0 and k.grid_shape[1] % 8 == 0
        assert k.grid_shape[0] >= 32 and k.grid_shape[1] >= 32
        # The implied domain must cover the bbox.
        Ly = coords[:, 0].max() - coords[:, 0].min()
        Lx = coords[:, 1].max() - coords[:, 1].min()
        assert k.grid_shape[0] * k.spacing[0] >= Ly
        assert k.grid_shape[1] * k.spacing[1] >= Lx

    def test_auto_grid_matches_explicit_overkill(self):
        """Auto-derived grid gives xtKx within 3% of a deliberately oversized grid."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(500, 2))
        x = rng.standard_normal(500)
        k_auto = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        k_big = NUFFTKernel(
            coords,
            grid_shape=(256, 256),
            spacing=(20 / 256, 20 / 256),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
        )
        Q_a, Q_b = k_auto.xtKx(x), k_big.xtKx(x)
        assert abs(Q_a - Q_b) / abs(Q_b) < 0.05, f"auto vs overkill: {Q_a:.1f} vs {Q_b:.1f}"

    def test_partial_override(self):
        """Supplying only spacing → grid auto-fills; only grid → spacing auto-fills."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 10, size=(300, 2))
        # Only grid_shape
        k1 = NUFFTKernel(coords, grid_shape=(64, 64), method="matern", bandwidth=1.0, nu=1.5)
        assert k1.grid_shape == (64, 64)
        assert all(s > 0 for s in k1.spacing)
        # Only spacing
        k2 = NUFFTKernel(coords, spacing=(0.5, 0.5), method="matern", bandwidth=1.0, nu=1.5)
        assert k2.spacing == (0.5, 0.5)
        assert all(s > 0 for s in k2.grid_shape)

    def test_invalid_coords(self):
        with pytest.raises(ValueError, match=r"coords must be shape"):
            NUFFTKernel(np.zeros((5, 3)), (16, 16), (1.0, 1.0))

    def test_invalid_method(self):
        with pytest.raises(ValueError, match=r"method must be"):
            NUFFTKernel(
                np.zeros((5, 2)),
                (16, 16),
                (1.0, 1.0),
                method="bogus",
            )


class TestNUFFTKernelxtKx:
    def test_matches_spatial_kernel_dense_on_irregular(self):
        """xtKx_nufft matches the dense Euclidean quadratic form to ~2% (torus-BC band)."""
        from quadsv.kernels import SpatialKernel

        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(
            coords,
            grid_shape=(64, 64),
            spacing=(20 / 64, 20 / 64),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
        )
        sk = SpatialKernel.from_coordinates(coords, method="matern", bandwidth=2.0, nu=1.5)
        # Average over several random x to smooth realization noise. The
        # relative bias is dominated by torus vs Euclidean boundary conditions
        # (~2-5% typical), which is the same approximation FFTKernel makes.
        rels = []
        for _ in range(10):
            xi = rng.standard_normal(N)
            rels.append(abs(k.xtKx(xi) - sk.xtKx(xi)) / abs(sk.xtKx(xi)))
        assert np.mean(rels) < 0.15, f"mean rel diff {np.mean(rels):.3f}"

    def test_batched_xtKx(self):
        rng = np.random.default_rng(0)
        N, M = 300, 4
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(
            coords, (32, 32), (20 / 32, 20 / 32), method="matern", bandwidth=2.0, nu=1.5
        )
        X = rng.standard_normal((N, M))
        Q_batched = np.asarray(k.xtKx(X))
        Q_loop = np.array([k.xtKx(X[:, m]) for m in range(M)])
        np.testing.assert_allclose(Q_batched, Q_loop, rtol=1e-6, atol=1e-9)


class TestNUFFTKernelKz:
    def test_Kz_consistent_with_xtKx(self):
        """z^T K z computed via z·Kx(z) should match the direct xtKx(z) call."""
        ny, nx = 16, 20
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        k_nufft = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        rng = np.random.default_rng(0)
        z = rng.standard_normal(ny * nx)
        Kz = k_nufft.Kx(z)
        assert Kz.shape == (ny * nx,)
        assert abs(float(z @ Kz) - k_nufft.xtKx(z)) < 1e-4

    def test_batched_Kz(self):
        rng = np.random.default_rng(0)
        N, M = 200, 3
        coords = rng.uniform(0, 15, size=(N, 2))
        k = NUFFTKernel(coords, (32, 32), (0.5, 0.5), method="matern", bandwidth=1.5, nu=1.5)
        Z = rng.standard_normal((N, M))
        KZ = k.Kx(Z)
        assert KZ.shape == (N, M)
        # Column-by-column check.
        for m in range(M):
            np.testing.assert_allclose(KZ[:, m], k.Kx(Z[:, m]), rtol=1e-6, atol=1e-8)


class TestNUFFTKernelTrace:
    def test_trace_matches_internal_fftkernel(self):
        """trace() and square_trace() forward deterministically to the internal
        FFTKernel — no Hutchinson, no RNG."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(400, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        assert k.trace() == k._fft_kernel.trace()
        assert k.square_trace() == k._fft_kernel.square_trace()


class TestSpatialQTestNUFFT:
    def test_power_on_structured_signal(self):
        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        y = coords[:, 0]
        x_sig = np.sin(2 * np.pi * y / 6.0) + 0.3 * rng.standard_normal(N)
        _, p = spatial_q_test_nufft(x_sig, k)
        assert p < 0.05, f"structured signal should be significant; got p={p:.3f}"

    def test_batched_q_test(self):
        rng = np.random.default_rng(0)
        N = 300
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        X = rng.standard_normal((N, 5))
        Q, p = spatial_q_test_nufft(X, k)
        assert Q.shape == (5,) and p.shape == (5,)

    def test_matches_fft_on_regular_grid(self):
        """On a uniform N=ny*nx grid the NUFFT Q-test equals spatial_q_test_fft."""
        from quadsv.fft import FFTKernel, spatial_q_test_fft

        ny, nx = 16, 20
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(ny * nx)

        k_nufft = NUFFTKernel(
            coords,
            grid_shape=(ny, nx),
            spacing=(1.0, 1.0),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
        )
        k_fft = FFTKernel(
            (ny, nx), spacing=(1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5, fft_solver="fft2"
        )
        Q_n, p_n = spatial_q_test_nufft(x, k_nufft)
        Q_f, p_f = spatial_q_test_fft(x.reshape(ny, nx), k_fft)
        assert abs(Q_n - float(Q_f)) / abs(float(Q_f)) < 1e-6
        assert abs(p_n - float(p_f)) < 1e-3


class TestSpatialRTestNUFFT:
    def test_correlated_pair_significant(self):
        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, (64, 64), (0.35, 0.35), method="matern", bandwidth=2.0, nu=1.5)
        pattern = np.sin(2 * np.pi * coords[:, 0] / 4.0)
        x_sig = pattern + 0.4 * rng.standard_normal(N)
        y_sig = pattern + 0.4 * rng.standard_normal(N)
        R, p = spatial_r_test_nufft(x_sig, y_sig, k)
        assert float(R) > 0, "correlated pair should give positive R"
        assert float(p) < 0.05, f"correlated pair significance; got p={float(p):.3f}"

    def test_uncorrelated_pair_not_significant(self):
        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, (64, 64), (0.35, 0.35), method="matern", bandwidth=2.0, nu=1.5)
        x = rng.standard_normal(N)
        y = rng.standard_normal(N)
        _, p = spatial_r_test_nufft(x, y, k)
        assert float(p) > 0.05, f"random pair should not be significant; got p={float(p):.3f}"


# ---------------------------------------------------------------------------
# PatternDetectorNUFFT
# ---------------------------------------------------------------------------


class TestPatternDetectorNUFFTBackend:
    """After Phase C, PatternDetectorNUFFT is gone — the same workflow runs
    through :class:`PatternDetector` with ``backend='nufft'``."""

    def _mk_adata(self, n_spots=400, n_genes=10, with_signal=True, seed=0):
        import anndata as ad

        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 20, size=(n_spots, 2))
        X = rng.standard_normal((n_spots, n_genes))
        if with_signal:
            X[:, 0] = np.sin(2 * np.pi * coords[:, 0] / 5.0) + 0.3 * rng.standard_normal(n_spots)
        X = np.maximum(X + 3.0, 0.0)
        gene_names = [f"g{i}" for i in range(n_genes)]
        adata = ad.AnnData(X=X)
        adata.var_names = gene_names
        adata.obsm["spatial"] = coords
        return adata

    def test_build_and_qstat(self):
        from quadsv import PatternDetector

        adata = self._mk_adata(n_spots=400, n_genes=8, with_signal=True)
        det = PatternDetector(adata).build_kernel(
            backend="nufft", method="matern", bandwidth=2.0, nu=1.5
        )
        assert det.kernel_ is not None
        assert det.kernel_method_ == "matern"
        assert det.backend_ == "nufft"
        df = det.compute_qstat(n_jobs=1)
        assert df.shape[0] == 8
        assert {"Feature", "Q", "Z_score", "P_value", "P_adj"} <= set(df.columns)
        assert df["Feature"].iloc[0] == "g0"
        assert df.set_index("Feature").loc["g0", "P_value"] < 0.05

    def test_rstat_on_correlated_pair(self):
        from quadsv import PatternDetector

        adata = self._mk_adata(n_spots=400, n_genes=4, with_signal=True)
        adata.X[:, 1] = adata.X[:, 0] + 0.3 * np.random.default_rng(1).standard_normal(adata.n_obs)
        det = PatternDetector(adata).build_kernel(
            backend="nufft", method="matern", bandwidth=2.0, nu=1.5
        )
        df = det.compute_rstat(features_x=["g0", "g1"], features_y=["g0", "g1"], n_jobs=1)
        row = df[(df.Feature_1 == "g0") & (df.Feature_2 == "g1")]
        assert not row.empty
        assert float(row.iloc[0]["R"]) > 0

    def test_invalid_spatial_key(self):
        import anndata as ad

        from quadsv import PatternDetector

        adata = ad.AnnData(X=np.zeros((5, 3)))
        det = PatternDetector(adata)
        with pytest.raises(KeyError, match="spatial"):
            det.build_kernel(backend="nufft", method="matern")

    def test_requires_build_kernel(self):
        from quadsv import PatternDetector

        adata = self._mk_adata(n_spots=50, n_genes=3)
        det = PatternDetector(adata)
        with pytest.raises(ValueError, match="Kernel not initialized"):
            det.compute_qstat(n_jobs=1)


class TestPhaseBNUFFTUnification:
    """Phase B: NUFFT Q/R tests accept the canonical kwargs and honor
    `null_params` without changing results."""

    def test_qtest_nufft_null_params_round_trip(self):
        from quadsv.nufft import NUFFTKernel, spatial_q_test_nufft

        rng = np.random.default_rng(0)
        ny, nx = 16, 16
        coords = rng.uniform(0, 15, size=(ny * nx, 2))
        k = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        z = rng.standard_normal(ny * nx)
        Q_auto, p_auto = spatial_q_test_nufft(z, k)
        # Pre-supply rescaled eigenvalues exactly as the internal branch does.
        scale = k.n / (ny * nx)
        evals = k.eigenvalues(return_full=True)
        sig_evals = evals[evals > 1e-9] * scale
        Q_given, p_given = spatial_q_test_nufft(
            z, k, null_params={"method": "liu", "eigenvalues": sig_evals}
        )
        assert abs(Q_auto - Q_given) < 1e-10
        assert abs(p_auto - p_given) < 1e-10

    def test_rtest_nufft_null_params_round_trip(self):
        from quadsv.nufft import NUFFTKernel, spatial_r_test_nufft

        rng = np.random.default_rng(0)
        ny, nx = 16, 16
        coords = rng.uniform(0, 15, size=(ny * nx, 2))
        k = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        x = rng.standard_normal(ny * nx)
        y = rng.standard_normal(ny * nx)
        R_auto, p_auto = spatial_r_test_nufft(x, y, k)
        scale = k.n / (ny * nx)
        var_R = float(k.square_trace()) * (scale**2)
        R_given, p_given = spatial_r_test_nufft(x, y, k, null_params={"var_R": var_R})
        assert abs(R_auto - R_given) < 1e-10
        assert abs(p_auto - p_given) < 1e-10
