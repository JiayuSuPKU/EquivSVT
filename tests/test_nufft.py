"""Tests for quadsv.nufft + ComparatorIrregular."""

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


from quadsv.nufft import NUFFTKernel
from quadsv.statistics import spatial_q_test, spatial_r_test


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
        assert k.stores_precision is False
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
        # 5% band: auto-derived grid (oversample=2 above the sampling Nyquist)
        # vs a deliberately oversized 256x256 grid; residual is NUFFT eps +
        # grid discretization on the auto side.
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
        from quadsv.kernels import MatrixKernel

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
        sk = MatrixKernel.from_coordinates(coords, method="matern", bandwidth=2.0, nu=1.5)
        # Average over several random x to smooth realization noise. The
        # 15% budget breaks down as: torus-BC vs Euclidean BC (~2-5%), NUFFT
        # precision at eps=1e-6 (~1-3%), kernel discretization on the
        # 64×64 k-grid (~2-5%). Same approximation stack the FFTKernel uses.
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


class TestNUFFTKernelKx:
    def test_Kx_consistent_with_xtKx(self):
        """x^T K x computed via x·Kx(x) should match the direct xtKx(x) call."""
        ny, nx = 16, 20
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        k_nufft = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        rng = np.random.default_rng(0)
        z = rng.standard_normal(ny * nx)
        Kz = k_nufft.Kx(z)
        assert Kz.shape == (ny * nx,)
        assert abs(float(z @ Kz) - k_nufft.xtKx(z)) < 1e-4

    def test_batched_Kx(self):
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
    def test_trace_analytic_scales_fftkernel(self):
        """Raw ``trace`` / ``square_trace`` (centering=False) rescale the
        internal FFTKernel's moments by ``N/n'`` and ``(N/n')²`` — the
        n-point-operator scaling used by :func:`spatial_q_test`."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(400, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5, centering=False)
        n_over_nprime = k.n / (k.grid_shape[0] * k.grid_shape[1])
        assert k.trace() == pytest.approx(k._fft_kernel.trace() * n_over_nprime, rel=1e-12)
        assert k.square_trace() == pytest.approx(
            k._fft_kernel.square_trace() * n_over_nprime**2, rel=1e-12
        )

    def test_trace_hutchinson_matches_analytic(self):
        """Hutchinson-estimated moments should agree with the analytic scaling
        within the probe-noise band on a typical NUFFT kernel."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(400, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        t_ana = k.trace(method="analytic")
        t_hut = k.trace(method="hutchinson", n_probes=64)
        assert abs(t_ana - t_hut) / abs(t_ana) < 0.25
        s_ana = k.square_trace(method="analytic")
        s_hut = k.square_trace(method="hutchinson", n_probes=64)
        assert abs(s_ana - s_hut) / abs(s_ana) < 0.25


class TestNUFFTTwoPathsAgree:
    def test_spectral_vs_matmul_on_irregular_coords(self):
        """``kernel.xtKx`` (Path A, spectral) and ``kernel.xtKx_matmul``
        (Path B) target the same band-limited quadratic form."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(500, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        x = rng.standard_normal(500)
        Q_a = k.xtKx(x)
        Q_b = k.xtKx_matmul(x)
        assert abs(Q_a - Q_b) / max(abs(Q_a), 1e-30) < 0.05

    def test_spectral_vs_matmul_on_regular_grid(self):
        """On a regular grid (n = n') both paths reduce to the same FFT —
        residual is float-precision only (~1e-8 typical)."""
        ny, nx = 16, 16
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        k = NUFFTKernel(
            coords,
            grid_shape=(ny, nx),
            spacing=(1.0, 1.0),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
        )
        # Loop a few seeds to confirm 1e-8 holds reliably, not just for one rng.
        for seed in range(5):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(ny * nx)
            Q_a = k.xtKx(x)
            Q_b = k.xtKx_matmul(x)
            assert (
                abs(Q_a - Q_b) / max(abs(Q_a), 1e-30) < 1e-8
            ), f"seed={seed}: Q_a={Q_a:.6e}, Q_b={Q_b:.6e}"

    def test_trace_analytic_vs_hutchinson_agree_on_irregular(self):
        """``trace(method='analytic')`` (Path A null) and
        ``trace(method='hutchinson')`` (Path B null) should agree within the
        Hutchinson probe noise on an irregular-coord kernel."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(500, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        t_ana = k.trace(method="analytic")
        t_hut = k.trace(method="hutchinson", n_probes=128)
        assert abs(t_ana - t_hut) / max(abs(t_ana), 1e-30) < 0.2
        s_ana = k.square_trace(method="analytic")
        s_hut = k.square_trace(method="hutchinson", n_probes=128)
        assert abs(s_ana - s_hut) / max(abs(s_ana), 1e-30) < 0.2

    def test_trace_analytic_vs_hutchinson_agree_on_regular_grid(self):
        """On a regular grid (N = n') the band-limited approximation is exact,
        so the analytic and Hutchinson estimators share the same population
        moments — the gap is only probe noise. Both estimators act on the
        raw operator; disable centering to isolate the scaling identity."""
        ny, nx = 16, 16
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        k = NUFFTKernel(
            coords,
            grid_shape=(ny, nx),
            spacing=(1.0, 1.0),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
            centering=False,
        )
        t_ana = k.trace(method="analytic")
        t_hut = k.trace(method="hutchinson", n_probes=128)
        # On a regular grid the torus-BC band is gone; 15 % probe-noise band
        # is a comfortable bound for n_probes=128.
        assert abs(t_ana - t_hut) / max(abs(t_ana), 1e-30) < 0.15
        s_ana = k.square_trace(method="analytic")
        s_hut = k.square_trace(method="hutchinson", n_probes=128)
        assert abs(s_ana - s_hut) / max(abs(s_ana), 1e-30) < 0.15

    def test_Kx_grid_round_trip_matches_xtKx(self):
        """``Kx_grid`` returns ``K x`` on the spatial grid; its inner product
        with the same grid signal must equal ``xtKx`` algebraically. The
        residual is bounded by the NUFFT precision ``eps`` (1e-6 default) —
        both methods run a type-1 NUFFT but ``Kx_grid`` additionally does a
        phase correction + ``ifft2`` round-trip, so the gap sits at ``eps``
        rather than float-precision."""
        ny, nx = 16, 16
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
        k = NUFFTKernel(
            coords,
            grid_shape=(ny, nx),
            spacing=(1.0, 1.0),
            method="matern",
            bandwidth=2.0,
            nu=1.5,
            eps=1e-10,  # push NUFFT precision down so the identity is tight.
            centering=False,
        )
        rng = np.random.default_rng(3)
        x = rng.standard_normal(ny * nx)
        x_grid_signal = x.reshape(ny, nx)
        Kx_on_grid = k.Kx_grid(x)  # (ny, nx) real
        assert Kx_on_grid.shape == (ny, nx)
        q_via_grid = float(np.sum(x_grid_signal * Kx_on_grid))
        q_direct = float(k.xtKx(x))
        assert abs(q_via_grid - q_direct) / max(abs(q_direct), 1e-30) < 1e-8


class TestSpatialQTestNUFFT:
    def test_power_on_structured_signal(self):
        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        y = coords[:, 0]
        x_sig = np.sin(2 * np.pi * y / 6.0) + 0.3 * rng.standard_normal(N)
        _, p = spatial_q_test(x_sig, k)
        assert p < 0.05, f"structured signal should be significant; got p={p:.3f}"

    def test_batched_q_test(self):
        rng = np.random.default_rng(0)
        N = 300
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
        X = rng.standard_normal((N, 5))
        Q, p = spatial_q_test(X, k)
        assert Q.shape == (5,) and p.shape == (5,)

    def test_matches_fft_on_regular_grid(self):
        """On a uniform N=ny*nx grid the NUFFT Q-test equals spatial_q_test (FFT kernel)."""
        from quadsv.fft import FFTKernel

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
        Q_n, p_n = spatial_q_test(x, k_nufft)
        Q_f, p_f = spatial_q_test(x.reshape(ny, nx), k_fft)
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
        R, p = spatial_r_test(x_sig, y_sig, k)
        assert float(R) > 0, "correlated pair should give positive R"
        assert float(p) < 0.05, f"correlated pair significance; got p={float(p):.3f}"

    def test_uncorrelated_pair_not_significant(self):
        rng = np.random.default_rng(0)
        N = 400
        coords = rng.uniform(0, 20, size=(N, 2))
        k = NUFFTKernel(coords, (64, 64), (0.35, 0.35), method="matern", bandwidth=2.0, nu=1.5)
        x = rng.standard_normal(N)
        y = rng.standard_normal(N)
        _, p = spatial_r_test(x, y, k)
        assert float(p) > 0.05, f"random pair should not be significant; got p={float(p):.3f}"


# ---------------------------------------------------------------------------
# DetectorNUFFT
# ---------------------------------------------------------------------------


class TestDetectorNUFFTBackend:
    """After Phase C, DetectorNUFFT is gone — the same workflow runs
    through :class:`DetectorIrregular` with ``backend='nufft'``."""

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
        from quadsv import DetectorIrregular

        adata = self._mk_adata(n_spots=400, n_genes=8, with_signal=True)
        det = DetectorIrregular(kernel_method="matern", backend="nufft", bandwidth=2.0, nu=1.5)
        det.setup_data(adata)
        assert det.kernel_ is not None
        assert det.kernel_method_ == "matern"
        assert det.backend_ == "nufft"
        df = det.compute_qstat(n_jobs=1)
        assert df.shape[0] == 8
        assert {"Feature", "Q", "Z_score", "P_value", "P_adj"} <= set(df.columns)
        assert df["Feature"].iloc[0] == "g0"
        assert df.set_index("Feature").loc["g0", "P_value"] < 0.05

    def test_rstat_on_correlated_pair(self):
        from quadsv import DetectorIrregular

        adata = self._mk_adata(n_spots=400, n_genes=4, with_signal=True)
        adata.X[:, 1] = adata.X[:, 0] + 0.3 * np.random.default_rng(1).standard_normal(adata.n_obs)
        det = DetectorIrregular(kernel_method="matern", backend="nufft", bandwidth=2.0, nu=1.5)
        det.setup_data(adata)
        df = det.compute_rstat(features_x=["g0", "g1"], features_y=["g0", "g1"], n_jobs=1)
        row = df[(df.Feature_1 == "g0") & (df.Feature_2 == "g1")]
        assert not row.empty
        assert float(row.iloc[0]["R"]) > 0

    def test_invalid_spatial_key(self):
        import anndata as ad

        from quadsv import DetectorIrregular

        adata = ad.AnnData(X=np.zeros((5, 3)))
        det = DetectorIrregular(kernel_method="matern", backend="nufft")
        with pytest.raises(KeyError, match="spatial"):
            det.setup_data(adata)

    def test_requires_build_kernel(self):
        from quadsv import DetectorIrregular

        det = DetectorIrregular(kernel_method="matern", backend="nufft")
        with pytest.raises((ValueError, RuntimeError), match="setup_data|Kernel not initialized"):
            det.compute_qstat(n_jobs=1)


class TestNUFFTKernelNullParamsRoundTrip:
    """NUFFT Q/R tests accept the canonical kwargs and honor ``null_params``
    without changing results."""

    def test_qtest_nufft_null_params_round_trip(self):
        from quadsv.nufft import NUFFTKernel
        from quadsv.statistics import spatial_q_test

        rng = np.random.default_rng(0)
        ny, nx = 16, 16
        coords = rng.uniform(0, 15, size=(ny * nx, 2))
        k = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        z = rng.standard_normal(ny * nx)
        Q_auto, p_auto = spatial_q_test(z, k)
        # Pre-supply rescaled eigenvalues exactly as the internal branch does.
        scale = k.n / (ny * nx)
        evals = k.eigenvalues(return_full_layout=True)
        sig_evals = evals[evals > 1e-9] * scale
        Q_given, p_given = spatial_q_test(
            z, k, null_params={"method": "liu", "eigenvalues": sig_evals}
        )
        assert abs(Q_auto - Q_given) < 1e-10
        assert abs(p_auto - p_given) < 1e-10

    def test_rtest_nufft_null_params_round_trip(self):
        from quadsv.nufft import NUFFTKernel
        from quadsv.statistics import spatial_r_test

        rng = np.random.default_rng(0)
        ny, nx = 16, 16
        coords = rng.uniform(0, 15, size=(ny * nx, 2))
        k = NUFFTKernel(coords, (ny, nx), (1.0, 1.0), method="matern", bandwidth=2.0, nu=1.5)
        x = rng.standard_normal(ny * nx)
        y = rng.standard_normal(ny * nx)
        R_auto, p_auto = spatial_r_test(x, y, k)
        # Since both X, Y are z-scored, var_R = trace((HKH)²). With the
        # default centering=True kernel, ``k.square_trace()`` returns
        # exactly that centered trace.
        var_R = float(k.square_trace())
        R_given, p_given = spatial_r_test(x, y, k, null_params={"var_R": var_R})
        assert abs(R_auto - R_given) < 1e-10
        assert abs(p_auto - p_given) < 1e-10


class TestNUFFTKNeighborsAPI:
    """``NUFFTKernel`` forwards ``k_neighbors`` to its internal ``FFTKernel``
    — same k-NN semantic as MatrixKernel.
    """

    def test_k_neighbors_forwarded(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(200, 2))
        k = NUFFTKernel(
            coords, grid_shape=(32, 32), spacing=(0.6, 0.6), method="moran", k_neighbors=4
        )
        assert k.params["neighbor_degree"] == 1

    def test_k_neighbors_for_car(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, size=(200, 2))
        k = NUFFTKernel(
            coords, grid_shape=(32, 32), spacing=(0.6, 0.6), method="car", k_neighbors=4, rho=0.9
        )
        assert k.params["neighbor_degree"] == 1
        assert k.params["rho"] == 0.9


class TestNUFFTEmpiricalNullMoments:
    """NUFFT graph kernels use z-scored (standardized) probes by default for
    ``trace`` / ``square_trace`` so the reported null moments match the
    realized variance of Q after :func:`spatial_q_test`'s z-scoring.
    Without this, CAR/Moran/graph_laplacian FPR collapsed to 0 under
    Welch/CLT (Fig 2C of the comparison notebook).
    """

    def _mk_irregular(self, n=1024, seed=11):
        from scipy.stats.qmc import PoissonDisk

        L = float(np.sqrt(n / 2))
        pts = PoissonDisk(d=2, radius=0.25 / L, seed=seed).random(n)
        return pts * L, L

    def test_centered_trace_strictly_less_than_raw(self):
        """Default ``trace()`` (centered) is less than raw ``trace(K)`` for
        graph kernels where the constant-mode eigenvalue is large.
        """
        coords, _ = self._mk_irregular()
        k_raw = NUFFTKernel(
            coords, method="car", k_neighbors=4, rho=0.9, workers=1, centering=False
        )
        k_cen = NUFFTKernel(coords, method="car", k_neighbors=4, rho=0.9, workers=1)
        assert k_cen.trace() < k_raw.trace()

    def test_welch_fpr_calibrated_on_nufft_car(self):
        from quadsv.statistics import compute_null_params

        coords, _ = self._mk_irregular()
        k = NUFFTKernel(coords, method="car", k_neighbors=4, rho=0.9, workers=1)
        X = np.random.default_rng(0).standard_normal((k.n, 400))
        params = compute_null_params(k, method="welch")
        _, pv = spatial_q_test(X, k, null_params=params)
        fpr = float((np.asarray(pv) < 0.05).mean())
        # Empirical null moments should land FPR within ±0.03 of nominal 0.05.
        assert abs(fpr - 0.05) < 0.03, f"NUFFT-CAR welch FPR {fpr} off target"

    def test_clt_fpr_calibrated_on_nufft_moran(self):
        from quadsv.statistics import compute_null_params

        coords, _ = self._mk_irregular()
        k = NUFFTKernel(coords, method="moran", k_neighbors=4, workers=1)
        X = np.random.default_rng(1).standard_normal((k.n, 400))
        params = compute_null_params(k, method="clt")
        _, pv = spatial_q_test(X, k, null_params=params)
        fpr = float((np.asarray(pv) < 0.05).mean())
        assert abs(fpr - 0.05) < 0.03, f"NUFFT-Moran clt FPR {fpr} off target"
