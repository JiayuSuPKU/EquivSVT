"""Tests for quadsv.spectral_compare."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy.stats import kstest

from quadsv.fft import power_spectrum_2d
from quadsv.spectral_compare import (
    SpectralComparator,
    align_spectra_by_rotation,
    benchmark_statistics,
    compare_two_groups,
    compare_two_groups_scalar,
    compute_sample_spectrum,
    normalize_by_background,
    radial_bin_spectrum,
    residualize_against_covariates,
    shape_normalize,
)

# ---------------------------------------------------------------------------
# Test helpers for the AnnData-based SpectralComparator API (Phase D)
# ---------------------------------------------------------------------------


def _grid_to_adata(sample: np.ndarray, gene_names, spacing=(1.0, 1.0)):
    """Wrap a pre-rasterized ``(n_genes, ny, nx)`` array into an AnnData with a
    regular-grid ``obsm['spatial']``. The NUFFT path will evaluate the spectrum
    on (approximately) the same grid, and radial binning with explicit
    ``spacing`` makes feature vectors cross-sample comparable."""
    n_genes, ny, nx = sample.shape
    X = sample.reshape(n_genes, ny * nx).T  # (ny*nx, n_genes)
    yy, xx = np.meshgrid(
        np.arange(ny) * spacing[0],
        np.arange(nx) * spacing[1],
        indexing="ij",
    )
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1)
    a = ad.AnnData(X=X.astype(np.float64))
    a.var_names = list(gene_names)
    a.obsm["spatial"] = coords
    return a


def _samples_to_adata_list(samples, gene_names, spacings=None):
    spacings = spacings if spacings is not None else [(1.0, 1.0)] * len(samples)
    return [_grid_to_adata(s, gene_names, spacing=spacings[i]) for i, s in enumerate(samples)]


# ---------------------------------------------------------------------------
# Sanity: power spectrum
# ---------------------------------------------------------------------------


class TestPowerSpectrumSanity:
    def test_constant_image_has_only_dc(self):
        img = 3.0 * np.ones((16, 16))
        P = power_spectrum_2d(img, fft_solver="fft2")
        # All power concentrated at DC (k=0).
        assert P[0, 0] == pytest.approx((3.0 * 16 * 16) ** 2)
        P[0, 0] = 0.0
        np.testing.assert_allclose(P, 0.0, atol=1e-20)

    def test_translation_invariance(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((24, 32))
        shifted = np.roll(img, shift=(5, -7), axis=(0, 1))
        P1 = power_spectrum_2d(img, fft_solver="fft2")
        P2 = power_spectrum_2d(shifted, fft_solver="fft2")
        np.testing.assert_allclose(P1, P2, atol=1e-9)

    def test_rfft2_shape(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((10, 16))
        P = power_spectrum_2d(img, fft_solver="rfft2")
        assert P.shape == (10, 9)  # 16 // 2 + 1


# ---------------------------------------------------------------------------
# Radial binning
# ---------------------------------------------------------------------------


class TestRadialBinning:
    def test_isotropic_gaussian_bump_decreases_radially(self):
        """An isotropic Gaussian bump in space has a Gaussian PSD; radial spectrum decreases."""
        ny, nx = 32, 32
        y, x = np.meshgrid(np.arange(ny) - ny / 2, np.arange(nx) - nx / 2, indexing="ij")
        r2 = y**2 + x**2
        bump = np.exp(-r2 / (2 * 4.0**2))
        P = power_spectrum_2d(bump, fft_solver="fft2")
        rb = radial_bin_spectrum(P, grid_shape=(ny, nx), n_bins=20, fft_solver="fft2")
        # First (DC excluded) bin should have the most energy; last bin the least.
        assert rb[0] > rb[-1]
        # Roughly monotonically decreasing (allow small reshuffles in mid-range).
        assert (np.diff(rb) <= 1e-6).sum() >= len(rb) - 4

    def test_radial_consistent_across_solvers(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((16, 24))
        P_full = power_spectrum_2d(img, fft_solver="fft2")
        P_half = power_spectrum_2d(img, fft_solver="rfft2")
        rb_full = radial_bin_spectrum(P_full, grid_shape=(16, 24), n_bins=8, fft_solver="fft2")
        rb_half = radial_bin_spectrum(P_half, grid_shape=(16, 24), n_bins=8, fft_solver="rfft2")
        np.testing.assert_allclose(rb_full, rb_half, rtol=1e-9, atol=1e-9)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="last two dims"):
            radial_bin_spectrum(np.zeros((10, 10)), grid_shape=(8, 8), fft_solver="fft2")


# ---------------------------------------------------------------------------
# Rotation alignment
# ---------------------------------------------------------------------------


class TestRotationAlignment:
    def _anisotropic_pattern(self, ny: int, nx: int) -> np.ndarray:
        """Stripes oriented horizontally: power concentrates along a known axis."""
        y = np.arange(ny)[:, None]
        return np.broadcast_to(np.sin(2 * np.pi * y / 8).astype(float), (ny, nx))

    def test_recovers_known_rotation_within_bin_width(self):
        import scipy.ndimage

        ny = nx = 48
        ref = self._anisotropic_pattern(ny, nx)
        true_angle = 25.0
        rotated = scipy.ndimage.rotate(
            ref, angle=true_angle, reshape=False, order=1, mode="reflect"
        )
        # Wrap as (n_genes=1, ny, nx).
        sp_ref = compute_sample_spectrum(ref[None, :, :], fft_solver="fft2")
        sp_rot = compute_sample_spectrum(rotated[None, :, :], fft_solver="fft2")
        n_theta = 360
        _, angles = align_spectra_by_rotation(
            [sp_ref, sp_rot],
            grid_shapes=[(ny, nx), (ny, nx)],
            fft_solver="fft2",
            reference_index=0,
            n_theta=n_theta,
        )
        # Reference angle is 0; recovered angle should be ~true_angle (mod 180).
        recovered = angles[1] % 180.0
        true_mod = true_angle % 180.0
        diff = min(abs(recovered - true_mod), 180.0 - abs(recovered - true_mod))
        # Tolerance: 2 angular bins (180 / n_theta = 0.5°) plus interpolation slack.
        assert diff < 5.0, f"recovered={recovered}, true={true_mod}, diff={diff}"


# ---------------------------------------------------------------------------
# Background normalization & residualization
# ---------------------------------------------------------------------------


class TestBackgroundNormalization:
    def test_identical_genes_become_unit_after_normalization(self):
        # Every gene has the same spectrum -> geo mean = same -> ratio = 1.
        spec = np.tile(np.arange(1.0, 6.0), (10, 1))  # (10 genes, K=5), all rows equal
        out = normalize_by_background(spec)
        np.testing.assert_allclose(out, np.ones_like(out), atol=1e-9)

    def test_preserves_shape(self):
        rng = np.random.default_rng(0)
        spec = rng.uniform(0.1, 10.0, size=(7, 12))
        out = normalize_by_background(spec)
        assert out.shape == spec.shape


class TestResidualization:
    def test_perfect_predictor_residual_is_near_zero(self):
        rng = np.random.default_rng(0)
        cov = rng.uniform(0.1, 5.0, size=(2, 8))  # 2 covariates, K=8
        gene = 1.5 * cov[0] - 0.7 * cov[1] + 2.0  # exact linear combo + intercept
        gene = np.tile(gene, (5, 1))
        out = residualize_against_covariates(gene, cov, fit_intercept=True)
        np.testing.assert_allclose(out, 0.0, atol=1e-9)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="Last axis"):
            residualize_against_covariates(np.zeros((3, 5)), np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# Two-group test: calibration & power
# ---------------------------------------------------------------------------


class TestTwoGroupNullCalibration:
    def test_log_l2_pvalues_are_uniform_under_h0(self):
        # All samples drawn from same distribution -> p-values should be ~Uniform(0,1).
        rng = np.random.default_rng(42)
        n_samples, n_genes, K = 8, 200, 12
        spectra = rng.uniform(0.5, 5.0, size=(n_samples, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=300, random_state=0)
        # KS test against U(0, 1).
        ks_stat, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"p-values not uniform under H0: KS p={ks_p:.4f}"


class TestTwoGroupPower:
    def test_implanted_difference_is_recovered(self):
        rng = np.random.default_rng(7)
        n_per = 6
        n_genes, K = 50, 10
        # Group A: spectra ~ N(1, 0.1)
        a = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        b = rng.normal(loc=1.0, scale=0.1, size=(n_per, n_genes, K))
        # Implant a strong shift on the first 5 genes' low-frequency bins for group B.
        b[:, :5, :3] += 0.8
        spectra = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=400, random_state=0)
        # The 5 implanted genes (named "0".."4") should rank in the top 10.
        top10 = set(df.head(10)["Feature"].astype(str))
        implanted = {"0", "1", "2", "3", "4"}
        recovered = len(top10 & implanted)
        assert recovered >= 4, f"only recovered {recovered}/5 top-10: {top10}"


class TestStatisticAliases:
    @pytest.mark.parametrize("stat", ["log_l2", "max_welch", "hotelling_lw", "mmd_rbf"])
    def test_each_statistic_runs(self, stat):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 8, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic=stat, n_perm=50, random_state=0)
        assert df.shape[0] == 8
        assert {"Feature", "Statistic", "P_value", "P_adj"} <= set(df.columns)
        assert df["P_value"].between(0, 1).all()

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            compare_two_groups(
                np.zeros((4, 3, 5)),
                np.array([0, 0, 1, 1]),
                statistic="bogus",
            )


class TestBenchmark:
    def test_benchmark_returns_one_df_per_statistic(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(8, 12, 8))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = benchmark_statistics(spectra, groups, n_perm=50, random_state=0)
        assert set(out.keys()) == {"log_l2", "hotelling_lw", "mmd_rbf", "max_welch"}
        for _stat, df in out.items():
            assert df.shape[0] == 12
            assert df["P_value"].between(0, 1).all()


# ---------------------------------------------------------------------------
# End-to-end: SpectralComparator
# ---------------------------------------------------------------------------


class TestSpectralComparatorEndToEnd:
    def test_pipeline_radial_runs_and_finds_implanted_gene(self):
        rng = np.random.default_rng(3)
        n_per = 4
        ny = nx = 32
        n_genes = 10
        gene_names = [f"g{i}" for i in range(n_genes)]

        def make_sample(group: int) -> np.ndarray:
            x = rng.standard_normal((n_genes, ny, nx)) * 0.1
            if group == 1:
                yy = np.arange(ny)[:, None]
                stripes = np.broadcast_to(np.sin(2 * np.pi * yy / 16.0), (ny, nx))
                x[0] += stripes * 1.5
            return x

        samples = [make_sample(0) for _ in range(n_per)] + [make_sample(1) for _ in range(n_per)]
        groups = np.array([0] * n_per + [1] * n_per)

        cmp = (
            SpectralComparator(_samples_to_adata_list(samples, gene_names), groups, gene_names)
            .fit()
            .normalize_background()
        )
        df = cmp.test(statistic="log_l2", n_perm=300, random_state=0)
        assert df["Feature"].iloc[0] == "g0", f"expected g0 first, got {df.head().to_dict()}"

    def test_pipeline_residualize_runs(self):
        rng = np.random.default_rng(0)
        n_per = 3
        ny = nx = 16
        gene_names = [f"g{i}" for i in range(4)]
        samples = [rng.standard_normal((4, ny, nx)) for _ in range(2 * n_per)]
        covariates = [rng.standard_normal((1, ny, nx)) for _ in range(2 * n_per)]
        groups = np.array([0] * n_per + [1] * n_per)
        cmp = SpectralComparator(_samples_to_adata_list(samples, gene_names), groups, gene_names)
        cmp.fit().residualize(covariates)
        df = cmp.test(statistic="log_l2", n_perm=50, random_state=0)
        assert df.shape[0] == 4

    def test_invalid_groups_raises(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4))] * 3, gene_names)
        with pytest.raises(ValueError, match="exactly two distinct"):
            SpectralComparator(adatas, np.array([0, 1, 2]), gene_names)

    def test_must_fit_before_test(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = SpectralComparator(adatas, np.array([0, 1]), gene_names)
        with pytest.raises(RuntimeError, match=r"\.fit\(\)"):
            cmp.test()


# ---------------------------------------------------------------------------
# DC/AC decomposition & DE-vs-pattern orthogonality
# ---------------------------------------------------------------------------


class TestMeanCenteringMakesDcZero:
    def test_mean_center_yields_dc_zero(self):
        """With center='mean', the spectrum's k=0 bin is numerically zero."""
        rng = np.random.default_rng(0)
        sample = rng.standard_normal((3, 12, 14)) + 5.0  # non-zero mean
        spec = compute_sample_spectrum(sample, fft_solver="fft2", center="mean")
        # The DC bin is at index (0, 0) for both fft2 and rfft2 layouts.
        np.testing.assert_allclose(spec[:, 0, 0], 0.0, atol=1e-18)

    def test_return_dc_reports_per_gene_grid_means(self):
        rng = np.random.default_rng(1)
        sample = rng.standard_normal((4, 8, 10)) + np.arange(4)[:, None, None]
        spec, dc = compute_sample_spectrum(sample, center="mean", return_dc=True)
        np.testing.assert_allclose(dc, sample.mean(axis=(1, 2)), rtol=1e-12)
        # Spectrum shape preserved.
        assert spec.shape[0] == 4

    def test_zscore_makes_std_unity(self):
        rng = np.random.default_rng(2)
        sample = 3.0 * rng.standard_normal((2, 16, 16)) + 7.0
        spec = compute_sample_spectrum(sample, fft_solver="fft2", center="zscore")
        # After z-scoring, total power (Parseval) ≈ N * (N * variance) / N = N.
        # i.e. sum(|X̂|²) / N == N_cells (since std is 1).
        n_cells = 16 * 16
        per_gene_total = spec.sum(axis=(1, 2)) / n_cells
        np.testing.assert_allclose(per_gene_total, n_cells, rtol=1e-6)

    def test_no_centering_preserves_dc(self):
        img = 2.5 * np.ones((8, 8))[None, :, :]
        spec = compute_sample_spectrum(img, fft_solver="fft2", center=None)
        assert spec[0, 0, 0] == pytest.approx((2.5 * 8 * 8) ** 2)


class TestScalarTestCalibration:
    def test_welch_permutation_is_uniform_under_h0(self):
        rng = np.random.default_rng(0)
        n_samples, n_genes = 10, 300
        values = rng.standard_normal((n_samples, n_genes))
        groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        df = compare_two_groups_scalar(values, groups, n_perm=300, random_state=0)
        ks_stat, ks_p = kstest(df.P_value.to_numpy(), "uniform")
        assert ks_p > 0.01, f"DE-test p-values not uniform under H0, KS p={ks_p:.4f}"

    def test_implanted_mean_shift_recovered(self):
        rng = np.random.default_rng(7)
        n_per, n_genes = 6, 40
        a = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b = rng.normal(loc=0.0, scale=1.0, size=(n_per, n_genes))
        b[:, :5] += 2.0  # large mean shift on genes 0..4
        values = np.concatenate([a, b], axis=0)
        groups = np.array([0] * n_per + [1] * n_per)
        df = compare_two_groups_scalar(values, groups, n_perm=400, random_state=0)
        top5 = set(df.head(5).Feature.astype(str).tolist())
        assert top5 == {"0", "1", "2", "3", "4"}


class TestDeAndPatternOrthogonality:
    def test_pure_dc_shift_does_not_light_up_pattern_test(self):
        """A gene with only a mean-shift in one group (identical pattern otherwise)
        should be highly significant for DE but NOT for the pattern test."""
        rng = np.random.default_rng(0)
        n_per = 5
        ny = nx = 24
        n_genes = 6
        # Shared spatial "pattern" per gene (shared across all samples).
        pattern = rng.standard_normal((n_genes, ny, nx))

        samples = []
        for _ in range(2 * n_per):
            samples.append(pattern + 0.05 * rng.standard_normal((n_genes, ny, nx)))
        # Add a big mean shift to gene 0 only for group 1.
        for i in range(n_per, 2 * n_per):
            samples[i][0] += 10.0

        groups = np.array([0] * n_per + [1] * n_per)
        gene_names = [f"g{i}" for i in range(n_genes)]
        cmp = SpectralComparator(
            _samples_to_adata_list(samples, gene_names),
            groups,
            gene_names=gene_names,
            n_radial_bins=8,
            center="mean",
        ).fit()

        de = cmp.test_expression(n_perm=400, random_state=0)
        pattern_df = cmp.test_pattern(n_perm=400, random_state=0)

        de_g0 = de.set_index("Feature").loc["g0"]
        pat_g0 = pattern_df.set_index("Feature").loc["g0"]

        # g0 should be the top DE hit.
        assert de.Feature.iloc[0] == "g0"
        assert de_g0.P_value < 0.05
        # The pattern test should NOT find g0 uniquely significant — its pattern
        # is identical between groups by construction.
        assert pat_g0.P_value > 0.05

    def test_center_none_keeps_dc_nonzero_mean_makes_it_zero(self):
        """Direct structural check: with `center='mean'` the DC bin is exactly 0
        in the returned spectrum (so no DE signal can leak into the pattern
        test), whereas with `center=None` it carries the mean-squared power.
        This is a more direct test than rerunning the full pattern statistic,
        which already strips DC via ``exclude_dc=True``.
        """
        rng = np.random.default_rng(2)
        sample = rng.standard_normal((3, 12, 12)) + 4.0  # non-zero mean
        spec_mean = compute_sample_spectrum(sample, fft_solver="fft2", center="mean")
        spec_none = compute_sample_spectrum(sample, fft_solver="fft2", center=None)
        np.testing.assert_allclose(spec_mean[:, 0, 0], 0.0, atol=1e-18)
        assert np.all(spec_none[:, 0, 0] > 0.0)


class TestSpectralComparatorDcAccess:
    def test_fit_populates_dc(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((3, 8, 10)) + s for s in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["a", "b", "c"]
        cmp = SpectralComparator(
            _samples_to_adata_list(samples, gene_names), groups, gene_names
        ).fit()
        assert cmp.dc_ is not None
        assert cmp.dc_.shape == (4, 3)
        # DC equals per-sample grid mean of the raw signal.
        expected = np.array([samples[i].mean(axis=(1, 2)) for i in range(4)])
        np.testing.assert_allclose(cmp.dc_, expected, rtol=1e-12)

    def test_test_expression_requires_fit(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = SpectralComparator(adatas, np.array([0, 1]), gene_names)
        with pytest.raises(RuntimeError, match="fit"):
            cmp.test_expression()


# ---------------------------------------------------------------------------
# shape_normalize: magnitude-invariant spectrum shapes
# ---------------------------------------------------------------------------


class TestShapeNormalize:
    def test_unit_geometric_mean_along_axis(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.1, 10.0, size=(4, 7, 12))
        out = shape_normalize(x, axis=-1)
        # exp(mean(log x)) == 1 for every row
        log_mean = np.log(out).mean(axis=-1)
        np.testing.assert_allclose(log_mean, 0.0, atol=1e-9)

    def test_cancels_scalar_rescale(self):
        """Two rows that differ only by a positive scalar get the same shape."""
        rng = np.random.default_rng(1)
        row = rng.uniform(0.5, 3.0, size=10)
        scales = np.array([[0.3], [1.0], [50.0]])
        stack = scales * row[None, :]
        out = shape_normalize(stack, axis=-1)
        # All three rows should be identical after normalization. Tolerance is
        # loose because the +eps floor before log breaks exact scale equivariance
        # at small values; practically the agreement is ~10 significant figures.
        np.testing.assert_allclose(out[0], out[1], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(out[1], out[2], rtol=1e-9, atol=1e-10)

    def test_preserves_shape(self):
        x = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 8, 6))
        assert shape_normalize(x).shape == x.shape

    def test_spectral_comparator_shape_normalize_chainable(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((4, 12, 14)) for _ in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2", "g3"]
        cmp = (
            SpectralComparator(
                _samples_to_adata_list(samples, gene_names), groups, gene_names, n_radial_bins=8
            )
            .fit()
            .normalize_background()
        )
        before_dc = cmp.dc_.copy()
        ret = cmp.shape_normalize()
        # Chainable: returns self
        assert ret is cmp
        # spectra_ now has unit geometric mean along the last axis
        geo_means = np.exp(np.log(cmp.spectra_).mean(axis=-1))
        np.testing.assert_allclose(geo_means, 1.0, atol=1e-9)
        # dc_ is untouched
        np.testing.assert_array_equal(cmp.dc_, before_dc)

    def test_shape_normalize_requires_fit(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = SpectralComparator(adatas, np.array([0, 1]), gene_names)
        with pytest.raises(RuntimeError, match="fit"):
            cmp.shape_normalize()


# ---------------------------------------------------------------------------
# Physical-frequency binning (radial_bin_spectrum with spacing/edges)
# ---------------------------------------------------------------------------


class TestPhysicalFrequencyBinning:
    def test_physical_spacing_changes_bin_scale(self):
        """With the same spectrum, larger spacing -> lower Nyquist -> smaller max bin edge."""
        rng = np.random.default_rng(0)
        img = rng.standard_normal((40, 40))

        P = power_spectrum_2d(img, fft_solver="rfft2")
        rb1 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(1.0, 1.0)
        )
        rb2 = radial_bin_spectrum(
            P, grid_shape=(40, 40), n_bins=10, fft_solver="rfft2", spacing=(10.0, 10.0)
        )
        # Per-bin values identical when edges auto-span [0, Nyquist] — only the
        # physical labelling of the axis changes between the two calls.
        np.testing.assert_allclose(rb1, rb2, rtol=1e-10)

    def test_common_edges_across_heterogeneous_grids(self):
        """Explicit edges let different-shape samples map onto the same bin grid."""
        rng = np.random.default_rng(0)

        img_a = rng.standard_normal((40, 40))
        img_b = rng.standard_normal((50, 60))
        Pa = power_spectrum_2d(img_a, fft_solver="rfft2")
        Pb = power_spectrum_2d(img_b, fft_solver="rfft2")
        edges = np.linspace(0, 0.3, 11)
        rb_a = radial_bin_spectrum(
            Pa, grid_shape=(40, 40), fft_solver="rfft2", spacing=(1.0, 1.0), edges=edges
        )
        rb_b = radial_bin_spectrum(
            Pb, grid_shape=(50, 60), fft_solver="rfft2", spacing=(1.0, 1.0), edges=edges
        )
        # exclude_dc=True by default drops the first bin, so output length = 10 - 1 = 9.
        assert rb_a.shape == rb_b.shape == (9,)

    def test_explicit_edges_non_monotonic_raises(self):
        P = np.zeros((8, 5))
        with pytest.raises(ValueError, match="monotonically"):
            radial_bin_spectrum(
                P, grid_shape=(8, 8), fft_solver="rfft2", edges=np.array([0.0, 0.5, 0.2])
            )


class TestSpectralComparatorWithSpacings:
    def test_physical_spacings_produce_comparable_bins(self):
        """SpectralComparator with per-sample auto-grids handles heterogeneous shapes."""
        rng = np.random.default_rng(0)
        shapes = [(32, 40), (30, 42), (34, 38), (33, 41)]
        samples = [rng.standard_normal((3, ny, nx)) for (ny, nx) in shapes]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2"]
        cmp = SpectralComparator(
            _samples_to_adata_list(samples, gene_names),
            groups,
            gene_names=gene_names,
            n_radial_bins=8,
        ).fit()
        # 8 edges -> 7 bins after DC-drop.
        assert cmp.spectra_.shape == (4, 3, 7)
        assert cmp.freq_edges is not None
        assert cmp.freq_edges.shape == (9,)
