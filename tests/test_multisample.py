"""Tests for quadsv.multisample."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy.stats import kstest

from quadsv.comparators import ComparatorIrregular
from quadsv.comparators.multisample import (
    align_spectra_by_rotation,
    apply_rotations_to_spectra,
    benchmark_statistics,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
    compute_sample_spectrum,
    estimate_rotations_from_landmarks,
    normalize_by_background,
    radial_bin_spectrum,
    residualize_against_covariates,
    shape_normalize,
)
from quadsv.kernels.fft import power_spectrum_2d

# ---------------------------------------------------------------------------
# Test helpers for the AnnData-based ComparatorIrregular API (Phase D)
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
    @staticmethod
    def _stripes(ny: int, nx: int, period: float = 8.0) -> np.ndarray:
        y = np.arange(ny)[:, None]
        return np.broadcast_to(np.sin(2 * np.pi * y / period).astype(float), (ny, nx)).copy()

    @staticmethod
    def _stripes_rotated(ny: int, nx: int, angle: float, period: float = 8.0) -> np.ndarray:
        import scipy.ndimage

        base = TestRotationAlignment._stripes(ny, nx, period=period)
        return scipy.ndimage.rotate(base, angle=angle, reshape=False, order=1, mode="reflect")

    def test_single_landmark_recovers_known_rotation(self):
        """One striped landmark per sample → recovered angle ≈ true angle."""
        ny = nx = 48
        true_angle = 25.0
        ref = self._stripes(ny, nx)
        rot = self._stripes_rotated(ny, nx, true_angle)

        sp_ref = compute_sample_spectrum(ref[None, :, :], fft_solver="fft2")
        sp_rot = compute_sample_spectrum(rot[None, :, :], fft_solver="fft2")
        _, angles = align_spectra_by_rotation(
            [sp_ref, sp_rot],
            grid_shapes=[(ny, nx), (ny, nx)],
            target_spectra=[sp_ref, sp_rot],
            fft_solver="fft2",
            reference_index=0,
            n_theta=360,
        )
        recovered = angles[1] % 180.0
        true_mod = true_angle % 180.0
        diff = min(abs(recovered - true_mod), 180.0 - abs(recovered - true_mod))
        assert diff < 5.0, f"recovered={recovered}, true={true_mod}, diff={diff}"

    def test_per_landmark_beats_mean_template(self):
        """With landmarks on perpendicular anisotropy axes, a mean-template
        would be near-isotropic and alignment would break down. Per-landmark
        alignment still locks onto the shared rotation because each landmark
        cross-correlates against its own same-index counterpart.
        """
        ny = nx = 96
        true_angle = 30.0

        # Two analytic landmarks with wave vectors on perpendicular axes.
        lm_h = TestRotationSimulation._sine_at_angle(ny, nx, 12.0, 0.0, 0.0)  # along +y
        lm_v = TestRotationSimulation._sine_at_angle(ny, nx, 0.0, 12.0, 0.0)  # along +x
        ref_stack = np.stack([lm_h, lm_v], axis=0)
        rot_stack = np.stack(
            [
                TestRotationSimulation._sine_at_angle(ny, nx, 12.0, 0.0, true_angle),
                TestRotationSimulation._sine_at_angle(ny, nx, 0.0, 12.0, true_angle),
            ],
            axis=0,
        )
        sp_ref = compute_sample_spectrum(ref_stack, fft_solver="fft2")
        sp_rot = compute_sample_spectrum(rot_stack, fft_solver="fft2")

        angles = estimate_rotations_from_landmarks(
            [sp_ref, sp_rot],
            grid_shapes=[(ny, nx), (ny, nx)],
            fft_solver="fft2",
            n_theta=720,
        )
        diff = TestRotationSimulation._canon_err(angles[1], true_angle)
        assert diff < 3.0, f"per-landmark recovered={angles[1]}, expected {true_angle}"

    def test_shape_validation(self):
        sp = compute_sample_spectrum(
            np.random.default_rng(0).standard_normal((3, 16, 16)), fft_solver="fft2"
        )
        sp_bad = compute_sample_spectrum(
            np.random.default_rng(0).standard_normal((2, 16, 16)), fft_solver="fft2"
        )
        with pytest.raises(ValueError, match="must match across samples"):
            align_spectra_by_rotation(
                [sp, sp_bad], grid_shapes=[(16, 16), (16, 16)], fft_solver="fft2"
            )
        with pytest.raises(ValueError, match="reference_index"):
            align_spectra_by_rotation(
                [sp, sp],
                grid_shapes=[(16, 16), (16, 16)],
                reference_index=9,
                fft_solver="fft2",
            )

    def test_apply_rotations_to_different_target(self):
        """Estimate rotation from one landmark, apply to an independent panel."""
        import scipy.ndimage

        ny = nx = 48
        true_angle = 18.0
        ref_landmark = self._stripes(ny, nx)
        cur_landmark = self._stripes_rotated(ny, nx, true_angle)

        # Target panel: three arbitrary genes per sample (distinct from the landmark).
        rng = np.random.default_rng(0)
        ref_target = rng.standard_normal((3, ny, nx))
        cur_target = np.stack(
            [
                scipy.ndimage.rotate(
                    ref_target[j], true_angle, reshape=False, order=1, mode="reflect"
                )
                for j in range(3)
            ],
            axis=0,
        )
        sp_ref_lm = compute_sample_spectrum(ref_landmark[None, :, :], fft_solver="fft2")
        sp_cur_lm = compute_sample_spectrum(cur_landmark[None, :, :], fft_solver="fft2")
        sp_ref_tgt = compute_sample_spectrum(ref_target, fft_solver="fft2")
        sp_cur_tgt = compute_sample_spectrum(cur_target, fft_solver="fft2")

        angles = estimate_rotations_from_landmarks(
            [sp_ref_lm, sp_cur_lm],
            grid_shapes=[(ny, nx), (ny, nx)],
            fft_solver="fft2",
            n_theta=360,
        )
        rotated = apply_rotations_to_spectra(
            [sp_ref_tgt, sp_cur_tgt],
            grid_shapes=[(ny, nx), (ny, nx)],
            angles_deg=angles,
            fft_solver="fft2",
        )
        # After applying the recovered rotation to the target spectra, the
        # L2 distance between sample 1 and the reference should drop.
        # Tolerance is loose because scipy.ndimage.rotate is itself lossy
        # on discrete grids (interpolation ringing around high-amplitude
        # FFT peaks) — even a perfectly recovered angle leaves substantial
        # residual mismatch; we only require a clear improvement.
        unrot_dist = np.linalg.norm(sp_cur_tgt - sp_ref_tgt)
        rot_dist = np.linalg.norm(rotated[1] - sp_ref_tgt)
        assert rot_dist < 0.95 * unrot_dist, (
            f"rotation failed to reduce distance: unrot={unrot_dist:.2g} " f"rot={rot_dist:.2g}"
        )


class TestRotationSimulation:
    """Simulation-based validation with **analytic** rotations (we rotate the
    wave vector of a sinusoidal landmark directly, so no pixel-interpolation
    bias is introduced by the simulator).

    The estimator's accuracy is fundamentally limited by the FFT grid: a
    sinusoid whose rotated wave vector lands between integer bins has a
    spectrum peak that is off the nearest bin by up to ~one angular bin
    (~180/n_theta degrees at the lowest landmark radius). Larger grids and
    multiple landmarks at different radii push this down.
    """

    @staticmethod
    def _sine_at_angle(ny, nx, ky0, kx0, phi_deg):
        """Generate ``sin(2π (ky·y + kx·x) / N)`` with ``(ky, kx)`` rotated
        by ``phi_deg``. No interpolation."""
        phi = np.deg2rad(phi_deg)
        c, s = np.cos(phi), np.sin(phi)
        ky = ky0 * c - kx0 * s
        kx = ky0 * s + kx0 * c
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return np.sin(2 * np.pi * (ky * yy / ny + kx * xx / nx))

    @staticmethod
    def _canon_err(a, b):
        """Angular error modulo 180°."""
        d = np.abs(np.mod(a, 180.0) - np.mod(b, 180.0))
        return np.minimum(d, 180.0 - d)

    def test_multi_sample_recovery_with_multiple_landmarks(self):
        """Per-sample rotations drawn from U(-80, 80). Using 4 sinusoidal
        landmarks at different frequencies — the per-landmark alignment
        (sum of per-gene cross-correlations) averages out the per-landmark
        aliasing, so every sample's angle is recovered to within a few
        angular bins."""
        ny = nx = 96
        rng = np.random.default_rng(42)
        n_samples = 8
        # Distinct k0 — higher radius → finer angular resolution.
        k0s = [10.0, 16.0, 22.0, 30.0]

        ref_landmarks = np.stack([self._sine_at_angle(ny, nx, k0, 0.0, 0.0) for k0 in k0s], axis=0)
        true_angles = np.concatenate([[0.0], rng.uniform(-80.0, 80.0, size=n_samples - 1)])
        samples = [ref_landmarks]
        for ang in true_angles[1:]:
            samples.append(
                np.stack(
                    [self._sine_at_angle(ny, nx, k0, 0.0, ang) for k0 in k0s],
                    axis=0,
                )
            )
        spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
        recovered = estimate_rotations_from_landmarks(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            fft_solver="fft2",
            n_theta=720,
        )
        errs = self._canon_err(recovered, true_angles)
        assert errs[0] == 0, "reference angle must be exactly 0"
        # With 4 landmarks on a 96×96 grid the recovered angle is within ~3°
        # of truth in every sample (aliasing-limited).
        assert errs[1:].max() < 4.0, (
            f"per-sample errors (deg): {errs[1:].round(2).tolist()}; "
            f"true angles: {true_angles[1:].round(2).tolist()}; "
            f"recovered: {recovered[1:].round(2).tolist()}"
        )
        # Mean error should be well under 2° — most of the budget is max-bin.
        assert errs[1:].mean() < 2.0

    def test_more_landmarks_reduces_bias(self):
        """Adding more landmarks (at distinct frequencies) should reduce the
        mean rotation-recovery error. This is a key property of the
        landmarks-first API — the previous mean-template design averaged the
        landmarks first and then cross-correlated, which did not have this
        property."""
        ny = nx = 96
        rng = np.random.default_rng(0)
        true_angles = np.concatenate([[0.0], rng.uniform(-70.0, 70.0, size=6)])

        def errs_for_k0s(k0s):
            samples = [
                np.stack([self._sine_at_angle(ny, nx, k, 0.0, a) for k in k0s], axis=0)
                for a in true_angles
            ]
            spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
            rec = estimate_rotations_from_landmarks(
                spectra,
                grid_shapes=[(ny, nx)] * len(samples),
                fft_solver="fft2",
                n_theta=720,
            )
            return self._canon_err(rec, true_angles)[1:]

        one = errs_for_k0s([12.0]).mean()
        many = errs_for_k0s([10.0, 14.0, 20.0, 26.0, 34.0]).mean()
        assert many <= one + 0.1, (
            f"multi-landmark bias ({many:.2f}°) did not improve over "
            f"single-landmark ({one:.2f}°)"
        )

    def test_apply_rotations_matches_raw_reference(self):
        """End-to-end: after rotation-correction, every sample's spectrum
        lands much closer to the reference than the un-corrected version.
        """
        ny = nx = 96
        rng = np.random.default_rng(7)
        n_samples = 5
        k0s = [10.0, 16.0, 24.0]

        ref_landmarks = np.stack([self._sine_at_angle(ny, nx, k0, 0.0, 0.0) for k0 in k0s], axis=0)
        true_angles = np.concatenate([[0.0], rng.uniform(-60.0, 60.0, size=n_samples - 1)])
        samples = [ref_landmarks]
        for ang in true_angles[1:]:
            samples.append(
                np.stack(
                    [self._sine_at_angle(ny, nx, k0, 0.0, ang) for k0 in k0s],
                    axis=0,
                )
            )
        spectra = [compute_sample_spectrum(s, fft_solver="fft2") for s in samples]
        angles = estimate_rotations_from_landmarks(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            fft_solver="fft2",
            n_theta=720,
        )
        corrected = apply_rotations_to_spectra(
            spectra,
            grid_shapes=[(ny, nx)] * n_samples,
            angles_deg=angles,
            fft_solver="fft2",
        )
        for i in range(1, n_samples):
            d_before = np.linalg.norm(spectra[i] - spectra[0])
            d_after = np.linalg.norm(corrected[i] - spectra[0])
            # Tolerance is loose because (a) scipy.ndimage.rotate on a
            # discrete spectrum is lossy — every applied angle ringing-blurs
            # high-amplitude FFT peaks — and (b) delta-like spectra of
            # sinusoids are the worst-case for bilinear rotation, so a
            # "perfect" recovery still leaves ~20-40% residual L2. The
            # recovery accuracy of ``angles`` itself is tested elsewhere;
            # here we only require a visible improvement.
            assert d_after < 0.95 * d_before, (
                f"sample {i}: rotation-correction did not tighten distance "
                f"(before={d_before:.2g}, after={d_after:.2g})"
            )


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
        # Force the sampling path (n_perm_max=0) so the KS test can check
        # continuous uniformity. Exact enumeration with only C(8,4)=70
        # distinct relabellings produces a discrete distribution on 71
        # values, which KS rejects by construction even under perfect
        # calibration; its own calibration is covered by
        # ``test_exact_permutation_on_small_samples`` below.
        df = compare_two_groups(
            spectra,
            groups,
            statistic="log_l2",
            n_perm=300,
            random_state=0,
            n_perm_max=0,
        )
        ks_stat, ks_p = kstest(df["P_value"].to_numpy(), "uniform")
        assert ks_p > 0.01, f"p-values not uniform under H0: KS p={ks_p:.4f}"

    def test_exact_permutation_on_small_samples(self):
        """With ``n_perm_max`` above ``C(n, n_a)`` the test enumerates every
        distinct relabelling. The resulting p-values are **discrete** with
        values in ``{1/(M+1), ..., (M+1)/(M+1)}`` where ``M = C(n, n_a)``,
        but remain calibrated: repeated runs are deterministic (no RNG
        sampling), and under H0 the rank of the observed statistic is
        Uniform on ``{1, ..., M+1}``.
        """
        rng = np.random.default_rng(0)
        n_genes, K = 50, 10
        # 4 vs 4: only C(8, 4) = 70 distinct relabellings → exact path.
        spectra = rng.uniform(0.5, 5.0, size=(8, n_genes, K))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        df_a = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=5000, random_state=0)
        # Re-run with a different seed → same p-values (exact → RNG-free).
        df_b = compare_two_groups(
            spectra, groups, statistic="log_l2", n_perm=5000, random_state=999
        )
        np.testing.assert_allclose(
            df_a.set_index("Feature").loc[df_b["Feature"], "P_value"].to_numpy(),
            df_b["P_value"].to_numpy(),
        )
        # The p-value alphabet is 71 = C(8,4) + 1 values.
        unique_pvals = set(df_a["P_value"].round(8).tolist())
        assert len(unique_pvals) <= 71
        # Mean p-value under H0 should still be ≈ 0.5 (up to finite-gene noise).
        assert 0.4 < df_a["P_value"].mean() < 0.6


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
    @pytest.mark.parametrize("stat", ["log_l2", "cauchy_welch", "hotelling_lw", "mmd_rbf"])
    def test_each_statistic_runs(self, stat):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 8, 6))
        groups = np.array([0, 0, 0, 1, 1, 1])
        df = compare_two_groups(spectra, groups, statistic=stat, n_perm=50, random_state=0)
        assert df.shape[0] == 8
        assert {"Feature", "Statistic", "P_value", "P_adj"} <= set(df.columns)
        assert df["P_value"].between(0, 1).all()
        if stat == "cauchy_welch":
            assert "P_value_per_bin" in df.columns
            # Each entry is an (K,) array of per-bin p-values in [0, 1].
            per_bin = np.stack(df["P_value_per_bin"].to_numpy())
            assert per_bin.shape == (8, 6)
            assert ((per_bin >= 0) & (per_bin <= 1)).all()

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError, match="Unknown statistic"):
            compare_two_groups(
                np.zeros((4, 3, 5)),
                np.array([0, 0, 1, 1]),
                statistic="bogus",
            )

    def test_log_l2_freq_weights(self):
        """Non-uniform weights should shift gene ranking compared to uniform."""
        rng = np.random.default_rng(0)
        n_samples, n_genes, K = 6, 4, 8
        # Gene 0: low-frequency difference only.
        # Gene 1: high-frequency difference only.
        base = rng.uniform(0.5, 1.5, size=(n_samples, n_genes, K))
        base[3:, 0, :2] *= 3.0  # low-freq bump in group B for gene 0
        base[3:, 1, -2:] *= 3.0  # high-freq bump in group B for gene 1
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Uniform weights: both genes score similarly.
        df_equal = compare_two_groups(base, groups, statistic="log_l2", n_perm=200, random_state=0)
        # Low-pass weights: gene 0 should come out on top.
        low_pass = np.concatenate([np.ones(2), np.zeros(K - 2)])
        df_low = compare_two_groups(
            base,
            groups,
            statistic="log_l2",
            n_perm=200,
            random_state=0,
            freq_weights=low_pass,
        )
        assert df_low["Feature"].iloc[0] == "0"
        # Sanity: equal-weights result is different from the low-pass ranking.
        assert df_equal["Feature"].iloc[0] != df_low["Feature"].iloc[-1]

    def test_log_l2_freq_weights_validation(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(4, 3, 6))
        groups = np.array([0, 0, 1, 1])
        # Wrong length:
        with pytest.raises(ValueError, match="length K="):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.ones(5),
            )
        # Negative weight:
        with pytest.raises(ValueError, match="non-negative"):
            compare_two_groups(
                spectra,
                groups,
                statistic="log_l2",
                n_perm=10,
                freq_weights=np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]),
            )


class TestBenchmark:
    def test_benchmark_returns_one_df_per_statistic(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(8, 12, 8))
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = benchmark_statistics(spectra, groups, n_perm=50, random_state=0)
        assert set(out.keys()) == {"log_l2", "hotelling_lw", "mmd_rbf", "cauchy_welch"}
        for _stat, df in out.items():
            assert df.shape[0] == 12
            assert df["P_value"].between(0, 1).all()
        # cauchy_welch specifically carries per-bin p-values.
        assert "P_value_per_bin" in out["cauchy_welch"].columns


# ---------------------------------------------------------------------------
# End-to-end: ComparatorIrregular
# ---------------------------------------------------------------------------


class TestComparatorIrregularEndToEnd:
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
            ComparatorIrregular(_samples_to_adata_list(samples, gene_names), groups, gene_names)
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
        cmp = ComparatorIrregular(_samples_to_adata_list(samples, gene_names), groups, gene_names)
        cmp.fit().residualize(covariates)
        df = cmp.test(statistic="log_l2", n_perm=50, random_state=0)
        assert df.shape[0] == 4

    def test_invalid_groups_raises(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4))] * 3, gene_names)
        with pytest.raises(ValueError, match="exactly two distinct"):
            ComparatorIrregular(adatas, np.array([0, 1, 2]), gene_names)

    def test_must_fit_before_test(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, np.array([0, 1]), gene_names)
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
        cmp = ComparatorIrregular(
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


class TestComparatorIrregularDcAccess:
    def test_fit_populates_dc(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((3, 8, 10)) + s for s in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["a", "b", "c"]
        cmp = ComparatorIrregular(
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
        cmp = ComparatorIrregular(adatas, np.array([0, 1]), gene_names)
        with pytest.raises(RuntimeError, match="fit"):
            cmp.test_expression()


# ---------------------------------------------------------------------------
# shape_normalize: magnitude-invariant spectrum shapes
# ---------------------------------------------------------------------------


class TestShapeNormalize:
    def test_sum_to_one_along_axis(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.1, 10.0, size=(4, 7, 12))
        out = shape_normalize(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-12)

    def test_cancels_scalar_rescale(self):
        """Two rows that differ only by a positive scalar get the same shape."""
        rng = np.random.default_rng(1)
        row = rng.uniform(0.5, 3.0, size=10)
        scales = np.array([[0.3], [1.0], [50.0]])
        stack = scales * row[None, :]
        out = shape_normalize(stack, axis=-1)
        # All three rows become identical probability vectors after L1
        # normalization (the shared shape of the row).
        np.testing.assert_allclose(out[0], out[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(out[1], out[2], rtol=1e-10, atol=1e-12)

    def test_preserves_shape(self):
        x = np.random.default_rng(0).uniform(0.1, 5.0, size=(3, 8, 6))
        assert shape_normalize(x).shape == x.shape

    def test_spectral_comparator_shape_normalize_chainable(self):
        rng = np.random.default_rng(0)
        samples = [rng.standard_normal((4, 12, 14)) for _ in range(4)]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2", "g3"]
        cmp = (
            ComparatorIrregular(
                _samples_to_adata_list(samples, gene_names), groups, gene_names, n_radial_bins=8
            )
            .fit()
            .normalize_background()
        )
        before_dc = cmp.dc_.copy()
        ret = cmp.shape_normalize()
        # Chainable: returns self
        assert ret is cmp
        # spectra_ now sums to 1 along the last axis (probability-vector shape)
        np.testing.assert_allclose(cmp.spectra_.sum(axis=-1), 1.0, rtol=1e-10)
        # dc_ is untouched
        np.testing.assert_array_equal(cmp.dc_, before_dc)

    def test_shape_normalize_requires_fit(self):
        gene_names = ["a", "b"]
        adatas = _samples_to_adata_list([np.zeros((2, 4, 4)), np.zeros((2, 4, 4))], gene_names)
        cmp = ComparatorIrregular(adatas, np.array([0, 1]), gene_names)
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


class TestComparatorIrregularWithSpacings:
    def test_physical_spacings_produce_comparable_bins(self):
        """ComparatorIrregular with per-sample auto-grids handles heterogeneous shapes."""
        rng = np.random.default_rng(0)
        shapes = [(32, 40), (30, 42), (34, 38), (33, 41)]
        samples = [rng.standard_normal((3, ny, nx)) for (ny, nx) in shapes]
        groups = np.array([0, 0, 1, 1])
        gene_names = ["g0", "g1", "g2"]
        cmp = ComparatorIrregular(
            _samples_to_adata_list(samples, gene_names),
            groups,
            gene_names=gene_names,
            n_radial_bins=8,
        ).fit()
        # 8 edges -> 7 bins after DC-drop.
        assert cmp.spectra_.shape == (4, 3, 7)
        assert cmp.freq_edges is not None
        assert cmp.freq_edges.shape == (9,)


class TestIncompleteData:
    def test_masked_matches_unmasked_when_full(self):
        """With all-True presence mask, masked == unmasked (same rng)."""
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 4, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        presence = np.ones((6, 4), dtype=bool)
        df_full = compare_two_groups(spectra, groups, statistic="log_l2", n_perm=50, random_state=0)
        df_mask = compare_two_groups_masked(
            spectra, groups, presence, statistic="log_l2", n_perm=50, random_state=0
        )
        # Features line up after sort; the statistic column agrees exactly.
        np.testing.assert_allclose(
            df_full.set_index("Feature").loc[df_mask["Feature"], "Statistic"].to_numpy(),
            df_mask["Statistic"].to_numpy(),
            rtol=1e-10,
        )
        # n_obs columns should be fully populated at n_samples each.
        assert (df_mask["n_obs_A"] == 3).all()
        assert (df_mask["n_obs_B"] == 3).all()

    def test_gene_skipped_when_below_min_samples_per_group(self):
        rng = np.random.default_rng(0)
        spectra = rng.uniform(0.1, 5.0, size=(6, 3, 5))
        groups = np.array([0, 0, 0, 1, 1, 1])
        # Gene 0 only observed in one sample of group B → must be skipped.
        presence = np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [False, True, True],
                [False, True, True],
                [True, True, True],
            ]
        )
        df = compare_two_groups_masked(
            spectra,
            groups,
            presence,
            statistic="log_l2",
            n_perm=20,
            random_state=0,
            min_samples_per_group=2,
        )
        skipped_row = df[df.Feature == "0"].iloc[0]
        assert np.isnan(skipped_row["P_value"])
        assert np.isnan(skipped_row["P_adj"])
        assert skipped_row["n_obs_A"] == 3
        assert skipped_row["n_obs_B"] == 1

    def test_presence_threshold_propagates_to_comparator(self):
        import anndata as ad

        rng = np.random.default_rng(0)
        ny = nx = 12
        gene_names = ["g0", "g1", "g2"]
        # Build four samples, each on the same regular grid.
        samples = []
        for _ in range(4):
            coords = (
                np.stack(
                    np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"),
                    axis=-1,
                )
                .reshape(-1, 2)
                .astype(float)
            )
            X = rng.uniform(0.1, 1.0, size=(ny * nx, 3))
            a = ad.AnnData(X=X)
            a.var_names = gene_names
            a.obsm["spatial"] = coords
            samples.append(a)
        # Zero gene 0 in samples 0 and 1 → presence_ will drop it there.
        for i in (0, 1):
            samples[i].X[:, 0] = 0.0
        groups = np.array([0, 0, 1, 1])
        cmp = ComparatorIrregular(
            samples,
            groups,
            gene_names,
            presence_threshold=0.5,
        ).fit()
        assert cmp.presence_.shape == (4, 3)
        # Gene 0 should be absent in the two samples we zeroed, present elsewhere.
        assert not cmp.presence_[0, 0]
        assert not cmp.presence_[1, 0]
        assert cmp.presence_[2, 0]
        # Running test_pattern should dispatch to the masked path and return
        # n_obs_A / n_obs_B columns.
        df = cmp.test_pattern(statistic="log_l2", n_perm=20, random_state=0)
        assert {"n_obs_A", "n_obs_B"}.issubset(df.columns)
