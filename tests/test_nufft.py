"""Tests for quadsv.nufft and SpectralComparator's NUFFT path."""

from __future__ import annotations

import numpy as np
import pytest

finufft = pytest.importorskip("finufft", reason="NUFFT tests require finufft")

from quadsv import SpectralComparator
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
# SpectralComparator.from_coords
# ---------------------------------------------------------------------------


class TestFromCoordsPipeline:
    def _mk_synthetic(self, n_samples=4, n_genes=3, seed=0):
        rng = np.random.default_rng(seed)
        coords = [rng.uniform(0, 100, size=(150 + 20 * i, 2)) for i in range(n_samples)]
        values = [rng.standard_normal((c.shape[0], n_genes)) for c in coords]
        groups = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        gene_names = [f"g{i}" for i in range(n_genes)]
        return coords, values, groups, gene_names

    def test_end_to_end_runs(self):
        coords, values, groups, gene_names = self._mk_synthetic()
        cmp = (
            SpectralComparator.from_coords(
                coords=coords,
                values=values,
                groups=groups,
                gene_names=gene_names,
                grid_shape=(24, 24),
                spacing=(5.0, 5.0),
                n_radial_bins=8,
            )
            .fit()
            .normalize_background()
        )
        assert cmp.mode == "nufft"
        assert cmp.spectra_.shape == (4, 3, 7)  # K - 1 after exclude_dc
        assert cmp.dc_.shape == (4, 3)
        df = cmp.test_pattern(n_perm=50, random_state=0)
        assert set(df.columns) == {"Feature", "Statistic", "P_value", "P_adj"}

    def test_unit_scales_per_sample(self):
        """Mixed-unit samples produce identical spectra when unit_scales convert them."""
        coords_um, values, groups, gene_names = self._mk_synthetic()
        coords_mm = [c / 1000.0 for c in coords_um]

        cmp_um = SpectralComparator.from_coords(
            coords=coords_um,
            values=values,
            groups=groups,
            gene_names=gene_names,
            grid_shape=(24, 24),
            spacing=(5.0, 5.0),
        ).fit()
        cmp_mm = SpectralComparator.from_coords(
            coords=coords_mm,
            values=values,
            groups=groups,
            gene_names=gene_names,
            grid_shape=(24, 24),
            spacing=(5.0, 5.0),
            unit_scales=[1000.0] * len(coords_mm),
        ).fit()
        np.testing.assert_allclose(cmp_um.spectra_, cmp_mm.spectra_, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cmp_um.dc_, cmp_mm.dc_, rtol=1e-12, atol=1e-12)

    def test_shape_validation(self):
        coords = [np.zeros((5, 2)), np.zeros((5, 2))]
        values = [np.zeros((5, 3)), np.zeros((5, 2))]  # gene-count mismatch
        with pytest.raises(ValueError, match="values"):
            SpectralComparator.from_coords(
                coords=coords,
                values=values,
                groups=np.array([0, 1]),
                gene_names=["a", "b", "c"],
                grid_shape=(8, 8),
                spacing=(1.0, 1.0),
            )

    def test_unit_scales_length_check(self):
        coords, values, groups, gene_names = self._mk_synthetic()
        with pytest.raises(ValueError, match="unit_scales"):
            SpectralComparator.from_coords(
                coords=coords,
                values=values,
                groups=groups,
                gene_names=gene_names,
                grid_shape=(16, 16),
                spacing=(5.0, 5.0),
                unit_scales=[1.0, 2.0],  # wrong length
            )


# ---------------------------------------------------------------------------
# Visium correctness: NUFFT on raw coords ≈ rasterized FFT (radial spectra)
# ---------------------------------------------------------------------------


VISIUM_H5AD = "/Users/jysumac/Projects/EquivSVT/data/idh_vs_gbm/adata/GBM_MGH258.h5ad"


@pytest.mark.skipif(
    not __import__("pathlib").Path(VISIUM_H5AD).exists(),
    reason="Visium cached .h5ad not present (run scripts/download_glioma_data.py).",
)
class TestVisiumAgreement:
    """Confirm correctness on a real Visium slide via the mathematical
    equivalence: ``FFT(zero-filled hex raster) == NUFFT(raw point coords)`` up
    to numerical precision, when both use the same grid shape and spacing and
    no coordinate centering.

    The NN-filled rasterization used in the notebooks is *not* equivalent —
    NN fill is a spatial low-pass filter that distorts the spectrum — so we
    use ``fill='zero'`` for this mathematical sanity check.
    """

    def test_fft_zero_fill_equals_nufft_on_raw_coords(self):
        import anndata as ad

        from quadsv import visium_hex_spacing_um, visium_to_grid

        adata = ad.read_h5ad(VISIUM_H5AD)
        adata.var_names_make_unique()

        # Pick 20 highly-expressed genes.
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene_scores = (X > 0).mean(axis=0) * X.mean(axis=0)
        top = np.argsort(-gene_scores)[:20]
        gene_names = [adata.var_names[i] for i in top]

        # Path A: rasterize with zero-fill (mathematical equivalent of a sparse
        # signal at the real spot locations), then FFT.
        grid, spacing = visium_to_grid(adata, genes=gene_names, grid="dense", fill="zero")
        ny, nx = grid.shape[1:]
        P_fft = np.stack(
            [power_spectrum_2d(grid[g], fft_solver="fft2") for g in range(len(gene_names))]
        )  # (n_genes, ny, nx)

        # Path B: NUFFT on raw (array_row * dy, array_col * dx) coordinates.
        rows = adata.obs["array_row"].to_numpy().astype(float)
        cols = adata.obs["array_col"].to_numpy().astype(float)
        dy, dx = visium_hex_spacing_um(grid="dense")
        assert (dy, dx) == spacing, "Visium spacing helper drifted from visium_to_grid."
        coords = np.stack([rows * dy, cols * dx], axis=1)
        vals = X[:, top].astype(np.float64)

        P_nufft_all = power_spectrum_2d_nufft(
            coords,
            vals,
            grid_shape=(ny, nx),
            spacing=spacing,
            eps=1e-10,
            center_coords=False,
        )  # (ny, nx, n_genes)
        P_nufft = np.moveaxis(P_nufft_all, -1, 0)  # (n_genes, ny, nx)

        # The two should agree to ~NUFFT tolerance on every gene.
        rel = np.linalg.norm(P_fft - P_nufft) / np.linalg.norm(P_fft)
        assert rel < 1e-6, f"FFT(zero-fill) vs NUFFT relative diff on real Visium = {rel:.3e}"

    def test_spectral_comparator_from_coords_on_visium(self):
        """from_coords() runs end-to-end on real Visium data with
        heterogeneous sample shapes and produces a sensible result table."""
        import anndata as ad

        from quadsv import visium_hex_spacing_um

        # 3 IDHm + 3 GBM slides, 30 shared highly-expressed genes.
        IDS = [
            "IDHm_BWH23O",
            "IDHm_BWH24A",
            "IDHm_BWH25A",
            "GBM_MGH258",
            "GBM_ZH1007inf",
            "GBM_ZH1007nec",
        ]
        dy, dx = visium_hex_spacing_um(grid="dense")

        first = ad.read_h5ad(VISIUM_H5AD)
        first.var_names_make_unique()
        scores = (first.X.toarray() > 0).mean(axis=0) * np.asarray(first.X.toarray()).mean(axis=0)
        gene_names = [first.var_names[i] for i in np.argsort(-scores)[:30]]

        coords_list: list[np.ndarray] = []
        values_list: list[np.ndarray] = []
        for sid in IDS:
            path = VISIUM_H5AD.rsplit("/", 1)[0] + f"/{sid}.h5ad"
            a = ad.read_h5ad(path)
            a.var_names_make_unique()
            missing = [g for g in gene_names if g not in a.var_names]
            if missing:
                pytest.skip(f"{sid} missing genes {missing[:3]}... ; skip cross-check.")
            idx = [a.var_names.get_loc(g) for g in gene_names]
            X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            rows = a.obs["array_row"].to_numpy().astype(float)
            cols = a.obs["array_col"].to_numpy().astype(float)
            coords_list.append(np.stack([rows * dy, cols * dx], axis=1))
            values_list.append(X[:, idx].astype(np.float64))

        groups = np.array([0, 0, 0, 1, 1, 1])
        cmp = (
            SpectralComparator.from_coords(
                coords=coords_list,
                values=values_list,
                groups=groups,
                gene_names=gene_names,
                grid_shape=(78, 128),
                spacing=(dy, dx),
                n_radial_bins=12,
            )
            .fit()
            .normalize_background()
        )
        assert cmp.mode == "nufft"
        assert cmp.spectra_.shape == (6, 30, 11)
        df = cmp.test_pattern(n_perm=100, random_state=0)
        assert df.shape[0] == 30
        assert df["P_value"].between(0, 1).all()
