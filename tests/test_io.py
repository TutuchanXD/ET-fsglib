import json

import numpy as np
from astropy.table import Table

from fsglib.common.io import load_dataset_batch, load_npz_frame


def test_load_dataset_batch(tmp_path):
    batch_root = tmp_path / "batch0"
    frames_dir = batch_root / "frames"
    frames_dir.mkdir(parents=True)

    np.savez(
        frames_dir / "scope0_coadd_000000_000000.npz",
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([0.0]),
    )

    run_meta = {
        "field_center_ra_deg": 10.0,
        "field_center_dec_deg": 20.0,
        "pixel_scale_arcsec_per_pix": 4.83,
        "detector_width_pix": 1119,
    }
    (batch_root / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    table = Table(
        rows=[(1.0, 2.0, 10.1, 20.2, 12.3, 0, 7)],
        names=["x0", "y0", "RA", "Dec", "Kepler Mag", "Field ID", "Star ID"],
    )
    table.write(batch_root / "stars.ecsv", format="ascii.ecsv")

    ctx = load_dataset_batch(batch_root, cfg={"dataset": {"truth_origin": "centered_pixels", "truth_y_axis_up": True}})

    assert ctx.batch_center_ra_deg == 10.0
    assert ctx.pixel_scale_arcsec_per_pix == 4.83
    assert len(ctx.frame_paths) == 1
    assert len(ctx.truth_stars) == 1
    assert np.isclose(ctx.truth_stars[0].x_pix, 560.0)
    assert np.isclose(ctx.truth_stars[0].y_pix, 557.0)


def test_load_dataset_batch_estimates_field_offset_from_truth(tmp_path):
    batch_root = tmp_path / "batch1"
    frames_dir = batch_root / "frames"
    frames_dir.mkdir(parents=True)

    np.savez(
        frames_dir / "scope0_coadd_000000_000000.npz",
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([0.0]),
    )

    ra0 = 10.0
    dec0 = 20.0
    scale = 4.83
    dx_pix = 5.0
    dy_pix = -7.0
    offset_x = 0.25
    offset_y = -0.40
    dec0_rad = np.radians(dec0)

    run_meta = {
        "field_center_ra_deg": ra0,
        "field_center_dec_deg": dec0,
        "pixel_scale_arcsec_per_pix": scale,
        "detector_width_pix": 1119,
        "detector_height_pix": 1119,
    }
    (batch_root / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    ra_star = ra0 + (dx_pix * scale / 3600.0) / np.cos(dec0_rad)
    dec_star = dec0 + (dy_pix * scale / 3600.0)
    table = Table(
        rows=[(dx_pix + offset_x, dy_pix + offset_y, ra_star, dec_star, 12.3, 0, 7)],
        names=["x0", "y0", "RA", "Dec", "Kepler Mag", "Field ID", "Star ID"],
    )
    table.write(batch_root / "stars.ecsv", format="ascii.ecsv")

    ctx = load_dataset_batch(batch_root, cfg={"dataset": {"truth_origin": "centered_pixels", "truth_y_axis_up": False}})

    assert ctx.field_offset_source == "estimated_truth"
    assert np.isclose(ctx.field_offset_x_pix, offset_x, atol=1e-6)
    assert np.isclose(ctx.field_offset_y_pix, offset_y, atol=1e-6)


def test_load_dataset_batch_ignores_non_unique_truth_star_ids(tmp_path):
    batch_root = tmp_path / "batch2"
    frames_dir = batch_root / "frames"
    frames_dir.mkdir(parents=True)

    np.savez(
        frames_dir / "scope0_coadd_000000_000000.npz",
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([0.0]),
    )

    run_meta = {
        "field_center_ra_deg": 10.0,
        "field_center_dec_deg": 20.0,
        "pixel_scale_arcsec_per_pix": 4.83,
        "detector_width_pix": 1119,
    }
    (batch_root / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    table = Table(
        rows=[
            (1.0, 2.0, 10.1, 20.2, 12.3, 0, 0),
            (2.0, 3.0, 10.2, 20.3, 12.4, 0, 0),
        ],
        names=["x0", "y0", "RA", "Dec", "Kepler Mag", "Field ID", "Star ID"],
    )
    table.write(batch_root / "stars.ecsv", format="ascii.ecsv")

    ctx = load_dataset_batch(batch_root, cfg={"dataset": {"truth_origin": "centered_pixels", "truth_y_axis_up": True}})

    assert len(ctx.truth_stars) == 2
    assert ctx.truth_stars[0].source_id is None
    assert ctx.truth_stars[1].source_id is None
    assert ctx.truth_stars[0].meta["raw_star_id"] == 0
    assert ctx.run_meta["truth_star_id_status"] == "non_unique_or_invalid"


def test_load_dataset_batch_prefers_truth_index_over_non_unique_star_id(tmp_path):
    batch_root = tmp_path / "batch3"
    frames_dir = batch_root / "frames"
    frames_dir.mkdir(parents=True)

    np.savez(
        frames_dir / "scope0_coadd_000000_000000.npz",
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([0.0]),
    )

    run_meta = {
        "field_center_ra_deg": 10.0,
        "field_center_dec_deg": 20.0,
        "pixel_scale_arcsec_per_pix": 4.83,
        "detector_width_pix": 1119,
    }
    (batch_root / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")

    table = Table(
        rows=[
            (0, 1.0, 2.0, 10.1, 20.2, 12.3, 0, 0),
            (1, 2.0, 3.0, 10.2, 20.3, 12.4, 0, 0),
        ],
        names=["Truth Index", "x0", "y0", "RA", "Dec", "Kepler Mag", "Field ID", "Star ID"],
    )
    table.write(batch_root / "stars.ecsv", format="ascii.ecsv")

    ctx = load_dataset_batch(batch_root, cfg={"dataset": {"truth_origin": "centered_pixels", "truth_y_axis_up": False}})

    assert [star.source_id for star in ctx.truth_stars] == [0, 1]
    assert ctx.truth_stars[0].meta["raw_star_id"] == 0
    assert ctx.run_meta["truth_star_id_column"] == "Truth Index"


def test_load_npz_frame_reads_frame_truth_and_falls_back_when_absent(tmp_path):
    npz_with_truth = tmp_path / "scope0_coadd_000000_000000.npz"
    np.savez(
        npz_with_truth,
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([0.0]),
        variant_ids=np.array([7]),
        truth_star_index=np.array([11, 12], dtype=np.int64),
        truth_ra_deg=np.array([10.1, 10.2], dtype=np.float64),
        truth_dec_deg=np.array([20.1, 20.2], dtype=np.float64),
        truth_mag=np.array([12.3, 12.4], dtype=np.float64),
        truth_static_x_centered_pix=np.array([1.0, 2.0], dtype=np.float64),
        truth_static_y_centered_pix=np.array([3.0, 4.0], dtype=np.float64),
        truth_static_x_image_pix=np.array([5.0, 6.0], dtype=np.float64),
        truth_static_y_image_pix=np.array([7.0, 8.0], dtype=np.float64),
        truth_x_centered_pix=np.array([[[1.5, 2.5]]], dtype=np.float64),
        truth_y_centered_pix=np.array([[[3.5, 4.5]]], dtype=np.float64),
        truth_x_image_pix=np.array([[[5.5, 6.5]]], dtype=np.float64),
        truth_y_image_pix=np.array([[[7.5, 8.5]]], dtype=np.float64),
        truth_valid_mask=np.array([[[True, False]]], dtype=bool),
        truth_dx_pointing_pix=np.array([[[0.1, 0.1]]], dtype=np.float64),
        truth_dy_pointing_pix=np.array([[[0.2, 0.2]]], dtype=np.float64),
        truth_dx_dva_pix=np.array([[[0.3, 0.3]]], dtype=np.float64),
        truth_dy_dva_pix=np.array([[[0.4, 0.4]]], dtype=np.float64),
        truth_dx_thermal_pix=np.array([[[0.5, 0.5]]], dtype=np.float64),
        truth_dy_thermal_pix=np.array([[[0.6, 0.6]]], dtype=np.float64),
        truth_dx_jitter_mean_pix=np.array([[[0.7, 0.7]]], dtype=np.float64),
        truth_dy_jitter_mean_pix=np.array([[[0.8, 0.8]]], dtype=np.float64),
    )

    raw = load_npz_frame(npz_with_truth)
    assert raw.variant_id == 7
    assert raw.meta["truth_source"] == "npz_frame_truth"
    assert len(raw.meta["truth_stars"]) == 1
    truth_star = raw.meta["truth_stars"][0]
    assert truth_star.source_id == 11
    assert np.isclose(truth_star.x_pix, 5.5)
    assert np.isclose(truth_star.y_pix, 7.5)
    assert np.isclose(truth_star.meta["truth_dx_jitter_mean_pix"], 0.7)

    npz_without_truth = tmp_path / "scope0_coadd_000001_000001.npz"
    np.savez(
        npz_without_truth,
        images=np.zeros((1, 1, 5, 5), dtype=np.float32),
        time_s=np.array([1.0]),
    )
    raw_fallback = load_npz_frame(npz_without_truth)
    assert "truth_stars" not in raw_fallback.meta
