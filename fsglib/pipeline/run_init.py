import numpy as np
from time import perf_counter

from fsglib.attitude.solver import dcm_to_quat, quat_to_dcm, solve_attitude
from fsglib.common.coords import radec_to_unit_vector
from fsglib.common.io import load_dataset_batch_for_frame, load_npz_frame
from fsglib.common.types import (
    AttitudeSolveInput,
    DatasetContext,
    FrameResult,
    MatchingContext,
)
from fsglib.ephemeris.pipeline import build_reference_stars
from fsglib.ephemeris.types import EphemerisContext
from fsglib.extract.pipeline import extract_stars
from fsglib.match.pipeline import match_stars
from fsglib.pipeline.convert import candidates_to_observed
from fsglib.pipeline.evaluate import evaluate_frame_result
from fsglib.preprocess.pipeline import preprocess_frame


def _coarse_attitude_from_boresight(boresight_inertial: np.ndarray) -> np.ndarray:
    boresight = np.asarray(boresight_inertial, dtype=np.float64)
    boresight /= np.linalg.norm(boresight)

    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(boresight, up_ref)) > 0.99:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_ref, boresight)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(boresight, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    c_bi = np.column_stack([x_axis, y_axis, boresight])
    c_ib = c_bi.T
    return dcm_to_quat(c_ib)


def _coarse_attitude_from_radec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra_rad = np.radians(float(ra_deg))
    dec_rad = np.radians(float(dec_deg))
    east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0], dtype=np.float64)
    north = np.array(
        [
            -np.sin(dec_rad) * np.cos(ra_rad),
            -np.sin(dec_rad) * np.sin(ra_rad),
            np.cos(dec_rad),
        ],
        dtype=np.float64,
    )
    center = np.array(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ],
        dtype=np.float64,
    )
    c_ib = np.vstack([east, north, center])
    return dcm_to_quat(c_ib)


def _boresight_from_attitude(prior_attitude_q: np.ndarray) -> np.ndarray:
    c_ib = quat_to_dcm(prior_attitude_q)
    boresight = c_ib.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return boresight / np.linalg.norm(boresight)


def _build_ephemeris_context(
    raw,
    cfg: dict,
    dataset_ctx: DatasetContext | None,
    mode: str = "init",
    prior_attitude_q: np.ndarray | None = None,
    track_catalog_ids: list[int] | None = None,
) -> EphemerisContext:
    boresight_inertial = None
    if dataset_ctx is not None:
        if dataset_ctx.batch_center_ra_deg is not None and dataset_ctx.batch_center_dec_deg is not None:
            boresight_inertial = radec_to_unit_vector(
                dataset_ctx.batch_center_ra_deg,
                dataset_ctx.batch_center_dec_deg,
            )

    if mode == "tracking" and prior_attitude_q is not None:
        boresight_inertial = _boresight_from_attitude(prior_attitude_q)

    if prior_attitude_q is None and dataset_ctx is not None:
        if dataset_ctx.batch_center_ra_deg is not None and dataset_ctx.batch_center_dec_deg is not None:
            prior_attitude_q = _coarse_attitude_from_radec(
                dataset_ctx.batch_center_ra_deg,
                dataset_ctx.batch_center_dec_deg,
            )
    if prior_attitude_q is None and boresight_inertial is not None:
        prior_attitude_q = _coarse_attitude_from_boresight(boresight_inertial)

    return EphemerisContext(
        mode=mode,
        time_s=raw.time_s,
        prior_attitude_q=prior_attitude_q,
        boresight_inertial=boresight_inertial,
        angular_rate_body=None,
        detector_model=cfg.get("detector", {}),
        optical_model=cfg.get("layout", {}),
        catalog_cfg=cfg["ephemeris"],
        correction_cfg=cfg.get("corrections", {}),
        track_catalog_ids=[] if track_catalog_ids is None else track_catalog_ids,
    )


def run_single_frame_init(
    npz_path: str,
    cfg: dict,
    models: dict,
    dataset_ctx: DatasetContext | None = None,
) -> FrameResult:
    if dataset_ctx is None:
        dataset_ctx = load_dataset_batch_for_frame(npz_path, cfg=cfg)

    if "projector" in models and hasattr(models["projector"], "set_field_center"):
        models["projector"].set_field_center(
            dataset_ctx.batch_center_ra_deg if dataset_ctx is not None else None,
            dataset_ctx.batch_center_dec_deg if dataset_ctx is not None else None,
            dataset_ctx.field_offset_x_pix if dataset_ctx is not None else None,
            dataset_ctx.field_offset_y_pix if dataset_ctx is not None else None,
        )

    timings: dict[str, float] = {}
    total_start = perf_counter()

    raw = load_npz_frame(npz_path, detector_id=int(cfg["layout"].get("default_detector_id", 0)))

    t0 = perf_counter()
    pre = preprocess_frame(raw, calib=models.get("calib", {}), cfg=cfg)
    timings["preprocess"] = perf_counter() - t0

    t0 = perf_counter()
    cand = extract_stars(pre, cfg=cfg)
    timings["extract"] = perf_counter() - t0

    t0 = perf_counter()
    obs = candidates_to_observed(cand, models["projector"], cfg)
    timings["convert"] = perf_counter() - t0

    t0 = perf_counter()
    eph_ctx = _build_ephemeris_context(raw, cfg, dataset_ctx=dataset_ctx, mode="init")
    ref = build_reference_stars(eph_ctx, models["catalog"], models["projector"], cfg)
    timings["ephemeris"] = perf_counter() - t0

    t0 = perf_counter()
    match_ctx = MatchingContext(
        mode="init",
        time_s=raw.time_s,
        observed_stars=obs,
        prior_attitude_q=eph_ctx.prior_attitude_q,
        detector_layout=cfg.get("layout", {}),
        optical_model=cfg.get("layout", {}),
        matching_cfg=cfg.get("match", {}),
        boresight_inertial=eph_ctx.boresight_inertial,
        reference_stars=ref,
    )
    matching = match_stars(match_ctx, ref, cfg)
    timings["match"] = perf_counter() - t0

    t0 = perf_counter()
    solve_input = AttitudeSolveInput(
        time_s=raw.time_s,
        matched_stars=matching.matched,
        prior_q_ib=eph_ctx.prior_attitude_q,
        mode="init",
        solver_cfg=cfg["attitude"],
    )
    solution = solve_attitude(solve_input, cfg)
    timings["attitude"] = perf_counter() - t0

    t0 = perf_counter()
    evaluation = evaluate_frame_result(raw, pre, cand, matching, solution, dataset_ctx, cfg=cfg)
    timings["evaluate"] = perf_counter() - t0
    timings["total"] = perf_counter() - total_start

    return FrameResult(
        raw=raw,
        preprocessed=pre,
        candidates=cand,
        observed=obs,
        reference=ref,
        matching=matching,
        solution=solution,
        evaluation=evaluation,
        meta={
            "dataset_batch_root": str(dataset_ctx.batch_root) if dataset_ctx is not None else None,
            "num_reference_stars": len(ref),
            "requested_mode": "init",
            "timings_s": timings,
        },
    )
