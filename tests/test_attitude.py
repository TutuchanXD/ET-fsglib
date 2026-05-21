import numpy as np
from scipy.spatial.transform import Rotation

from fsglib.attitude.solver import (
    dcm_to_quat,
    quat_to_dcm,
    scalar_first_quat_to_scipy_xyzw,
    solve_attitude,
)
from fsglib.common.types import AttitudeSolveInput, MatchedStar


ARCSEC_PER_RAD = 206264.80624709636


def _attitude_cfg(min_operational: int = 3) -> dict:
    return {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": min_operational,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 30.0,
            "weight_mode": "variance_snr_hybrid",
        },
        "project": {"mode": "init"},
    }


def _basis_vectors() -> list[np.ndarray]:
    return [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]


def _matched_identity_stars(
    vectors: list[np.ndarray],
    *,
    sigma_arcsec: float | None,
) -> list[MatchedStar]:
    stars = []
    for idx, vec in enumerate(vectors):
        flags = {}
        weight = 1.0
        if sigma_arcsec is not None:
            flags["sigma_angle_arcsec"] = sigma_arcsec
            weight = 1.0 / sigma_arcsec**2
        stars.append(
            MatchedStar(
                idx,
                idx,
                idx,
                vec.copy(),
                vec.copy(),
                weight=weight,
                flags=flags,
            )
        )
    return stars


def _robust_attitude_cfg(min_operational: int = 4) -> dict:
    cfg = _attitude_cfg(min_operational=min_operational)
    cfg["attitude"].update(
        {
            "outlier_reject_enable": True,
            "outlier_reject_mode": "sigma_clip_iterative",
            "outlier_max_residual_arcsec": 30.0,
            "outlier_sigma_clip": 3.0,
            "max_iterations": 5,
            "outlier_mad_fallback_enable": True,
        }
    )
    return cfg


def _rotate_vector_arcsec(
    vec: np.ndarray,
    arcsec: float,
    axis_hint: np.ndarray | None = None,
) -> np.ndarray:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if axis_hint is not None:
        axis = np.asarray(axis_hint, dtype=np.float64)
    axis = axis - np.dot(axis, vec) * vec
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, vec) * vec
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * (float(arcsec) / ARCSEC_PER_RAD)).apply(vec)


def test_scalar_first_quaternion_convention_matches_scipy_rotation():
    q_ib = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])

    scipy_xyzw = scalar_first_quat_to_scipy_xyzw(q_ib)
    assert np.allclose(scipy_xyzw, [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])

    c_ib = quat_to_dcm(q_ib)
    assert np.allclose(c_ib @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(c_ib, Rotation.from_quat(scipy_xyzw).as_matrix())
    assert np.allclose(dcm_to_quat(c_ib), q_ib)


def test_attitude_solution_exposes_quality_fields():
    stars = [
        MatchedStar(0, 0, 0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), weight=3.0),
        MatchedStar(1, 1, 1, np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), weight=2.0),
        MatchedStar(2, 2, 2, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), weight=4.0),
        MatchedStar(3, 3, 3, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), weight=5.0),
    ]
    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 30.0,
        },
        "project": {"mode": "init"},
    }

    sol = solve_attitude(AttitudeSolveInput(0.0, stars, None, "tracking", cfg["attitude"]), cfg)
    assert sol.valid
    assert sol.quality_flag == "VALID"
    assert sol.degraded_level == "NORMAL_4D"
    assert sol.active_detector_ids == [0, 1, 2, 3]
    assert sol.solver_iterations == 1
    assert sol.num_rejected == 0


def test_attitude_solution_reports_lost_when_underconstrained():
    stars = [MatchedStar(0, 0, 0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    cfg = {
        "attitude": {
            "min_stars_mathematical": 2,
            "min_stars_operational": 4,
            "outlier_reject_enable": False,
            "outlier_max_residual_arcsec": 30.0,
        },
        "project": {"mode": "init"},
    }

    sol = solve_attitude(stars, cfg)
    assert not sol.valid
    assert sol.quality_flag == "LOST"
    assert sol.degraded_level == "LOST"


def test_attitude_solution_reports_covariance_from_matched_star_sigmas():
    sigma_arcsec = 1.0
    stars = _matched_identity_stars(_basis_vectors(), sigma_arcsec=sigma_arcsec)

    sol = solve_attitude(stars, _attitude_cfg())

    assert sol.valid
    assert sol.covariance_rad2 is not None
    assert sol.covariance_rad2.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(sol.covariance_rad2) > 0.0)
    assert np.isclose(sol.sigma_non_roll_arcsec, 1.0, rtol=1e-6)
    assert np.isclose(sol.sigma_roll_arcsec, 1.0 / np.sqrt(2.0), rtol=1e-6)
    assert np.isclose(sol.attitude_condition_number, 1.0, rtol=1e-12)
    assert sol.quality["meta"]["attitude_covariance"]["available"] is True
    assert sol.quality["meta"]["attitude_covariance"]["source"] == "sigma_angle_arcsec"
    assert sol.quality["meta"]["weight_mode"] == "variance_snr_hybrid"
    assert sol.quality["meta"]["sigma_angle_arcsec"] == [1.0, 1.0, 1.0]


def test_attitude_solution_does_not_fabricate_covariance_without_sigmas():
    stars = _matched_identity_stars(_basis_vectors(), sigma_arcsec=None)

    sol = solve_attitude(stars, _attitude_cfg())

    assert sol.valid
    assert sol.covariance_rad2 is None
    assert sol.sigma_non_roll_arcsec is None
    assert sol.sigma_roll_arcsec is None
    assert sol.attitude_condition_number is None
    cov_meta = sol.quality["meta"]["attitude_covariance"]
    assert cov_meta["available"] is False
    assert cov_meta["reason"] == "missing_sigma_angle_arcsec"
    assert cov_meta["missing_sigma_count"] == 3


def test_sigma_clip_iterative_rejects_outlier_below_hard_gate():
    rng = np.random.default_rng(20)
    vectors = _random_unit_vectors(rng, 9)
    stars = _matched_identity_stars(vectors, sigma_arcsec=1.0)
    stars[-1].los_body = _rotate_vector_arcsec(vectors[-1], 20.0)

    sol = solve_attitude(stars, _robust_attitude_cfg(min_operational=4))

    assert sol.valid
    assert sol.num_rejected == 1
    assert sol.num_matched == 8
    audit = sol.quality["meta"]["robust_rejection"]
    assert audit["enabled"] is True
    assert audit["mode"] == "sigma_clip_iterative"
    assert audit["converged"] is True
    assert audit["num_rejected"] == 1
    rejected = audit["rejected_stars"][0]
    assert rejected["catalog_id"] == 8
    assert "sigma_clip" in rejected["reasons"]
    assert rejected["residual_arcsec"] < 30.0
    assert rejected["normalized_residual"] > 3.0


def test_partial_config_defaults_to_sigma_clip_iterative_mode():
    rng = np.random.default_rng(25)
    vectors = _random_unit_vectors(rng, 9)
    stars = _matched_identity_stars(vectors, sigma_arcsec=1.0)
    stars[-1].los_body = _rotate_vector_arcsec(vectors[-1], 20.0)
    cfg = _attitude_cfg(min_operational=4)
    cfg["attitude"].update(
        {
            "outlier_reject_enable": True,
            "outlier_max_residual_arcsec": 30.0,
            "max_iterations": 5,
        }
    )

    sol = solve_attitude(stars, cfg)

    assert sol.valid
    assert sol.num_rejected == 1
    audit = sol.quality["meta"]["robust_rejection"]
    assert audit["mode"] == "sigma_clip_iterative"
    assert audit["rejected_stars"][0]["catalog_id"] == 8
    assert "sigma_clip" in audit["rejected_stars"][0]["reasons"]


def test_iterative_rejection_degrades_when_operational_support_is_lost():
    rng = np.random.default_rng(21)
    vectors = _random_unit_vectors(rng, 4)
    stars = _matched_identity_stars(vectors, sigma_arcsec=1.0)
    stars[-1].los_body = _rotate_vector_arcsec(vectors[-1], 90.0)

    sol = solve_attitude(stars, _robust_attitude_cfg(min_operational=4))

    assert not sol.valid
    assert sol.quality_flag == "DEGRADED"
    assert sol.num_rejected == 1
    assert sol.num_matched == 3
    audit = sol.quality["meta"]["robust_rejection"]
    assert audit["final_support_count"] == 3
    assert audit["final_active_detector_count"] == 3
    assert audit["rejected_stars"][0]["catalog_id"] == 3


def test_mad_fallback_rejects_outlier_when_measurement_sigmas_are_missing():
    rng = np.random.default_rng(22)
    vectors = _random_unit_vectors(rng, 9)
    stars = _matched_identity_stars(vectors, sigma_arcsec=None)
    stars[-1].los_body = _rotate_vector_arcsec(vectors[-1], 25.0)

    sol = solve_attitude(stars, _robust_attitude_cfg(min_operational=4))

    assert sol.valid
    assert sol.num_rejected == 1
    rejected = sol.quality["meta"]["robust_rejection"]["rejected_stars"][0]
    assert rejected["catalog_id"] == 8
    assert "mad_sigma_clip" in rejected["reasons"]


def test_min_active_detectors_valid_can_force_degraded_solution():
    vectors = _basis_vectors() + [np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)]
    stars = _matched_identity_stars(vectors, sigma_arcsec=1.0)
    for star in stars:
        star.detector_id = 0
    cfg = _robust_attitude_cfg(min_operational=4)
    cfg["attitude"]["min_active_detectors_valid"] = 2

    sol = solve_attitude(stars, cfg)

    assert not sol.valid
    assert sol.quality_flag == "DEGRADED"
    assert sol.quality["meta"]["active_detector_ids"] == [0]
    assert sol.quality["meta"]["min_active_detectors_valid"] == 2


def test_sigma_floor_prevents_rejecting_nominal_model_residuals():
    rng = np.random.default_rng(23)
    vectors = _random_unit_vectors(rng, 8)
    stars = _matched_identity_stars(vectors, sigma_arcsec=0.1)
    for idx, star in enumerate(stars):
        axis_hint = rng.normal(size=3)
        star.los_body = _rotate_vector_arcsec(vectors[idx], 1.0 + 0.1 * idx, axis_hint)
    cfg = _robust_attitude_cfg(min_operational=4)
    cfg["attitude"]["outlier_sigma_floor_arcsec"] = 2.0

    sol = solve_attitude(stars, cfg)

    assert sol.valid
    assert sol.num_rejected == 0
    assert sol.quality["meta"]["robust_rejection"]["converged"] is True


def test_partial_config_uses_documented_sigma_floor_default():
    rng = np.random.default_rng(26)
    vectors = _random_unit_vectors(rng, 8)
    stars = _matched_identity_stars(vectors, sigma_arcsec=0.1)
    for idx, star in enumerate(stars):
        axis_hint = rng.normal(size=3)
        star.los_body = _rotate_vector_arcsec(vectors[idx], 1.0 + 0.1 * idx, axis_hint)
    cfg = _robust_attitude_cfg(min_operational=4)
    cfg["attitude"].pop("outlier_sigma_floor_arcsec", None)

    sol = solve_attitude(stars, cfg)

    assert sol.valid
    assert sol.num_rejected == 0
    decision = sol.quality["meta"]["robust_rejection"]["iterations"][0]["decisions"][0]
    assert decision["sigma_floor_arcsec"] == 2.0


def test_single_pass_mode_keeps_legacy_all_hard_gate_rejections():
    rng = np.random.default_rng(24)
    vectors = _random_unit_vectors(rng, 7)
    stars = _matched_identity_stars(vectors, sigma_arcsec=1.0)
    stars[-1].los_body = _rotate_vector_arcsec(vectors[-1], 120.0)
    stars[-2].los_body = _rotate_vector_arcsec(vectors[-2], 100.0)
    cfg = _robust_attitude_cfg(min_operational=4)
    cfg["attitude"]["outlier_reject_mode"] = "single_pass"
    cfg["attitude"]["outlier_max_residual_arcsec"] = 30.0

    sol = solve_attitude(stars, cfg)

    assert sol.num_rejected >= 2
    assert sol.quality["meta"]["robust_rejection"]["mode"] == "single_pass"


def _random_unit_vectors(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    vectors = []
    for _ in range(count):
        vec = rng.normal(size=3)
        vec /= np.linalg.norm(vec)
        vectors.append(vec.astype(np.float64))
    return vectors


def _tangent_basis(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, vec))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b1 = ref - np.dot(ref, vec) * vec
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(vec, b1)
    b2 /= np.linalg.norm(b2)
    return b1, b2


def _perturb_los(
    rng: np.random.Generator,
    vec: np.ndarray,
    sigma_rad: float,
) -> np.ndarray:
    b1, b2 = _tangent_basis(vec)
    dx, dy = rng.normal(scale=sigma_rad, size=2)
    perturbed = vec + dx * b1 + dy * b2
    return perturbed / np.linalg.norm(perturbed)


def test_attitude_covariance_matches_small_monte_carlo_scatter_scale():
    rng = np.random.default_rng(19)
    sigma_arcsec = 2.0
    sigma_rad = sigma_arcsec / ARCSEC_PER_RAD
    inertial_vectors = _random_unit_vectors(rng, 8)
    reference_stars = _matched_identity_stars(
        inertial_vectors,
        sigma_arcsec=sigma_arcsec,
    )
    reference_solution = solve_attitude(reference_stars, _attitude_cfg())
    predicted_cov = reference_solution.covariance_rad2
    assert predicted_cov is not None

    rotvecs = []
    for _ in range(250):
        noisy_stars = []
        for idx, vec in enumerate(inertial_vectors):
            noisy_los = _perturb_los(rng, vec, sigma_rad)
            noisy_stars.append(
                MatchedStar(
                    idx,
                    idx,
                    idx,
                    noisy_los,
                    vec,
                    weight=1.0 / sigma_arcsec**2,
                    flags={"sigma_angle_arcsec": sigma_arcsec},
                )
            )
        sol = solve_attitude(noisy_stars, _attitude_cfg())
        rotvecs.append(Rotation.from_matrix(sol.c_ib).as_rotvec())

    empirical_cov = np.cov(np.asarray(rotvecs).T, ddof=1)
    empirical_sigma = np.sqrt(np.diag(empirical_cov))
    predicted_sigma = np.sqrt(np.diag(predicted_cov))
    ratio = empirical_sigma / predicted_sigma
    assert np.all(ratio > 0.5)
    assert np.all(ratio < 1.6)
