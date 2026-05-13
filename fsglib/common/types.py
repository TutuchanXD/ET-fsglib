from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np

class SolveMode(str, Enum):
    INIT_KNOWN_FIELD = "init_known_field"
    TRACKING = "tracking"
    LOCAL_REACQUIRE = "local_reacquire"
    LOST_IN_SPACE = "lost_in_space"
    SAFE_LOST = "safe_lost"


@dataclass
class RawFrame:
    detector_id: int
    image: np.ndarray
    time_s: float
    cadence_s: float | None = None
    coadd_start: int | None = None
    coadd_stop: int | None = None
    unit: str | None = None
    variant_id: int | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class PreprocessedFrame:
    detector_id: int
    image: np.ndarray
    background: np.ndarray | float
    noise_map: np.ndarray | float
    valid_mask: np.ndarray
    variance_map: np.ndarray | float | None = None
    preprocess_meta: dict = field(default_factory=dict)
    artifact_masks: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class StarCandidate:
    detector_id: int
    source_id: int
    x: float
    y: float
    flux: float
    peak: float
    area: int
    snr: float
    bbox: tuple[int, int, int, int]
    shape: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

@dataclass
class ObservedStar:
    detector_id: int
    source_id: int
    x: float
    y: float
    los_body: np.ndarray
    flux: float
    snr: float
    weight: float = 1.0
    flags: dict = field(default_factory=dict)

@dataclass
class MatchedStar:
    detector_id: int
    source_id: int
    catalog_id: int
    los_body: np.ndarray
    los_inertial: np.ndarray
    residual_arcsec: float | None = None
    weight: float = 1.0
    match_score: float = 0.0
    flags: dict = field(default_factory=dict)

@dataclass
class AttitudeSolution:
    q_ib: np.ndarray
    c_ib: np.ndarray | None
    euler_zyx: np.ndarray | None
    valid: bool
    mode: str
    num_matched: int
    residual_rms_arcsec: float
    residual_max_arcsec: float
    quality: dict = field(default_factory=dict)
    num_rejected: int = 0
    quality_flag: str = "UNKNOWN"
    degraded_level: str = "UNKNOWN"
    active_detector_ids: list[int] = field(default_factory=list)
    solver_iterations: int = 0

@dataclass
class TruthStar:
    source_id: int | None
    x_pix: float
    y_pix: float
    ra_deg: float
    dec_deg: float
    mag: float | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class DatasetContext:
    batch_root: Path
    frame_paths: list[Path]
    batch_center_ra_deg: float | None
    batch_center_dec_deg: float | None
    pixel_scale_arcsec_per_pix: float | None
    detector_width_pix: int | None
    detector_height_pix: int | None = None
    field_offset_x_pix: float | None = None
    field_offset_y_pix: float | None = None
    field_offset_source: str | None = None
    truth_stars: list[TruthStar] = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)

@dataclass
class MatchingContext:
    mode: str
    time_s: float
    observed_stars: list[ObservedStar]
    prior_attitude_q: np.ndarray | None
    detector_layout: dict
    optical_model: dict
    matching_cfg: dict
    boresight_inertial: np.ndarray | None = None
    reference_stars: list[Any] = field(default_factory=list)
    match_cache: Any | None = None

@dataclass
class MatchingResult:
    matched: list[MatchedStar]
    unmatched_observed_ids: list[int]
    unmatched_catalog_ids: list[int]
    mode: str
    success: bool
    score: float
    debug: dict = field(default_factory=dict)

@dataclass
class TrackState:
    catalog_id: int
    detector_id: int | None
    last_xy: tuple[float, float] | None
    last_seen_time_s: float
    miss_count: int = 0
    quality_score: float = 0.0
    active: bool = True
    last_match_score: float = 0.0
    last_residual_pix: float | None = None

@dataclass
class SolveStateMachine:
    mode: str | SolveMode
    consecutive_tracking_failures: int = 0
    consecutive_init_failures: int = 0
    consecutive_reacquire_failures: int = 0
    consecutive_lost_in_space_failures: int = 0
    reacquire_count: int = 0
    lost_count: int = 0
    safe_lost_count: int = 0
    transition_reason: str = "startup"
    total_tracking_frames: int = 0
    total_tracking_successes: int = 0
    total_init_frames: int = 0
    total_init_successes: int = 0
    total_reacquire_frames: int = 0
    total_reacquire_successes: int = 0
    total_lost_in_space_frames: int = 0
    total_lost_in_space_successes: int = 0
    requested_mode: str | None = None
    requested_match_algorithm: str | None = None
    selected_match_strategy: str | None = None
    validation_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, SolveMode):
            self.mode = self.mode.value

@dataclass
class AttitudeQuality:
    num_input: int
    num_used: int
    num_rejected: int
    residual_rms_arcsec: float
    residual_max_arcsec: float
    degraded: bool
    mode: str
    meta: dict = field(default_factory=dict)

@dataclass
class AttitudeSolveInput:
    time_s: float
    matched_stars: list[MatchedStar]
    prior_q_ib: np.ndarray | None
    mode: str
    solver_cfg: dict

@dataclass
class FrameEvaluation:
    num_truth_stars: int
    num_candidate_truth_matches: int
    centroid_mae_pix: float | None
    centroid_max_pix: float | None
    matched_catalog_truth_support: int
    matched_catalog_truth_ratio: float | None
    boresight_error_arcsec: float | None
    non_roll_error_arcsec: float | None = None
    roll_error_arcsec: float | None = None
    total_attitude_error_arcsec: float | None = None
    centroid_mean_dx_pix: float | None = None
    centroid_mean_dy_pix: float | None = None
    centroid_mean_abs_dx_pix: float | None = None
    centroid_mean_abs_dy_pix: float | None = None
    centroid_rms_dx_pix: float | None = None
    centroid_rms_dy_pix: float | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class FrameResult:
    raw: RawFrame
    preprocessed: PreprocessedFrame
    candidates: list[StarCandidate]
    observed: list[ObservedStar]
    reference: list[Any]
    matching: MatchingResult
    solution: AttitudeSolution
    evaluation: FrameEvaluation | None = None
    meta: dict = field(default_factory=dict)

@dataclass
class SequenceResult:
    frame_results: list[FrameResult]
    track_states: list[TrackState]
    mode_history: list[str]
    state_history: list[SolveStateMachine] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
