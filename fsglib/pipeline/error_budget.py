from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fsglib.common.types import ErrorBudgetLedger, ErrorBudgetTerm


_FAKE_ASSET_MARKERS = ("pr09_fake", "fake", "dummy", "noop", "no-op")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _rms(value: Any) -> float | None:
    arr = _finite_array(value)
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr**2)))


def _median(values: list[float | None]) -> float | None:
    finite = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _image_unit(raw: Any | None, preprocessed: Any | None) -> str:
    unit = getattr(raw, "unit", None)
    if unit:
        return str(unit)
    meta = getattr(preprocessed, "preprocess_meta", {}) or {}
    return str(meta.get("input_unit") or "image_unit")


def _variance_base_unit(preprocess_meta: dict[str, Any], raw: Any | None, preprocessed: Any | None) -> str:
    variance_unit = str(preprocess_meta.get("variance_unit") or "")
    if variance_unit.endswith("^2"):
        return variance_unit[:-2]
    return _image_unit(raw, preprocessed)


def _frame_id(raw: Any | None) -> str | None:
    meta = getattr(raw, "meta", {}) or {}
    npz_path = meta.get("npz_path")
    if npz_path:
        return Path(str(npz_path)).stem
    detector_id = getattr(raw, "detector_id", None)
    time_s = _safe_float(getattr(raw, "time_s", None))
    if detector_id is not None and time_s is not None:
        return f"det{detector_id}_t{time_s:g}"
    return None


def _is_fake_asset(path: Any) -> bool:
    if not path:
        return False
    text = str(path).lower()
    return any(marker in text for marker in _FAKE_ASSET_MARKERS)


def _iter_detector_contexts(
    raw: Any | None,
    preprocessed: Any | None,
    candidates: list[Any],
    detector_contexts: dict[str, dict[str, Any]] | None,
) -> list[tuple[str, Any | None, Any | None, list[Any]]]:
    if detector_contexts:
        items = []
        for detector_id, ctx in detector_contexts.items():
            items.append(
                (
                    str(detector_id),
                    ctx.get("raw"),
                    ctx.get("preprocessed"),
                    list(ctx.get("selected_candidates", ctx.get("all_candidates", [])) or []),
                )
            )
        return items
    detector_id = getattr(raw, "detector_id", getattr(preprocessed, "detector_id", "frame"))
    return [(str(detector_id), raw, preprocessed, list(candidates or []))]


def _term(
    *,
    name: str,
    stage: str,
    value: float | int | None,
    unit: str,
    source: str | None,
    assumption: str | None = None,
    scope: str = "frame",
    detector_id: int | str | None = None,
    star_id: int | str | None = None,
    available: bool = True,
    reason: str | None = None,
    angular_equivalent_arcsec: float | None = None,
    meta: dict[str, Any] | None = None,
) -> ErrorBudgetTerm:
    numeric = _safe_float(value)
    int_value = _safe_int(value)
    if available and value is not None and numeric is None:
        available = False
        reason = reason or "nonfinite_value"
    if available and value is None:
        available = False
        reason = reason or "value_unavailable"
    if isinstance(value, (int, np.integer)) and int_value is not None:
        stored_value: float | int | None = int_value
    else:
        stored_value = numeric
    angular = _safe_float(angular_equivalent_arcsec)
    return ErrorBudgetTerm(
        name=name,
        stage=stage,
        value=stored_value,
        unit=unit,
        source=source,
        assumption=assumption,
        scope=scope,
        detector_id=detector_id,
        star_id=star_id,
        available=bool(available),
        reason=reason,
        angular_equivalent_arcsec=angular,
        meta={} if meta is None else dict(meta),
    )


def _unavailable_term(name: str, stage: str, unit: str, reason: str, source: str | None = None) -> ErrorBudgetTerm:
    return _term(
        name=name,
        stage=stage,
        value=None,
        unit=unit,
        source=source,
        available=False,
        reason=reason,
    )


def _preprocess_meta(detector_items: list[tuple[str, Any | None, Any | None, list[Any]]]) -> dict[str, Any]:
    for _, _, preprocessed, _ in detector_items:
        meta = getattr(preprocessed, "preprocess_meta", None)
        if isinstance(meta, dict):
            return meta
    return {}


def _collect_noise_arrays(detector_items: list[tuple[str, Any | None, Any | None, list[Any]]]) -> list[np.ndarray]:
    arrays = []
    for _, _, preprocessed, _ in detector_items:
        arr = _finite_array(getattr(preprocessed, "noise_map", None))
        if arr.size:
            arrays.append(arr)
    return arrays


def _collect_image_arrays(detector_items: list[tuple[str, Any | None, Any | None, list[Any]]]) -> list[np.ndarray]:
    arrays = []
    for _, _, preprocessed, _ in detector_items:
        arr = _finite_array(getattr(preprocessed, "image", None))
        if arr.size:
            arrays.append(arr)
    return arrays


def _detector_summaries(
    detector_items: list[tuple[str, Any | None, Any | None, list[Any]]],
    observed: list[Any],
    matching: Any,
) -> dict[str, dict[str, Any]]:
    observed_by_detector: dict[str, list[Any]] = {}
    for star in observed or []:
        observed_by_detector.setdefault(str(getattr(star, "detector_id", "unknown")), []).append(star)

    matched_by_detector: dict[str, list[Any]] = {}
    for star in getattr(matching, "matched", []) or []:
        matched_by_detector.setdefault(str(getattr(star, "detector_id", "unknown")), []).append(star)

    summaries: dict[str, dict[str, Any]] = {}
    for detector_id, _, preprocessed, candidates in detector_items:
        noise_rms = _rms(getattr(preprocessed, "noise_map", None))
        valid_mask = getattr(preprocessed, "valid_mask", None)
        valid_fraction = None
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.size:
                valid_fraction = float(np.count_nonzero(valid) / valid.size)
        meta = getattr(preprocessed, "preprocess_meta", {}) or {}
        summaries[str(detector_id)] = {
            "num_candidates": int(len(candidates)),
            "num_observed": int(len(observed_by_detector.get(str(detector_id), []))),
            "num_matched": int(len(matched_by_detector.get(str(detector_id), []))),
            "noise_rms": noise_rms,
            "valid_fraction": valid_fraction,
            "num_saturated_pixels": int((meta.get("artifact_counts") or {}).get("saturated", 0)),
        }
    return summaries


def _fake_asset_count(preprocess_meta: dict[str, Any]) -> tuple[int, list[str]]:
    paths = []
    calibration = preprocess_meta.get("calibration") or {}
    if isinstance(calibration, dict):
        for payload in calibration.values():
            if isinstance(payload, dict) and payload.get("applied", False):
                path = payload.get("path")
                if _is_fake_asset(path):
                    paths.append(str(path))
    return len(paths), paths


def _centroid_sigma_pix(candidates: list[Any]) -> list[float]:
    sigmas = []
    for candidate in candidates:
        cov = getattr(candidate, "centroid_cov_pix", None)
        if cov is None:
            continue
        arr = np.asarray(cov, dtype=np.float64)
        if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
            continue
        sigmas.append(float(np.sqrt(max(float(np.trace(arr)), 0.0))))
    return sigmas


def _sigma_angle(star: Any) -> float | None:
    value = getattr(star, "sigma_angle_arcsec", None)
    if value is None:
        flags = getattr(star, "flags", {}) or {}
        value = flags.get("sigma_angle_arcsec")
    return _safe_float(value)


def _matched_residuals(matching: Any) -> list[float]:
    values = []
    for star in getattr(matching, "matched", []) or []:
        value = _safe_float(getattr(star, "residual_arcsec", None))
        if value is not None:
            values.append(value)
    return values


def _robust_rejection_payload(solution: Any) -> dict[str, Any]:
    quality = getattr(solution, "quality", {}) or {}
    meta = quality.get("meta") if isinstance(quality, dict) else {}
    if not isinstance(meta, dict):
        return {}
    robust = meta.get("robust_rejection")
    return robust if isinstance(robust, dict) else {}


def _per_star_records(observed: list[Any], matching: Any, solution: Any, max_records: int | None) -> list[dict[str, Any]]:
    observed_by_source = {str(getattr(star, "source_id", "")): star for star in observed or []}
    robust = _robust_rejection_payload(solution)
    rejected_by_source: dict[str, list[str]] = {}
    for rejected in robust.get("rejected_stars", []) or []:
        if not isinstance(rejected, dict):
            continue
        source_id = rejected.get("source_id")
        if source_id is None:
            continue
        reasons = rejected.get("reasons") or []
        rejected_by_source[str(source_id)] = [str(reason) for reason in reasons]

    records = []
    for star in getattr(matching, "matched", []) or []:
        source_id = str(getattr(star, "source_id", ""))
        observed_star = observed_by_source.get(source_id)
        flags = getattr(star, "flags", {}) or {}
        record = {
            "detector_id": getattr(star, "detector_id", None),
            "source_id": source_id,
            "catalog_id": getattr(star, "catalog_id", None),
            "matched": True,
            "sigma_angle_arcsec": _sigma_angle(star) or _sigma_angle(observed_star),
            "residual_arcsec": _safe_float(getattr(star, "residual_arcsec", None)),
            "residual_pix": _safe_float(flags.get("residual_pix")),
            "weight": _safe_float(getattr(star, "weight", None)),
            "match_score": _safe_float(getattr(star, "match_score", None)),
            "snr": None if observed_star is None else _safe_float(getattr(observed_star, "snr", None)),
            "rejected": source_id in rejected_by_source,
            "rejection_reasons": rejected_by_source.get(source_id, []),
        }
        records.append(record)
        if max_records is not None and len(records) >= max_records:
            break
    return records


def _summary(terms: list[ErrorBudgetTerm]) -> dict[str, Any]:
    available = [term for term in terms if term.available]
    unavailable = [term for term in terms if not term.available]
    angular_terms = [
        term
        for term in available
        if term.angular_equivalent_arcsec is not None and np.isfinite(term.angular_equivalent_arcsec)
    ]
    dominant = None
    if angular_terms:
        stage_priority = {
            "attitude": 5,
            "attitude_validation": 4,
            "matching": 3,
            "centroid": 2,
            "optics": 1,
            "catalog": 1,
        }
        term = max(
            angular_terms,
            key=lambda item: (
                float(item.angular_equivalent_arcsec),
                stage_priority.get(item.stage, 0),
            ),
        )
        dominant = {
            "name": term.name,
            "stage": term.stage,
            "value": term.value,
            "unit": term.unit,
            "angular_equivalent_arcsec": term.angular_equivalent_arcsec,
            "source": term.source,
        }
    return {
        "num_terms": int(len(terms)),
        "num_available_terms": int(len(available)),
        "num_unavailable_terms": int(len(unavailable)),
        "dominant_angular_term": dominant,
        "missing_terms": [
            {"name": term.name, "stage": term.stage, "reason": term.reason}
            for term in unavailable
        ],
    }


def build_error_budget_ledger(
    *,
    raw: Any | None,
    preprocessed: Any | None,
    candidates: list[Any],
    observed: list[Any] | None,
    matching: Any,
    solution: Any,
    evaluation: Any | None,
    dataset_ctx: Any | None,
    cfg: dict | None,
    detector_contexts: dict[str, dict[str, Any]] | None = None,
) -> ErrorBudgetLedger:
    cfg = {} if cfg is None else cfg
    budget_cfg = dict(cfg.get("evaluation", {}).get("error_budget", {}))
    enabled = bool(budget_cfg.get("enabled", True))
    if not enabled:
        return ErrorBudgetLedger(enabled=False, frame_id=_frame_id(raw), summary={"enabled": False})

    detector_items = _iter_detector_contexts(raw, preprocessed, candidates, detector_contexts)
    all_candidates = [candidate for _, _, _, item_candidates in detector_items for candidate in item_candidates]
    if not all_candidates:
        all_candidates = list(candidates or [])
    first_raw = raw if raw is not None else next((item_raw for _, item_raw, _, _ in detector_items if item_raw is not None), None)
    first_pre = preprocessed if preprocessed is not None else next(
        (item_pre for _, _, item_pre, _ in detector_items if item_pre is not None),
        None,
    )
    preprocess_meta = _preprocess_meta(detector_items)
    variance_model = str(preprocess_meta.get("variance_model_effective") or preprocess_meta.get("variance_model") or "unknown")
    image_unit = _variance_base_unit(preprocess_meta, first_raw, first_pre)
    terms: list[ErrorBudgetTerm] = []
    assumptions: list[str] = []

    noise_arrays = _collect_noise_arrays(detector_items)
    aggregate_noise = np.concatenate(noise_arrays) if noise_arrays else np.zeros(0, dtype=np.float64)
    terms.append(
        _term(
            name="detector.noise.empirical_rms",
            stage="detector",
            value=_rms(aggregate_noise),
            unit=image_unit,
            source="PreprocessedFrame.noise_map",
            assumption="empirical robust RMS includes unresolved detector and background noise"
            if variance_model != "poisson_read_noise"
            else None,
        )
    )

    variance_components = preprocess_meta.get("variance_components") or {}
    if variance_model == "poisson_read_noise":
        gain = _safe_float(variance_components.get("gain_e_per_output_unit"))
        read_noise = _safe_float(variance_components.get("read_noise_e"))
        quant_noise = _safe_float(variance_components.get("quantization_noise_e"))
        image_arrays = _collect_image_arrays(detector_items)
        signal_e = None
        if gain is not None and image_arrays:
            image_values = np.concatenate(image_arrays)
            signal_e = np.maximum(image_values, 0.0) * gain
        terms.extend(
            [
                _term(
                    name="detector.noise.read",
                    stage="detector",
                    value=read_noise,
                    unit="e-",
                    source="preprocess.variance_components.read_noise_e",
                ),
                _term(
                    name="detector.noise.quantization",
                    stage="detector",
                    value=quant_noise,
                    unit="e-",
                    source="preprocess.variance_components.quantization_noise_e",
                ),
                _term(
                    name="detector.noise.photon_shot",
                    stage="detector",
                    value=None if signal_e is None else _rms(np.sqrt(signal_e)),
                    unit="e-",
                    source="PreprocessedFrame.image * gain_e_per_output_unit",
                    assumption="uses calibrated nonnegative frame signal as photon-noise proxy",
                ),
            ]
        )
        dark_source = str(variance_components.get("dark_current_source") or "none")
        terms.append(
            _unavailable_term(
                "detector.noise.dark_current",
                "detector",
                "e-",
                "dark_current_value_not_recorded" if dark_source != "none" else "dark_current_not_configured",
                source=f"preprocess.variance_components.dark_current_source={dark_source}",
            )
        )
        if bool(variance_components.get("flat_uncertainty_included", False)):
            terms.append(
                _term(
                    name="detector.noise.flat_residual",
                    stage="detector",
                    value=0.0,
                    unit="relative",
                    source="preprocess.variance_components.flat_uncertainty_included",
                    assumption="flat residual included by upstream variance model",
                )
            )
        else:
            terms.append(
                _unavailable_term(
                    "detector.noise.flat_residual",
                    "detector",
                    "relative",
                    "flat_uncertainty_not_propagated",
                    source="preprocess.variance_components.flat_uncertainty_included",
                )
            )
    else:
        for name, unit in (
            ("detector.noise.read", "e-"),
            ("detector.noise.dark_current", "e-"),
            ("detector.noise.photon_shot", "e-"),
            ("detector.noise.quantization", "e-"),
            ("detector.noise.flat_residual", "relative"),
        ):
            terms.append(
                _unavailable_term(
                    name,
                    "detector",
                    unit,
                    "requires_poisson_read_noise_variance_model",
                    source="preprocess.variance_model_effective",
                )
            )

    adc_clip = preprocess_meta.get("adc_clip") or {}
    terms.append(
        _term(
            name="preprocess.adc_clip.high_pixels",
            stage="preprocess",
            value=adc_clip.get("num_clipped_high_pixels", 0),
            unit="pix",
            source="preprocess_meta.adc_clip.num_clipped_high_pixels",
            assumption="ADC-clipped pixels are handled by saturation/artifact guards before centroiding",
        )
    )

    fake_count, fake_paths = _fake_asset_count(preprocess_meta)
    fake_assumption = None
    if fake_count:
        fake_assumption = "fake calibration assets are placeholders and must not be treated as real detector calibration"
        assumptions.append(fake_assumption)
    terms.append(
        _term(
            name="preprocess.calibration.fake_asset_count",
            stage="preprocess",
            value=fake_count,
            unit="asset",
            source="preprocess_meta.calibration[].path",
            assumption=fake_assumption,
            meta={"paths": fake_paths},
        )
    )

    centroid_sigmas = _centroid_sigma_pix(all_candidates)
    terms.append(
        _term(
            name="centroid.uncertainty.sigma_radial_median",
            stage="centroid",
            value=_median(centroid_sigmas),
            unit="pix",
            source="StarCandidate.centroid_cov_pix",
        )
    )
    sigma_angles = [_sigma_angle(star) for star in observed or []]
    if not sigma_angles:
        sigma_angles = [_sigma_angle(star) for star in getattr(matching, "matched", []) or []]
    sigma_angle_median = _median(sigma_angles)
    terms.append(
        _term(
            name="centroid.uncertainty.sigma_angle_median",
            stage="centroid",
            value=sigma_angle_median,
            unit="arcsec",
            source="ObservedStar.sigma_angle_arcsec",
            angular_equivalent_arcsec=sigma_angle_median,
        )
    )
    centroid_error_radial = None
    if evaluation is not None:
        rms_dx = _safe_float(getattr(evaluation, "centroid_rms_dx_pix", None))
        rms_dy = _safe_float(getattr(evaluation, "centroid_rms_dy_pix", None))
        if rms_dx is not None and rms_dy is not None:
            centroid_error_radial = float(np.hypot(rms_dx, rms_dy))
        else:
            centroid_error_radial = _safe_float(getattr(evaluation, "centroid_mae_pix", None))
    pixel_scale = _safe_float(getattr(dataset_ctx, "pixel_scale_arcsec_per_pix", None))
    terms.append(
        _term(
            name="centroid.error.rms_radial",
            stage="centroid",
            value=centroid_error_radial,
            unit="pix",
            source="FrameEvaluation.centroid_rms_*_pix",
            angular_equivalent_arcsec=None
            if centroid_error_radial is None or pixel_scale is None
            else centroid_error_radial * pixel_scale,
        )
    )

    catalog_uncertainty = _safe_float(budget_cfg.get("catalog_uncertainty_arcsec"))
    terms.append(
        _term(
            name="catalog.reference_uncertainty",
            stage="catalog",
            value=catalog_uncertainty,
            unit="arcsec",
            source="evaluation.error_budget.catalog_uncertainty_arcsec",
            available=catalog_uncertainty is not None,
            reason=None if catalog_uncertainty is not None else "catalog_uncertainty_not_configured",
            angular_equivalent_arcsec=catalog_uncertainty,
        )
    )
    alignment_residual = _safe_float(budget_cfg.get("optical_alignment_residual_arcsec"))
    terms.append(
        _term(
            name="optics.alignment_residual",
            stage="optics",
            value=alignment_residual,
            unit="arcsec",
            source="evaluation.error_budget.optical_alignment_residual_arcsec",
            available=alignment_residual is not None,
            reason=None if alignment_residual is not None else "alignment_uncertainty_not_configured",
            angular_equivalent_arcsec=alignment_residual,
        )
    )

    match_residual_pix = _safe_float((getattr(matching, "debug", {}) or {}).get("mean_residual_pix"))
    terms.append(
        _term(
            name="matching.residual.rms_pix",
            stage="matching",
            value=match_residual_pix,
            unit="pix",
            source="MatchingResult.debug.mean_residual_pix",
            available=match_residual_pix is not None,
            reason=None if match_residual_pix is not None else "matching_pixel_residual_not_recorded",
            angular_equivalent_arcsec=None if match_residual_pix is None or pixel_scale is None else match_residual_pix * pixel_scale,
        )
    )
    matched_residual_rms = _rms(_matched_residuals(matching))
    terms.append(
        _term(
            name="matching.residual.rms",
            stage="matching",
            value=matched_residual_rms,
            unit="arcsec",
            source="MatchedStar.residual_arcsec",
            available=matched_residual_rms is not None,
            reason=None if matched_residual_rms is not None else "matched_residuals_not_recorded",
            angular_equivalent_arcsec=matched_residual_rms,
        )
    )

    residual_rms = _safe_float(getattr(solution, "residual_rms_arcsec", None))
    residual_max = _safe_float(getattr(solution, "residual_max_arcsec", None))
    terms.extend(
        [
            _term(
                name="attitude.residual.rms",
                stage="attitude",
                value=residual_rms,
                unit="arcsec",
                source="AttitudeSolution.residual_rms_arcsec",
                angular_equivalent_arcsec=residual_rms,
            ),
            _term(
                name="attitude.residual.max",
                stage="attitude",
                value=residual_max,
                unit="arcsec",
                source="AttitudeSolution.residual_max_arcsec",
                angular_equivalent_arcsec=residual_max,
            ),
            _term(
                name="attitude.covariance.sigma_non_roll",
                stage="attitude",
                value=_safe_float(getattr(solution, "sigma_non_roll_arcsec", None)),
                unit="arcsec",
                source="AttitudeSolution.sigma_non_roll_arcsec",
                angular_equivalent_arcsec=_safe_float(getattr(solution, "sigma_non_roll_arcsec", None)),
            ),
            _term(
                name="attitude.covariance.sigma_roll",
                stage="attitude",
                value=_safe_float(getattr(solution, "sigma_roll_arcsec", None)),
                unit="arcsec",
                source="AttitudeSolution.sigma_roll_arcsec",
                angular_equivalent_arcsec=_safe_float(getattr(solution, "sigma_roll_arcsec", None)),
            ),
        ]
    )
    robust = _robust_rejection_payload(solution)
    terms.append(
        _term(
            name="attitude.robust_rejection.count",
            stage="attitude_validation",
            value=robust.get("num_rejected", getattr(solution, "num_rejected", 0)),
            unit="star",
            source="AttitudeSolution.quality.meta.robust_rejection",
            assumption="control term: rejected measurements explain validation behavior, not a physical noise source",
        )
    )

    max_records = budget_cfg.get("max_per_star_records")
    max_records_int = None if max_records is None else int(max_records)
    per_detector = _detector_summaries(detector_items, observed or [], matching)
    per_star = _per_star_records(observed or [], matching, solution, max_records_int)
    summary = _summary(terms)
    summary["num_per_star_records"] = int(len(per_star))
    summary["num_per_detector_records"] = int(len(per_detector))
    summary["variance_model"] = variance_model

    return ErrorBudgetLedger(
        enabled=True,
        frame_id=_frame_id(first_raw),
        terms=terms,
        summary=summary,
        per_detector=per_detector,
        per_star=per_star,
        assumptions=assumptions,
    )


def summarize_error_budget_ledgers(ledgers: list[ErrorBudgetLedger | dict[str, Any]]) -> dict[str, Any]:
    term_values: dict[str, list[tuple[float, str]]] = {}
    for ledger in ledgers:
        payload = ledger.to_dict() if isinstance(ledger, ErrorBudgetLedger) else ledger
        if not payload.get("enabled", False):
            continue
        for term in payload.get("terms", []) or []:
            if isinstance(term, ErrorBudgetTerm):
                term = term.to_dict()
            if not term.get("available", False):
                continue
            value = _safe_float(term.get("angular_equivalent_arcsec"))
            unit = "arcsec"
            if value is None:
                value = _safe_float(term.get("value"))
                unit = str(term.get("unit"))
            if value is None:
                continue
            term_values.setdefault(str(term.get("name")), []).append((value, unit))

    summary_terms: dict[str, dict[str, Any]] = {}
    for name, values_with_units in sorted(term_values.items()):
        values = [value for value, _ in values_with_units]
        unit = values_with_units[0][1]
        summary_terms[name] = {
            "count": int(len(values)),
            "unit": unit,
            "mean": float(np.mean(values)),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "max": float(np.max(values)),
        }
    return {
        "num_ledgers": int(sum(1 for ledger in ledgers if (ledger.to_dict() if isinstance(ledger, ErrorBudgetLedger) else ledger).get("enabled", False))),
        "terms": summary_terms,
    }


def error_budget_csv_rows(ledger: ErrorBudgetLedger | dict[str, Any]) -> list[dict[str, Any]]:
    payload = ledger.to_dict() if isinstance(ledger, ErrorBudgetLedger) else ledger
    rows = []
    for term in payload.get("terms", []) or []:
        if isinstance(term, ErrorBudgetTerm):
            term = term.to_dict()
        rows.append(
            {
                "name": term.get("name"),
                "stage": term.get("stage"),
                "scope": term.get("scope"),
                "detector_id": term.get("detector_id"),
                "star_id": term.get("star_id"),
                "available": term.get("available"),
                "value": term.get("value"),
                "unit": term.get("unit"),
                "angular_equivalent_arcsec": term.get("angular_equivalent_arcsec"),
                "source": term.get("source"),
                "assumption": term.get("assumption"),
                "reason": term.get("reason"),
            }
        )
    return rows
