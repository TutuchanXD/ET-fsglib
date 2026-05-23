import json

from fsglib.common.debug import (
    _safe_artifact_mask_filename,
    _write_validation_debug_artifacts,
)


def test_safe_artifact_mask_filename_rejects_path_traversal_characters():
    assert _safe_artifact_mask_filename("../bad/mask") == "artifact_mask_bad_mask.npy"
    assert _safe_artifact_mask_filename("saturation_guard") == (
        "artifact_mask_saturation_guard.npy"
    )


def test_validation_debug_artifacts_write_error_budget_json(tmp_path):
    payload = {
        "enabled": True,
        "summary": {
            "num_terms": 1,
            "dominant_angular_term": {
                "name": "attitude.residual.rms",
                "angular_equivalent_arcsec": 1.5,
            },
        },
        "terms": [
            {
                "name": "attitude.residual.rms",
                "stage": "attitude",
                "value": 1.5,
                "unit": "arcsec",
                "available": True,
            }
        ],
    }

    _write_validation_debug_artifacts(tmp_path, payload)

    output = tmp_path / "validation" / "error_budget.json"
    assert output.exists()
    saved = json.loads(output.read_text())
    assert saved["summary"]["dominant_angular_term"]["name"] == "attitude.residual.rms"
