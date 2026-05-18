from fsglib.common.debug import _safe_artifact_mask_filename


def test_safe_artifact_mask_filename_rejects_path_traversal_characters():
    assert _safe_artifact_mask_filename("../bad/mask") == "artifact_mask_bad_mask.npy"
    assert _safe_artifact_mask_filename("saturation_guard") == (
        "artifact_mask_saturation_guard.npy"
    )

