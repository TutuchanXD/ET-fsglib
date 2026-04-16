import numpy as np
from fsglib.common.types import StarCandidate, ObservedStar

def candidates_to_observed(candidates: list[StarCandidate], projector, cfg: dict) -> list[ObservedStar]:
    """
    Convert image plane StarCandidates into 3D ObservedStars using the projector / camera model.
    """
    observed = []
    
    for cand in candidates:
        los_body = projector.pixel_to_los_body(cand.detector_id, cand.x, cand.y)
        
        # Calculate uncertainty/weight - could be based on SNR
        weight = cand.snr if cand.snr > 0 else 1.0
        
        observed.append(ObservedStar(
            detector_id=cand.detector_id,
            source_id=cand.source_id,
            x=cand.x,
            y=cand.y,
            los_body=los_body,
            flux=cand.flux,
            snr=cand.snr,
            weight=weight,
            flags=cand.flags
        ))
        
    return observed
