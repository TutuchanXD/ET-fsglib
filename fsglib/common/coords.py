import numpy as np

def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float64)

def unit_vector_to_radec(v: np.ndarray) -> tuple[float, float]:
    v_norm = v / np.linalg.norm(v)
    x, y, z = v_norm
    dec_rad = np.arcsin(z)
    ra_rad = np.arctan2(y, x)
    if ra_rad < 0:
        ra_rad += 2 * np.pi
    return np.rad2deg(ra_rad), np.rad2deg(dec_rad)
