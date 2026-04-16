import numpy as np
import pytest
import os
from fsglib.match.triangle import TriangleMatcher
from fsglib.common.types import ObservedStar

@pytest.fixture
def gsc_path():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "catalogs", "gsc_8mag.npz")
    if not os.path.exists(path):
        pytest.skip(f"Index {path} not found. Run tools.build_gsc first.")
    return path

def test_triangle_matching(gsc_path):
    # Load raw catalog data to pick real stars
    data = np.load(gsc_path)
    cat_ids = data['catalog_ids']
    cat_vecs = data['catalog_vectors']
    
    # Pick 1 random center star
    np.random.seed(42)
    center_idx = np.random.randint(len(cat_ids))
    center_vec = cat_vecs[center_idx]
    
    # Compute angles to all other stars
    dots = np.dot(cat_vecs, center_vec)
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    
    # Find stars within 5 degrees
    nearby_indices = np.where((angles_deg > 0.0) & (angles_deg < 5.0))[0]
    
    if len(nearby_indices) < 4:
        pytest.skip(f"Not enough nearby stars around center {center_idx}")
        
    chosen_neighbors = np.random.choice(nearby_indices, 4, replace=False)
    indices = np.append([center_idx], chosen_neighbors)
    np.random.shuffle(indices)
    
    selected_ids = cat_ids[indices]
    selected_vecs = cat_vecs[indices]
    
    # Create an arbitrary rotation matrix (Wahba's problem transformation)
    # E.g. rotate roughly 30 degrees around Z axis
    theta = np.radians(30)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Add tiny noise (like 1 arcsecond = 5e-6 rad)
    noise = np.random.normal(0, 5e-6, selected_vecs.shape)
    
    rotated_vecs = (Rz @ selected_vecs.T).T + noise
    rotated_vecs /= np.linalg.norm(rotated_vecs, axis=1)[:, np.newaxis]
    
    observed = []
    for i in range(5):
        obs = ObservedStar(
            detector_id="main",
            source_id=100 + i, # arbitrary
            x=0.0,
            y=0.0,
            los_body=rotated_vecs[i],
            flux=1000.0 - i, # Give highest flux so they are picked
            snr=100.0
        )
        observed.append(obs)
        
    matcher = TriangleMatcher(gsc_path=gsc_path, angle_tol_deg=0.001, max_stars=5)
    matched = matcher.match(observed)
    
    print(f"Found {len(matched)} matches!")
    assert len(matched) == 5
    
    # Verify their mapped catalog_ids match selected_ids
    mapped_ids = {m.source_id: m.catalog_id for m in matched}
    
    for i in range(5):
        obs_id = 100 + i
        if obs_id in mapped_ids:    
            print(f"Index {i} -> Expected: {selected_ids[i]}, Got: {mapped_ids[obs_id]}")
            assert mapped_ids[obs_id] == selected_ids[i]
        else:
            print(f"Index {i} -> Expected: {selected_ids[i]}, Got: NOT FOUND")
            assert False, "Star not matched!"
