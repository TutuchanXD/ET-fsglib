import argparse
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def generate_kvector(values, q=1.0e-5):
    """
    Builds a 1D K-Vector indexing structure.
    y = mx + q (creates the linear bins)
    """
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    
    n = len(sorted_vals)
    if n == 0:
        return None, None, None, None
        
    y_min = sorted_vals[0]
    y_max = sorted_vals[-1]
    
    # Line equation: length = m * v + q
    # We map value v to an index k
    # k = (v - y_min) / (y_max - y_min) * (n - 1)
    
    m = (n - 1) / (y_max - y_min + 1e-12)
    b = -m * y_min
    
    k_vec = np.zeros(n + 1, dtype=int)
    
    for i in range(1, n):
        # The expected index based on the line
        k_val = m * sorted_vals[i] + b
        int_k = int(np.floor(k_val))
        
        if int_k >= n:
            int_k = n - 1
            
        k_vec[int_k + 1] = i
        
    # Fill gaps
    for i in range(1, len(k_vec)):
        if k_vec[i] == 0:
            k_vec[i] = k_vec[i-1]
            
    k_vec[-1] = n
    return sorted_idx, sorted_vals, m, b, k_vec

import multiprocessing as mp

def process_file(args):
    f, max_mag = args
    try:
        df = pd.read_csv(f, usecols=['source_id', 'ra', 'dec', 'g_mean_mag'])
        df = df[df['g_mean_mag'] <= max_mag]
        if not df.empty:
            return df
    except Exception:
        pass
    return None

def build_gsc(gaia_dir: str, output_path: str, max_mag: float = 12.0, max_fov_deg: float = 20.0, min_fov_deg: float = 0.5, max_files: int = None):
    """
    Reads Gaia subset, extracts vectors, creates all pairs within FOV bounds, and saves K-Vector NPZ.
    """
    print(f"Loading Gaia catalog from {gaia_dir} (mag <= {max_mag})...")
    files = glob.glob(os.path.join(gaia_dir, "healpix_n05_nested_*.csv"))
    if max_files is not None:
        files = files[:max_files]
        
    total_files = len(files)
    print(f"Found {total_files} CSV files to process.", flush=True)
    
    pool_args = [(f, max_mag) for f in files]
    all_rows = []
    with mp.Pool(processes=os.cpu_count() or 4) as pool:
        for i, res in enumerate(pool.imap_unordered(process_file, pool_args)):
            if res is not None:
                all_rows.append(res)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{total_files} files...", flush=True)
        
    if len(all_rows) == 0:
        print("No stars found!", flush=True)
        return
        
    cat = pd.concat(all_rows, ignore_index=True)
    print(f"Total guide stars selected: {len(cat)}", flush=True)
    
    # Calculate Unit vectors
    ra_rad = np.radians(cat['ra'].values)
    dec_rad = np.radians(cat['dec'].values)
    
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    vectors = np.vstack([x, y, z]).T
    
    star_ids = cat['source_id'].values
    
    print("Building pair-wise distances... (This may take a while depending on star count)")
    
    # We only want pairs where the angle is within max/min FOV.
    # To do this fast, we can use scipy KDTree or just chunked dot products since it's an offline script.
    from scipy.spatial import cKDTree
    tree = cKDTree(vectors)
    
    # Euclidean distance equivalent for sphere
    # cos(theta) = 1 - (d^2) / 2
    # d = sqrt(2 - 2*cos(theta))
    max_d = np.sqrt(2 - 2 * np.cos(np.radians(max_fov_deg)))
    min_d = np.sqrt(2 - 2 * np.cos(np.radians(min_fov_deg)))
    
    pairs = tree.query_pairs(r=max_d)
    
    pair_dists = []
    pair_ids = []
    
    # We want to store COSINE of the angle to avoid arccos in realtime
    # cos(theta) = dot(v1, v2)
    # distance matching is perfectly monotonic with cos(theta)
    # Actually wait! If we store dot products, the value decreases as angle increases. 
    # Let's just track Euclidean distances exactly, or angles exactly. 
    # Angles usually have linear distribution over solid angle pairs.
    # Euclidean distances are faster to compute. We will index on angular distance (degrees) for intuition.
    
    # Convert memory efficient structure
    pair_array = np.array(list(pairs), dtype=np.int32)
    
    # Vectorized dot product
    v1 = vectors[pair_array[:, 0]]
    v2 = vectors[pair_array[:, 1]]
    dots = np.sum(v1 * v2, axis=1)
    
    # Fix precision issues
    dots = np.clip(dots, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    
    # Filter min_fov
    valid_mask = angles_deg >= min_fov_deg
    valid_pairs = pair_array[valid_mask]
    valid_angles = angles_deg[valid_mask]
    
    print(f"Total valid pairs within ({min_fov_deg} < theta < {max_fov_deg} deg): {len(valid_angles)}")
    
    print("Building K-Vector Index...")
    sorted_idx, sorted_angles, m, b, k_vec = generate_kvector(valid_angles)
    
    sorted_pairs = valid_pairs[sorted_idx]
    
    # Also save the catalog itself so the matcher doesn't need to load thousands of CSV files to find star positions!
    print("Saving to HDF5/NPZ bundle...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        catalog_ids=star_ids,
        catalog_vectors=vectors,
        catalog_mags=cat['g_mean_mag'].values,
        pair_indices=sorted_pairs,
        pair_angles=sorted_angles,
        k_m=np.array([m]),
        k_b=np.array([b]),
        k_vec=k_vec
    )
    print(f"Done! Guide Star Index saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GSC offline index")
    parser.add_argument("--gaia_dir", type=str, required=True, help="Path to Gaia healpix subset")
    parser.add_argument("--out", type=str, required=True, help="Path to output npz")
    parser.add_argument("--mag_limit", type=float, default=12.0, help="Magnitude limit for Guide Stars")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process")
    args = parser.parse_args()
    
    build_gsc(args.gaia_dir, args.out, max_mag=args.mag_limit, max_files=args.max_files)
