import numpy as np
import itertools
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from fsglib.common.types import ObservedStar, MatchedStar

class TriangleMatcher:
    def __init__(self, gsc_path: str, angle_tol_deg: float = 0.05, max_stars: int = 15):
        self.gsc_path = gsc_path
        self.angle_tol_deg = angle_tol_deg
        self.max_stars = max_stars
        self._loaded = False
        
    def _load_gsc(self):
        if self._loaded:
            return
        # print(f"Loading Triangle Index from {self.gsc_path}...")
        data = np.load(self.gsc_path)
        self.cat_ids = data['catalog_ids']
        self.cat_vecs = data['catalog_vectors']
        self.pair_indices = data['pair_indices']
        self.pair_angles = data['pair_angles']
        self.k_m = data['k_m'][0]
        self.k_b = data['k_b'][0]
        self.k_vec = data['k_vec']
        self._loaded = True
        
    def query_kvector(self, angle_deg: float) -> np.ndarray:
        min_angle = max(0.0, angle_deg - self.angle_tol_deg)
        max_angle = angle_deg + self.angle_tol_deg
        
        n = len(self.k_vec) - 1
        
        # Buffer the k index to handle approximation errors
        k_min = int(np.floor(self.k_m * min_angle + self.k_b)) - 2
        k_max = int(np.ceil(self.k_m * max_angle + self.k_b)) + 2
        
        k_min = max(0, min(k_min, n))
        k_max = max(0, min(k_max, n))
        
        idx_start = self.k_vec[k_min]
        idx_end = self.k_vec[k_max]
        
        # Use searchsorted to get exact bounds within the buffered range
        sub_angles = self.pair_angles[idx_start:idx_end]
        exact_start = idx_start + np.searchsorted(sub_angles, min_angle, side='left')
        exact_end = idx_start + np.searchsorted(sub_angles, max_angle, side='right')
        
        if exact_start >= exact_end:
            return np.array([], dtype=np.int32)
            
        return self.pair_indices[exact_start:exact_end]

    def match(self, observed: List[ObservedStar]) -> List[MatchedStar]:
        self._load_gsc()
        
        # 1. Sort observed by flux (highest first), take top max_stars
        if hasattr(observed[0], 'flux') and observed[0].flux is not None:
            sorted_obs = sorted(observed, key=lambda s: s.flux, reverse=True)
        else:
            # Fallback to positional sorting or arbitrary if flux is missing
            sorted_obs = observed
            
        top_obs = sorted_obs[:self.max_stars]
        n_obs = len(top_obs)
        if n_obs < 3:
            return []
            
        # 2. Precompute all pair angles and query database
        obs_pairs = {}
        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                # Cosine distance
                dot = np.dot(top_obs[i].los_body, top_obs[j].los_body)
                dot = np.clip(dot, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot))
                
                cat_pairs = self.query_kvector(angle_deg)
                obs_pairs[(i, j)] = cat_pairs
                
        # 3. Find matching triangles
        # A triangle is formed by 3 observed stars (i, j, k)
        # We need their 3 edge combinations: (i,j), (i,k), (j,k)
        
        best_votes = defaultdict(int)
        
        for i, j, k in itertools.combinations(range(n_obs), 3):
            pairs_ij = obs_pairs[(i, j)]
            pairs_ik = obs_pairs[(i, k)]
            pairs_jk = obs_pairs[(j, k)]
            
            if len(pairs_ij) == 0 or len(pairs_ik) == 0 or len(pairs_jk) == 0:
                continue
                
            # Which catalog stars are in pairs_ij?
            # pairs_ij is Nx2 array of catalog star indices
            # we want to find a triplet of catalog stars (A, B, C) that match
            # To do this fast, compute the intersection of star IDs
            
            set_ij = set(pairs_ij.flatten())
            set_ik = set(pairs_ik.flatten())
            set_jk = set(pairs_jk.flatten())
            
            # Potential catalog stars forming this triangle must form a clique
            
            # Map observed edge -> set of Catalog edges
            edges_ij = set(map(tuple, pairs_ij)) | set(map(tuple, pairs_ij[:, ::-1]))
            edges_ik = set(map(tuple, pairs_ik)) | set(map(tuple, pairs_ik[:, ::-1]))
            edges_jk = set(map(tuple, pairs_jk)) | set(map(tuple, pairs_jk[:, ::-1]))
            
            # To avoid O(V^3), iterate over edges_ij, and look for intersection of neighbors
            # Build adjacency sets for ik and jk
            adj_ik = defaultdict(set)
            for u, v in edges_ik:
                adj_ik[u].add(v)
            
            adj_jk = defaultdict(set)
            for u, v in edges_jk:
                adj_jk[u].add(v)
            
            for A, B in edges_ij:
                if A == B: continue
                # Possible C's are nodes connected to A in ik, AND connected to B in jk
                possible_C = adj_ik[A].intersection(adj_jk[B])
                
                for C in possible_C:
                    if C == A or C == B: continue
                    
                    # Geometric chirality check to avoid mirror flipped matches
                    obs_cross = np.cross(top_obs[i].los_body, top_obs[j].los_body)
                    obs_vol = np.dot(obs_cross, top_obs[k].los_body)
                    
                    cat_cross = np.cross(self.cat_vecs[A], self.cat_vecs[B])
                    cat_vol = np.dot(cat_cross, self.cat_vecs[C])
                    
                    if obs_vol * cat_vol > 0:
                        # Valid match found!
                        best_votes[(top_obs[i].source_id, A)] += 1
                        best_votes[(top_obs[j].source_id, B)] += 1
                        best_votes[(top_obs[k].source_id, C)] += 1

        if not best_votes:
            return []
            
        # 4. Resolve matches
        # For each observed source_id, pick the catalog index with highest votes
        obs_to_cat = {}
        for (obs_sid, cat_idx), votes in best_votes.items():
            if obs_sid not in obs_to_cat or votes > obs_to_cat[obs_sid][1]:
                obs_to_cat[obs_sid] = (cat_idx, votes)
                
        # Optional: Require a minimum number of votes (e.g. 2 triangles)
        # But even 1 triangle (1 vote) is 3 matched stars. 
        
        matched = []
        for obs in top_obs:
            if obs.source_id in obs_to_cat:
                cat_idx, votes = obs_to_cat[obs.source_id]
                # We could filter by minimum votes if we want strictness
                
                matched.append(MatchedStar(
                    detector_id=obs.detector_id,
                    source_id=obs.source_id,
                    catalog_id=self.cat_ids[cat_idx],
                    los_body=obs.los_body,
                    los_inertial=self.cat_vecs[cat_idx],
                    residual_arcsec=None,
                    weight=1.0,
                    match_score=float(votes),
                    flags={"triangle_votes": votes}
                ))
                
        return matched
