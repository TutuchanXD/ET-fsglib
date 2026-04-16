import os
import numpy as np
import pandas as pd
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy import units as astropy_units

from fsglib.ephemeris.types import CatalogStar

class HealpixCatalogProvider:
    """
    Catalog provider that loads Gaia DR3 stars from nested HEALPix CSV files.
    """
    def __init__(self, cfg: dict):
        self.root_dir = cfg["ephemeris"]["gaia_root_dir"]
        self.mag_limit = float(cfg["ephemeris"]["mag_limit"])
        
        # As given by user: Npix = 12 * NSIDE^2 = 12 * 32^2 = 12288 -> NSIDE=32
        self.hp = HEALPix(nside=32, order='nested', frame='icrs')
        
    def query_region(
        self,
        boresight_vec: np.ndarray,
        radius_deg: float,
        mag_limit: float | None = None,
    ) -> list[CatalogStar]:
        """
        Queries the catalog for stars within a given radius around the boresight vector.
        """
        if boresight_vec is None:
            return []

        # Convert vector to ra, dec
        # vector is in ICRS
        norm = np.linalg.norm(boresight_vec)
        v = boresight_vec / norm
        mag_cut = self.mag_limit if mag_limit is None else float(mag_limit)
        
        dec_rad = np.arcsin(v[2])
        ra_rad = np.arctan2(v[1], v[0])
        
        if ra_rad < 0:
            ra_rad += 2 * np.pi
            
        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)
        
        # Center coordinate
        center = SkyCoord(ra=ra_deg * astropy_units.deg, dec=dec_rad * astropy_units.rad)
        
        # Find intersecting healpix pixels
        # cone_search_skycoord returns pixel indices inside the cone
        pixels = self.hp.cone_search_skycoord(center, radius_deg * astropy_units.deg)
        
        all_stars = []
        
        for p in pixels:
            # File format is healpix_n05_nested_xxxxx.csv -> 5 digits zero-padded
            file_path = os.path.join(self.root_dir, f"healpix_n05_nested_{p:05d}.csv")
            
            if not os.path.exists(file_path):
                # We might not have all pixels downloaded locally, silently skip missing
                continue
                
            try:
                # Based on the user's `head` output: 
                # source_id,ra,dec,g_mean_mag,bp_mean_mag,rp_mean_mag,pmra,pmdec,ref_epoch,parallax
                df = pd.read_csv(file_path)
                
                # Filter by magnitude
                # Handle potential NaN magnitudes
                valid_mask = df['g_mean_mag'].notna() & (df['g_mean_mag'] <= mag_cut)
                df_filtered = df.loc[valid_mask]
                
                for _, row in df_filtered.iterrows():
                    star_ra = float(row['ra'])
                    star_dec = float(row['dec'])
                    star_coord = SkyCoord(
                        ra=star_ra * astropy_units.deg,
                        dec=star_dec * astropy_units.deg,
                    )
                    if center.separation(star_coord).deg > radius_deg:
                        continue
                    
                    # Handling NaNs for astrometric params
                    pmra = float(row['pmra']) if pd.notna(row['pmra']) else 0.0
                    pmdec = float(row['pmdec']) if pd.notna(row['pmdec']) else 0.0
                    plx = float(row['parallax']) if pd.notna(row['parallax']) else 0.0
                    
                    bp_rp = 0.0
                    if pd.notna(row['bp_mean_mag']) and pd.notna(row['rp_mean_mag']):
                        bp_rp = float(row['bp_mean_mag'] - row['rp_mean_mag'])

                    star = CatalogStar(
                        catalog_id=int(row['source_id']),
                        ra_deg=star_ra,
                        dec_deg=star_dec,
                        pm_ra_mas_per_yr=pmra,
                        pm_dec_mas_per_yr=pmdec,
                        parallax_mas=plx,
                        rv_km_s=0.0,
                        mag_g=float(row['g_mean_mag']),
                        color_bp_rp=bp_rp,
                        meta={}
                    )
                    all_stars.append(star)
                    
            except Exception as e:
                print(f"Failed to load catalog partition {file_path}: {e}")
                
        return all_stars

    def query_tracking_targets(self, ctx) -> list[CatalogStar]:
        if ctx.boresight_inertial is None:
            return []

        radius_deg = float(ctx.catalog_cfg.get("tracking_catalog_radius_deg", 2.0))
        stars = self.query_region(
            boresight_vec=ctx.boresight_inertial,
            radius_deg=radius_deg,
            mag_limit=ctx.catalog_cfg.get("mag_limit"),
        )
        if ctx.track_catalog_ids:
            tracked = set(ctx.track_catalog_ids)
            tracked_stars = [star for star in stars if star.catalog_id in tracked]
            if tracked_stars:
                return tracked_stars
        return stars
