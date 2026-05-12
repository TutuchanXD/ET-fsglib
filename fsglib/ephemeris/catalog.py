import copy
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy import units as astropy_units

from fsglib.ephemeris.types import CatalogStar


_HEALPIX_NSIDE = 32
_HEALPIX_ORDER = "nested"
_MISSING_PARTITION_POLICIES = {"ignore", "warn", "error"}


def _optional_float(row, name: str) -> float | None:
    if name not in row.index or pd.isna(row[name]):
        return None
    return float(row[name])


def _first_optional_float(row, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _optional_float(row, name)
        if value is not None:
            return value
    return None


class HealpixCatalogProvider:
    """
    Catalog provider that loads Gaia DR3 stars from nested HEALPix CSV files.
    """
    def __init__(self, cfg: dict):
        ephemeris_cfg = cfg["ephemeris"]
        self.root_dir = str(ephemeris_cfg["gaia_root_dir"])
        self.mag_limit = float(ephemeris_cfg["mag_limit"])
        self.partition_cache_size = max(
            int(ephemeris_cfg.get("gaia_partition_cache_size", 8)),
            0,
        )
        self.missing_partition_policy = str(
            ephemeris_cfg.get("missing_partition_policy", "ignore")
        ).lower()
        if self.missing_partition_policy not in _MISSING_PARTITION_POLICIES:
            valid = ", ".join(sorted(_MISSING_PARTITION_POLICIES))
            raise ValueError(
                "ephemeris.missing_partition_policy must be one of "
                f"{valid}, got {self.missing_partition_policy!r}"
            )

        self._partition_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._last_query_stats: dict = {}

        # As given by user: Npix = 12 * NSIDE^2 = 12 * 32^2 = 12288 -> NSIDE=32
        self.hp = HEALPix(nside=_HEALPIX_NSIDE, order=_HEALPIX_ORDER, frame="icrs")

    @property
    def last_query_stats(self) -> dict:
        return copy.deepcopy(self._last_query_stats)

    def _partition_path(self, pixel: int) -> str:
        # File format is healpix_n05_nested_xxxxx.csv -> 5 digits zero-padded
        return os.path.join(self.root_dir, f"healpix_n05_nested_{pixel:05d}.csv")

    def _new_query_stats(
        self,
        *,
        radius_deg: float,
        mag_limit: float,
        candidate_pixels: list[int],
    ) -> dict:
        return {
            "catalog_provider": "HealpixCatalogProvider",
            "root_dir": self.root_dir,
            "healpix_nside": _HEALPIX_NSIDE,
            "healpix_order": _HEALPIX_ORDER,
            "radius_deg": float(radius_deg),
            "mag_limit": float(mag_limit),
            "missing_partition_policy": self.missing_partition_policy,
            "candidate_pixels": candidate_pixels,
            "loaded_pixels": [],
            "cache_hit_pixels": [],
            "cache_miss_pixels": [],
            "cache_evicted_pixels": [],
            "missing_pixels": [],
            "failed_pixels": [],
            "failed_partitions": [],
            "num_candidate_pixels": len(candidate_pixels),
            "num_loaded_partitions": 0,
            "num_cache_hits": 0,
            "num_cache_misses": 0,
            "num_cache_evictions": 0,
            "num_missing_partitions": 0,
            "num_failed_partitions": 0,
            "num_rows_loaded": 0,
            "num_rows_after_mag_filter": 0,
            "num_rows_inside_cone": 0,
            "num_stars_returned": 0,
        }

    def _finalize_query_stats(self, stats: dict, num_stars_returned: int) -> None:
        stats["num_loaded_partitions"] = len(stats["loaded_pixels"])
        stats["num_cache_hits"] = len(stats["cache_hit_pixels"])
        stats["num_cache_misses"] = len(stats["cache_miss_pixels"])
        stats["num_cache_evictions"] = len(stats["cache_evicted_pixels"])
        stats["num_missing_partitions"] = len(stats["missing_pixels"])
        stats["num_failed_partitions"] = len(stats["failed_pixels"])
        stats["num_stars_returned"] = int(num_stars_returned)

    def _record_missing_partition(self, pixel: int, file_path: str, stats: dict) -> None:
        stats["missing_pixels"].append(pixel)
        stats["num_missing_partitions"] = len(stats["missing_pixels"])
        message = f"missing Gaia catalog partition {file_path}"
        if self.missing_partition_policy == "error":
            raise FileNotFoundError(message)
        if self.missing_partition_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _record_failed_partition(
        self,
        pixel: int,
        file_path: str,
        error: Exception,
        stats: dict,
    ) -> None:
        stats["failed_pixels"].append(pixel)
        stats["failed_partitions"].append(
            {"pixel": pixel, "file_path": file_path, "error": str(error)}
        )
        stats["num_failed_partitions"] = len(stats["failed_pixels"])
        message = f"failed to load Gaia catalog partition {file_path}: {error}"
        if self.missing_partition_policy == "error":
            raise RuntimeError(message) from error
        if self.missing_partition_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _load_partition(
        self,
        pixel: int,
        stats: dict,
    ) -> tuple[pd.DataFrame | None, str]:
        file_path = self._partition_path(pixel)

        if pixel in self._partition_cache:
            self._partition_cache.move_to_end(pixel)
            stats["cache_hit_pixels"].append(pixel)
            return self._partition_cache[pixel], file_path

        if not os.path.exists(file_path):
            self._record_missing_partition(pixel, file_path, stats)
            return None, file_path

        stats["cache_miss_pixels"].append(pixel)
        try:
            df = pd.read_csv(file_path)
        except Exception as error:
            self._record_failed_partition(pixel, file_path, error, stats)
            return None, file_path

        stats["loaded_pixels"].append(pixel)
        if self.partition_cache_size > 0:
            self._partition_cache[pixel] = df
            self._partition_cache.move_to_end(pixel)
            while len(self._partition_cache) > self.partition_cache_size:
                evicted_pixel, _ = self._partition_cache.popitem(last=False)
                stats["cache_evicted_pixels"].append(evicted_pixel)
        return df, file_path

    def query_region(
        self,
        boresight_vec: np.ndarray,
        radius_deg: float,
        mag_limit: float | None = None,
    ) -> list[CatalogStar]:
        """
        Queries the catalog for stars within a given radius around the boresight vector.
        """
        mag_cut = self.mag_limit if mag_limit is None else float(mag_limit)
        if boresight_vec is None:
            stats = self._new_query_stats(
                radius_deg=radius_deg,
                mag_limit=mag_cut,
                candidate_pixels=[],
            )
            self._last_query_stats = stats
            return []

        # Convert vector to ra, dec
        # vector is in ICRS
        norm = np.linalg.norm(boresight_vec)
        v = boresight_vec / norm

        dec_rad = np.arcsin(v[2])
        ra_rad = np.arctan2(v[1], v[0])

        if ra_rad < 0:
            ra_rad += 2 * np.pi

        ra_deg = np.degrees(ra_rad)
        dec_deg = np.degrees(dec_rad)

        # Center coordinate
        center = SkyCoord(
            ra=ra_deg * astropy_units.deg,
            dec=dec_deg * astropy_units.deg,
        )

        # Find intersecting healpix pixels
        # cone_search_skycoord returns pixel indices inside the cone
        pixels = [
            int(pixel)
            for pixel in self.hp.cone_search_skycoord(
                center,
                radius_deg * astropy_units.deg,
            )
        ]
        stats = self._new_query_stats(
            radius_deg=radius_deg,
            mag_limit=mag_cut,
            candidate_pixels=pixels,
        )
        self._last_query_stats = stats

        all_stars = []

        for p in pixels:
            df, file_path = self._load_partition(p, stats)
            if df is None:
                continue

            try:
                # Based on the user's `head` output:
                # source_id,ra,dec,g_mean_mag,bp_mean_mag,rp_mean_mag,pmra,pmdec,ref_epoch,parallax
                stats["num_rows_loaded"] += len(df)

                # Filter by magnitude
                # Handle potential NaN magnitudes
                valid_mask = df["g_mean_mag"].notna() & (df["g_mean_mag"] <= mag_cut)
                df_filtered = df.loc[valid_mask]
                stats["num_rows_after_mag_filter"] += len(df_filtered)

                for _, row in df_filtered.iterrows():
                    star_ra = float(row["ra"])
                    star_dec = float(row["dec"])
                    star_coord = SkyCoord(
                        ra=star_ra * astropy_units.deg,
                        dec=star_dec * astropy_units.deg,
                    )
                    if center.separation(star_coord).deg > radius_deg:
                        continue

                    stats["num_rows_inside_cone"] += 1

                    # Handling NaNs for astrometric params
                    pmra = float(row["pmra"]) if pd.notna(row["pmra"]) else 0.0
                    pmdec = float(row["pmdec"]) if pd.notna(row["pmdec"]) else 0.0
                    plx = float(row["parallax"]) if pd.notna(row["parallax"]) else 0.0
                    ref_epoch = _optional_float(row, "ref_epoch")
                    rv_km_s = _first_optional_float(
                        row,
                        (
                            "radial_velocity",
                            "radial_velocity_km_s",
                            "rv_km_s",
                            "rv",
                        ),
                    )

                    bp_rp = 0.0
                    if pd.notna(row["bp_mean_mag"]) and pd.notna(row["rp_mean_mag"]):
                        bp_rp = float(row["bp_mean_mag"] - row["rp_mean_mag"])

                    star = CatalogStar(
                        catalog_id=int(row["source_id"]),
                        ra_deg=star_ra,
                        dec_deg=star_dec,
                        pm_ra_mas_per_yr=pmra,
                        pm_dec_mas_per_yr=pmdec,
                        parallax_mas=plx,
                        rv_km_s=rv_km_s,
                        mag_g=float(row["g_mean_mag"]),
                        color_bp_rp=bp_rp,
                        ref_epoch=ref_epoch,
                        meta={
                            "catalog_provider": "HealpixCatalogProvider",
                            "catalog_root_dir": self.root_dir,
                            "catalog_file": file_path,
                            "healpix_pixel": p,
                            "healpix_nside": _HEALPIX_NSIDE,
                            "healpix_order": _HEALPIX_ORDER,
                            "query_radius_deg": float(radius_deg),
                            "query_mag_limit": mag_cut,
                        },
                    )
                    all_stars.append(star)
            except Exception as error:
                self._record_failed_partition(p, file_path, error, stats)
                continue

        self._finalize_query_stats(stats, len(all_stars))
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
