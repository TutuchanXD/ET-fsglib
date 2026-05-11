from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fsglib.match.lost_in_space import build_lis_index_from_arrays, save_lis_index

ARCSEC_PER_RAD = 206264.80624709636


def _radec_to_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra_rad = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec_rad = np.radians(np.asarray(dec_deg, dtype=np.float64))
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.vstack([x, y, z]).T


def _drop_fainter_close_neighbors(df: pd.DataFrame, isolation_radius_arcsec: float) -> pd.DataFrame:
    if isolation_radius_arcsec <= 0.0 or len(df) < 2:
        return df

    vectors = _radec_to_vectors(df["ra"].to_numpy(), df["dec"].to_numpy())
    keep = np.ones(len(df), dtype=bool)
    cos_limit = float(np.cos(float(isolation_radius_arcsec) / ARCSEC_PER_RAD))
    ordered_indices = sorted(
        range(len(df)),
        key=lambda idx: (float(df.iloc[idx]["g_mean_mag"]), int(df.iloc[idx]["source_id"])),
    )

    for pos, idx in enumerate(ordered_indices):
        if not keep[idx]:
            continue
        for other in ordered_indices[pos + 1:]:
            if not keep[other]:
                continue
            if float(np.dot(vectors[idx], vectors[other])) >= cos_limit:
                keep[other] = False

    return df.loc[keep].reset_index(drop=True)


def _load_gaia_csv_rows(gaia_root: Path, mag_limit: float, max_files: int | None) -> pd.DataFrame:
    files = sorted(gaia_root.glob("healpix_n05_nested_*.csv"))
    if max_files is not None:
        files = files[:max_files]

    rows = []
    for path in files:
        frame = pd.read_csv(path, usecols=["source_id", "ra", "dec", "g_mean_mag"])
        valid = frame["g_mean_mag"].notna() & (frame["g_mean_mag"] <= float(mag_limit))
        filtered = frame.loc[valid, ["source_id", "ra", "dec", "g_mean_mag"]]
        if not filtered.empty:
            rows.append(filtered)

    if not rows:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "g_mean_mag"])
    return pd.concat(rows, ignore_index=True)


def build_lis_index_from_gaia_csv(
    *,
    gaia_root: str | Path,
    mag_limit: float,
    epoch: float,
    bandpass: str,
    isolation_radius_arcsec: float,
    max_files: int | None = None,
):
    root = Path(gaia_root).expanduser().resolve()
    filtered = _load_gaia_csv_rows(root, mag_limit, max_files)
    isolated = _drop_fainter_close_neighbors(filtered, isolation_radius_arcsec)
    config_snapshot = {
        "gaia_root": str(root),
        "mag_limit": float(mag_limit),
        "epoch": float(epoch),
        "bandpass": bandpass,
        "isolation_radius_arcsec": float(isolation_radius_arcsec),
        "max_files": max_files,
    }
    config_snapshot_json = json.dumps(config_snapshot, sort_keys=True)
    vectors = _radec_to_vectors(isolated["ra"].to_numpy(), isolated["dec"].to_numpy())
    return build_lis_index_from_arrays(
        catalog_ids=isolated["source_id"].to_numpy(dtype=np.int64),
        vectors=vectors,
        magnitudes=isolated["g_mean_mag"].to_numpy(dtype=np.float64),
        config_snapshot={
            "gaia_root": str(root),
            "epoch": float(epoch),
            "bandpass": bandpass,
            "filters": {
                "mag_limit": float(mag_limit),
                "isolation_radius_arcsec": float(isolation_radius_arcsec),
                "neighbor_policy": "drop_fainter",
            },
            "config_snapshot_json": config_snapshot_json,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline lost-in-space guide-star index")
    parser.add_argument("--gaia-root", required=True, help="Root directory containing Gaia HEALPix CSV partitions")
    parser.add_argument("--out", required=True, help="Output .lis_index.npz path")
    parser.add_argument("--mag-limit", type=float, required=True, help="Faint-end Gaia G magnitude limit")
    parser.add_argument("--epoch", type=float, required=True, help="Catalog epoch recorded in index metadata")
    parser.add_argument("--bandpass", default="gaia_g", help="Magnitude bandpass recorded in index metadata")
    parser.add_argument(
        "--isolation-radius-arcsec",
        type=float,
        default=0.0,
        help="Drop fainter stars within this angular radius",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on input partitions for fixtures")
    args = parser.parse_args(argv)

    index = build_lis_index_from_gaia_csv(
        gaia_root=args.gaia_root,
        mag_limit=args.mag_limit,
        epoch=args.epoch,
        bandpass=args.bandpass,
        isolation_radius_arcsec=args.isolation_radius_arcsec,
        max_files=args.max_files,
    )
    save_lis_index(index, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
