from fsglib.ephemeris.catalog import HealpixCatalogProvider
from fsglib.ephemeris.projector import RealOpticalProjector
from fsglib.preprocess.calibration import load_calibration_products


def build_models(cfg: dict) -> dict:
    return {
        "catalog": HealpixCatalogProvider(cfg),
        "projector": RealOpticalProjector(cfg),
        "calib": load_calibration_products(cfg),
    }
