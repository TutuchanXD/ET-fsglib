from fsglib.ephemeris.catalog import HealpixCatalogProvider
from fsglib.ephemeris.projector import RealOpticalProjector

def build_models(cfg: dict) -> dict:
    return {
        "catalog": HealpixCatalogProvider(cfg),
        "projector": RealOpticalProjector(cfg),
        "calib": {}
    }
