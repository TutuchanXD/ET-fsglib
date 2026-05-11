PYRAMID_CAPABLE_ALGORITHMS = {
    "local_pyramid",
    "predicted_position_with_pyramid_reacquire",
    "predicted_position_and_local_pyramid",
}


def uses_local_pyramid(cfg: dict) -> bool:
    algorithm = cfg.get("match", {}).get("algorithm", "predicted_position")
    return algorithm in PYRAMID_CAPABLE_ALGORITHMS


def get_match_cache(models: dict, cfg: dict):
    cache = models.get("match_cache")
    if cache is not None or not uses_local_pyramid(cfg):
        return cache

    from fsglib.match.pyramid import LocalPyramidCache

    cache = LocalPyramidCache()
    models["match_cache"] = cache
    return cache
