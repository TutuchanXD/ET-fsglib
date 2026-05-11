from fsglib.match.cache import get_match_cache


def test_get_match_cache_skips_non_pyramid_algorithms():
    models = {}

    cache = get_match_cache(models, {"match": {"algorithm": "predicted_position"}})

    assert cache is None
    assert "match_cache" not in models


def test_get_match_cache_lazily_creates_cache_for_pyramid_algorithms():
    from fsglib.match.pyramid import LocalPyramidCache

    models = {}
    cfg = {"match": {"algorithm": "local_pyramid"}}

    cache = get_match_cache(models, cfg)
    again = get_match_cache(models, cfg)

    assert isinstance(cache, LocalPyramidCache)
    assert again is cache
    assert models["match_cache"] is cache
