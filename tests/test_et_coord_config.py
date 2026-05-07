import sys
import types


def test_load_et_coord_uses_named_config_factory(monkeypatch, tmp_path):
    from fsglib.pipeline.run_guide_init import _load_et_coord

    calls = []

    class DummyConfig:
        @classmethod
        def microlens_guide_only(cls):
            return "microlens-guide-only-config"

    class DummyTransformer:
        def __init__(self, registry):
            self.registry = registry

    class DummyCatalog:
        def __init__(self, root):
            self.root = root

    def load_registry(path, config=None):
        calls.append((str(path), config))
        return {"path": str(path), "config": config}

    fake_et_coord = types.SimpleNamespace(
        ETCoordConfig=DummyConfig,
        GaiaCatalog=DummyCatalog,
        GaiaSourceFilter=object,
        Transformer=DummyTransformer,
        load_registry=load_registry,
    )
    monkeypatch.setitem(sys.modules, "et_coord", fake_et_coord)

    registry, transformer, _, _ = _load_et_coord(
        {
            "et_coord": {
                "src_dir": str(tmp_path),
                "data_dir": str(tmp_path / "data_microlens"),
                "gaia_root_dir": str(tmp_path / "gaia"),
                "config_factory": "microlens_guide_only",
            }
        }
    )

    assert registry["config"] == "microlens-guide-only-config"
    assert transformer.registry is registry
    assert calls == [
        (str((tmp_path / "data_microlens").resolve()), "microlens-guide-only-config")
    ]
