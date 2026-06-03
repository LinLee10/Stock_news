import importlib
import sys
import types


class _StubFeatureFlags:
    def get_all_flags(self):
        return {"enable_test_flag": True}


def test_config_import_does_not_print_secret_values(monkeypatch, capsys):
    sentinel_key = "sentinel-alpha-vantage-key"

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: False

    fake_feature_flags = types.ModuleType("config.feature_flags")
    fake_feature_flags.feature_flags = _StubFeatureFlags()

    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setitem(sys.modules, "config.feature_flags", fake_feature_flags)
    monkeypatch.delitem(sys.modules, "config.config", raising=False)
    monkeypatch.setenv("ALPHA_VANTAGE_KEY", sentinel_key)

    config_module = importlib.import_module("config.config")
    captured = capsys.readouterr()

    assert config_module.ALPHA_VANTAGE_KEY == sentinel_key
    assert captured.out == ""
    assert captured.err == ""
    assert sentinel_key not in captured.out
    assert sentinel_key not in captured.err
