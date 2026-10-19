from riskcausal.config import load_config


def test_dev_profile_is_small_prototype():
    cfg = load_config("dev")
    assert 5 <= len(cfg.etfs) <= 6
    assert len(cfg.fred_series) <= 4
    assert set(cfg.graph_nodes.etfs) <= set(cfg.etfs)


def test_full_profile_expands_universe():
    cfg = load_config("full")
    assert len(cfg.etfs) >= 20


def test_config_hash_changes_with_settings():
    a = load_config("dev")
    b = load_config("dev", overrides={"graph": {"window": 126}})
    assert a.hash() != b.hash()
    assert a.hash() == load_config("dev").hash()
