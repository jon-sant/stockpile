"""Tests for fetch.py's persistent-cache-aware fetch helper.

Exercises `_fetch_chain_with_cache` directly (the piece that decides
cache-hit vs. live-fetch vs. force_live) rather than the full
`fetch_and_enrich_cached`/`fetch_position_cached` wrappers, since
`_enrich()` does real earnings/realized-vol lookups that would need
network — out of scope here, this file only proves the caching
contract. No test touches real Yahoo/Selenium; `chain.fetch_chain` is
always monkeypatched.
"""

import pandas as pd
import pytest

from options_scanner import fetch


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_CHAIN_CACHE_DB", str(tmp_path / "c.db"))


def _chain(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["call"] * n,
        "strike": [100.0 + i for i in range(n)],
        "dte": [30] * n,
    })


def test_second_call_served_from_cache(monkeypatch):
    calls = []

    def _fake_fetch_chain(ticker, opt_type, min_dte, max_dte, provider,
                          schwab_config=None, moomoo_config=None):
        calls.append(ticker)
        return _chain()

    monkeypatch.setattr("options_scanner.chain.fetch_chain", _fake_fetch_chain)

    df1, _, from_cache1 = fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "yahoo-headless", None, None, True, force_live=False)
    df2, _, from_cache2 = fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "yahoo-headless", None, None, True, force_live=False)

    assert len(calls) == 1  # second call hit the persistent cache
    assert from_cache1 is False
    assert from_cache2 is True
    pd.testing.assert_frame_equal(df1, df2)


def test_force_live_always_bypasses_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "options_scanner.chain.fetch_chain",
        lambda *a, **k: calls.append(1) or _chain(),
    )

    fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "yahoo-headless", None, None, True, force_live=False)
    fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "yahoo-headless", None, None, True, force_live=True)

    assert len(calls) == 2  # force_live bypassed the now-populated cache


def test_non_headless_provider_never_uses_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "options_scanner.chain.fetch_chain",
        lambda *a, **k: calls.append(1) or _chain(),
    )

    fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "schwab", None, None, True, force_live=False)
    fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "schwab", None, None, True, force_live=False)

    assert len(calls) == 2  # schwab fetches never check/populate chain_cache


def test_only_yahoo_headless_fetches_populate_cache(monkeypatch):
    monkeypatch.setattr("options_scanner.chain.fetch_chain",
                        lambda *a, **k: _chain())

    fetch._fetch_chain_with_cache(
        "AMD", "both", 30, 90, "schwab", None, None, True, force_live=False)

    from options_scanner import chain_cache
    assert not chain_cache.has_fresh_snapshot("AMD", 30, 90, "schwab")
    assert not chain_cache.has_fresh_snapshot("AMD", 30, 90, "yahoo-headless")
