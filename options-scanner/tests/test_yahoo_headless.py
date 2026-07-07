"""Tests for the pure HTML-parsing half of yahoo_headless.py.

The Selenium plumbing (driver pool, page loads) is exercised manually;
these tests pin down the parsers it feeds — spot extraction, expiry
discovery, table location/cleaning, and canonical row building — using
small synthetic page fragments.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from options_scanner.yahoo_headless import (
    chain_matches_expiration,
    extract_expiration_timestamps,
    extract_spot,
    parse_chain_tables,
    rows_from_tables,
)


def _ts_days_out(days: int) -> int:
    d = date.today() + timedelta(days=days)
    return int(datetime(d.year, d.month, d.day,
                        tzinfo=timezone.utc).timestamp())


def _chain_table_html(contract_prefix: str, side: str,
                      strike: str = "150.00", bid: str = "2.10",
                      ask: str = "2.30", iv: str = "28.45%",
                      volume: str = "1,234", oi: str = "5,678") -> str:
    letter = "C" if side == "call" else "P"
    return f"""
    <table>
      <thead><tr>
        <th>Contract Name</th><th>Last Trade Date</th><th>Strike</th>
        <th>Last Price</th><th>Bid</th><th>Ask</th><th>Change</th>
        <th>% Change</th><th>Volume</th><th>Open Interest</th>
        <th>Implied Volatility</th>
      </tr></thead>
      <tbody><tr>
        <td>{contract_prefix}260918{letter}00150000</td>
        <td>7/2/2026 3:59 PM EDT</td><td>{strike}</td><td>2.20</td>
        <td>{bid}</td><td>{ask}</td><td>+0.10</td><td>+4.76%</td>
        <td>{volume}</td><td>{oi}</td><td>{iv}</td>
      </tr></tbody>
    </table>
    """


PAGE = f"""
<html><body>
  <fin-streamer data-symbol="AAPL" data-field="regularMarketPrice"
      data-value="212.44">212.44</fin-streamer>
  <a href="/quote/AAPL/options?date={_ts_days_out(45)}">exp</a>
  <a href="/quote/AAPL/options?date={_ts_days_out(80)}">exp</a>
  <a href="/quote/AAPL/options?date={_ts_days_out(400)}">exp</a>
  {_chain_table_html("AAPL", "call")}
  {_chain_table_html("AAPL", "put", bid="1.05", ask="1.15", iv="31.20%")}
</body></html>
"""


# ── extract_spot ─────────────────────────────────────────────────────────

def test_spot_from_fin_streamer_data_value():
    assert extract_spot(PAGE, "AAPL") == pytest.approx(212.44)


def test_spot_from_fin_streamer_text_content():
    html = ('<fin-streamer data-symbol="TSLA" '
            'data-field="regularMarketPrice">1,024.50</fin-streamer>')
    assert extract_spot(html, "TSLA") == pytest.approx(1024.50)


def test_spot_from_qsp_price_testid():
    html = '<span data-testid="qsp-price" class="x">98.76</span>'
    assert extract_spot(html, "AMD") == pytest.approx(98.76)


def test_spot_from_after_hours_text():
    html = "212.44 +1.30 (+0.62%) At close: 4:00PM EDT"
    assert extract_spot(html, "AAPL") == pytest.approx(212.44)


def test_spot_missing_returns_none():
    assert extract_spot("<html><body>nothing here</body></html>",
                        "AAPL") is None


# ── extract_expiration_timestamps ────────────────────────────────────────

def test_expirations_from_date_links():
    got = extract_expiration_timestamps(PAGE)
    assert got == sorted([_ts_days_out(45), _ts_days_out(80),
                          _ts_days_out(400)])


def test_expirations_from_json_array_and_options():
    ts1, ts2 = _ts_days_out(30), _ts_days_out(60)
    html = (f'"expirationDates":[{ts1},{ts2}]'
            f'<option value="{ts1}">Jul</option>')
    assert extract_expiration_timestamps(html) == [ts1, ts2]


def test_expirations_reject_implausible_timestamps():
    html = 'options?date=100 data-value="9999999999"'
    assert extract_expiration_timestamps(html) == []


# ── parse_chain_tables ───────────────────────────────────────────────────

def test_parse_splits_calls_and_puts_by_contract_name():
    chain = parse_chain_tables(PAGE)
    assert chain["calls"] is not None and chain["puts"] is not None
    call = chain["calls"].iloc[0]
    assert call["Strike"] == pytest.approx(150.0)
    assert call["Bid"] == pytest.approx(2.10)
    assert call["Volume"] == pytest.approx(1234)
    assert call["Open Interest"] == pytest.approx(5678)
    # IV parsed from "28.45%" into a ratio
    assert call["Implied Volatility"] == pytest.approx(0.2845)
    assert chain["puts"].iloc[0]["Implied Volatility"] == pytest.approx(0.312)


def test_parse_positional_fallback_without_contract_name():
    table = """
    <table><thead><tr><th>Strike</th><th>Bid</th><th>Ask</th>
    <th>Implied Volatility</th></tr></thead>
    <tbody><tr><td>100</td><td>1.00</td><td>1.20</td><td>25.00%</td></tr>
    </tbody></table>
    """
    chain = parse_chain_tables(f"<html>{table}{table}</html>")
    assert chain["calls"] is not None and chain["puts"] is not None


def test_parse_drops_rows_missing_strike_or_iv():
    html = _chain_table_html("AAPL", "call", iv="-")
    chain = parse_chain_tables(html)
    assert chain["calls"] is not None and chain["calls"].empty


def test_parse_no_tables():
    chain = parse_chain_tables("<html><body>maintenance</body></html>")
    assert chain == {"calls": None, "puts": None}


def test_parse_concats_split_side_tables_and_dedupes():
    # A side split across two tables (plus one duplicated row from a
    # sticky-header clone) must concatenate without double-counting.
    t1 = _chain_table_html("AAPL", "call", strike="150.00")
    t2 = _chain_table_html("AAPL", "call", strike="155.00").replace(
        "AAPL260918C00150000", "AAPL260918C00155000")
    chain = parse_chain_tables(f"<html>{t1}{t2}{t1}</html>")
    assert chain["calls"] is not None
    assert sorted(chain["calls"]["Strike"]) == [150.0, 155.0]
    assert chain["puts"] is None


def test_parse_second_same_side_table_not_dropped():
    # Regression: the old first-match-wins logic silently discarded a
    # second table that detected as the same side.
    t1 = _chain_table_html("AAPL", "put", strike="140.00").replace(
        "AAPL260918P00150000", "AAPL260918P00140000")
    t2 = _chain_table_html("AAPL", "put", strike="145.00").replace(
        "AAPL260918P00150000", "AAPL260918P00145000")
    chain = parse_chain_tables(f"<html>{t1}{t2}</html>")
    assert chain["puts"] is not None
    assert sorted(chain["puts"]["Strike"]) == [140.0, 145.0]


# ── chain_matches_expiration ─────────────────────────────────────────────

def test_expiration_match_accepts_right_date():
    chain = parse_chain_tables(PAGE)  # contracts dated 260918
    assert chain_matches_expiration(chain, "2026-09-18")


def test_expiration_match_rejects_redirected_page():
    chain = parse_chain_tables(PAGE)
    assert not chain_matches_expiration(chain, "2026-10-16")


def test_expiration_match_trusts_pages_without_contract_names():
    table = """
    <table><thead><tr><th>Strike</th><th>Bid</th><th>Ask</th>
    <th>Implied Volatility</th></tr></thead>
    <tbody><tr><td>100</td><td>1.00</td><td>1.20</td><td>25.00%</td></tr>
    </tbody></table>
    """
    chain = parse_chain_tables(f"<html>{table}{table}</html>")
    assert chain_matches_expiration(chain, "2026-10-16")


# ── rows_from_tables ─────────────────────────────────────────────────────

def _one_expiration_chain():
    return parse_chain_tables(PAGE)


def test_rows_canonical_schema_and_greeks():
    rows = rows_from_tables(_one_expiration_chain(), spot=212.44,
                            exp_str="2026-09-18", dte=74, opt_type="both")
    assert {r["type"] for r in rows} == {"call", "put"}
    call = next(r for r in rows if r["type"] == "call")
    assert call["strike"] == pytest.approx(150.0)
    assert call["expiration"] == "2026-09-18"
    assert call["dte"] == 74
    assert call["mid"] == pytest.approx((2.10 + 2.30) / 2)
    assert call["iv"] == pytest.approx(0.2845)
    assert 0.0 < call["delta"] <= 1.0   # deep ITM call → delta near 1
    put = next(r for r in rows if r["type"] == "put")
    assert -1.0 <= put["delta"] < 0.0
    # freshness column populated from Last Trade Date
    assert call["last_trade_days"] >= 0


def test_rows_last_trade_from_tz_suffixed_header():
    # Current Yahoo layout: zone in the header, not the cells.
    html = _chain_table_html("AAPL", "call").replace(
        "<th>Last Trade Date</th>", "<th>Last Trade Date (EDT)</th>"
    ).replace("7/2/2026 3:59 PM EDT", "7/2/2026 3:59 PM")
    rows = rows_from_tables(parse_chain_tables(html), spot=212.44,
                            exp_str="2026-09-18", dte=74, opt_type="both")
    assert rows and rows[0]["last_trade_days"] >= 0


def test_rows_respects_opt_type_filter():
    rows = rows_from_tables(_one_expiration_chain(), spot=212.44,
                            exp_str="2026-09-18", dte=74, opt_type="puts")
    assert rows and all(r["type"] == "put" for r in rows)


def test_rows_keeps_last_priced_contracts_when_quotes_zeroed():
    # Overnight Yahoo zeroes bid/ask before the chain goes dark; rows
    # must survive priced off the last trade (mid = Last Price).
    chain = parse_chain_tables(
        _chain_table_html("AAPL", "call", bid="0.00", ask="0.00"))
    rows = rows_from_tables(chain, spot=212.44, exp_str="2026-09-18",
                            dte=74, opt_type="both")
    assert len(rows) == 1
    assert rows[0]["mid"] == pytest.approx(2.20)  # the Last Price cell


def test_rows_drops_contracts_with_no_price_at_all():
    html = _chain_table_html("AAPL", "call", bid="0.00", ask="0.00"
                             ).replace("<td>2.20</td>", "<td>0.00</td>")
    rows = rows_from_tables(parse_chain_tables(html), spot=212.44,
                            exp_str="2026-09-18", dte=74, opt_type="both")
    assert rows == []


def test_parse_placeholder_page_yields_empty_tables():
    # Yahoo's overnight "no data" rendering: a table whose only row is
    # the placeholder sentence in every cell.
    cells = "".join(f"<td>There are no calls.</td>" for _ in range(4))
    html = ("<table><thead><tr><th>Contract Name</th><th>Strike</th>"
            "<th>Bid</th><th>Implied Volatility</th></tr></thead>"
            f"<tbody><tr>{cells}</tr></tbody></table>")
    chain = parse_chain_tables(html)
    assert chain["calls"] is not None and chain["calls"].empty
