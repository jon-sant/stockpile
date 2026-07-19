"""Scan-provenance stamps: data source + scan timestamp shown on every
chart and below every table so the context survives screenshots, HTML
exports, and Reddit reposts.

Also home to the canonical provider identity constants
(PROVIDER_LABELS, PROVIDER_COLORS) since they're consumed by the
stamp helpers and by the top-bar pill / sidebar source chip — one
source of truth keeps the green/blue convention in sync across the
chart subtitles, table captions, buttons, and pill chips.

Convention: stamp reads `scan_provider` (snapshotted at scan time),
NOT the live data-source dropdown. That way the stamp reflects what
was actually used to fetch the displayed data, even after the user
flips the dropdown post-scan.
"""

from __future__ import annotations

import streamlit as st


PROVIDER_LABELS = {
    "yahoo":          "Yahoo Finance",
    "yahoo-headless": "Yahoo (headless)",
    "schwab":         "Schwab",
    "moomoo":         "Moomoo",
}
PROVIDER_COLORS = {
    "yahoo":          "#16a34a",   # green
    "yahoo-headless": "#0d9488",   # teal
    "schwab":         "#2563eb",   # blue
    "moomoo":         "#f97316",   # orange
}


def tz_abbr(ts) -> str:
    """3–4 char timezone abbreviation that works across platforms.

    Python's strftime('%Z') gives the full name on Windows ('Eastern
    Daylight Time') but the short form on POSIX ('EDT'). Normalize by
    taking the uppercase initials when the name is long.
    """
    name = ts.tzname() or ""
    if not name:
        return ""
    if len(name) <= 4:
        return name
    return "".join(w[0] for w in name.split() if w[:1].isupper())[:4]


def scan_stamp_text() -> str:
    """Format like 'Schwab · 2026-05-16 14:32 EDT · Per-expiry'.

    Reads `scan_provider` and `scan_surface_label` (both snapshotted at
    scan time) so the stamp reflects what was actually used to fetch and
    fit the displayed data, even after the user changes the dropdowns.

    When `scan_from_cache` is set (the data came from a same-day
    background scan — see background_scan.py / chain_cache.py — rather
    than a fetch made just now), a `(background)` qualifier is appended
    so the stamp doesn't imply a live scan just happened.
    """
    ts = st.session_state.get("scan_ts")
    if not ts:
        return ""
    provider = st.session_state.get("scan_provider", "yahoo")
    label = PROVIDER_LABELS.get(provider, provider)
    stamp = f"{label} · {ts.strftime('%Y-%m-%d %H:%M')} {tz_abbr(ts)}".rstrip()
    if st.session_state.get("scan_from_cache"):
        stamp += " (background)"
    surface = st.session_state.get("scan_surface_label", "")
    if surface:
        stamp += f" · {surface}"
    return stamp


def scan_stamp_color() -> str:
    """Hex color for the stamp text, based on the provider at scan time."""
    provider = st.session_state.get("scan_provider", "yahoo")
    return PROVIDER_COLORS.get(provider, "#94a3b8")


def stamp_caption() -> None:
    """Render the scan stamp as a colored caption below a table."""
    text = scan_stamp_text()
    if not text:
        return
    color = scan_stamp_color()
    st.markdown(
        f'<div style="color:{color}; font-size:0.85rem; '
        f'margin-top:-4px;">{text}</div>',
        unsafe_allow_html=True,
    )
