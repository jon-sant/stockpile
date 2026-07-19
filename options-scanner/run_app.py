"""Streamlit web UI for the options scanner."""

import asyncio
import json
import sys
from datetime import datetime, timedelta

# Streamlit's internal async handling is incompatible with Windows's default
# ProactorEventLoop on Python 3.12+. Switch to the Selector policy before
# Streamlit starts its own loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from pathlib import Path

import streamlit as st

from options_scanner.ui_theme import (
    badge,
    disclaimer_chip,
    footer as ui_footer,
    inject_theme,
    metric_card,
    register_altair_theme,
    section_header,
)
from options_scanner.display.scan_stamp import PROVIDER_LABELS, PROVIDER_COLORS
from options_scanner.tabs.backtest import tab_backtest
from options_scanner.tabs.gex import tab_gex
from options_scanner.tabs.live_charts import tab_live_charts
from options_scanner.tabs.portfolio import tab_portfolio, tab_watchlist
from options_scanner.tabs.single import tab_single
from options_scanner.tabs.spreads import tab_directional, tab_neutral, tab_spreads
from options_scanner.tabs.trades import tab_trades

_FAVICON_PATH = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="Scanner",
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.exists() else "•",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Background scan scheduler ────────────────────────────────────────────────
# st.cache_resource caches by process, not by session/rerun — this factory
# runs exactly once per server process no matter how many browser tabs/
# reruns hit the app, which is what "start one background thread" needs.
# See options_scanner/background_scan.py for what the thread actually does.
@st.cache_resource(show_spinner=False)
def _background_scan_singleton():
    from options_scanner.background_scan import start_background_scan_once
    return start_background_scan_once()


_background_scan_singleton()


# Inject the global stylesheet and Altair theme as early as possible so
# every downstream widget renders in the redesigned visual language.
inject_theme()
register_altair_theme()


# ── Legacy theme switcher (kept for backward-compat session_state keys) ─────
# The new design system replaces the old four-way theme picker. We leave a
# no-op so any existing references / saved preferences don't crash.

THEMES: dict[str, None] = {"Default": None}


def _apply_theme(theme_name: str) -> None:  # noqa: ARG001 — preserved for compat
    """Compatibility shim: the new ui_theme.inject_theme() supersedes this."""
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

# Config must load first so data_source_choice is seeded into session_state
# before we compute the accent colors below.
from options_scanner.config import (
    load_config, get_provider,
    get_schwab_config as _get_schwab_cfg,
    get_moomoo_config as _get_moomoo_cfg,
)
_app_cfg = load_config()
_cfg_provider = get_provider(_app_cfg)
_cfg_schwab = _get_schwab_cfg(_app_cfg)
_cfg_moomoo = _get_moomoo_cfg(_app_cfg)
_schwab_configured = (
    bool(_cfg_schwab.get("app_key"))
    and not _cfg_schwab["app_key"].startswith("your-")
    and bool(_cfg_schwab.get("app_secret"))
    and not _cfg_schwab["app_secret"].startswith("your-")
)

# Schwab refresh-token countdown — shown in the data-source toggle label and
# (color-coded) the sidebar badge. Computed once per rerun; updates on any
# interaction (no auto-refresh — a day/hour readout doesn't need to-the-second
# accuracy, and an expired token also surfaces via the re-auth hint on scan).
_schwab_tok_label = None   # "6D" / "13H" / "46M" / "expired" / "re-auth"
_schwab_tok_color = None   # set => override sidebar badge (amber soon / red)
_schwab_tok_tip = None     # exact remaining, shown as hover text on the toggle
if _schwab_configured:
    from stocks_shared.schwab_live import token_remaining_seconds
    _secs = token_remaining_seconds(_cfg_schwab.get("token_file", ""))
    if _secs is None:
        _schwab_tok_label, _schwab_tok_color = "re-auth", "#ef4444"
        _schwab_tok_tip = "Schwab token not found - run schwab_auth.py"
    elif _secs <= 0:
        _schwab_tok_label, _schwab_tok_color = "expired", "#ef4444"
        _schwab_tok_tip = "Schwab token expired - run schwab_auth.py"
    else:
        if _secs >= 86400:
            _schwab_tok_label = f"{int(_secs // 86400)}D"
        elif _secs >= 3600:
            _schwab_tok_label = f"{int(_secs // 3600)}H"
        else:
            _schwab_tok_label = f"{max(int(_secs // 60), 1)}M"
        _schwab_tok_color = ("#ef4444" if _secs < 7200          # red < 2h
                             else "#f59e0b" if _secs < 86400     # amber < 1d
                             else None)
        _d, _r = divmod(int(_secs), 86400)
        _h, _r = divmod(_r, 3600)
        _m = _r // 60
        _exp = datetime.now().astimezone() + timedelta(seconds=_secs)
        _schwab_tok_tip = (f"Schwab token: {_d}d {_h}h {_m}m left - "
                           f"re-auth by {_exp:%a %b %d %I:%M %p}")

if "data_source_choice" not in st.session_state:
    if _cfg_provider == "schwab" and _schwab_configured:
        st.session_state["data_source_choice"] = "schwab"
    elif _cfg_provider == "moomoo":
        st.session_state["data_source_choice"] = "moomoo"
    elif _cfg_provider == "yahoo-headless":
        st.session_state["data_source_choice"] = "yahoo-headless"
    else:
        st.session_state["data_source_choice"] = "yahoo"

# Compute accent colors from the current data-source choice. Reads
# `data_source_choice` (the widget key) — NOT the effective
# `data_source` — so the color flips on the same rerun the dropdown
# changed, not one rerun later.
_BTN_COLORS = {
    "yahoo":  ("#16a34a", "#15803d"),   # normal, hover
    "yahoo-headless": ("#0d9488", "#0f766e"),
    "schwab": ("#2563eb", "#1d4ed8"),
    "moomoo": ("#f97316", "#ea6e0e"),
}
_btn_bg, _btn_hover = _BTN_COLORS.get(
    st.session_state.get("data_source_choice", "yahoo"),
    _BTN_COLORS["yahoo"],
)

# Static layout rules via st.markdown so they land in the main document.
# st.html() renders in an iframe and cannot affect position:fixed elements
# in the main page — st.markdown(unsafe_allow_html=True) injects directly.
_STYLES_CSS = (
    Path(__file__).parent / "options_scanner" / "styles.css"
).read_text(encoding="utf-8")
st.markdown(f"<style>{_STYLES_CSS}</style>", unsafe_allow_html=True)

# Dynamic accent colors — injected fresh each rerun so button colors
# flip immediately when the data source toggle changes. st.markdown is
# used (not st.html) so the rules reach the main document.
st.markdown(f"""<style>
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {{
    background-color: {_btn_bg} !important;
    border-color: {_btn_bg} !important;
}}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {{
    background-color: {_btn_hover} !important;
    border-color: {_btn_hover} !important;
}}
[class*="st-key-data_source_pill"] button[aria-pressed="true"],
[class*="st-key-data_source_pill"] button[aria-selected="true"],
[class*="st-key-data_source_pill"] button[data-testid*="Active"] {{
    color: {_btn_bg} !important;
    border-color: {_btn_bg} !important;
    box-shadow: inset 0 0 0 1px {_btn_bg} !important;
}}
[class*="st-key-data_source_pill"] button[aria-pressed="true"] p,
[class*="st-key-data_source_pill"] button[aria-selected="true"] p,
[class*="st-key-data_source_pill"] button[data-testid*="Active"] p {{
    color: {_btn_bg} !important;
}}
</style>""", unsafe_allow_html=True)

# Sidebar-state observer: watches the actual sidebar element's rendered
# width and writes data-sidebar-open onto body so the header-bar CSS
# above can respond. Identical to the previous implementation — Streamlit
# offers no native hook for this.
st.iframe(
    r"""
    <script>
    (function() {
        const doc = window.parent.document;
        const SCHWAB_TIP = __SCHWAB_TIP__;

        // Sidebar-open state — drives pill positioning CSS.
        const syncSidebar = () => {
            const sb = doc.querySelector('[data-testid="stSidebar"]');
            if (!sb) return;
            const w = sb.getBoundingClientRect().width;
            doc.body.dataset.sidebarOpen = w > 60 ? 'true' : 'false';
        };

        // Theme detection — reads the actual rendered background of the
        // Streamlit app container so we respond to the hamburger toggle,
        // not just the OS preference. Sets data-osc-theme="dark"|"light"
        // on <html> so CSS can branch on it.
        const syncTheme = () => {
            const app = doc.querySelector('[data-testid="stApp"]');
            if (!app) return;
            const bg = window.parent.getComputedStyle(app).backgroundColor;
            const m = bg.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            if (!m) return;
            const brightness = (+m[1] + +m[2] + +m[3]) / 3;
            doc.documentElement.setAttribute(
                'data-osc-theme', brightness < 80 ? 'dark' : 'light'
            );
        };

        // Exact token-expiry tooltip on the Schwab segment, re-applied after
        // every rerun (Streamlit rebuilds the buttons and drops the title).
        const syncSchwabTip = () => {
            if (!SCHWAB_TIP) return;
            const pill = doc.querySelector('[class*="st-key-data_source_pill"]');
            if (!pill) return;
            pill.querySelectorAll('button').forEach((b) => {
                if ((b.innerText || '').trim().indexOf('Schwab') === 0) {
                    b.title = SCHWAB_TIP;
                }
            });
        };

        const sync = () => { syncSidebar(); syncTheme(); syncSchwabTip(); };
        sync();
        const obs = new MutationObserver(sync);
        obs.observe(doc.body, {
            childList: true, subtree: true,
            attributes: true,
            attributeFilter: ['style', 'class', 'aria-expanded'],
        });
        window.addEventListener('resize', sync);
    })();
    </script>
    """.replace("__SCHWAB_TIP__", json.dumps(_schwab_tok_tip)),
    height=1, width=1,
)


# Title-bar data-source switch — pinned via CSS to the right of the
# rescan pill so it's always visible without opening the sidebar.
def _source_label(s: str) -> str:
    if s == "yahoo":
        return "Yahoo Finance"
    if s == "yahoo-headless":
        return "Yahoo (headless)"
    if s == "moomoo":
        return "Moomoo (live)"
    if not _schwab_configured:
        return "Schwab (unconfigured)"
    return f"Schwab ({_schwab_tok_label})" if _schwab_tok_label else "Schwab"

with st.container(key="data_source_pill"):
    _source_raw = st.segmented_control(
        "Data source",
        ["yahoo", "yahoo-headless", "schwab", "moomoo"],
        format_func=_source_label,
        label_visibility="collapsed",
        key="data_source_choice",
    )
if _source_raw is None:
    _source_raw = "yahoo"

if _source_raw == "schwab" and _schwab_configured:
    data_source = "schwab"
elif _source_raw == "moomoo":
    data_source = "moomoo"
elif _source_raw == "yahoo-headless":
    data_source = "yahoo-headless"
else:
    data_source = "yahoo"
st.session_state["data_source"] = data_source
st.session_state["schwab_config"] = _cfg_schwab if data_source == "schwab" else None
st.session_state["moomoo_config"] = _cfg_moomoo if data_source == "moomoo" else None

# Pick up a re-authenticated Schwab token without a server restart. A fresh
# token (re-auth button or schwab_auth.py) rewrites the token file; the
# client cache already rebuilds on the new mtime, but the @st.cache_data
# chain/spot fetches would keep serving the failure they cached under
# identical args until their TTL — which looked like "must restart the
# server". Detect the mtime change here and drop those caches so the next
# scan uses the new token.
if data_source == "schwab" and _cfg_schwab:
    from stocks_shared.schwab_live import token_mtime as _schwab_token_mtime
    _cur_tok_mtime = _schwab_token_mtime(_cfg_schwab.get("token_file", ""))
    if ("_schwab_token_mtime" in st.session_state
            and st.session_state["_schwab_token_mtime"] != _cur_tok_mtime):
        from options_scanner.fetch import fetch_and_enrich, fetch_position
        from options_scanner.display.spot_meta import fetch_spot_meta
        fetch_and_enrich.clear()
        fetch_position.clear()
        fetch_spot_meta.clear()
    st.session_state["_schwab_token_mtime"] = _cur_tok_mtime


# ── Page header chips ────────────────────────────────────────────────────
# Sidebar: an "About" panel — the legacy theme picker is gone (we now ship
# one canonical design system). Add helpful links and a status indicator.
with st.sidebar:
    st.markdown(
        "<div style='padding: 0.5rem 0 0.75rem 0;'>"
        + badge("WORKSPACE", "neutral")
        + "</div>",
        unsafe_allow_html=True,
    )
    section_header(
        title="Stockpile",
        subtitle=(
            "Options Analytics made for:<br>"
            "• Income generation<br>"
            "• Directional bets<br>"
            "• Defined-risk spreads<br>"
            "• GEX analysis"
        ),
    )
    st.markdown(
        disclaimer_chip("Research tool · Not investment advice"),
        unsafe_allow_html=True,
    )
    st.markdown("---")
    section_header("Data source", eyebrow="ACTIVE PROVIDER")
    _src_label = _source_label(data_source)
    _src_color = PROVIDER_COLORS.get(data_source, "#94a3b8")
    # Tint the badge by token life: amber < 1 day, red < 2h / expired.
    if data_source == "schwab" and _schwab_tok_color:
        _src_color = _schwab_tok_color
    st.markdown(
        f"<div style='font-size:0.86rem; margin-bottom:0.4rem;'>"
        f"<span style='display:inline-block; padding:0.2rem 0.65rem; "
        f"border-radius:6px; font-weight:500; color:#FFFFFF; "
        f"background-color:{_src_color};'>{_src_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Switch between Yahoo Finance (free, 15-min delay), Yahoo "
        "(headless) (browser scrape — slower, but returns quotes after "
        "hours when the Yahoo API serves none), Schwab (authenticated, "
        "live), and Moomoo (live via OpenD). Use the toggle in the top bar."
    )
    st.markdown("---")
    section_header("About", eyebrow="HOW THIS WORKS")
    st.caption(
        "Surface contracts whose implied volatility sits above (or below) "
        "the fitted surface. Filter by DTE, delta, liquidity; export a "
        "shareable HTML report."
    )
    st.caption(
        "For every option in the chain, we fit a smooth volatility "
        "surface across strike and DTE, then rank contracts by how much "
        "their IV exceeds the fit (IV+pp). 3pp ≈ noise; 5+pp is signal."
    )
    st.markdown("---")
    section_header("Documentation", eyebrow="REFERENCE")
    st.markdown(
        "- [README](https://github.com/) — overview & install\n"
        "- [Interpreting IV](https://github.com/) — what IV+pp means\n"
        "- [Spreads](https://github.com/) — strategy glossary",
        unsafe_allow_html=False,
    )

# Compatibility shim — keep `_apply_theme(theme_choice)` working in case
# any deferred code path references it. With the new design system in
# place this is a no-op.
_apply_theme("Default")

# Brand wordmark — now in normal document flow just below the fixed
# title bar (rescan / data-source / surface-fit pills), so it scrolls
# away with the page instead of staying pinned.
st.markdown(
    """
    <div class='osc-wordmark-inline'>
      <span class='osc-wm-dot'></span>
      <span class='osc-wm-brand'>STOCKPILE</span>
      <span class='osc-wm-suffix'>· OPTIONS SCANNER</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Confirmation queued by an action that then reran (e.g. placing a trade):
# a toast created right before st.rerun() is discarded with that run, so the
# producer stashes the message and we show it here on the next run. Rendered
# as a centered, fully-visible banner (st.toast is top-right + truncates):
# fades out after ~6s and is click-through. $ → &#36; so it renders literally
# rather than as LaTeX math.
_pending_toast = st.session_state.pop("_osc_toast", None)
if _pending_toast:
    # Build the centered banner in the parent document via JS (st.markdown
    # strips inline JS, so a click-to-dismiss × needs a component iframe). It
    # stays until the user clicks × or 60s elapse; the timeout lives on
    # window.parent so it survives this iframe being torn down on a rerun.
    st.iframe(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          const prev = doc.getElementById('osc-center-toast');
          if (prev) prev.remove();
          const box = doc.createElement('div');
          box.id = 'osc-center-toast';
          box.style.cssText = ['position:fixed','top:50%','left:50%',
            'transform:translate(-50%,-50%)','z-index:1000000','max-width:560px',
            'background:#16a34a','color:#fff','padding:1.2rem 2.6rem 1.2rem 1.6rem',
            'border-radius:12px','box-shadow:0 10px 40px rgba(0,0,0,.4)',
            'font-size:1.08rem','line-height:1.45','text-align:center'].join(';');
          const msg = doc.createElement('span');
          msg.textContent = __MSG__;
          const x = doc.createElement('span');
          x.textContent = '\\u00d7';
          x.title = 'Dismiss';
          x.style.cssText = ['position:absolute','top:6px','right:12px',
            'cursor:pointer','font-size:1.4rem','line-height:1',
            'font-weight:700'].join(';');
          x.onclick = function() { box.remove(); };
          box.appendChild(x);
          box.appendChild(msg);
          doc.body.appendChild(box);
          window.parent.setTimeout(function() {
            const b = doc.getElementById('osc-center-toast');
            if (b) b.remove();
          }, 60000);
        })();
        </script>
        """.replace("__MSG__", json.dumps(_pending_toast)),
        height=1, width=1,
    )

(
    panel_single, panel_watchlist, panel_trades, panel_portfolio, panel_gex,
    panel_spreads, panel_directional, panel_neutral,
    panel_backtest, panel_live,
) = st.tabs(
    ["Single Ticker", "Watchlist", "Trades", "Portfolio", "GEX",
     "Spreads", "Directional", "Neutral", "Backtest", "Live Charts"]
)

with panel_single:
    tab_single()

with panel_watchlist:
    tab_watchlist()

with panel_trades:
    tab_trades()

with panel_portfolio:
    tab_portfolio()

with panel_gex:
    tab_gex()

with panel_spreads:
    tab_spreads()

with panel_directional:
    tab_directional()

with panel_neutral:
    tab_neutral()

with panel_backtest:
    tab_backtest()

with panel_live:
    tab_live_charts()

# ── Footer ───────────────────────────────────────────────────────────────
ui_footer()
