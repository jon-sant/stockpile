"""Market-view stance table: (Direction x Option Type) -> outlook.

Pure data/logic, no Streamlit dependency — shared by the display layer
(`display/outlook_card.py` renders it as a card; `display/
portfolio_action_card.py` reuses the tone palette) and the compute layer
(`mc_batch.py` derives Monte Carlo drift from it). Moved here from
outlook_card.py so the compute path never has to import a Streamlit
module to reach this data — the exact mistake already made once in this
codebase for ranking logic (see display/rank_filter.py's docstring).
"""

from __future__ import annotations

OUTLOOK_TABLE: dict[tuple[bool, str], dict[str, str]] = {
    # (buy?, opt_type) -> {stance, tone, summary, examples}
    # 'tone' picks the accent color: pos = green, neg = red, neutral = amber, vol = purple
    (False, "Calls"): {
        "stance": "Bearish / neutral-down",
        "tone": "neg",
        "summary": "Collect premium on calls you expect to expire worthless. "
                   "Profits if the underlying stays below the strike — the "
                   "classic 'covered call' or 'short call' setup. IV-rich "
                   "premium boosts the credit you receive.",
        "examples": "Covered call · Short call · Credit call spread",
    },
    (False, "Puts"): {
        "stance": "Bullish / neutral-up",
        "tone": "pos",
        "summary": "Collect premium on puts you expect to expire worthless. "
                   "Profits if the underlying stays above the strike. The "
                   "'cash-secured put' is the bullish income trade — you're "
                   "paid to wait for a price you'd be happy to buy at.",
        "examples": "Cash-secured put · Short put · Credit put spread",
    },
    (False, "Both"): {
        "stance": "Range-bound (short volatility)",
        "tone": "neutral",
        "summary": "Sell premium on both sides because you expect the "
                   "underlying to stay inside a range. Profits if IV "
                   "contracts AND the move is small. Beware of binary "
                   "events (earnings, FDA) that can crush range-bound bets.",
        "examples": "Iron condor · Short strangle · Short straddle",
    },
    (True, "Calls"): {
        "stance": "Bullish",
        "tone": "pos",
        "summary": "Pay premium for upside leverage. Profits if the "
                   "underlying rises enough to cover the debit. IV-cheap "
                   "candidates give you a better entry point because you "
                   "buy when volatility is under-priced.",
        "examples": "Long call · Debit call spread · Diagonal / PMCC",
    },
    (True, "Puts"): {
        "stance": "Bearish",
        "tone": "neg",
        "summary": "Pay premium for downside exposure. Profits if the "
                   "underlying falls enough to cover the debit. IV-cheap "
                   "candidates make the directional bet more efficient "
                   "because vol isn't already priced in.",
        "examples": "Long put · Debit put spread · Protective put",
    },
    (True, "Both"): {
        "stance": "Volatility expansion (long vol)",
        "tone": "vol",
        "summary": "Pay premium for a big move in either direction. "
                   "Profits if realized vol exceeds implied vol OR if IV "
                   "expands. Best entered when IV is low AND a catalyst "
                   "is approaching (earnings, FDA). Beware vol crush.",
        "examples": "Long straddle · Long strangle · Calendar spread",
    },
}


OUTLOOK_TONE_HEX = {
    "pos":     "#059669",   # green — success
    "neg":     "#DC2626",   # red — destructive
    "neutral": "#D97706",   # amber — accent
    "vol":     "#8B5CF6",   # purple — vol expansion
}

# stance string -> tone, derived once from OUTLOOK_TABLE so there's a
# single source of truth even though drift_for_stance() only has the
# stance string to work from (iv_history persists the stance, not the
# (buy, opt_type) pair it came from).
_STANCE_TONE: dict[str, str] = {
    cfg["stance"]: cfg["tone"] for cfg in OUTLOOK_TABLE.values()
}

DEFAULT_DRIFT_MAGNITUDE = 0.10  # +/-10%/yr for a directional (bullish/bearish) stance


def stance_for(buy: bool, opt_type: str) -> str | None:
    """The outlook card's stance string for (buy, opt_type), or None if
    the pair isn't in the table (e.g. an opt_type outside "Calls"/
    "Puts"/"Both")."""
    cfg = OUTLOOK_TABLE.get((buy, opt_type))
    return cfg["stance"] if cfg else None


def drift_for_stance(stance: str | None,
                      magnitude: float = DEFAULT_DRIFT_MAGNITUDE) -> float:
    """Monte Carlo drift (SimulationConfig.drift — additional annualized
    drift above the risk-free rate) implied by a persisted market_view
    stance string.

    Directional stances (tone pos/neg) get +/-magnitude. Non-directional
    stances (Range-bound, Volatility expansion — tone neutral/vol) get 0:
    neither expresses a view on which way the underlying moves, only on
    how much it moves or how IV behaves. None or an unrecognized stance
    (no market_view recorded for that row) also gets 0 — "no state
    found" means assume neutral, same treatment as the non-directional
    stances.
    """
    tone = _STANCE_TONE.get(stance or "")
    if tone == "pos":
        return magnitude
    if tone == "neg":
        return -magnitude
    return 0.0
