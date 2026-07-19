"""Tests for the Stars-rating helpers in format.py."""

import numpy as np
import pandas as pd

from options_scanner.format import fmt_stars, stars_for


def test_stars_for_evenly_spaced_scores():
    # 5 evenly-spaced scores -> percentile ranks 0.2, 0.4, 0.6, 0.8, 1.0
    # (pandas default rank(pct=True) with no ties) -> *5*2 rounded /2.
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stars = stars_for(scores)
    ranks = scores.rank(pct=True).to_numpy()
    expected = np.round(ranks * 5.0 * 2.0) / 2.0
    assert np.allclose(stars, expected)
    assert stars[-1] == 5.0  # top score -> full 5 stars
    assert 0.0 <= stars.min()
    assert stars.max() <= 5.0


def test_stars_for_nan_score_is_nan():
    scores = pd.Series([1.0, np.nan, 3.0])
    stars = stars_for(scores)
    assert np.isnan(stars[1])
    assert not np.isnan(stars[0])
    assert not np.isnan(stars[2])


def test_stars_for_single_value_is_top():
    scores = pd.Series([42.0])
    stars = stars_for(scores)
    assert stars[0] == 5.0


def test_fmt_stars_boundaries():
    assert fmt_stars(2.0) == "★★"
    assert fmt_stars(2.5) == "★★½"
    assert fmt_stars(0.0) == ""
    assert fmt_stars(5.0) == "★★★★★"
    assert fmt_stars(4.5) == "★★★★½"


def test_fmt_stars_nan_is_blank():
    assert fmt_stars(float("nan")) == ""
