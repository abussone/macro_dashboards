from __future__ import annotations

"""Market Memory Explorer

How it works
------------
1) Pulls daily adjusted closes from Yahoo Finance (yfinance, auto_adjust=True).
2) Defines the current pattern window from *Start* -> latest available close (length L trading days).
3) Scans historical rolling windows of length L, requiring a full 3L segment entirely before *Start*:
      candidate_start + 3L <= current_start_index
4) Computes Pearson correlation on the cumulative return path.
5) Picks TOP_N matches.

"""

import datetime
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =============================================================================
# SESSION STATE INIT (MUST BE BEFORE WIDGETS)
# =============================================================================
DEFAULT_TICKER = "^GSPC"
DEFAULT_START_DATE = "2025-06-01"

if "mm_ticker" not in st.session_state:
    st.session_state["mm_ticker"] = DEFAULT_TICKER
if "mm_start_date" not in st.session_state:
    st.session_state["mm_start_date"] = DEFAULT_START_DATE
if "mm_top_n" not in st.session_state:
    st.session_state["mm_top_n"] = 5
if "mm_min_corr" not in st.session_state:
    st.session_state["mm_min_corr"] = 0.50
if "mm_buffer_frac" not in st.session_state:
    st.session_state["mm_buffer_frac"] = 0.50


# =============================================================================
# CONFIG
# =============================================================================
LOOK_AHEAD_FACTOR = 3  # hard requirement in your use-case
CACHE_TTL_SECONDS = 3600
DISK_CACHE_DIR = Path(".cache_yf_mm")

# DISTINCT_ANALOGS is HARD-CODED ON (not an option).


# =============================================================================
# UI HELPERS (aligned to US Dashboard page style)
# =============================================================================
def _inject_light_ui() -> None:
    """Subtle, professional, mostly-white UI polish (same approach as 2_US_Dashboard.py)."""
    st.markdown(
        """
        <style>
            .stApp { background: #F7F8FB; }
            .block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; max-width: 1200px; }
            h1, h2, h3 { letter-spacing: -0.01em; }

            /* Show header so the sidebar toggle is visible on mobile */
            header { visibility: visible !important; height: auto !important; }
            footer { visibility: hidden; height: 0; }

            /* Fixed sidebar width for readability and stability */
            [data-testid="stSidebar"] {
                min-width: 260px;
                max-width: 300px;
            }

            /* Sticky Navigation Bar (if used) */
            div[data-testid="stRadio"] {
              position: sticky;
              top: 2.85rem;
              z-index: 1000;
              background: #F7F8FB;
              padding: 1rem 0;
              border-bottom: 1px solid rgba(15, 23, 42, 0.05);
              margin-bottom: 1rem;
            }

            /* Neutral detail card */
            .details-box {
              background: rgba(255,255,255,0.92);
              border: 1px solid rgba(15,23,42,0.06);
              border-radius: 14px;
              padding: 0.9rem 1.0rem;
              margin-top: 0.75rem;
              box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fig_from_spec(data: list[dict], layout: dict, height: int) -> go.Figure:
    """Copy of the US dashboard helper: forces black fonts and avoids 'undefined' title/hover issues."""
    fig = go.Figure(data=data, layout=layout)

    # Defensive Plotly hygiene
    try:
        if getattr(fig.layout.xaxis, "type", None) in (None, "") and fig.data:
            xs = getattr(fig.data[0], "x", None)
            if xs is not None and len(xs) > 0:
                idxs = np.linspace(0, len(xs) - 1, num=min(5, len(xs)), dtype=int)
                sample = [xs[i] for i in idxs]
                parsed = pd.to_datetime(sample, errors="coerce")
                if hasattr(parsed, "notna") and float(parsed.notna().mean()) >= 0.6:
                    fig.update_xaxes(type="date")
    except Exception:
        pass

    for tr in fig.data:
        try:
            nm = getattr(tr, "name", None)
            if nm is None or str(nm).strip().lower() == "undefined":
                tr.name = ""
                tr.showlegend = False
                tr.hoverinfo = "skip"
        except Exception:
            pass

    base_margin = layout.get("margin", dict(l=60, r=60, t=30, b=70))
    fig.update_layout(
        hovermode=layout.get("hovermode", "x unified"),
        height=height,
        margin=base_margin,
    )
    fig.update_layout(
        font=dict(color="black"),
        hoverlabel=dict(namelength=-1),
        legend_font_color="black",
    )

    try:
        t_text = None
        if getattr(fig.layout, "title", None) is not None:
            t_text = getattr(fig.layout.title, "text", None)
        if t_text is None or str(t_text).strip().lower() == "undefined":
            fig.update_layout(title_text="")
        fig.update_layout(title=dict(font=dict(color="black")))
    except Exception:
        pass

    fig.update_xaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        linecolor="black",
        gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        linecolor="black",
        gridcolor="rgba(0,0,0,0.05)")

    # Make room for horizontal legends below the plot (avoid scrollbars)
    try:
        leg = fig.layout.legend
        if leg and getattr(leg, "orientation", None) == "h":
            ly = getattr(leg, "y", None)
            if ly is not None and float(ly) < 0:
                n_items = 0
                for tr in fig.data:
                    show = getattr(tr, "showlegend", True)
                    if show is False:
                        continue
                    n_items += 1
                per_row = 4
                rows = max(1, int(math.ceil(n_items / per_row)))
                legend_px = 26 * rows + 28
                needed_b = max(int(base_margin.get("b", 0) or 0), 70 + legend_px)
                fig.update_layout(
                    margin=dict(**base_margin, b=needed_b),
                    height=height + max(0, needed_b - base_margin.get("b", 0)),
                )
    except Exception:
        pass

    return fig


def _show_chart(
    title: str,
    fig: go.Figure | None,
    observations: list[str] | None,
    sources: list[str] | None,
    *,
    show_details: bool = True,
) -> None:
    st.subheader(title)
    if fig is None:
        st.warning("Chart unavailable (missing data).")
        return

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "responsive": True,
            "displayModeBar": True,
        },
    )

    if not show_details:
        return

    st.markdown('<div class="details-box">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key observations")
        if observations:
            st.markdown("\n".join([f"- {o}" for o in observations]))
        else:
            st.write("—")
    with c2:
        st.markdown("#### Data sources")
        if sources:
            st.markdown("\n".join([f"- {s}" for s in sources]))
        else:
            st.write("—")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MATH + ENGINE
# =============================================================================
def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _first_trading_index_on_or_after(dates: pd.DatetimeIndex, start_date: pd.Timestamp) -> int:
    pos = dates.searchsorted(start_date, side="left")
    if pos >= len(dates):
        raise ValueError("START_DATE is after the last available trading day.")
    return int(pos)


def _cumret(prices: np.ndarray) -> np.ndarray:
    return (prices / prices[0]) - 1.0


def _to_rebased_100_from_cumret(cumret: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 + cumret)


def _corr_batch(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Correlation between each row of W (shape [n, L]) and x (shape [L]). ddof=0."""
    x0 = x - x.mean()
    x_std = x0.std(ddof=0)
    if x_std == 0:
        return np.full(W.shape[0], np.nan, dtype=np.float64)

    W0 = W - W.mean(axis=1, keepdims=True)
    W_std = W0.std(axis=1, ddof=0)

    L = x.shape[0]
    cov = (W0 @ x0) / L
    denom = W_std * x_std

    out = np.full(W.shape[0], np.nan, dtype=np.float64)
    mask = denom > 0
    out[mask] = cov[mask] / denom[mask]
    return out


def _safe_cache_stem(symbol: str) -> str:
    s = symbol.strip()
    return "".join(ch if ch.isalnum() else "_" for ch in s) or "EMPTY"


def _ensure_cache_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_data_daily(symbol: str) -> pd.DataFrame:
    """
    Full daily history with auto_adjust=True, plus a simple disk cache fallback.

    FIX: yfinance MultiIndex columns
    -------------------------------
    Depending on yfinance version/settings, yf.download() can return MultiIndex columns.
    In that case hist["Close"] may be a DataFrame (not a Series), and using it inside .loc
    can raise: "Cannot index with multidimensional key".

    We therefore extract a 1D Close series robustly (single-level or MultiIndex),
    then build a clean DataFrame with one 'Close' column.
    """
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("Ticker is empty. Please enter a valid Yahoo ticker (e.g., ^GSPC).")

    _ensure_cache_dir(DISK_CACHE_DIR)
    cache_path = DISK_CACHE_DIR / f"{_safe_cache_stem(symbol)}_auto_adjust_close_1d.pkl"
    now = time.time()

    # Fresh disk cache
    if cache_path.exists() and (now - cache_path.stat().st_mtime) <= CACHE_TTL_SECONDS:
        df = pd.read_pickle(cache_path)
        if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) > 10:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df[["Close"]]

    def _extract_close(hist: pd.DataFrame) -> pd.Series:
        # MultiIndex: could be ("Close", SYMBOL) or (SYMBOL, "Close") or other variants.
        if isinstance(hist.columns, pd.MultiIndex):
            if ("Close", symbol) in hist.columns:
                s = hist[("Close", symbol)]
            elif (symbol, "Close") in hist.columns:
                s = hist[(symbol, "Close")]
            else:
                s = None
                for level in (0, 1):
                    try:
                        xs = hist.xs("Close", axis=1, level=level)
                        if isinstance(xs, pd.DataFrame):
                            if xs.shape[1] == 0:
                                continue
                            s = xs.iloc[:, 0]
                        else:
                            s = xs
                        break
                    except Exception:
                        continue
                if s is None:
                    raise ValueError("Could not locate 'Close' in MultiIndex columns.")
        else:
            if "Close" in hist.columns:
                s = hist["Close"]
            elif "Adj Close" in hist.columns:
                s = hist["Adj Close"]
            else:
                close_like = hist.filter(like="Close")
                if close_like.shape[1] == 0:
                    raise ValueError("Missing Close column.")
                s = close_like.iloc[:, 0]

        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            raise ValueError("Close series is empty after cleaning.")
        return s

    # Yahoo retry loop
    delay = 1.0
    last_err: Optional[Exception] = None
    for _ in range(4):
        try:
            hist = yf.download(
                symbol,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if hist is None or hist.empty:
                raise ValueError("Yahoo returned empty history.")

            close_s = _extract_close(hist)
            df = close_s.to_frame("Close")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()

            df.to_pickle(cache_path)
            return df[["Close"]]
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 2

    # Fallback to stale disk cache (better than nothing)
    if cache_path.exists():
        df = pd.read_pickle(cache_path)
        if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) > 10:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df[["Close"]]

    raise RuntimeError(f"Failed to download {symbol}: {last_err}")
@dataclass(frozen=True)
class AnalogPick:
    start_idx: int
    rho: float


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def compute_analogs_payload(
    symbol: str,
    start_date: str,
    top_n: int,
    min_corr: float,
    distinct_buffer_fraction: float,
) -> tuple[dict, pd.DataFrame]:
    """Compute analogs + precomputed series for plotting (cached).

    Matching mechanism:
    - Pearson correlation is computed on *cumulative return paths* over the current window (length L):
        cumret = P/P0 - 1

    Display mechanism (this page):
    - We display *prices rebased to 100 at the match end* (x=0):
        index100 = 100 * P / P_match_end
      This makes all series equal 100 at x=0 (the end of the overlap window).

    Returns reporting:
    - We EXCLUDE the overlap window [0, L) from "outcome" returns.
    - We report returns over:
        [L, 2L] => from P(L-1) to P(2L-1)
        [L, 3L] => from P(L-1) to P(3L-1)
      i.e., performance *after* the overlap period.
    """
    if LOOK_AHEAD_FACTOR != 3:
        raise ValueError("This page enforces LOOK_AHEAD_FACTOR = 3.")
    if not (0.0 <= min_corr <= 1.0):
        raise ValueError("MIN_CORR must be in [0, 1].")
    if top_n < 1:
        raise ValueError("TOP_N must be >= 1.")

    df = fetch_data_daily(symbol)
    close = df["Close"].dropna().copy()
    if len(close) < 300:
        raise ValueError("Not enough daily history for this ticker.")

    dates = close.index
    s_ts = _to_ts(start_date)
    start_idx = _first_trading_index_on_or_after(dates, s_ts)

    # Current window: START_DATE -> now (length L)
    target = close.iloc[start_idx:].copy()
    if target.empty:
        raise ValueError("No data after START_DATE.")

    L = int(len(target))
    H = int(LOOK_AHEAD_FACTOR * L)

    # Strict no-overlap: candidate start i must satisfy i + 3L <= start_idx
    max_start = start_idx - H
    if max_start < 0:
        raise ValueError(
            f"Not enough pre-start history for strict 3L no-overlap. "
            f"Need at least {H} trading days before START_DATE; have only {start_idx}."
        )

    p = close.to_numpy(dtype=np.float64)

    # Matching target vector (cumret over L days)
    cur_prices = p[start_idx : start_idx + L]
    x = _cumret(cur_prices)

    cand_count = max_start + 1
    elem = cand_count * L
    vectorize_ok = elem <= 6_000_000  # keep memory bounded

    rhos: np.ndarray

    if vectorize_ok:
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            Wp_all = sliding_window_view(p, L)      # shape ~ [N-L+1, L]
            Wp = Wp_all[:cand_count, :]             # candidates fully before overlap (via cand_count)
            W = (Wp / Wp[:, [0]]) - 1.0             # cumret windows [cand_count, L]
            rhos = _corr_batch(W.astype(np.float64), x.astype(np.float64))
        except Exception:
            vectorize_ok = False

    if not vectorize_ok:
        rhos = np.full(cand_count, np.nan, dtype=np.float64)
        for i in range(cand_count):
            Wp = p[i : i + L]
            W = _cumret(Wp)
            rhos[i] = float(_corr_batch(W.reshape(1, -1), x)[0])

    ok = np.isfinite(rhos) & (rhos >= min_corr)
    good = np.where(ok)[0]
    if good.size == 0:
        raise ValueError("No candidates remain after MIN_CORR.")

    ranked = good[np.argsort(rhos[good])[::-1]]

    # DISTINCT always ON (non-maximum suppression)
    buffer = max(1, int(round(L * distinct_buffer_fraction)))
    picks: list[AnalogPick] = []
    for k in ranked:
        if len(picks) >= top_n:
            break
        i = int(k)
        rho = float(rhos[k])
        if any(abs(i - p0.start_idx) < buffer for p0 in picks):
            continue
        picks.append(AnalogPick(start_idx=i, rho=rho))

    if not picks:
        raise ValueError("No picks selected (try lowering MIN_CORR or reducing the buffer).")

    # --- Build payload for rebased-to-match-end chart ---
    # x=0 at match end, i.e. day index (L-1). index100 = 100 * P / P_at_match_end
    cur_base = float(cur_prices[-1])
    cur_index100 = (100.0 * (cur_prices / cur_base)).astype(np.float64)

    rows: list[dict] = []
    analogs_payload: list[dict] = []

    for pick in picks:
        i = pick.start_idx
        seg_prices = p[i : i + H]  # full 3L segment

        # Match end price (x=0)
        base = float(seg_prices[L - 1])
        seg_index100 = (100.0 * (seg_prices / base)).astype(np.float64)

        # Outcome returns EXCLUDING the overlap window:
        ret_L_2L = float(seg_prices[2 * L - 1] / seg_prices[L - 1] - 1.0)
        ret_L_3L = float(seg_prices[3 * L - 1] / seg_prices[L - 1] - 1.0)

        rows.append(
            {
                "start_date": dates[i].date(),
                "match_end_date": dates[i + (L - 1)].date(),
                "horizon_end_date": dates[i + (H - 1)].date(),
                "rho": float(pick.rho),
                "ret_L_2L": ret_L_2L,
                "ret_L_3L": ret_L_3L,
            }
        )

        analogs_payload.append(
            {
                "label": f"{dates[i].strftime('%Y-%m')} (ρ={pick.rho:.2f})",
                "rho": float(pick.rho),
                "index100": seg_index100.tolist(),
            }
        )

    results = pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)

    payload = {
        "symbol": symbol,
        "start_date": str(pd.Timestamp(s_ts).date()),
        "L": L,
        "H": H,
        "buffer": buffer,
        "candidate_count": int(cand_count),
        "current": {
            "label": f"Current: {target.index[0].date()} → {target.index[-1].date()}",
            "ret_L": float(cur_prices[-1] / cur_prices[0] - 1.0),
            "index100": cur_index100.tolist(),
        },
        "analogs": analogs_payload,
    }
    return payload, results

# =============================================================================
# FIGURES
# =============================================================================
def _build_rebased_index_fig(payload: dict) -> go.Figure:
    """Build the single chart: index rebased to 100 at match end (x=0).

    Axis semantics:
    - x=0 is the end of the overlap window ("now" for the current series).
    - The overlap window spans x in [-(L-1), 0].
    - The historical outcome window spans x in [1, H-L].
    """
    L = int(payload["L"])
    H = int(payload["H"])
    current = payload["current"]
    analogs = payload["analogs"]

    # X ranges
    x_cur = list(np.arange(L) - (L - 1))
    x_full = list(np.arange(H) - (L - 1))
    x_min, x_max = int(-(L - 1)), int(H - L)

    rhos = [a["rho"] for a in analogs] or [1.0]
    rho_max = max(rhos) if max(rhos) > 0 else 1.0

    data: list[dict] = []

    # Analogs (full 3L)
    for a in analogs:
        y = a["index100"]
        alpha = 0.35 + 0.45 * (float(a["rho"]) / rho_max)

        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": x_full,
                "y": y,
                "name": a["label"],
                "opacity": float(alpha),
                "line": {"width": 2, "dash": "dash"},
            }
        )

        # marker at x=0 (match end) should be exactly 100
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [0],
                "y": [y[L - 1]],
                "showlegend": False,
                "marker": {"size": 6},
            }
        )

    # Current (only overlap window)
    y_cur = current["index100"]
    data.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": x_cur,
            "y": y_cur,
            "name": current["label"],
            "line": {"width": 4, "color": "black"},
        }
    )

    shapes = [
        {
            "type": "line",
            "x0": 0,
            "x1": 0,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {"width": 2, "dash": "dash", "color": "rgba(0,0,0,0.5)"},
        }
    ]

    annotations = [
        {
            "x": 2,
            "y": 0.95,
            "xref": "x",
            "yref": "paper",
            "text": "HISTORICAL<br>OUTCOME",
            "showarrow": False,
            "font": {"size": 10, "color": "gray"},
        }
    ]

    title = f"{payload['symbol']} — Rebased Index (100 at match end / x=0)"

    layout = {
        "title": {"text": title},
        "hovermode": "x unified",
        "xaxis": {
            "title": "Trading Days Relative to Match End (0 = Now)",
            "range": [x_min, x_max],
            "type": "linear",
            "zeroline": False,
        },
        "yaxis": {
            "title": "Index (100 at Match End)",
            "tickformat": ",.0f",
        },
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
        "shapes": shapes,
        "annotations": annotations,
    }

    return _fig_from_spec(data=data, layout=layout, height=520)

# =============================================================================
# PAGE
# =============================================================================
def main() -> None:
    _inject_light_ui()

    st.sidebar.title("Controls")
    if st.sidebar.button("Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.text_input(
        "Ticker (Yahoo Finance)",
        key="mm_ticker",
        help="Examples: ^GSPC, ^NDX, AAPL, BTC-USD, EURUSD=X",
    )
    st.sidebar.text_input("Start (YYYY-MM-DD)", key="mm_start_date")
    st.sidebar.slider("Top analogs", 1, 10, key="mm_top_n")
    st.sidebar.slider("Min ρ", 0.0, 1.0, step=0.05, key="mm_min_corr")

    with st.sidebar.expander("Distinctness (always ON)", expanded=False):
        st.markdown("When we scan history, we evaluate a candidate start index every trading day (i.e., a rolling window). If a specific historical episode matches well, then its neighbors (start shifted by 1 day, 2 days, …) will usually also match well. So the top correlations tend to come in clusters around the same event. “Distinctness” prevents your Top-N list from being “the same match 5 times with a 1–2 day shift”.")
        st.slider(
            "Buffer fraction",
            min_value=0.10,
            max_value=1.00,
            step=0.05,
            key="mm_buffer_frac",
            help="Non-maximum suppression: suppress other matches within L * buffer_fraction trading days.",
        )

    st.sidebar.caption("Tip: results are cached, so tweaks should feel snappy after the first run.")

    # Header row
    h1, h2, h3 = st.columns([0.72, 0.18, 0.10], vertical_alignment="bottom")
    with h1:
        st.title("📈 Market Memory Explorer")
    with h2:
        st.caption(f"Local time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with h3:
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    ticker = str(st.session_state["mm_ticker"]).strip()
    start_date = str(st.session_state["mm_start_date"]).strip()
    top_n = int(st.session_state["mm_top_n"])
    min_corr = float(st.session_state["mm_min_corr"])
    buffer_frac = float(st.session_state["mm_buffer_frac"])

    with st.spinner("Loading / computing (cached)…"):
                payload, results = compute_analogs_payload(
            symbol=ticker,
            start_date=start_date,
            top_n=top_n,
            min_corr=min_corr,
            distinct_buffer_fraction=buffer_frac,
        )

    # KPIs
    cur_last = float(payload["current"]["ret_L"])
    med_L_3L = float(results["ret_L_3L"].median()) if not results.empty else float("nan")
    sig_L_3L = float(results["ret_L_3L"].std(ddof=0)) if not results.empty else float("nan")

    k1, k2, k3 = st.columns(3)
    k1.metric("Current window return (L)", f"{cur_last:.2%}")
    k2.metric("Median analog return (L→3L)", f"{med_L_3L:.2%}" if np.isfinite(med_L_3L) else "n/a")
    k3.metric("Analog dispersion (σ) (L→3L)", f"{sig_L_3L:.2%}" if np.isfinite(sig_L_3L) else "n/a")

    st.markdown("---")

    # Table (no rank bar; compact table)
    show = results.copy()
    show.insert(0, "#", np.arange(1, len(show) + 1))
    show["rho"] = show["rho"].map(lambda x: f"{x:.3f}")
    for c in ["ret_L_2L", "ret_L_3L"]:
        show[c] = show[c].map(lambda x: f"{x:.2%}")
    st.dataframe(show, width='stretch', hide_index=True)

    # Figures
    fig_idx = _build_rebased_index_fig(payload)

    obs = [
        "Matching uses Pearson correlation on the cumulative return path over the current window (length L). This chart is rebased to 100 at the match end (0=Now). Returns reported below exclude the overlap window and measure performance after x=0.",
        f"Strict no-overlap: historical analogs must have a full 3L segment ending before {payload['start_date']}.",
        f"Distinctness is always ON: matches are separated by ≥ {payload['buffer']} trading days.",
    ]
    src = ["Yahoo Finance via yfinance (auto_adjust=True, daily closes)"]

    _show_chart("Rebased Indexes (100 at match end, ie at 0=Now)", fig_idx, obs, src)

if __name__ == "__main__":
    main()
