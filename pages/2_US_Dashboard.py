import datetime
import io as _io
import json
import math
import os
import random
import re
import threading
import time
import zipfile
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _resolve_fred_api_key() -> str:
    """Resolve FRED API key from Streamlit secrets or environment variables."""
    try:
        key = str(st.secrets.get("FRED_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    return str(os.getenv("FRED_API_KEY", "")).strip()


FRED_API_KEY = _resolve_fred_api_key()

# --- FOMC / Monetary policy inputs (from original US_Dashboard.py) ---
FED_MONETARYPOLICY_BASE = "https://www.federalreserve.gov/monetarypolicy/"
SEP_DISCOVERY_PAGES = [
    f"{FED_MONETARYPOLICY_BASE}fomccalendars.htm",
    f"{FED_MONETARYPOLICY_BASE}monetarypolicytools.htm",
]
SEP_FALLBACK_URL = f"{FED_MONETARYPOLICY_BASE}fomcprojtabl20250917.htm"
SEP_URL_RE = re.compile(r"fomcprojtabl(\d{8})\.htm", re.IGNORECASE)
MPT_XLSX_URL = "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cenfis/market-probability-tracker/mpt_histdata.xlsx"
START_DATE_FOMC = "2018-12-31"
LOOKBACK_YEARS = 6

INFLATION_SERIES = {
    5: "T5YIEM",
    7: "T7YIEM",
    10: "T10YIEM",
    20: "T20YIEM",
    30: "T30YIEM",
}

# --- Yield curve inputs ---
YIELD_CURVE_SERIES = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}
YIELD_CURVE_ORDER = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# --- Money market inputs ---
MM_SERIES_IDS = ["DPCREDIT", "IORB", "DFF", "OBFR", "SOFR", "RRPONTSYAWARD", "RRPONTTLD"]
MM_FUNDS_IDS = ["DFEDTARL", "DFEDTARU"]
MM_OFR_MNEMONICS = {"BGCR": "FNYR-BGCR-A", "TGCR": "FNYR-TGCR-A"}

MM_SERIES_NAMES = [
    "Discount Window Primary Credit (DPCREDIT)",
    "IORB",
    "DFF",
    "OBFR",
    "SOFR",
    "ON RRP Award Rate",
    "ON RRP Total Volume (RRPONTTLD)",
    "Broad General Collateral (BGCR)",
    "Tri-Party General Collateral (TGCR)",
]

# --- Hedge fund inputs (CFTC COT) ---
CFTC_USER_AGENT = {"User-Agent": "Mozilla/5.0"}
CFTC_HISTORY_BLOCK = "https://www.cftc.gov/files/dea/history/fut_fin_txt_2010_2024.zip"
CFTC_YEAR_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

# --- Gold / Liquidity / Flows inputs ---
GOLD_START_DATE = "2023-10-01"
IDS0182_URL = "https://apps.bea.gov/international/zip/IDS0182.zip"

GOLD_USE_BOP_ADJUSTED = True
if GOLD_USE_BOP_ADJUSTED:
    GOLD_SHEET_NAME = "BP-based, NSA"
    GOLD_IMPORT_CODE = "MNMGLD"
    GOLD_EXPORT_CODE = "XNMGLD"
else:
    GOLD_SHEET_NAME = "Census-based, NSA"
    GOLD_IMPORT_CODE = "M14270"
    GOLD_EXPORT_CODE = "X12260"

# Disk cache (kept from original to handle Yahoo throttling / BEA zip)
GOLD_CACHE_DIR = Path("./cache_gold")
GOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GOLD_CACHE_TTL_SEC = 60 * 60 * 6
GOLD_M2_CACHE_TTL_SEC = 60 * 60 * 12
GOLD_BEA_CACHE_TTL_SEC = 60 * 60 * 12

GOLD_MIN_SECONDS_BETWEEN_YAHOO_CALLS = 30.0
_gold_fetch_lock = threading.Lock()
_last_yahoo_call_ts = 0.0

# =============================================================================
# NETWORK SESSIONS
# =============================================================================
def make_session(user_agent: str) -> requests.Session:
    """Create requests session with retries."""
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
        }
    )
    return s


SESSION = make_session("Mozilla/5.0 (compatible; US dashboard)")
GOLD_HTTP_SESSION = make_session(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)
GOLD_HTTP_SESSION.headers.update(
    {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
)

# =============================================================================
# PLOTLY + UI HELPERS
# =============================================================================
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _pct_fmt(x, nd=2) -> str:
    return f"{x:.{nd}f}%" if np.isfinite(x) else "n/a"

def _resample_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end resampling that works across pandas versions.

    Pandas historically used the 'M' alias for month-end; newer versions also accept 'ME'.
    We try both, and fall back to a group-by month-end if needed.
    """
    if df is None or df.empty:
        return df
    for rule in ("ME", "M"):
        try:
            return df.resample(rule).last()
        except Exception:
            continue

    # Fallback: take last obs per calendar month
    out = df.copy()
    out["_ym"] = out.index.to_period("M")
    out = out.groupby("_ym").tail(1).drop(columns=["_ym"])
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def _fig_from_spec(data: list[dict], layout: dict, height: int) -> go.Figure:
    """Build Plotly figure from dict specs (keeps original trace/layout logic).

    Keeps horizontal legends with negative y *inside* the canvas to avoid Streamlit scrollbars.
    """
    fig = go.Figure(data=data, layout=layout)

    # ---- Defensive Plotly hygiene ----
    # Plotly/Streamlit + unified hover can show a bold "undefined" when:
    #   - x-axis auto-type inference misfires for datetime-like x values, or
    #   - a helper trace has name=None.
    # We harden both cases without changing the visual output.
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
    # Force black font for all elements using safest update methods
    fig.update_layout(
        font=dict(color="black"),
        hoverlabel=dict(namelength=-1),
        legend_font_color="black",
    )

    # Avoid Plotly rendering a bold "undefined" title when only title styling is applied.
    # This can happen if a figure has no explicit title text but we set title font/color.
    try:
        t_text = None
        if getattr(fig.layout, "title", None) is not None:
            t_text = getattr(fig.layout.title, "text", None)

        if t_text is None or str(t_text).strip().lower() == "undefined":
            fig.update_layout(title_text="")

        fig.update_layout(title=dict(font=dict(color="black")))
    except Exception:
        pass
    fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"), linecolor="black", gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"), linecolor="black", gridcolor="rgba(0,0,0,0.05)")

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
                fig.update_layout(margin=dict(**base_margin, b=needed_b), height=height + max(0, needed_b - base_margin.get("b", 0)))
    except Exception:
        pass

    return fig


def _inject_light_ui():
    """Subtle, professional, mostly-white UI polish (same approach as app_eu_dashboard)."""
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


          /* Sticky Navigation Bar */
          div[data-testid="stRadio"] {
            position: sticky;
            top: 2.85rem;
            z-index: 1000;
            background: #F7F8FB;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
          }

          /* Adjust block container for sticky header */
          .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_chart(
    title: str,
    fig: go.Figure | None,
    observations: list[str] | None,
    sources: list[str] | None,
    label_guide: list[str] | None = None,
    *,
    show_details: bool = True,
):
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

    # Static "Details" area (no clickable header) — aligns with the neutral card style.
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

    if label_guide:
        st.markdown("---")
        st.markdown("#### Label guide")
        st.markdown("\n".join([f"- {lg}" for lg in label_guide]))

    st.markdown("</div>", unsafe_allow_html=True)


def _series_last_and_prev(s: pd.Series) -> tuple[pd.Timestamp | None, float | None, pd.Timestamp | None, float | None]:
    if s is None:
        return None, None, None, None
    ss = pd.to_numeric(s, errors="coerce").dropna()
    if ss.empty:
        return None, None, None, None
    last_dt = pd.to_datetime(ss.index[-1])
    last_val = float(ss.iloc[-1])
    if len(ss) >= 2:
        prev_dt = pd.to_datetime(ss.index[-2])
        prev_val = float(ss.iloc[-2])
        return last_dt, last_val, prev_dt, prev_val
    return last_dt, last_val, None, None


def _metric_from_series(
    name: str,
    s: pd.Series,
    unit: str = "%",
    decimals: int = 2,
    *,
    delta_mode: str = "diff",  # "diff" or "pct"
) -> dict | None:
    last_dt, last_val, prev_dt, prev_val = _series_last_and_prev(s)
    if last_dt is None or last_val is None:
        return None

    label = f"{name} (as of {last_dt.strftime('%Y-%m-%d')})"
    value = f"{last_val:.{decimals}f}{unit}"

    delta = None
    if prev_dt is not None and prev_val is not None and np.isfinite(prev_val) and prev_val != 0:
        if delta_mode == "pct":
            d = (last_val - prev_val) / prev_val * 100.0
            delta = f"{d:+.{max(1, min(2, decimals))}f}% vs {prev_dt.strftime('%Y-%m-%d')}"
        else:
            d = last_val - prev_val
            suffix = " pp" if unit == "%" else ""
            delta = f"{d:+.{decimals}f}{suffix} vs {prev_dt.strftime('%Y-%m-%d')}"

    return {"label": label, "value": value, "delta": delta}


# =============================================================================
# DATA FETCHING (CACHED)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id: str, observation_start: str | None = None) -> pd.Series:
    """Fetch a FRED series as a pd.Series (DatetimeIndex).

    Primary: FRED JSON API (uses API key)
    Fallback: fredgraph CSV endpoint (no key), which is often more tolerant when keys / rate-limits are an issue.
    """
    series_id = str(series_id).strip()

    def _from_api() -> pd.Series:
        if not FRED_API_KEY:
            return pd.Series(dtype=float)

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
        if observation_start:
            params["observation_start"] = observation_start

        backoff = 0.8
        for attempt in range(6):
            try:
                r = SESSION.get(url, params=params, timeout=50)
                # If we still get throttled / server errors, backoff and retry.
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff + random.random() * 0.2)
                    backoff = min(backoff * 1.8, 8.0)
                    continue
                r.raise_for_status()
                data = r.json()
                df = pd.DataFrame(data.get("observations", []))
                if df.empty:
                    return pd.Series(dtype=float)
                df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
                df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
                df = df.dropna(subset=["date", "value"])
                if df.empty:
                    return pd.Series(dtype=float)
                return df.set_index("date")["value"].sort_index()
            except Exception:
                time.sleep(backoff + random.random() * 0.2)
                backoff = min(backoff * 1.8, 8.0)
        return pd.Series(dtype=float)

    def _from_fredgraph() -> pd.Series:
        # fredgraph CSV doesn't need an API key
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        params = {"id": series_id, "cosd": observation_start or "1900-01-01", "coed": end}
        backoff = 0.8
        for attempt in range(5):
            try:
                r = SESSION.get(url, params=params, timeout=50)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff + random.random() * 0.2)
                    backoff = min(backoff * 1.8, 8.0)
                    continue
                r.raise_for_status()
                text = (r.text or "").strip()
                if not text or text.lower().startswith("<!doctype html") or text.lower().startswith("<html"):
                    return pd.Series(dtype=float)

                df = pd.read_csv(StringIO(text))
                if df.empty:
                    return pd.Series(dtype=float)

                # Header differs sometimes; be defensive.
                col0 = df.columns[0]
                date_col = "DATE" if "DATE" in df.columns else ("date" if "date" in df.columns else col0)
                val_col = series_id if series_id in df.columns else df.columns[-1]

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                df = df.dropna(subset=[date_col, val_col])
                if df.empty:
                    return pd.Series(dtype=float)
                return df.set_index(date_col)[val_col].sort_index()
            except Exception:
                time.sleep(backoff + random.random() * 0.2)
                backoff = min(backoff * 1.8, 8.0)
        return pd.Series(dtype=float)

    s = _from_api()
    if s.empty:
        s = _from_fredgraph()
    return s

def ofr_series(mnemonic: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Fetch OFR series (BGCR/TGCR) and return DF with columns date,value.
    
    Note: The OFR API doesn't use startDate parameter in the URL. 
    We fetch all data and filter afterwards (matches original US_Dashboard.py approach).
    """
    url = "https://data.financialresearch.gov/v1/series/timeseries"
    params = {"mnemonic": mnemonic}  # Don't include startDate - API doesn't support it
    r = SESSION.get(url, params=params, timeout=40)
    r.raise_for_status()
    j = r.json()
    # Handle both list of dicts and nested structure
    if isinstance(j, list):
        data = j
    else:
        data = j.get("data", [])
    if not data:
        return pd.DataFrame(columns=["date", "value"])
    out = pd.DataFrame(data, columns=["date", "value"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    return out[["date", "value"]].sort_values("date")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_mpt_data() -> pd.DataFrame:
    r = SESSION.get(MPT_XLSX_URL, timeout=60)
    r.raise_for_status()
    mpt_df = pd.read_excel(BytesIO(r.content), sheet_name="DATA")
    mpt_df["date"] = pd.to_datetime(mpt_df["date"], errors="coerce")
    mpt_df["reference_start"] = pd.to_datetime(mpt_df["reference_start"], errors="coerce")
    return mpt_df.dropna(subset=["date", "reference_start"])


def _sep_date_from_url(url: str) -> pd.Timestamp | None:
    m = SEP_URL_RE.search(url or "")
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d", errors="raise")
    except Exception:
        return None


def _extract_sep_urls_from_html(html: str, base_url: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for href in re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE):
        m = SEP_URL_RE.search(href)
        if not m:
            continue
        date_key = m.group(1)
        clean_href = href.split("#", 1)[0].split("?", 1)[0]
        out[date_key] = urljoin(base_url, clean_href)

    if out:
        return out

    # Fallback parser in case href extraction changes.
    for m in SEP_URL_RE.finditer(html or ""):
        date_key = m.group(1)
        out[date_key] = urljoin(base_url, m.group(0))
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def discover_latest_sep_url() -> tuple[str, str]:
    candidates: dict[str, str] = {}
    for page_url in SEP_DISCOVERY_PAGES:
        try:
            r = SESSION.get(page_url, timeout=40)
            r.raise_for_status()
            candidates.update(_extract_sep_urls_from_html(r.text, page_url))
        except Exception:
            continue

    if candidates:
        latest_key = max(candidates.keys())
        latest_dt = pd.to_datetime(latest_key, format="%Y%m%d", errors="coerce")
        latest_iso = latest_dt.strftime("%Y-%m-%d") if pd.notna(latest_dt) else "n/a"
        return candidates[latest_key], latest_iso

    fallback_dt = _sep_date_from_url(SEP_FALLBACK_URL)
    fallback_iso = fallback_dt.strftime("%Y-%m-%d") if fallback_dt is not None else "n/a"
    return SEP_FALLBACK_URL, fallback_iso


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for tpl in out.columns.to_flat_index():
            parts = [str(x).strip() for x in tpl if str(x).strip() and str(x).strip().lower() not in {"nan", "none"}]
            cols.append(" | ".join(parts))
        out.columns = cols
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _select_sep_figure2_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for tbl in tables:
        if tbl is None or tbl.empty:
            continue
        df = _flatten_columns(tbl)
        first_col = str(df.columns[0]).lower() if len(df.columns) else ""
        if "midpoint of target range or target level" not in first_col:
            continue
        year_cols = sum(bool(re.search(r"\b20\d{2}\b", str(c))) for c in df.columns[1:])
        if year_cols >= 2:
            return df
    raise ValueError("Figure 2 federal funds projection table not found")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sep_figure2_table() -> tuple[pd.DataFrame, str, str]:
    latest_url, latest_date = discover_latest_sep_url()
    candidate_urls = [latest_url]
    if latest_url != SEP_FALLBACK_URL:
        candidate_urls.append(SEP_FALLBACK_URL)

    errors = []
    for url in candidate_urls:
        try:
            r = SESSION.get(url, timeout=40)
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))
            df = _select_sep_figure2_table(tables)
            page_dt = _sep_date_from_url(url)
            page_date = page_dt.strftime("%Y-%m-%d") if page_dt is not None else latest_date
            return df, url, page_date
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue

    raise RuntimeError("Could not fetch Figure 2 SEP table. " + " | ".join(errors))


# =============================================================================
# SECTION 1 — FOMC / POLICY
# =============================================================================
def _extract_distribution_levels(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(vals).sum() >= max(3, int(0.5 * len(vals))):
        return vals

    mids = []
    for raw in series.astype(str):
        s = re.sub(r"\s+", "", raw)
        m = re.match(r"([0-9.]+)-([0-9.]+)", s)
        mids.append((float(m.group(1)) + float(m.group(2))) / 2.0 if m else np.nan)
    return np.array(mids, dtype=float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")

    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(np.interp(q, cum / total, v))


def _sep_percentiles_for_column(df: pd.DataFrame, col_idx: int) -> tuple[float, float, float]:
    levels = _extract_distribution_levels(df.iloc[:, 0])
    counts = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    p10 = _weighted_quantile(levels, counts, 0.10)
    p50 = _weighted_quantile(levels, counts, 0.50)
    p90 = _weighted_quantile(levels, counts, 0.90)
    return p50, p10, p90


def _sep_projection_columns(df: pd.DataFrame) -> list[tuple[str, int, str]]:
    selected: dict[str, int] = {}
    for i, col in enumerate(df.columns[1:], start=1):
        label = str(col)
        m = re.search(r"\b(20\d{2})\b", label)
        if m:
            selected[m.group(1)] = i
            continue
        if re.search(r"longer\s*run", label, flags=re.IGNORECASE):
            selected["Longer run"] = i

    years = sorted(int(y) for y in selected if y.isdigit())
    out: list[tuple[str, int, str]] = []
    for y in years:
        out.append((str(y), selected[str(y)], f"{y}-12-31"))

    if "Longer run" in selected:
        anchor_year = years[-1] + 2 if years else (pd.Timestamp.today().year + 2)
        out.append(("Longer run", selected["Longer run"], f"{anchor_year}-12-31"))
    return out

def value_on_or_before(series: pd.Series, t: pd.Timestamp) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    idx = s.index[s.index <= t]
    if len(idx) == 0:
        return float("nan")
    return float(s.loc[idx[-1]])


@st.cache_data(ttl=1800, show_spinner=False)
def build_fomc_figs():
    mpt_df = fetch_mpt_data()
    sep_source_url = "n/a"
    sep_source_date = "n/a"
    try:
        df_sep, sep_source_url, sep_source_date = fetch_sep_figure2_table()
    except Exception:
        df_sep = pd.DataFrame()

    # FRED series
    effr = fred_series("DFF", START_DATE_FOMC)
    t10y = fred_series("DGS10", START_DATE_FOMC)

    # ---- Plot 1: Monetary policy path ----
    unique_dates = np.sort(mpt_df["date"].dropna().unique())
    latest_date = pd.to_datetime(unique_dates[-1])
    lookback_cutoff = latest_date - pd.DateOffset(years=LOOKBACK_YEARS)

    historical_dates = []
    current_month = None
    for d in unique_dates:
        dt = pd.to_datetime(d)
        if lookback_cutoff <= dt < latest_date:
            ym = (dt.year, dt.month)
            if ym != current_month:
                historical_dates.append(d)
                current_month = ym

    latest_curve = (
        mpt_df[mpt_df["date"] == latest_date].pivot_table(index="reference_start", columns="field", values="value") / 100.0
    ).sort_index()

    data1 = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "10Y Treasury Yield (DGS10)",
            "x": t10y.index,
            "y": t10y.values.astype(float),
            "line": {"width": 1.8, "color": "black"},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Effective Fed Funds Rate (DFF)",
            "x": effr.index,
            "y": effr.values.astype(float),
            "line": {"width": 2.2, "color": "#9E9E9E"},
        },
    ]

    for d in historical_dates:
        curve = (mpt_df[mpt_df["date"] == d].pivot_table(index="reference_start", columns="field", values="value") / 100.0).sort_index()
        if "Rate: mean" not in curve.columns:
            continue
        data1.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"MPT mean ({pd.to_datetime(d).strftime('%Y-%m')})",
                "x": curve.index,
                "y": curve["Rate: mean"].values.astype(float),
                "line": {"width": 1, "color": "#e0e0e0"},
                "opacity": 0.9,
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    if "Rate: 25th percentile" in latest_curve.columns and "Rate: 75th percentile" in latest_curve.columns:
        x_band = latest_curve.index
        lo = latest_curve["Rate: 25th percentile"].values.astype(float)
        hi = latest_curve["Rate: 75th percentile"].values.astype(float)
        data1.append(
            {"type": "scatter", "mode": "lines", "name": "Market 25–75th percentile band", "x": x_band, "y": lo, "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"}
        )
        data1.append(
            {"type": "scatter", "mode": "lines", "name": "Market 25–75th percentile band", "x": x_band, "y": hi, "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(11, 61, 145, 0.18)"}
        )

    if "Rate: mean" in latest_curve.columns:
        data1.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"Mean implied path (MPT, {latest_date.strftime('%d-%b-%Y')})",
                "x": latest_curve.index,
                "y": latest_curve["Rate: mean"].values.astype(float),
                "line": {"width": 3.2, "color": "#0B3D91"},
            }
        )
    if "Rate: mode" in latest_curve.columns:
        data1.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"Mode implied path (MPT, {latest_date.strftime('%d-%b-%Y')})",
                "x": latest_curve.index,
                "y": latest_curve["Rate: mode"].values.astype(float),
                "line": {"width": 2.2, "color": "#0B3D91"},
                "marker": {"symbol": "star", "size": 10, "color": "#0B3D91"},
            }
        )

    # SEP dots with 10–90th error bars (Figure 2 federal funds distribution table)
    xs, ys, err_plus, err_minus, texts = [], [], [], [], []
    sep_cols = _sep_projection_columns(df_sep) if not df_sep.empty else []
    for label, col_idx, date_str in sep_cols:
        med, p10, p90 = _sep_percentiles_for_column(df_sep, col_idx)
        if np.isfinite(med):
            xs.append(pd.to_datetime(date_str))
            ys.append(float(med))
            err_plus.append(float(p90 - med) if np.isfinite(p90) else 0.0)
            err_minus.append(float(med - p10) if np.isfinite(p10) else 0.0)
            texts.append(str(label).upper())

    if xs:
        data1.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "name": "FOMC SEP median (with 10–90th)",
                "x": xs,
                "y": ys,
                "marker": {"size": 10, "color": "#0B3D91"},
                "text": texts,
                "textposition": "top center",
                "error_y": {"type": "data", "array": err_plus, "arrayminus": err_minus, "visible": True, "thickness": 2, "width": 6, "color": "#0B3D91"},
            }
        )

    layout1 = {
        "xaxis": {"title": "", "tickformat": "%Y"},
        "yaxis": {"title": "Rate / Yield (%)", "range": [0, 7]},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 35, "b": 95},
    }
    fig1 = _fig_from_spec(data1, layout1, height=520)

    effr_last = _safe_float(effr.dropna().iloc[-1]) if len(effr.dropna()) else np.nan
    t10_last = _safe_float(t10y.dropna().iloc[-1]) if len(t10y.dropna()) else np.nan
    mpt_mean0 = _safe_float(latest_curve["Rate: mean"].iloc[0]) if "Rate: mean" in latest_curve.columns else np.nan
    mpt_end = _safe_float(latest_curve["Rate: mean"].dropna().iloc[-1]) if "Rate: mean" in latest_curve.columns and len(latest_curve["Rate: mean"].dropna()) else np.nan

    obs1 = [
        "SEP dots show committee median & dispersion (10–90th)",
        (
            f"Figure 2 source: {sep_source_date}"
            if sep_source_date != "n/a"
            else "Figure 2 source unavailable (using chart without SEP dots if fetch fails)"
        ),
    ]
    src1 = [
        "FRED: DFF (EFFR), DGS10 (10Y)",
        "Atlanta Fed: Market Probability Tracker (MPT)",
        (
            f"Federal Reserve Board (Figure 2): {sep_source_url}"
            if sep_source_url != "n/a"
            else "Federal Reserve Board (Figure 2): unavailable"
        ),
    ]

    # ---- Plot 2: Breakeven inflation curve ----
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=30)

    inflation_data = {}
    maturities = list(INFLATION_SERIES.keys())
    for mat, sid in INFLATION_SERIES.items():
        ser = fred_series(sid, start.strftime("%Y-%m-%d"))
        inflation_data[mat] = ser.astype(float)

    df_infl = pd.DataFrame(inflation_data).sort_index()
    latest_common = df_infl.dropna().index.max()
    if pd.isna(latest_common):
        fig2 = None
        obs2 = ["Breakeven curve unavailable (no common date across maturities)."]
        src2 = [f"FRED series: {', '.join(INFLATION_SERIES.values())}"]
        return (fig1, obs1, src1, latest_date.strftime("%Y-%m-%d")), (fig2, obs2, src2)

    snap_targets = [latest_common - pd.DateOffset(months=k) for k in range(0, 25)]
    curves, used_dates = [], []
    for t in snap_targets:
        row = {m: value_on_or_before(df_infl[m], t) for m in maturities}
        curves.append([row[m] for m in maturities])
        used_dates.append(t)
    curves = np.array(curves, dtype=float)

    xcats = [f"{m}Y" for m in maturities]
    data2 = []
    for i in range(24, 1, -1):
        data2.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": f"{used_dates[i].strftime('%Y-%m')}",
                "x": xcats,
                "y": [float(x) for x in curves[i]],
                "line": {"width": 2, "color": "lightgrey"},
                "marker": {"size": 6, "color": "lightgrey"},
                "opacity": 0.35,
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    data2.append(
        {"type": "scatter", "mode": "lines+markers", "name": f"1 month earlier ({used_dates[1].strftime('%Y-%m')})", "x": xcats, "y": [float(x) for x in curves[1]], "line": {"width": 3, "color": "black"}, "marker": {"size": 7, "color": "black"}}
    )
    data2.append(
        {"type": "scatter", "mode": "lines+markers", "name": f"Latest ({used_dates[0].strftime('%Y-%m')})", "x": xcats, "y": [float(x) for x in curves[0]], "line": {"width": 3.5, "color": "#0B3D91"}, "marker": {"size": 8, "color": "#0B3D91"}}
    )

    layout2 = {
        "xaxis": {"title": "Maturity", "type": "category"},
        "yaxis": {"title": "Percent"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 35, "b": 95},
    }
    fig2 = _fig_from_spec(data2, layout2, height=480)

    latest_vals = {m: curves[0][i] for i, m in enumerate(maturities)}
    prev_vals = {m: curves[1][i] for i, m in enumerate(maturities)}
    slope_latest = latest_vals[max(maturities)] - latest_vals[min(maturities)]
    slope_prev = prev_vals[max(maturities)] - prev_vals[min(maturities)]
    obs2 = [
        "Breakevens are proxies — nominal yields embed term premia and, in principle, liquidity/credit premia. Moves may not reflect pure inflation expectations.",
        f"Curve slope (max–min maturity): {slope_latest:.2f}pp",
        "Grey lines show the prior 24 monthly snapshots",
    ]
    src2 = [f"FRED series: {', '.join(INFLATION_SERIES.values())}"]

    return (fig1, obs1, src1, latest_date.strftime("%Y-%m-%d")), (fig2, obs2, src2)


# =============================================================================
# SECTION 2 — YIELD CURVE
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def build_yield_curve_figs(lookback_months: int = 24):
    end_dt = pd.Timestamp.today().normalize()
    start_dt = (end_dt - pd.DateOffset(months=lookback_months + 1)).strftime("%Y-%m-%d")

    series = {}
    for sid, label in YIELD_CURVE_SERIES.items():
        s = fred_series(sid, start_dt)
        if s.empty:
            continue
        series[label] = s.rename(label)

    if not series:
        return (None, None, ["No yield data."], ["FRED"]), (None, None, ["No yield data."], ["FRED"]), "n/a"

    df_wide = pd.concat(series.values(), axis=1).sort_index().dropna(how="all")
    if df_wide.empty:
        return (None, None, ["No yield data."], ["FRED"]), (None, None, ["No yield data."], ["FRED"]), "n/a"

    df_ff = df_wide.ffill()
    df_monthly = _resample_month_end(df_ff).dropna()
    if len(df_monthly) < 2:
        return (None, None, ["Not enough history for monthly snapshots."], ["FRED"]), (None, None, ["Not enough history."], ["FRED"]), "n/a"

    df_monthly = df_monthly.tail(lookback_months + 1)
    latest_dt = df_monthly.index[-1]
    prev_dt = df_monthly.index[-2]

    xcats = [t for t in YIELD_CURVE_ORDER if t in df_monthly.columns]
    data1 = []
    for dt in df_monthly.index[:-2]:
        row = df_monthly.loc[dt]
        data1.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": dt.strftime("%Y-%m"),
                "x": xcats,
                "y": [float(row[t]) for t in xcats],
                "line": {"width": 1, "color": "rgba(160,160,160,0.55)"},
                "marker": {"size": 4, "color": "rgba(160,160,160,0.55)"},
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )

    for dt, color, name in [(prev_dt, "black", f"Prev ({prev_dt.strftime('%Y-%m')})"), (latest_dt, "#0B3D91", f"Latest ({latest_dt.strftime('%Y-%m')})")]:
        row = df_monthly.loc[dt]
        data1.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": name,
                "x": xcats,
                "y": [float(row[t]) for t in xcats],
                "line": {"width": 3.2, "color": color},
                "marker": {"size": 6, "color": color},
            }
        )

    latest_curve = df_monthly.loc[latest_dt]
    y2 = _safe_float(latest_curve.get("2Y"))
    y10 = _safe_float(latest_curve.get("10Y"))
    y30 = _safe_float(latest_curve.get("30Y"))
    s2s10 = y10 - y2 if np.isfinite(y10) and np.isfinite(y2) else np.nan
    y3m = _safe_float(latest_curve.get("3M"))
    s3m10 = y10 - y3m if np.isfinite(y10) and np.isfinite(y3m) else np.nan
    y5 = _safe_float(latest_curve.get("5Y"))
    s5s30 = y30 - y5 if np.isfinite(y30) and np.isfinite(y5) else np.nan

    obs1 = [
        f"Curve slopes: 10Y–2Y {s2s10:+.2f}pp · 10Y–3M {s3m10:+.2f}pp · 30Y–5Y {s5s30:+.2f}pp",
        "Fading curves are monthly snapshots over the last ~24 months",
    ]
    src = [f"FRED series: {', '.join(YIELD_CURVE_SERIES.keys())}"]

    layout1 = {
        "xaxis": {"title": "Maturity", "type": "category"},
        "yaxis": {"title": "Yield (%)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 35, "b": 95},
    }
    fig1 = _fig_from_spec(data1, layout1, height=480)

    # Daily time series
    df_plot = df_ff.tail(int(lookback_months * 31))
    data2 = []
    for tenor in YIELD_CURVE_ORDER:
        if tenor not in df_plot.columns:
            continue
        s = df_plot[tenor].dropna()
        if s.empty:
            continue
        data2.append({"type": "scatter", "mode": "lines", "name": tenor, "x": s.index, "y": s.values.astype(float), "line": {"width": 1.4}})

    layout2 = {
        "xaxis": {"title": "", "tickformat": "%Y-%m"},
        "yaxis": {"title": "Yield (%)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.30},
        "margin": {"l": 60, "r": 60, "t": 35, "b": 105},
    }
    fig2 = _fig_from_spec(data2, layout2, height=520)

    last_row = df_plot.dropna(how="all").iloc[-1] if not df_plot.dropna(how="all").empty else latest_curve
    obs2 = [
        f"Latest daily observation: {df_plot.index.max().strftime('%Y-%m-%d') if len(df_plot.index) else 'n/a'}",
        "Lines are daily yields over the last ~24 months (forward-filled over non-trading days)",
    ]

    return (fig1, obs1, src), (fig2, obs2, src), latest_dt.strftime("%Y-%m-%d")


# =============================================================================
# SECTION 3 — MONEY MARKET
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def mm_fetch_all_data():
    dfs = []
    for sid in MM_SERIES_IDS:
        df = fred_series(sid, observation_start="2018-01-01")
        dfs.append(pd.DataFrame({"date": df.index, "value": df.values}) if not df.empty else pd.DataFrame(columns=["date", "value"]))

    dfs_funds = []
    for sid in MM_FUNDS_IDS:
        s = fred_series(sid, observation_start="2018-01-01")
        dfs_funds.append(pd.DataFrame({"date": s.index, "value": s.values}) if not s.empty else pd.DataFrame(columns=["date", "value"]))

    for key in ["BGCR", "TGCR"]:
        try:
            df_ofr = ofr_series(MM_OFR_MNEMONICS[key], start_date="2018-01-01")
        except Exception:
            df_ofr = pd.DataFrame(columns=["date", "value"])
        dfs.append(df_ofr)

    return dfs, dfs_funds, MM_SERIES_NAMES


def mm_plot_rates_complex(
    dfs,
    dfs_funds,
    labels,
    cut_date: str,
    title: str,
    *,
    observations_override: list[str] | None = None,
):
    cut_date = pd.to_datetime(cut_date)
    has_rrp_vol = False
    latest_vals = {}

    data = []
    for i, df in enumerate(dfs):
        if df.empty:
            continue
        df_filt = df[df["date"] > cut_date].copy()
        if df_filt.empty:
            continue

        label = labels[i]
        val = float(df_filt["value"].iloc[-1])
        latest_vals[label] = val
        label_val = f"{label} ({val:.2f})"

        if "RRPONTTLD" in label:
            has_rrp_vol = True
            data.append(
                {"type": "bar", "name": label_val, "x": df_filt["date"], "y": df_filt["value"].astype(float), "yaxis": "y2", "opacity": 0.25}
            )
        else:
            data.append(
                {"type": "scatter", "mode": "lines", "name": label_val, "x": df_filt["date"], "y": df_filt["value"].astype(float), "line": {"width": 1.8}}
            )

    # Funds target range band
    if len(dfs_funds) >= 2:
        df_l, df_u = dfs_funds[0], dfs_funds[1]
        if (not df_l.empty) and (not df_u.empty):
            df_l = df_l[df_l["date"] > cut_date].set_index("date")
            df_u = df_u[df_u["date"] > cut_date].set_index("date")
            common = df_l.index.intersection(df_u.index)
            if not common.empty:
                lo_last = float(df_l.loc[common[-1], "value"])
                up_last = float(df_u.loc[common[-1], "value"])

                x_band = common
                y_lo = df_l.loc[common, "value"].values.astype(float)
                y_hi = df_u.loc[common, "value"].values.astype(float)
                data.append(
                    {"type": "scatter", "mode": "lines", "name": f"Funds target ({lo_last:.2f}-{up_last:.2f})", "x": x_band, "y": y_lo, "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"}
                )
                data.append(
                    {"type": "scatter", "mode": "lines", "name": f"Funds target ({lo_last:.2f}-{up_last:.2f})", "x": x_band, "y": y_hi, "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(120,120,120,0.10)"}
                )

    if observations_override is not None:
        obs = observations_override
    else:
        def _lv_contains(substr):
            for k, v in latest_vals.items():
                if substr in k:
                    return v
            return np.nan

        iorb = _lv_contains("IORB")
        sofr = _lv_contains("SOFR")
        dff = _lv_contains("DFF")
        tgcr = _lv_contains("Tri-Party")
        rrp = _lv_contains("ON RRP Award Rate")

        obs = []
        if np.isfinite(sofr) and np.isfinite(iorb):
            obs.append(f"SOFR–IORB: {sofr - iorb:+.2f}pp (repo rate vs administered rate)")
        if np.isfinite(dff) and np.isfinite(iorb):
            obs.append(f"DFF–IORB: {dff - iorb:+.2f}pp (reserve tightness proxy)")
        if np.isfinite(tgcr) and np.isfinite(rrp):
            obs.append(f"TGCR–RRP: {tgcr - rrp:+.2f}pp (private repo demand proxy)")

    src = ["FRED (API): rates, RRP award & volume, discount window", "OFR: BGCR/TGCR via data.financialresearch.gov"]

    layout = {"xaxis": {"title": ""}, "yaxis": {"title": "Rate (%)"}, "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25}, "margin": {"l": 60, "r": 60, "t": 35, "b": 95}}
    if has_rrp_vol:
        layout["yaxis2"] = {"title": "Billions USD", "overlaying": "y", "side": "right", "showgrid": False}

    fig = _fig_from_spec(data, layout, height=520)
    return fig, obs, src


def mm_plot_stress_spread(
    dfs,
    idx1: int,
    idx2: int,
    labels,
    cut_date: str,
    title: str,
    *,
    observations_override: list[str] | None = None,
):
    cut_date = pd.to_datetime(cut_date)
    df1, df2 = dfs[idx1], dfs[idx2]
    if df1.empty or df2.empty:
        return None, ["Spread unavailable (missing data)."], ["FRED/OFR"]

    df1 = df1[df1["date"] > cut_date].copy()
    df2 = df2[df2["date"] > cut_date].copy()
    merged = pd.merge(df1, df2, on="date", suffixes=("_1", "_2"))
    if merged.empty:
        return None, ["Spread unavailable (no overlap)."], ["FRED/OFR"]

    diff = merged["value_1"] - merged["value_2"]
    last_val = float(diff.iloc[-1])

    status = ""
    if "Cash vs Collateral" in title:
        status = "Excess Cash" if last_val < 0 else "Excess Collateral"
    elif "Scarcity of Reserves" in title:
        status = "No stress" if last_val < 0 else "Stress"
    elif "Bank activity" in title:
        status = "Banks lend cash" if last_val < 0 else "Banks borrow cash"

    label = f"{labels[idx1]} - {labels[idx2]} (Latest: {last_val:.2f} {status})"

    data = [
        {"type": "scatter", "mode": "lines", "name": label, "x": merged["date"], "y": diff.values.astype(float), "line": {"width": 2.6, "color": "#0B3D91"}}
    ]
    shapes = [
        {"type": "line", "xref": "x", "yref": "y", "x0": merged["date"].iloc[0], "x1": merged["date"].iloc[-1], "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}}
    ]

    if observations_override is not None:
        obs = observations_override
    else:
        obs = [f"Latest spread: {last_val:+.2f}pp ({status or 'interpretation depends on context'})", "0-line is a useful threshold for 'stress' vs 'no stress' read-through"]

    src = ["FRED (API): SOFR, IORB, DFF, RRP award", "OFR: TGCR/BGCR"]

    layout = {"xaxis": {"title": ""}, "yaxis": {"title": "Spread (pp)"}, "shapes": shapes, "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25}, "margin": {"l": 60, "r": 60, "t": 35, "b": 95}}
    fig = _fig_from_spec(data, layout, height=480)
    return fig, obs, src


@st.cache_data(ttl=1800, show_spinner=False)
def build_money_market_figs():
    dfs, dfs_funds, names = mm_fetch_all_data()

    obs_specs_plot1 = [
        "DPCREDIT: Discount Window Primary Credit Rate",
        "IORB - UNSEC: Interest Rate on Reserve Balances",
        "DFF - UNSEC: Federal Funds Effective Rate",
        "OBFR - UNSEC: Overnight Bank Funding Rate",
        "SOFR - SEC: Secured Overnight Financing Rate",
        "RRPONTSYAWARD: Overnight Reverse Repurchase Agreements Award Rate (Treasury collateral)",
        "RRPONTTLD: Overnight Reverse Repurchase Agreements total volume (Treasury collateral)",
    ]

    prev_year = datetime.datetime.now().year - 1
    prev_year_jan = f"{prev_year}-01-01"

    fig_a, obs_a, src_a = mm_plot_rates_complex(dfs, dfs_funds, names, "2020-01-01", "Money Market Rates (Since 2020)", observations_override=obs_specs_plot1)
    fig_b, obs_b, src_b = mm_plot_rates_complex(dfs, dfs_funds, names, prev_year_jan, f"Money Market Rates (Since {prev_year_jan})")

    title_c = "PRIVATE REPO DEMAND (TGCR - ON RRP): Demand for Cash vs Collateral"
    title_d = "BANK REPOS (SOFR - IORB): Bank activity in Repo"
    title_e = "RESERVE DEMAND (Fed Funds - IORB): Scarcity of Reserves"

    fig_c, obs_c, src_c = mm_plot_stress_spread(dfs, 8, 5, names, "2020-01-01", title_c)
    fig_d, obs_d, src_d = mm_plot_stress_spread(
        dfs,
        4,
        1,
        names,
        "2020-01-01",
        title_d,
        observations_override=["SOFR above IORB can indicate sizeable bank activity in repo; the sign matters for liquidity read-through."],
    )
    fig_e, obs_e, src_e = mm_plot_stress_spread(
        dfs,
        2,
        1,
        names,
        "2020-01-01",
        title_e,
        observations_override=["Spread between Fed Funds & IORB: if positive, suggests scarcer reserves historically; the Fed may add reserves."],
    )

    return {
        "plot_a": (fig_a, obs_a, src_a, "2020-01-01"),
        "plot_b": (fig_b, obs_b, src_b, prev_year_jan),
        "plot_c": (fig_c, obs_c, src_c),
        "plot_d": (fig_d, obs_d, src_d),
        "plot_e": (fig_e, obs_e, src_e),
        "titles": {"c": title_c, "d": title_d, "e": title_e},
        "raw": (dfs, dfs_funds, names),
    }


# =============================================================================
# SECTION 4 — HEDGE FUNDS (CFTC)
# =============================================================================
@st.cache_data(ttl=12 * 3600, show_spinner=False)
def hf_fetch_all_cftc_years(start_year=2010) -> pd.DataFrame | None:
    current_year = datetime.datetime.now().year
    combined = []
    urls = [CFTC_HISTORY_BLOCK] + [CFTC_YEAR_URL.format(year=y) for y in range(start_year, current_year + 1)]
    for url in urls:
        try:
            resp = SESSION.get(url, headers=CFTC_USER_AGENT, timeout=60)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(_io.BytesIO(resp.content)) as z:
                fname = z.namelist()[0]
                with z.open(fname) as f:
                    df = pd.read_csv(f, low_memory=False)
                    combined.append(df)
        except Exception:
            pass
    if not combined:
        return None
    full = pd.concat(combined, ignore_index=True)
    full.columns = full.columns.str.strip()
    return full


def hf_clean_cftc_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    date_col_1 = "Report_Date_as_YYYY-MM-DD"
    date_col_2 = "Report_Date_as_MM_DD_YYYY"

    d1 = pd.to_datetime(df[date_col_1], format="%Y-%m-%d", errors="coerce") if date_col_1 in df.columns else pd.Series([pd.NaT] * len(df))
    d2 = pd.to_datetime(df[date_col_2], format="%m-%d-%Y", errors="coerce") if date_col_2 in df.columns else pd.Series([pd.NaT] * len(df))

    df["Report_Date"] = d1.fillna(d2)
    df = df.dropna(subset=["Report_Date"])

    cols_to_drop = [c for c in [date_col_1, date_col_2] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.sort_values("Report_Date").drop_duplicates(subset=["Market_and_Exchange_Names", "Report_Date"])
    return df


def hf_extract_treasury_positions(df: pd.DataFrame) -> pd.DataFrame:
    target_patterns = {
        "2Y": ["2-YEAR U.S. TREASURY NOTES", "UST 2Y NOTE"],
        "5Y": ["5-YEAR U.S. TREASURY NOTES", "UST 5Y NOTE"],
        "10Y": ["10-YEAR U.S. TREASURY NOTES", "UST 10Y NOTE"],
        "10Y_Ultra": ["ULTRA 10-YEAR U.S. T-NOTES", "ULTRA UST 10Y"],
        "30Y": ["U.S. TREASURY BONDS", "UST BOND"],
        "30Y_Ultra": ["ULTRA U.S. TREASURY BONDS", "ULTRA UST BOND"],
    }

    plot_list = []
    for label, patterns in target_patterns.items():
        regex_pattern = "|".join(patterns)
        mask = df["Market_and_Exchange_Names"].astype(str).str.contains(regex_pattern, case=False, na=False)
        temp = df[mask].copy()

        temp["Lev_Money_Positions_Short_All"] = pd.to_numeric(temp.get("Lev_Money_Positions_Short_All"), errors="coerce")
        temp["Lev_Money_Positions_Long_All"] = pd.to_numeric(temp.get("Lev_Money_Positions_Long_All"), errors="coerce")

        temp = temp.groupby("Report_Date").agg(
            {
                "Lev_Money_Positions_Short_All": "sum",
                "Lev_Money_Positions_Long_All": "sum",
            }
        )

        temp.columns = [f"{label}_Shorts", f"{label}_Longs"]
        plot_list.append(temp)

    multi = pd.concat(plot_list, axis=1).sort_index().fillna(0)
    return multi


def build_hedge_fund_figs():
    raw_df = hf_fetch_all_cftc_years(start_year=2010)
    if raw_df is None:
        return None

    df = hf_clean_cftc_data(raw_df)
    multi_ust = hf_extract_treasury_positions(df)

    multiplier = 100_000
    trillion = 1e12
    tenors = ["2Y", "5Y", "10Y", "10Y_Ultra", "30Y", "30Y_Ultra"]

    plot_df = pd.DataFrame(index=multi_ust.index)
    for t in tenors:
        plot_df[f"{t}_Short"] = (multi_ust[f"{t}_Shorts"] * multiplier) / trillion
        plot_df[f"{t}_Long"] = (multi_ust[f"{t}_Longs"] * multiplier) / trillion
    plot_df = plot_df.loc["2018-01-01":].copy()
    if plot_df.empty:
        return None

    repo_date = pd.to_datetime("2019-09-17")
    covid_date = pd.to_datetime("2020-03-15")
    surge_start = pd.to_datetime("2023-01-01")
    last_x = plot_df.index.max()

    common_shapes = [
        {"type": "line", "xref": "x", "yref": "paper", "x0": repo_date, "x1": repo_date, "y0": 0, "y1": 1, "line": {"color": "red", "width": 1.5, "dash": "dash"}},
        {"type": "line", "xref": "x", "yref": "paper", "x0": covid_date, "x1": covid_date, "y0": 0, "y1": 1, "line": {"color": "darkred", "width": 1.5, "dash": "dash"}},
        {"type": "rect", "xref": "x", "yref": "paper", "x0": surge_start, "x1": last_x, "y0": 0, "y1": 1, "fillcolor": "rgba(255,165,0,0.10)", "line": {"width": 0}},
    ]

    # Shorts
    data_short = []
    for t in tenors:
        data_short.append({"type": "scatter", "mode": "lines", "name": t, "x": plot_df.index, "y": plot_df[f"{t}_Short"].values.astype(float), "stackgroup": "shorts", "line": {"width": 0.6}})
    layout_short = {
        "yaxis": {"title": "Notional (Trillions $)", "tickprefix": "$", "ticksuffix": "T"},
        "xaxis": {"title": ""},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
        "shapes": common_shapes,
    }
    fig_short = _fig_from_spec(data_short, layout_short, height=420)

    # Longs
    data_long = []
    for t in tenors:
        data_long.append({"type": "scatter", "mode": "lines", "name": t, "x": plot_df.index, "y": plot_df[f"{t}_Long"].values.astype(float), "stackgroup": "longs", "line": {"width": 0.6}})
    layout_long = {
        "yaxis": {"title": "Notional (Trillions $)", "tickprefix": "$", "ticksuffix": "T"},
        "xaxis": {"title": "Report Date"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
        "shapes": common_shapes,
    }
    fig_long = _fig_from_spec(data_long, layout_long, height=420)

    latest = plot_df.iloc[-1]
    tot_short = float(sum(latest[f"{t}_Short"] for t in tenors))
    tot_long = float(sum(latest[f"{t}_Long"] for t in tenors))
    top_short_t = max(tenors, key=lambda t: float(latest[f"{t}_Short"]))
    top_long_t = max(tenors, key=lambda t: float(latest[f"{t}_Long"]))

    label_guide = [
        "2Y: deliverable U.S. Treasury notes with 1y9m–2y remaining maturity",
        "5Y: deliverable U.S. Treasury notes with remaining maturity around 4y2m–5y3m (contract spec varies by exchange rulebook)",
        "10Y: deliverable U.S. Treasury notes with remaining maturity typically ≥6y6m and <8y (approx)",
        "10Y_Ultra: Ultra 10-year note future (longer DV01 bucket than classic 10Y)",
        "30Y: classic U.S. Treasury bond future",
        "30Y_Ultra: Ultra bond future (longest duration bucket)",
        "Positioning: CFTC Disaggregated COT 'Leveraged Money' — Tuesday snapshot, released Fridays",
    ]
    obs_gross = [
        "Post-2023 shading highlights the surge often discussed as 'basis trade' activity",
    ]
    src = ["CFTC COT: Financial Futures (Disaggregated), 'Leveraged Money'", "Weekly reports (Tuesday positioning, released Friday)"]

    # Net positions
    net_df = pd.DataFrame(index=multi_ust.index)
    for t in tenors:
        net_df[f"{t}_Net"] = ((multi_ust[f"{t}_Longs"] - multi_ust[f"{t}_Shorts"]) * multiplier) / trillion
    net_df["Aggregate_Net"] = net_df[[f"{t}_Net" for t in tenors]].sum(axis=1)
    net_plot = net_df.loc["2018-01-01":].copy()

    data_net = []
    for t in tenors:
        s = net_plot[f"{t}_Net"].dropna()
        if s.empty:
            continue
        data_net.append({"type": "scatter", "mode": "lines", "name": f"{t} Net", "x": s.index, "y": s.values.astype(float), "line": {"width": 2}})
    data_net.append({"type": "scatter", "mode": "lines", "name": "TOTAL AGGREGATE NET", "x": net_plot.index, "y": net_plot["Aggregate_Net"].values.astype(float), "line": {"width": 3, "dash": "dash", "color": "black"}})

    if not net_plot.empty:
        shapes2 = common_shapes + [{"type": "line", "xref": "x", "yref": "y", "x0": net_plot.index.min(), "x1": net_plot.index.max(), "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}}]
    else:
        shapes2 = common_shapes
    layout_net = {
        "xaxis": {"title": ""}, "yaxis": {"title": "Net Notional (Trillions $)"}, "shapes": shapes2, "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25}, "margin": {"l": 60, "r": 40, "t": 45, "b": 95}
        }
    fig_net = _fig_from_spec(data_net, layout_net, height=520)

    latest_net = float(net_plot["Aggregate_Net"].dropna().iloc[-1]) if len(net_plot["Aggregate_Net"].dropna()) else np.nan
    driver = max(tenors, key=lambda t: abs(float(net_plot[f"{t}_Net"].dropna().iloc[-1]))) if len(net_plot) else "n/a"
    obs_net = [f"Latest aggregate net: {latest_net:+.2f}T (positive = net long)", f"Largest net contributor (abs): {driver}", "Dashed black line is total across the curve"]

    return {
        "gross": (fig_short, fig_long, obs_gross, src, label_guide),
        "net": (fig_net, obs_net, src, label_guide),
    }


# =============================================================================
# SECTION 5 — GOLD & LIQUIDITY
# =============================================================================
_gold_mem_cache = {}
_gold_cache_lock = threading.Lock()


def _gold_cache_paths(key: str):
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in key)
    return (GOLD_CACHE_DIR / f"{safe}.pkl", GOLD_CACHE_DIR / f"{safe}.meta")


def _gold_cache_get(key: str, ttl_sec: int):
    now = time.time()
    with _gold_cache_lock:
        entry = _gold_mem_cache.get(key)
        if entry and (now - entry["ts"] <= ttl_sec):
            return entry["data"], entry.get("meta", {})
    pkl, meta = _gold_cache_paths(key)
    if pkl.exists() and meta.exists():
        try:
            ts = float(meta.read_text(encoding="utf-8").strip().splitlines()[0])
            if now - ts <= ttl_sec:
                data = pd.read_pickle(pkl)
                m = {"cached_at": datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")}
                with _gold_cache_lock:
                    _gold_mem_cache[key] = {"ts": ts, "data": data, "meta": m}
                return data, m
        except Exception:
            pass
    return None, {}


def _gold_cache_get_stale(key: str):
    with _gold_cache_lock:
        entry = _gold_mem_cache.get(key)
        if entry:
            return entry["data"], entry.get("meta", {})
    pkl, meta = _gold_cache_paths(key)
    if pkl.exists():
        try:
            data = pd.read_pickle(pkl)
            m = {}
            if meta.exists():
                ts = float(meta.read_text(encoding="utf-8").strip().splitlines()[0])
                m["cached_at"] = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            return data, m
        except Exception:
            pass
    return None, {}


def _gold_cache_set(key: str, obj, meta_dict: dict | None = None):
    ts = time.time()
    pkl, meta = _gold_cache_paths(key)
    try:
        pd.to_pickle(obj, pkl)
        meta.write_text(f"{ts}\n", encoding="utf-8")
    except Exception:
        pass
    m = {"cached_at": datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")}
    if meta_dict:
        m.update(meta_dict)
    with _gold_cache_lock:
        _gold_mem_cache[key] = {"ts": ts, "data": obj, "meta": m}
    return m


@st.cache_data(ttl=3600, show_spinner=False)
def gold_fred_series(series_id: str, start_date: str, end_date: str):
    """Fetch a FRED time series as a pandas Series indexed by date.

    We use the fredgraph CSV endpoint (no API key required). The CSV usually has a
    `DATE` column, but we defensively fall back to the first column if headers
    differ (e.g., `date`, `observation_date`) to avoid KeyError issues.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": series_id, "cosd": start_date, "coed": end_date}
    r = SESSION.get(url, params=params, timeout=40)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) < 2:
        return pd.Series(dtype=float)

    # Identify the date column (normally "DATE")
    date_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("date", "observation_date", "time", "timestamp", "datetime"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Identify the first non-date column as the value column
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        return pd.Series(dtype=float)
    value_col = value_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    s = (
        df.dropna(subset=[date_col, value_col])
        .set_index(date_col)[value_col]
        .sort_index()
    )
    s.name = series_id
    return s


def gold_fetch_m2(start_date: str):
    cache_key = f"m2_{start_date}"
    cached, meta = _gold_cache_get(cache_key, GOLD_M2_CACHE_TTL_SEC)
    if cached is not None and isinstance(cached, pd.DataFrame):
        return cached["M2_billions"], cached["M2_yoy_pct"], {"source": "FRED (WM2NS)", **meta}

    end_dt = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    start_dt = pd.to_datetime(start_date)
    m2_start = (start_dt - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    m2 = gold_fred_series("WM2NS", m2_start, end_dt).rename("M2_billions")
    m2_yoy = ((m2 / m2.shift(12)) - 1.0) * 100
    m2_yoy = m2_yoy.rename("M2_yoy_pct")

    m2 = m2.loc[start_dt:]
    m2_yoy = m2_yoy.loc[start_dt:]

    out = pd.DataFrame({"M2_billions": m2, "M2_yoy_pct": m2_yoy})
    meta = _gold_cache_set(cache_key, out, meta_dict={"source": "FRED (WM2NS)"})
    return out["M2_billions"], out["M2_yoy_pct"], meta


def gold_fetch_gold_gc_f(start_date: str, session: requests.Session):
    global _last_yahoo_call_ts
    cache_key = f"gold_gc_f_chart_{start_date}"

    cached, meta = _gold_cache_get(cache_key, GOLD_CACHE_TTL_SEC)
    if cached is not None and isinstance(cached, pd.Series) and not cached.empty:
        return cached, {"source": "Yahoo chart (GC=F)", "cache": "fresh", **meta}

    stale, stale_meta = _gold_cache_get_stale(cache_key)
    has_stale = stale is not None and isinstance(stale, pd.Series) and not stale.empty

    with _gold_fetch_lock:
        cached2, meta2 = _gold_cache_get(cache_key, GOLD_CACHE_TTL_SEC)
        if cached2 is not None and isinstance(cached2, pd.Series) and not cached2.empty:
            return cached2, {"source": "Yahoo chart (GC=F)", "cache": "fresh", **meta2}

        now = time.time()
        wait = GOLD_MIN_SECONDS_BETWEEN_YAHOO_CALLS - (now - _last_yahoo_call_ts)
        if wait > 0:
            time.sleep(wait + random.random())

        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
        end_ts = int((pd.Timestamp.today(tz="UTC").normalize() + pd.Timedelta(days=1)).timestamp())

        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,splits"}

        try:
            r = session.get(url, params=params, timeout=25)
            _last_yahoo_call_ts = time.time()

            if r.status_code == 429:
                msg = "429 Too Many Requests"
                if has_stale:
                    return stale, {"source": "Yahoo chart (GC=F)", "cache": "stale", "warning": msg, **(stale_meta or {})}
                r.raise_for_status()

            r.raise_for_status()
            j = r.json()
            result = j["chart"]["result"][0]
            ts = result["timestamp"]
            closes = result["indicators"]["quote"][0]["close"]
            idx = pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert(None)
            s = pd.Series(closes, index=idx, name="Close").dropna()
            if s.empty:
                raise RuntimeError("Parsed gold series is empty")
            meta_saved = _gold_cache_set(cache_key, s, meta_dict={"source": "Yahoo chart (GC=F)"})
            return s, {"source": "Yahoo chart (GC=F)", "cache": "fresh", **meta_saved}
        except Exception as e:
            if has_stale:
                return stale, {"source": "Yahoo chart (GC=F)", "cache": "stale", "warning": f"Fetch failed: {e}", **(stale_meta or {})}
            raise


def _gold_pick_member(names, must_contain):
    names_l = [(n, n.lower()) for n in names]
    matches = [n for n, nl in names_l if all(s.lower() in nl for s in must_contain)]
    if not matches:
        raise ValueError(f"No zip member found containing: {must_contain}")

    def score(n):
        nl = n.lower()
        s = 0
        if "historical" in nl:
            s += 30
        if "hist" in nl:
            s += 20
        if "current" in nl:
            s += 10
        return s

    return sorted(matches, key=score, reverse=True)[0]


def _gold_read_ids0182_table_from_zip(zip_bytes, kind, sheet_name):
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as z:
        members = z.namelist()
        if kind.lower() == "imports":
            xlsx_name = _gold_pick_member(members, must_contain=("imports",))
        else:
            xlsx_name = _gold_pick_member(members, must_contain=("exports",))
        xbytes = z.read(xlsx_name)
    df = pd.read_excel(_io.BytesIO(xbytes), sheet_name=sheet_name, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Enduse Code" not in df.columns or "Year" not in df.columns:
        raise ValueError(f"Unexpected format in {xlsx_name} / sheet={sheet_name}")
    return df


def _gold_wide_code_table_to_monthly_series(df, code):
    code = str(code).strip()
    sub = df[df["Enduse Code"].astype(str).str.strip() == code].copy()
    if sub.empty:
        raise ValueError(f"Code '{code}' not found in IDS-0182 table")
    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    out_idx, out_val = [], []
    for _, row in sub.iterrows():
        year = int(row["Year"])
        for mi, mname in enumerate(month_cols, start=1):
            v = row.get(mname)
            if pd.isna(v) or v == "":
                continue
            try:
                vnum = float(v)
            except Exception:
                continue
            out_idx.append(pd.Timestamp(year, mi, 1))
            out_val.append(vnum)
    s = pd.Series(out_val, index=pd.DatetimeIndex(out_idx)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def gold_fetch_bea_net_flow(start_date: str):
    cache_key = f"bea_netflow_{start_date}_{GOLD_SHEET_NAME}_{GOLD_IMPORT_CODE}_{GOLD_EXPORT_CODE}"
    cached, meta = _gold_cache_get(cache_key, GOLD_BEA_CACHE_TTL_SEC)
    if cached is not None and isinstance(cached, pd.Series):
        return cached, {"source": "BEA IDS-0182", **meta}

    resp = GOLD_HTTP_SESSION.get(IDS0182_URL, timeout=180)
    resp.raise_for_status()
    zip_bytes = resp.content

    imports_df = _gold_read_ids0182_table_from_zip(zip_bytes, kind="imports", sheet_name=GOLD_SHEET_NAME)
    exports_df = _gold_read_ids0182_table_from_zip(zip_bytes, kind="exports", sheet_name=GOLD_SHEET_NAME)

    imports_series = _gold_wide_code_table_to_monthly_series(imports_df, GOLD_IMPORT_CODE).rename("imports")
    exports_series = _gold_wide_code_table_to_monthly_series(exports_df, GOLD_EXPORT_CODE).rename("exports")

    net = (exports_series - imports_series).rename("NetGoldFlow").loc[pd.to_datetime(start_date):]
    meta = _gold_cache_set(cache_key, net, meta_dict={"source": "BEA IDS-0182"})
    return net, meta


@st.cache_data(ttl=1800, show_spinner=False)
def build_gold_figs():
    start_date = GOLD_START_DATE
    gold, gold_meta = gold_fetch_gold_gc_f(start_date, GOLD_HTTP_SESSION)
    m2, m2_yoy, _m2_meta = gold_fetch_m2(start_date)
    net_flow, _bea_meta = gold_fetch_bea_net_flow(start_date)

    # Defensive cleanup (prevents empty/NaN series from blanking the whole section)
    gold = pd.to_numeric(gold, errors="coerce").dropna().sort_index()
    if gold.empty or len(gold) < 10:
        obs = ["Gold price series unavailable (Yahoo Finance fetch/cache returned no usable observations)."]
        src = ["Yahoo Finance chart: GC=F (gold futures continuous proxy)"]
        return None, None, obs, src

    # Trend band on gold (exp trend in log-space)
    days = (gold.index - gold.index[0]).days.values.astype(float)
    b1, b0 = np.polyfit(days, np.log(gold.values), 1)
    trend = np.exp(b0 + b1 * days)
    lower = trend * 0.90
    upper = trend * 1.10

    # Top chart
    data_top = [
        {"type": "scatter", "mode": "lines", "name": "Gold trend band (±10%)", "x": gold.index, "y": lower.astype(float), "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "name": "Gold trend band (±10%)", "x": gold.index, "y": upper.astype(float), "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(158,158,158,0.18)"},
        {"type": "scatter", "mode": "lines", "name": "Gold exp. trend", "x": gold.index, "y": trend.astype(float), "line": {"dash": "dot", "width": 1.4, "color": "#9E9E9E"}},
        {"type": "scatter", "mode": "lines", "name": "Gold (GC=F, USD/oz)", "x": gold.index, "y": gold.values.astype(float), "line": {"width": 1.8, "color": "black"}},
    ]

    if m2 is not None and len(m2.dropna()):
        m2s = m2.dropna()
        data_top.append({"type": "scatter", "mode": "lines", "name": "US M2 (WM2NS, bn)", "x": m2s.index, "y": m2s.values.astype(float), "yaxis": "y2", "line": {"dash": "dash", "width": 1.4, "color": "#0B3D91"}})

    if m2_yoy is not None and len(m2_yoy.dropna()):
        my = m2_yoy.dropna()
        data_top.append({"type": "bar", "name": "M2 YoY (%)", "x": my.index, "y": my.values.astype(float), "yaxis": "y3", "opacity": 0.25})

    # Shared x-range
    active_series = [s.dropna() for s in [gold, m2, m2_yoy, net_flow] if s is not None and hasattr(s, "dropna") and not s.dropna().empty]
    if active_series:
        _xmin = min([as_s.index.min() for as_s in active_series])
        _xmax = max([as_s.index.max() for as_s in active_series])
        _xrange = [_xmin, _xmax]
    else:
        _xrange = [None, None]

    layout_top = {
        "xaxis": {"title": "", "tickformat": "%b %Y", "domain": [0.0, 0.86], "range": _xrange},
        "yaxis": {"title": "Gold (USD/oz)"},
        "yaxis2": {"title": "US M2 (bn)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.89},
        "yaxis3": {"title": "US M2 YoY (%)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.98},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 110, "t": 45, "b": 95},
    }
    fig_top = _fig_from_spec(data_top, layout_top, height=560)

    # Bottom chart
    fig_bot = None
    if net_flow is not None and len(net_flow.dropna()):
        net_b = (net_flow.dropna() / 1000.0).rename("NetGoldFlow_B")
        data_bot = [{"type": "bar", "name": "Net gold flow (USD bn)", "x": net_b.index, "y": net_b.values.astype(float), "opacity": 0.75}]
        shapes = [{"type": "line", "xref": "x", "yref": "y", "x0": net_b.index.min(), "x1": net_b.index.max(), "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}}]
        layout_bot = {
            "xaxis": {"title": "", "tickformat": "%b %Y", "domain": [0.0, 0.86], "range": _xrange},
            "yaxis": {"title": "Net gold flow (USD bn)"},
            "shapes": shapes,
            "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
            "margin": {"l": 60, "r": 110, "t": 45, "b": 95},
            "hovermode": "x unified",
        }
        fig_bot = _fig_from_spec(data_bot, layout_bot, height=420)

    gold_last = float(gold.dropna().iloc[-1]) if len(gold.dropna()) else np.nan
    band_low = float(lower[-1]) if len(lower) else np.nan
    band_high = float(upper[-1]) if len(upper) else np.nan
    where = "within" if (np.isfinite(gold_last) and np.isfinite(band_low) and band_low <= gold_last <= band_high) else "outside"
    m2y_last = float(m2_yoy.dropna().iloc[-1]) if (m2_yoy is not None and len(m2_yoy.dropna())) else np.nan
    flow_last = float(net_flow.dropna().iloc[-1] / 1000.0) if (net_flow is not None and len(net_flow.dropna())) else np.nan

    obs = [
        f"Gold latest: {gold_last:,.0f} (USD/oz), {where} the ±10% trend band", 
           ]
    if np.isfinite(flow_last):
        obs.append(f"Latest BEA net gold flow: {flow_last:+.2f} bn (exports − imports)")
    if isinstance(gold_meta, dict) and gold_meta.get("cache") == "stale":
        obs.append(f"Gold served from stale cache (Yahoo throttling): {gold_meta.get('warning','')}".strip())

    src = ["Yahoo Finance chart: GC=F (gold futures proxy)", "FRED: WM2NS (M2 money stock) via fredgraph.csv", f"BEA IDS-0182: {GOLD_SHEET_NAME} (codes {GOLD_EXPORT_CODE} − {GOLD_IMPORT_CODE})"]

    return fig_top, fig_bot, obs, src


# =============================================================================
# DASHBOARD DATA PREFETCH
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_dashboard_data() -> dict:
    """Fetch all dashboard data (cached).

    The original US_Dashboard.py would render "missing data" cards instead of crashing.
    Streamlit should behave similarly: one flaky endpoint shouldn't blank out the whole app.
    """
    # KPI series (fast window)
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=45)).strftime("%Y-%m-%d")
    dff = fred_series("DFF", start)
    iorb = fred_series("IORB", start)
    sofr = fred_series("SOFR", start)
    dgs10 = fred_series("DGS10", start)

    gold_kpi = pd.Series(dtype=float)
    try:
        gold_kpi, _ = gold_fetch_gold_gc_f((pd.Timestamp.today().normalize() - pd.DateOffset(days=60)).strftime("%Y-%m-%d"), GOLD_HTTP_SESSION)
    except Exception:
        pass

    kpis: list[dict] = []
    for payload in [
        _metric_from_series("EFFR (DFF)", dff, unit="%", decimals=2),
        _metric_from_series("IORB", iorb, unit="%", decimals=2),
        _metric_from_series("SOFR", sofr, unit="%", decimals=2),
        _metric_from_series("10Y (DGS10)", dgs10, unit="%", decimals=2),
        _metric_from_series("Gold (USD/oz)", gold_kpi, unit="", decimals=0, delta_mode="pct") if not gold_kpi.empty else None,
    ]:
        if payload:
            kpis.append(payload)

    def _err(msg: str) -> list[str]:
        return [f"⚠️ {msg}"] if msg else ["⚠️ Data unavailable."]

    # ===== FOMC =====
    try:
        (fomc1, obs1, src1, asof), (fomc2, obs2, src2) = build_fomc_figs()
    except Exception as e:
        msg = f"FOMC section failed: {e}"
        (fomc1, obs1, src1, asof) = (None, _err(msg), ["FRED / Atlanta Fed / Fed SEP"], "n/a")
        (fomc2, obs2, src2) = (None, _err(msg), ["FRED"],)

    # ===== YIELD CURVE =====
    try:
        yc1, yc2, yc_asof = build_yield_curve_figs()
    except Exception as e:
        msg = f"Yield curve section failed: {e}"
        yc1 = (None, _err(msg), ["FRED"])
        yc2 = (None, _err(msg), ["FRED"])
        yc_asof = "n/a"

    # ===== MONEY MARKET =====
    try:
        mm = build_money_market_figs()
    except Exception as e:
        msg = f"Money market section failed: {e}"
        mm = {
            "plot_a": (None, _err(msg), ["FRED / OFR"], "2020-01-01"),
            "plot_b": (None, _err(msg), ["FRED / OFR"], f"{pd.Timestamp.today().year - 1}-01-01"),
            "plot_c": (None, _err(msg), ["FRED / OFR"]),
            "plot_d": (None, _err(msg), ["FRED / OFR"]),
            "plot_e": (None, _err(msg), ["FRED / OFR"]),
            "titles": {"c": "Private repo demand", "d": "Bank repos", "e": "Reserve demand"},
        }

    # ===== HEDGE FUNDS =====
    try:
        hf = build_hedge_fund_figs()
    except Exception:
        hf = None

    # ===== GOLD =====
    try:
        gold_top, gold_bot, gold_obs, gold_src = build_gold_figs()
    except Exception as e:
        msg = f"Gold section failed: {e}"
        gold_top, gold_bot, gold_obs, gold_src = None, None, _err(msg), ["Yahoo Finance / FRED / BEA"]

    return {
        "kpis": kpis,
        "fomc": {"policy": (fomc1, obs1, src1, asof), "breakevens": (fomc2, obs2, src2)},
        "yield_curve": {"snapshots": yc1, "timeseries": yc2, "asof": yc_asof},
        "money_market": mm,
        "hedge_funds": hf,
        "gold": (gold_top, gold_bot, gold_obs, gold_src),
    }

def main():
    _inject_light_ui()

    with st.spinner("Loading data (cached)…"):
        data = load_dashboard_data()

    # Sidebar
    st.sidebar.title("Controls")
    if st.sidebar.button("Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: data is prefetched so switching sections is instant.")
    if not FRED_API_KEY:
        st.sidebar.warning(
            "FRED_API_KEY is not set. Public fallbacks are used where possible."
        )

    # Header row
    h1, h2, h3 = st.columns([0.72, 0.18, 0.10], vertical_alignment="bottom")
    with h1:
        st.title("🇺🇸 United States Macro Dashboard")
    with h2:
        st.caption(f"Local time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with h3:
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()


    # Removed KPI row

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Monetary Policy", "Yield Curve", "Money Market", "Hedge Funds", "Gold & Liquidity"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Removed Overview section logic

    # ===== POLICY =====
    if page == "Monetary Policy":
        fig1, obs1, src1, asof = data["fomc"]["policy"]
        st.caption(f"Market Probability Tracker snapshot as of: {asof}")
        _show_chart("Monetary Policy Path: EFFR vs Market-implied path vs FOMC SEP", fig1, obs1, src1)

        st.markdown("---")
        fig2, obs2, src2 = data["fomc"]["breakevens"]
        _show_chart("Breakeven Market-implied Breakeven Inflation Expectations", fig2, obs2, src2)

    # ===== YIELD CURVE =====
    elif page == "Yield Curve":
        fig1, obs1, src1 = data["yield_curve"]["snapshots"]
        _show_chart("US Treasury Yield Curve (Monthly Snapshots, last 24 months)", fig1, obs1, src1)

        st.markdown("---")
        fig2, obs2, src2 = data["yield_curve"]["timeseries"]
        _show_chart("US Treasury Yields (Daily Time Series, last 24 months)", fig2, obs2, src2)

    # ===== MONEY MARKET =====
    elif page == "Money Market":
        mm = data["money_market"]

        fig, obs, src, _ = mm["plot_a"]
        _show_chart("Market Rates Complex (Since 2020)", fig, obs, src)

        st.markdown("---")
        fig, obs, src, _ = mm["plot_b"]
        _show_chart("Market Rates Complex (Since last year)", fig, obs, src)

        st.markdown("---")
        _show_chart(mm["titles"]["c"], *mm["plot_c"])

        st.markdown("---")
        _show_chart(mm["titles"]["d"], *mm["plot_d"])

        st.markdown("---")
        _show_chart(mm["titles"]["e"], *mm["plot_e"])

    # ===== HEDGE FUNDS =====
    elif page == "Hedge Funds":
        hf = data.get("hedge_funds")
        if not hf:
            st.warning("Hedge fund section unavailable (CFTC fetch failed).")
        else:
            fig_s, fig_l, obs_g, src_g, guide_g = hf["gross"]
            fig_net, obs_n, src_n, guide_n = hf["net"]

            _show_chart("Hedge Fund UST futures – Gross SHORTS", fig_s, obs_g, src_g, guide_g, show_details=False)
            st.markdown("---")
            _show_chart("Hedge Fund UST futures – Gross LONGS", fig_l, obs_g, src_g, guide_g, show_details=False)
            st.markdown("---")
            _show_chart("Hedge Fund UST futures – Net positioning", fig_net, obs_n, src_n, guide_n, show_details=False)


            # Static details area (no clickable expander header)
            st.markdown('<div class="details-box">', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Key observations")
                st.markdown("**Gross positioning**")
                if obs_g:
                    st.markdown("\n".join([f"- {o}" for o in obs_g]))
                else:
                    st.write("—")

                st.markdown("**Net positioning**")
                if obs_n:
                    st.markdown("\n".join([f"- {o}" for o in obs_n]))
                else:
                    st.write("—")

            with c2:
                st.markdown("#### Data sources")
                _src = []
                for s in (src_g or []) + (src_n or []):
                    if s and s not in _src:
                        _src.append(s)
                if _src:
                    st.markdown("\n".join([f"- {s}" for s in _src]))
                else:
                    st.write("—")

            _guide = []
            for g in (guide_g or []) + (guide_n or []):
                if g and g not in _guide:
                    _guide.append(g)
            if _guide:
                st.markdown("---")
                st.markdown("#### Label guide")
                st.markdown("\n".join([f"- {g}" for g in _guide]))

            st.markdown("</div>", unsafe_allow_html=True)
    # ===== GOLD =====
    elif page == "Gold & Liquidity":
        fig_top, fig_bot, obs, src = data["gold"]
        _show_chart("Gold Trajectory vs Liquidity (M2)", fig_top, obs, src, show_details=False)
        st.markdown("---")
        _show_chart("Physical Gold Flows (Non-Monetary)", fig_bot, obs, src, show_details=False)


        # Static details area (no clickable expander header)
        st.markdown('<div class="details-box">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Key observations")
            if obs:
                st.markdown("\n".join([f"- {o}" for o in obs]))
            else:
                st.write("—")

        with c2:
            st.markdown("#### Data sources")
            if src:
                st.markdown("\n".join([f"- {s}" for s in src]))
            else:
                st.write("—")

        st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
