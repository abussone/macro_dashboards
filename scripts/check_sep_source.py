#!/usr/bin/env python3
from __future__ import annotations

import re
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests

FED_MONETARYPOLICY_BASE = "https://www.federalreserve.gov/monetarypolicy/"
SEP_DISCOVERY_PAGES = [
    f"{FED_MONETARYPOLICY_BASE}fomccalendars.htm",
    f"{FED_MONETARYPOLICY_BASE}monetarypolicytools.htm",
]
SEP_FALLBACK_URL = f"{FED_MONETARYPOLICY_BASE}fomcprojtabl20250917.htm"
SEP_URL_RE = re.compile(r"fomcprojtabl(\d{8})\.htm", re.IGNORECASE)


def extract_sep_urls_from_html(html: str, base_url: str) -> dict[str, str]:
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

    for m in SEP_URL_RE.finditer(html or ""):
        out[m.group(1)] = urljoin(base_url, m.group(0))
    return out


def discover_latest_sep_url(session: requests.Session) -> tuple[str, str]:
    candidates: dict[str, str] = {}
    for page_url in SEP_DISCOVERY_PAGES:
        try:
            r = session.get(page_url, timeout=40)
            r.raise_for_status()
            candidates.update(extract_sep_urls_from_html(r.text, page_url))
        except Exception:
            continue

    if not candidates:
        m = SEP_URL_RE.search(SEP_FALLBACK_URL)
        date_iso = "n/a"
        if m:
            date_iso = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")
        return SEP_FALLBACK_URL, date_iso

    latest_key = max(candidates.keys())
    latest_iso = pd.to_datetime(latest_key, format="%Y%m%d", errors="coerce").strftime("%Y-%m-%d")
    return candidates[latest_key], latest_iso


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def select_figure2_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for tbl in tables:
        if tbl is None or tbl.empty:
            continue
        df = flatten_columns(tbl)
        first_col = str(df.columns[0]).lower() if len(df.columns) else ""
        if "midpoint of target range or target level" not in first_col:
            continue
        year_cols = sum(bool(re.search(r"\b20\d{2}\b", str(c))) for c in df.columns[1:])
        if year_cols >= 2:
            return df
    raise ValueError("Figure 2 federal funds projection table not found")


def validate_figure2_table(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    levels = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    horizon_cols = [str(c) for c in df.columns[1:] if re.search(r"\b20\d{2}\b", str(c)) or re.search(r"longer\s*run", str(c), re.IGNORECASE)]

    issues: list[str] = []
    info: list[str] = []

    if levels.notna().sum() < 8:
        issues.append("Too few numeric midpoint levels in Figure 2 first column.")
    else:
        info.append(f"numeric_midpoint_levels={int(levels.notna().sum())}")

    if len(horizon_cols) < 3:
        issues.append("Too few horizon columns in Figure 2 table.")
    else:
        info.append(f"horizons={horizon_cols}")

    non_empty_horizons = 0
    for col in horizon_cols:
        counts = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if float(counts.sum()) > 0:
            non_empty_horizons += 1
    if non_empty_horizons < max(2, len(horizon_cols) // 2):
        issues.append("Most Figure 2 horizon columns are empty or zero.")
    else:
        info.append(f"non_empty_horizons={non_empty_horizons}")

    return issues, info


def main() -> int:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; SEP monitor)",
            "Accept": "*/*",
        }
    )

    errors: list[str] = []
    latest_url = "n/a"
    latest_date = "n/a"
    try:
        latest_url, latest_date = discover_latest_sep_url(session)
        print(f"latest_discovered_url={latest_url}")
        print(f"latest_discovered_date={latest_date}")
    except Exception as e:
        errors.append(f"Failed to discover latest SEP URL: {e}")

    candidate_urls: list[str] = []
    if latest_url != "n/a":
        candidate_urls.append(latest_url)
    if SEP_FALLBACK_URL not in candidate_urls:
        candidate_urls.append(SEP_FALLBACK_URL)

    for url in candidate_urls:
        try:
            r = session.get(url, timeout=40)
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))
            fig2 = select_figure2_table(tables)
            issues, info = validate_figure2_table(fig2)
            if issues:
                raise RuntimeError(" ; ".join(issues))
            print(f"checked_url={url}")
            for entry in info:
                print(entry)
            print("status=ok")
            return 0
        except Exception as e:
            errors.append(f"{url}: {e}")

    print("status=failed")
    for e in errors:
        print(f"error={e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
