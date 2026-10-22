"""
Label days as "flood day" or not, using flood-event catalogs for the
Western Black Sea -- so the synoptic analog days from 03_find_analogs.py can
be checked against a real-world outcome, not just a matching atmospheric
pattern.

Why not ESWD: ESWD's report-search/export requires a login, and that login
has been unreachable (password rejected, reset links invalid). Rather than
block the pipeline on an account problem, this script uses two sources that
need no login at all:

  1. Dartmouth Flood Observatory (DFO) "Global Active Archive of Large
     Flood Events" -- a global flood catalog, 1985-present, with begin/end
     dates and a location per event. Manual one-time download (same pattern
     as the ERA5/CDS downloads): https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo
     Save the file (CSV or XLSX, whatever HDX gives you) into
     data/raw/flood_events/. Covers 1985-2024 -- 1979-1984 has no DFO
     coverage, a known small gap.
  2. Copernicus Emergency Management Service (EMS) Rapid Mapping
     activations -- the EU's satellite flood-mapping activations, fetched
     automatically by this script from their public API (no download, no
     login). Sparser (only the largest, officially-activated events, and
     only from ~2012 onward) but high-confidence and independently
     verifiable -- Bozkurt itself is very likely in there under its own
     EMSR activation code.
  3. (Optional) If an ESWD export CSV ever does show up in data/raw/eswd/,
     it's picked up too and merged in -- no harm in having both.

Usage:
    python scripts/04_build_flood_labels.py   (run from the project root)

Outputs:
    data/labels/flood_labels.csv                    every day 1979-2024,
                                                      with flood_core /
                                                      flood_extended flags
    data/processed/analogs_significant_labeled.csv   the 03_find_analogs.py
                                                      significant set, with
                                                      the same two flags

NOTE: the exact column names in the DFO download, and the exact field names
in the Copernicus EMS API response, haven't been confirmed against real
data yet -- both parsers are written defensively (print what they found /
skip what they can't parse rather than crash) so we can adjust quickly once
we see real output.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OUT_LABELS_CSV = config.DATA_LABELS / "flood_labels.csv"
OUT_ANALOGS_LABELED_CSV = config.DATA_PROCESSED / "analogs_significant_labeled.csv"

COPERNICUS_EMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


# ----------------------------------------------------------------------
# Source 1: Dartmouth Flood Observatory archive (manual one-time download)
# ----------------------------------------------------------------------
def load_dfo_archive(path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [c.strip().upper().replace(" ", "_") for c in raw.columns]
    print(f"Columns found in {path.name}: {list(raw.columns)}")

    began_col = find_column(raw.columns, ["BEGAN", "BEGIN", "START", "DATE_BEGIN"])
    ended_col = find_column(raw.columns, ["ENDED", "END", "DATE_END"])
    lat_col = find_column(raw.columns, ["LAT", "LATITUDE", "CENTROID_Y", "Y"])
    lon_col = find_column(raw.columns, ["LONG", "LON", "LONGITUDE", "CENTROID_X", "X"])

    missing = [n for n, c in [("began", began_col), ("latitude", lat_col), ("longitude", lon_col)] if c is None]
    if missing:
        raise ValueError(
            f"Couldn't find columns for: {missing} in {path.name}. "
            f"Available columns: {list(raw.columns)} -- share this list so "
            f"load_dfo_archive() can be pointed at the right column names."
        )

    out = pd.DataFrame({
        "begin": pd.to_datetime(raw[began_col], errors="coerce"),
        "end": pd.to_datetime(raw[ended_col], errors="coerce") if ended_col else pd.NaT,
        "latitude": pd.to_numeric(raw[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(raw[lon_col], errors="coerce"),
    })
    out["end"] = out["end"].fillna(out["begin"])
    out["source"] = "DFO"
    n_before = len(out)
    out = out.dropna(subset=["begin", "latitude", "longitude"])
    if len(out) < n_before:
        print(f"Dropped {n_before - len(out)}/{n_before} DFO rows with unparseable begin/lat/lon.")
    return out


# ----------------------------------------------------------------------
# Source 2: Copernicus EMS Rapid Mapping activations (public API, no login,
# fetched live). Best-effort: a network hiccup or an API shape we didn't
# anticipate shouldn't block the rest of the pipeline, so failures are
# caught and just reduce coverage rather than crashing.
# ----------------------------------------------------------------------
def fetch_copernicus_ems():
    try:
        resp = requests.get(COPERNICUS_EMS_API, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Copernicus EMS fetch failed ({e}) -- continuing without it.")
        return pd.DataFrame(columns=["begin", "end", "latitude", "longitude", "source"])

    items = data if isinstance(data, list) else data.get("results", data.get("activations", []))
    rows = []
    for item in items:
        category = str(item.get("category") or item.get("type") or item.get("event_type") or "").lower()
        if "flood" not in category:
            continue
        date_str = (item.get("event_date") or item.get("activation_date")
                    or item.get("date") or item.get("start_date"))
        centroid = item.get("centroid") or item.get("geom_centroid") or item.get("location")
        if date_str is None or centroid is None:
            continue
        match = re.search(r"POINT\s*\(\s*([\-\d.]+)\s+([\-\d.]+)\s*\)", str(centroid))
        if not match:
            continue
        lon, lat = float(match.group(1)), float(match.group(2))
        rows.append({
            "begin": pd.to_datetime(date_str, errors="coerce"),
            "end": pd.to_datetime(date_str, errors="coerce"),
            "latitude": lat,
            "longitude": lon,
            "source": f"Copernicus EMS ({item.get('code', '?')})",
        })

    out = pd.DataFrame(rows, columns=["begin", "end", "latitude", "longitude", "source"])
    if len(out):
        out = out.dropna(subset=["begin", "latitude", "longitude"])
    print(f"Copernicus EMS: {len(out)} flood activations parsed from {len(items)} total activations.")
    return out


# ----------------------------------------------------------------------
# Source 3 (optional): ESWD, if a CSV export ever shows up after all.
# ----------------------------------------------------------------------
def load_eswd_if_present():
    if not config.DATA_RAW_ESWD.exists():
        return None
    files = sorted(config.DATA_RAW_ESWD.glob("*.csv"))
    if not files:
        return None
    print(f"\nAlso found an ESWD export ({files[0].name}), including it...")
    raw = pd.read_csv(files[0], sep=None, engine="python")
    raw.columns = [c.strip().upper() for c in raw.columns]
    time_col = find_column(raw.columns, ["TIME_EVENT", "TIME_EVENT_BEGIN", "DATE", "TIME"])
    lat_col = find_column(raw.columns, ["LATITUDE", "LAT"])
    lon_col = find_column(raw.columns, ["LONGITUDE", "LON", "LONG"])
    if not (time_col and lat_col and lon_col):
        print(f"  Couldn't recognise ESWD columns ({list(raw.columns)}) -- skipping it.")
        return None
    out = pd.DataFrame({
        "begin": pd.to_datetime(raw[time_col], errors="coerce", dayfirst=True),
        "latitude": pd.to_numeric(raw[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(raw[lon_col], errors="coerce"),
    })
    out["end"] = out["begin"]
    out["source"] = "ESWD"
    return out.dropna(subset=["begin", "latitude", "longitude"])


# ----------------------------------------------------------------------
def in_box(df, box):
    return (
        (df["latitude"] >= box["south"]) & (df["latitude"] <= box["north"]) &
        (df["longitude"] >= box["west"]) & (df["longitude"] <= box["east"])
    )


def expand_to_days(events):
    """One row per event -> one row per calendar day the event spans
    (most events are single-day; DFO events can run for a week or more)."""
    rows = []
    for _, ev in events.iterrows():
        for day in pd.date_range(ev["begin"].normalize(), ev["end"].normalize(), freq="D"):
            rows.append({"date": day, "latitude": ev["latitude"], "longitude": ev["longitude"]})
    return pd.DataFrame(rows, columns=["date", "latitude", "longitude"])


def event_days_in_box(daily_events, box):
    if len(daily_events) == 0:
        return set()
    sub = daily_events[in_box(daily_events, box)]
    return set(pd.to_datetime(sub["date"].unique()))


def label_days(candidate_dates, event_days, window_days):
    if not event_days:
        return [False] * len(candidate_dates)
    event_arr = np.array(sorted(event_days), dtype="datetime64[ns]")
    labels = []
    for d in candidate_dates:
        day = np.datetime64(pd.Timestamp(d).normalize())
        diffs_days = np.abs((event_arr - day) / np.timedelta64(1, "D"))
        labels.append(bool((diffs_days <= window_days).any()))
    return labels


# ----------------------------------------------------------------------
def main():
    sources = []

    dfo_files = sorted(config.DATA_RAW_FLOODS.glob("*.xlsx")) + sorted(config.DATA_RAW_FLOODS.glob("*.csv"))
    if dfo_files:
        print(f"Reading {dfo_files[0].name}...")
        sources.append(load_dfo_archive(dfo_files[0]))
    else:
        print(f"No DFO archive file found in {config.DATA_RAW_FLOODS} -- skipping it "
              f"(download from https://data.humdata.org/dataset/global-active-archive-of-large-flood-events-dfo "
              f"and save the file there to include it).")

    print("\nFetching Copernicus EMS activations (public API, no login)...")
    sources.append(fetch_copernicus_ems())

    eswd = load_eswd_if_present()
    if eswd is not None:
        sources.append(eswd)

    sources = [s for s in sources if len(s)]
    events = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(
        columns=["begin", "end", "latitude", "longitude", "source"])
    print(f"\nTotal flood events from all sources combined: {len(events)}")
    if len(events) == 0:
        raise RuntimeError(
            f"No flood event data from any source. At minimum, download the DFO archive and "
            f"place it in {config.DATA_RAW_FLOODS} -- see this script's docstring."
        )
    print(events["source"].value_counts().to_string() if "source" in events else "")

    daily_events = expand_to_days(events)
    core_days = event_days_in_box(daily_events, config.CORE_LABEL_REGION)
    extended_days = event_days_in_box(daily_events, config.EXTENDED_LABEL_REGION)
    print(f"\nDistinct event-days in the core region: {len(core_days)}")
    print(f"Distinct event-days in the extended region: {len(extended_days)}")

    # ------------------------------------------------------------------
    # Label the full 1979-2024 record -- this is the eventual ML target.
    # ------------------------------------------------------------------
    all_scores = pd.read_csv(config.DATA_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])
    all_days = sorted(set(pd.to_datetime(all_scores["time"]).dt.normalize()))

    labels = pd.DataFrame({"date": all_days})
    labels["flood_core"] = label_days(labels["date"], core_days, config.LABEL_MATCH_WINDOW_DAYS)
    labels["flood_extended"] = label_days(labels["date"], extended_days, config.LABEL_MATCH_WINDOW_DAYS)

    config.DATA_LABELS.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT_LABELS_CSV, index=False)
    print(f"\nSaved daily flood labels for the full record: {OUT_LABELS_CSV}")
    print(f"  {labels['flood_core'].sum()} / {len(labels)} days flagged (core region)")
    print(f"  {labels['flood_extended'].sum()} / {len(labels)} days flagged (extended region)")

    # ------------------------------------------------------------------
    # Cross-check against the significant synoptic analog set.
    # ------------------------------------------------------------------
    significant = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    significant["date"] = significant["time"].dt.normalize()
    significant["flood_core"] = label_days(significant["date"], core_days, config.LABEL_MATCH_WINDOW_DAYS)
    significant["flood_extended"] = label_days(significant["date"], extended_days, config.LABEL_MATCH_WINDOW_DAYS)
    significant.to_csv(OUT_ANALOGS_LABELED_CSV, index=False)

    n_hit_core = int(significant["flood_core"].sum())
    n_hit_ext = int(significant["flood_extended"].sum())
    print(f"\nOf the {len(significant)} significant synoptic analog days:")
    print(f"  {n_hit_core} ({100 * n_hit_core / len(significant):.0f}%) coincide with a core-region flood day")
    print(f"  {n_hit_ext} ({100 * n_hit_ext / len(significant):.0f}%) coincide with an extended-region flood day")
    print(f"Saved: {OUT_ANALOGS_LABELED_CSV}")


if __name__ == "__main__":
    main()
