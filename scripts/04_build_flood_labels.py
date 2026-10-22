"""
Label days as "flood day" or not, using flood-event catalogs for the
Western Black Sea -- so the synoptic analog days from 03_find_analogs.py can
be checked against a real-world outcome, not just a matching atmospheric
pattern.

Sources, at two resolutions:

  1. Dartmouth Flood Observatory (DFO) archive (flood_all.csv) --
     COUNTRY-LEVEL only (no lat/lon). Coarse: "a large flood somewhere in
     Turkey". -> flood_country_core / flood_country_extended
  2. Copernicus EMS Rapid Mapping activations -- precise (lat/lon).
     -> flood_core / flood_extended
       a) MANUAL CSV (preferred): data/raw/copernicus/*.csv, hand-built
          from the Rapid Mapping portal, columns:
              code,event_date,latitude,longitude,country,note
          e.g. EMSR526,2021-08-11,41.95,33.99,Turkey,Bozkurt/Kastamonu
       b) Live public API (best-effort fallback, follows pagination).
  3. EM-DAT (data/raw/emdat/*.xlsx) -- Location / Admin Units text with
     province names. The only source covering the Turkish core region
     (Bozkurt 2021, Akcakoca 2019, Zonguldak 2004, ...). 2000 onward.
  4. HANZE v3 (data/raw/hanze/HANZE_events*.csv) -- NUTS3 regions; covers
     the Bulgarian/Romanian coast (Varna, Dobrich, Burgas, Constanta,
     Tulcea, Galati), 1870-2025. No Turkey.
  EM-DAT and HANZE are mapped to approximate province/region centroids, then
  go through the same box test as Copernicus -> flood_core / flood_extended.

Usage:
    python scripts/04_build_flood_labels.py   (run from the project root)

Outputs:
    data/labels/flood_labels.csv
    data/processed/analogs_significant_labeled.csv

Treat flood_core/flood_extended as the trustworthy columns. The
flood_country_* columns over-count (any big flood anywhere in the country)
and are supplementary evidence only.
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
DATA_RAW_COPERNICUS = config.DATA_RAW / "copernicus"

COPERNICUS_EMS_API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/"

CORE_COUNTRIES = {"TURKEY", "TÜRKİYE", "TURKIYE", "TÜRKIYE"}
EXTENDED_COUNTRIES = CORE_COUNTRIES | {"BULGARIA", "ROMANIA"}


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


# ----------------------------------------------------------------------
# Source 1: DFO archive (point or country-level variant)
# ----------------------------------------------------------------------
def load_dfo_archive(path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [c.strip().upper().replace(" ", "_") for c in raw.columns]
    print(f"Columns found in {path.name}: {list(raw.columns)}")

    began_col = find_column(raw.columns, ["BEGAN", "BEGIN", "START", "START_DATE", "DATE_BEGIN"])
    ended_col = find_column(raw.columns, ["ENDED", "END", "END_DATE", "DATE_END"])
    lat_col = find_column(raw.columns, ["LAT", "LATITUDE", "CENTROID_Y", "Y"])
    lon_col = find_column(raw.columns, ["LONG", "LON", "LONGITUDE", "CENTROID_X", "X"])
    country_col = find_column(raw.columns, ["COUNTRY"])

    if began_col is None:
        raise ValueError(f"No begin-date column in {path.name}: {list(raw.columns)}")

    base = pd.DataFrame({
        "begin": pd.to_datetime(raw[began_col], errors="coerce", format="mixed"),
        "end": pd.to_datetime(raw[ended_col], errors="coerce", format="mixed") if ended_col else pd.NaT,
    })
    base["end"] = base["end"].fillna(base["begin"])

    if lat_col and lon_col:
        base["latitude"] = pd.to_numeric(raw[lat_col], errors="coerce")
        base["longitude"] = pd.to_numeric(raw[lon_col], errors="coerce")
        base["source"] = "DFO (point)"
        base = base.dropna(subset=["begin", "latitude", "longitude"])
        print(f"DFO: point-resolution file, {len(base)} usable rows.")
        return {"geo": base, "country": None}

    if country_col:
        base["country"] = raw[country_col].astype(str).str.strip().str.upper()
        base["source"] = "DFO (country-level)"
        base = base.dropna(subset=["begin"])
        print(f"DFO: country-level file (no lat/lon), {len(base)} usable rows.")
        return {"geo": None, "country": base}

    raise ValueError(f"No lat/lon or country column in {path.name}: {list(raw.columns)}")


# ----------------------------------------------------------------------
# Source 2a: Copernicus EMS -- hand-built CSV (preferred)
# ----------------------------------------------------------------------
# The portal export (mapping.emergency.copernicus.eu/activations) has no
# coordinates: Code;Title;Date;Event Type;Country. Locations are recovered
# from the Title with this keyword table (approximate place coordinates).
# A Title matching nothing here is dropped as "not in our region" -- the
# script prints those so you can add a keyword if one was missed.
PLACE_KEYWORDS = {
    # core: Western Black Sea, Turkey
    "bozkurt": (41.95, 34.01), "kastamonu": (41.38, 33.78), "bartin": (41.63, 32.34),
    "bartın": (41.63, 32.34), "sinop": (42.03, 35.15), "zonguldak": (41.45, 31.79),
    "duzce": (40.84, 31.16), "düzce": (40.84, 31.16), "karabuk": (41.20, 32.62),
    "karabük": (41.20, 32.62), "western black sea": (41.6, 33.0),
    # extended: Bulgarian / Romanian Black Sea coast
    "varna": (43.21, 27.91), "burgas": (42.50, 27.47), "dobrich": (43.57, 27.83),
    "constanta": (44.17, 28.64), "constanța": (44.17, 28.64), "tulcea": (45.18, 28.80),
    "black sea": (43.0, 28.5),
}


# Activations checked one by one (title alone doesn't say where / when).
# code -> (lat, lon, event_date or None to keep the export's date, note)
# The export's Date is the ACTIVATION date, which for EMSN (on-demand) products
# can be weeks after the event -- so the real event date is set here where known.
CODE_OVERRIDES = {
    "EMSR092": (43.19, 27.92, "2014-06-19", "Varna (Asparuhovo) / Dobrich flash flood"),
    "EMSN141": (42.17, 27.85, "2023-09-05", "Southern Bulgarian Black Sea coast (Tsarevo), "
                                           "Storm Daniel -- event date to verify on the activation page"),
    "EMSR758": (45.43, 28.05, None, "Galati county (Vaslui part is inland, outside the box)"),
}
# Checked and deliberately excluded (outside the label boxes or duplicates):
#   EMSR364 Arakli/Trabzon -- eastern Black Sea, different orographic regime
#   EMSN224 Suceava/Neamt, EMSN193 Tazlau, EMSR082 SW Romania, EMSR445 Prut basin -- inland
#   EMSN022 Bulgaria -- recovery/risk product for the 2014 Varna event (already EMSR092)
EXCLUDED_CODES = {"EMSR364", "EMSN224", "EMSN193", "EMSR082", "EMSR445", "EMSN022"}


def _locate_from_title(title):
    t = str(title).lower()
    for kw in sorted(PLACE_KEYWORDS, key=len, reverse=True):  # most specific first
        if kw in t:
            return PLACE_KEYWORDS[kw]
    return None


def _load_portal_export(raw, fname):
    rows, unmatched = [], []
    for _, r in raw.iterrows():
        if "flood" not in str(r.get("event type", "")).lower():
            continue
        code = str(r.get("code", "")).strip().upper()
        if code in EXCLUDED_CODES:
            continue
        t = pd.to_datetime(str(r["date"]).split(",")[0].strip(), format="%d/%m/%Y", errors="coerce")
        if code in CODE_OVERRIDES:
            lat, lon, event_date, note = CODE_OVERRIDES[code]
            if event_date:
                t = pd.Timestamp(event_date)
            rows.append({"begin": t, "end": t, "latitude": lat, "longitude": lon,
                         "source": f"Copernicus EMS ({code}: {note})"})
            continue
        loc = _locate_from_title(r.get("title", ""))
        if loc is None:
            unmatched.append(f"{code}: {r.get('title')} ({r.get('country')})")
            continue
        rows.append({"begin": t, "end": t, "latitude": loc[0], "longitude": loc[1],
                     "source": f"Copernicus EMS ({code})"})
    relevant = [u for u in unmatched
                if any(c in u.upper() for c in ("TURKEY", "TÜRKİYE", "TURKIYE", "BULGARIA", "ROMANIA"))]
    if relevant:
        print(f"  {fname}: these TR/BG/RO floods had no known place keyword in the title "
              f"(add one to PLACE_KEYWORDS if they're in the region):")
        for u in relevant:
            print(f"    {u}")
    print(f"  {fname}: {len(rows)} of {len(raw)} activations located in/near the Black Sea region.")
    return pd.DataFrame(rows, columns=["begin", "end", "latitude", "longitude", "source"])


def load_copernicus_manual():
    if not DATA_RAW_COPERNICUS.exists():
        return None
    files = sorted(DATA_RAW_COPERNICUS.glob("*.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        raw = pd.read_csv(f, sep=None, engine="python")
        raw.columns = [c.strip().lower() for c in raw.columns]
        if {"code", "title", "date", "country"}.issubset(raw.columns):
            frames.append(_load_portal_export(raw, f.name).dropna(subset=["begin"]))
            continue
        need = {"event_date", "latitude", "longitude"}
        if not need.issubset(raw.columns):
            raise ValueError(f"{f.name} needs columns {sorted(need)}, found {list(raw.columns)}")
        out = pd.DataFrame({
            "begin": pd.to_datetime(raw["event_date"], errors="coerce", dayfirst=False),
            "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
        })
        out["end"] = out["begin"]
        codes = raw["code"] if "code" in raw.columns else "?"
        out["source"] = "Copernicus EMS (" + pd.Series(codes, index=raw.index).astype(str) + ")"
        frames.append(out.dropna(subset=["begin", "latitude", "longitude"]))
    out = pd.concat(frames, ignore_index=True)
    print(f"Copernicus EMS (manual CSV): {len(out)} activations from {len(files)} file(s).")
    return out


# ----------------------------------------------------------------------
# Source 2b: Copernicus EMS -- live API, best-effort, follows pagination
# ----------------------------------------------------------------------
def fetch_copernicus_ems(max_pages=200):
    items, url, pages = [], COPERNICUS_EMS_API, 0
    try:
        while url and pages < max_pages:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pages += 1
            if isinstance(data, list):
                items.extend(data)
                break
            items.extend(data.get("results", data.get("activations", [])))
            url = data.get("next")
    except Exception as e:
        print(f"Copernicus EMS API fetch stopped ({e}) after {pages} page(s).")

    if items:
        print(f"Copernicus EMS API: {len(items)} activations over {pages} page(s). "
              f"Example keys: {sorted(items[0].keys())}")

    rows = []
    for item in items:
        blob = " ".join(str(item.get(k, "")) for k in
                        ("category", "type", "event_type", "hazard", "eventType", "name")).lower()
        if "flood" not in blob:
            continue
        date_str = (item.get("eventTime") or item.get("event_date") or item.get("activation_date")
                    or item.get("activationTime") or item.get("date") or item.get("start_date"))
        centroid = item.get("centroid") or item.get("geom_centroid") or item.get("location")
        if date_str is None or centroid is None:
            continue
        match = re.search(r"POINT\s*\(\s*([\-\d.]+)\s+([\-\d.]+)\s*\)", str(centroid))
        if not match:
            continue
        lon, lat = float(match.group(1)), float(match.group(2))
        t = pd.to_datetime(date_str, errors="coerce")
        rows.append({"begin": t, "end": t, "latitude": lat, "longitude": lon,
                     "source": f"Copernicus EMS ({item.get('code', '?')})"})

    out = pd.DataFrame(rows, columns=["begin", "end", "latitude", "longitude", "source"])
    out = out.dropna(subset=["begin", "latitude", "longitude"])
    print(f"Copernicus EMS API: {len(out)} flood activations parsed.")
    return out


# ----------------------------------------------------------------------
# Province / region centroids used to place EM-DAT and HANZE events.
# Only coastal provinces of the label regions are listed on purpose: an event
# naming none of them is "not in our region" (e.g. Istanbul, Samsun, Trabzon).
# ----------------------------------------------------------------------
PROVINCE_COORDS = {
    # core: Western Black Sea, Turkey
    "duzce": (40.84, 31.16), "akcakoca": (41.09, 31.12), "zonguldak": (41.45, 31.79),
    "bartin": (41.63, 32.34), "karabuk": (41.20, 32.62), "kastamonu": (41.38, 33.78),
    "bozkurt": (41.95, 34.01), "sinop": (42.03, 35.15),
    # extended: Turkish Thrace coast, Bulgarian and Romanian coast
    "kirklareli": (41.73, 27.22), "varna": (43.21, 27.91), "dobrich": (43.57, 27.83),
    "burgas": (42.50, 27.47), "constanta": (44.17, 28.64), "tulcea": (45.18, 28.80),
    "galati": (45.43, 28.05),
}
NUTS3_COORDS = {
    "BG331": PROVINCE_COORDS["varna"], "BG332": PROVINCE_COORDS["dobrich"],
    "BG341": PROVINCE_COORDS["burgas"], "RO223": PROVINCE_COORDS["constanta"],
    "RO225": PROVINCE_COORDS["tulcea"], "RO224": PROVINCE_COORDS["galati"],
}
# Long river floods (snowmelt, weeks-long) would flag dozens of days; for those
# only the onset matters for precursors, so they're cut to their first days.
MAX_EVENT_DAYS = 7
ONSET_DAYS_IF_LONG = 3


def _ascii(s):
    import unicodedata
    s = str(s).replace("ı", "i").replace("İ", "I")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def load_emdat():
    folder = config.DATA_RAW / "emdat"
    files = sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        print("No EM-DAT file in data/raw/emdat/ -- skipping it.")
        return None
    raw = pd.read_excel(files[0]) if files[0].suffix == ".xlsx" else pd.read_csv(files[0])
    raw = raw[raw["Disaster Type"].astype(str).str.contains("Flood", case=False)]

    rows, no_day = [], 0
    for _, r in raw.iterrows():
        if pd.isna(r.get("Start Day")):
            no_day += 1
            continue
        begin = pd.Timestamp(int(r["Start Year"]), int(r["Start Month"]), int(r["Start Day"]))
        try:
            end = pd.Timestamp(int(r["End Year"]), int(r["End Month"]), int(r["End Day"]))
        except (TypeError, ValueError):
            end = begin
        text = _ascii(f"{r.get('Location', '')} {r.get('Admin Units', '')}")
        hits = {k for k in PROVINCE_COORDS if re.search(rf"\b{k}\b", text)}
        if "bozkurt" in hits:
            hits.discard("kastamonu")   # same event, keep the more precise point
        for k in hits:
            lat, lon = PROVINCE_COORDS[k]
            rows.append({"begin": begin, "end": end, "latitude": lat, "longitude": lon,
                         "source": f"EM-DAT {r['DisNo.']} ({k})"})
    out = pd.DataFrame(rows, columns=["begin", "end", "latitude", "longitude", "source"])
    print(f"EM-DAT: {len(raw)} flood records, {out['source'].str.split(' ').str[1].nunique() if len(out) else 0} "
          f"in the label regions ({no_day} skipped for missing start day).")
    return out


def load_hanze():
    folder = config.DATA_RAW / "hanze"
    files = sorted(folder.glob("HANZE_events*.csv")) if folder.exists() else []
    if not files:
        print("No HANZE events file in data/raw/hanze/ -- skipping it.")
        return None
    raw = pd.read_csv(files[0])
    raw = raw[raw["Type"].astype(str).str.lower() != "coastal"]   # storm surges aren't rain-driven
    rows = []
    for _, r in raw.iterrows():
        regions = set(str(r["Regions affected (NUTS 3)"]).split(";")) & set(NUTS3_COORDS)
        for code in regions:
            lat, lon = NUTS3_COORDS[code]
            rows.append({"begin": pd.to_datetime(r["Start date"]), "end": pd.to_datetime(r["End date"]),
                         "latitude": lat, "longitude": lon,
                         "source": f"HANZE {r['ID']} ({code}, {r['Type']})"})
    out = pd.DataFrame(rows, columns=["begin", "end", "latitude", "longitude", "source"])
    print(f"HANZE: {len(raw)} rain-driven flood events, "
          f"{out['source'].str.split(' ').str[1].nunique() if len(out) else 0} touching the coastal regions.")
    return out


def cap_long_events(df):
    df = df.copy()
    df["end"] = df["end"].fillna(df["begin"])
    long_ = (df["end"] - df["begin"]).dt.days > MAX_EVENT_DAYS
    df.loc[long_, "end"] = df.loc[long_, "begin"] + pd.Timedelta(days=ONSET_DAYS_IF_LONG - 1)
    return df


# ----------------------------------------------------------------------
def in_box(df, box):
    return (
        (df["latitude"] >= box["south"]) & (df["latitude"] <= box["north"]) &
        (df["longitude"] >= box["west"]) & (df["longitude"] <= box["east"])
    )


def expand_to_days(events, extra_cols=()):
    rows = []
    cols = list(extra_cols)
    for _, ev in events.iterrows():
        for day in pd.date_range(ev["begin"].normalize(), ev["end"].normalize(), freq="D"):
            row = {"date": day}
            for c in cols:
                row[c] = ev[c]
            rows.append(row)
    return pd.DataFrame(rows, columns=["date"] + cols)


def geo_event_days_in_box(daily_events, box):
    if daily_events is None or len(daily_events) == 0:
        return set()
    sub = daily_events[in_box(daily_events, box)]
    return set(pd.to_datetime(sub["date"].unique()))


def country_event_days(daily_events, countries):
    if daily_events is None or len(daily_events) == 0:
        return set()
    sub = daily_events[daily_events["country"].isin(countries)]
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


FLAG_COLS = ["flood_core", "flood_extended", "flood_country_core", "flood_country_extended"]


def add_flags(df, core_days, extended_days, country_core_days, country_extended_days):
    w = config.LABEL_MATCH_WINDOW_DAYS
    df["flood_core"] = label_days(df["date"], core_days, w)
    df["flood_extended"] = label_days(df["date"], extended_days, w)
    df["flood_country_core"] = label_days(df["date"], country_core_days, w)
    df["flood_country_extended"] = label_days(df["date"], country_extended_days, w)
    return df


# ----------------------------------------------------------------------
def main():
    geo_sources, country_sources = [], []

    dfo_files = sorted(config.DATA_RAW_FLOODS.glob("*.csv")) + sorted(config.DATA_RAW_FLOODS.glob("*.xlsx"))
    if dfo_files:
        print(f"Reading {dfo_files[0].name}...")
        dfo = load_dfo_archive(dfo_files[0])
        if dfo["geo"] is not None:
            geo_sources.append(dfo["geo"])
        if dfo["country"] is not None:
            country_sources.append(dfo["country"])
    else:
        print(f"No DFO file found in {config.DATA_RAW_FLOODS} -- skipping it.")

    manual = load_copernicus_manual()
    if manual is not None and len(manual):
        geo_sources.append(manual)
    else:
        print("\nNo manual Copernicus CSV -- trying the live API...")
        geo_sources.append(fetch_copernicus_ems())

    for loader in (load_emdat, load_hanze):
        src = loader()
        if src is not None and len(src):
            geo_sources.append(src)

    geo_sources = [s for s in geo_sources if len(s)]
    geo_events = pd.concat(geo_sources, ignore_index=True) if geo_sources else pd.DataFrame(
        columns=["begin", "end", "latitude", "longitude", "source"])
    if len(geo_events):
        geo_events = cap_long_events(geo_events)
        geo_events = geo_events[(geo_events["begin"] >= pd.Timestamp("1979-01-01")) &
                                (geo_events["begin"] <= pd.Timestamp("2024-12-31"))].sort_values("begin")
    country_sources = [s for s in country_sources if len(s)]
    country_events = pd.concat(country_sources, ignore_index=True) if country_sources else pd.DataFrame(
        columns=["begin", "end", "country", "source"])

    print(f"\nGeo-located events (precise): {len(geo_events)}")
    print(f"Country-level events (coarse): {len(country_events)}")

    daily_geo = expand_to_days(geo_events, extra_cols=["latitude", "longitude"]) if len(geo_events) else None
    daily_country = expand_to_days(country_events, extra_cols=["country"]) if len(country_events) else None

    core_days = geo_event_days_in_box(daily_geo, config.CORE_LABEL_REGION)
    extended_days = geo_event_days_in_box(daily_geo, config.EXTENDED_LABEL_REGION)
    country_core_days = country_event_days(daily_country, CORE_COUNTRIES)
    country_extended_days = country_event_days(daily_country, EXTENDED_COUNTRIES)

    print(f"\nPrecise event-days -- core: {len(core_days)}, extended: {len(extended_days)}")
    if len(geo_events):
        for _, r in geo_events.iterrows():
            where = ("core" if in_box(pd.DataFrame([r]), config.CORE_LABEL_REGION).iloc[0]
                     else "extended" if in_box(pd.DataFrame([r]), config.EXTENDED_LABEL_REGION).iloc[0]
                     else "outside")
            print(f"  {where:8s} {r['begin'].date()}  {r['source']}")
    print(f"Coarse event-days -- Turkey: {len(country_core_days)}, "
          f"Turkey+Bulgaria+Romania: {len(country_extended_days)}")

    all_scores = pd.read_csv(config.DATA_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])
    all_days = sorted(set(pd.to_datetime(all_scores["time"]).dt.normalize()))
    labels = add_flags(pd.DataFrame({"date": all_days}),
                       core_days, extended_days, country_core_days, country_extended_days)

    config.DATA_LABELS.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT_LABELS_CSV, index=False)
    print(f"\nSaved: {OUT_LABELS_CSV}")
    for col in FLAG_COLS:
        print(f"  {col}: {labels[col].sum()} / {len(labels)} days flagged "
              f"({100 * labels[col].mean():.1f}%)")

    significant = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    significant["date"] = significant["time"].dt.normalize()
    significant = add_flags(significant, core_days, extended_days, country_core_days, country_extended_days)
    significant.to_csv(OUT_ANALOGS_LABELED_CSV, index=False)

    n = len(significant)
    print(f"\nOf the {n} significant synoptic analog days:")
    for col in FLAG_COLS:
        k = int(significant[col].sum())
        print(f"  {col}: {k} ({100 * k / n:.1f}%)  vs base rate {100 * labels[col].mean():.1f}%")
    print(f"Saved: {OUT_ANALOGS_LABELED_CSV}")


if __name__ == "__main__":
    main()
