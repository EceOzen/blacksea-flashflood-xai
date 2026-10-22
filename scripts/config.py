"""
Shared configuration for the Western Black Sea flash-flood project.

Every other script imports its region boxes, date ranges, and file paths
from here, so a change in one place (e.g. widening the analog search box)
propagates everywhere instead of drifting between scripts.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW_ERA5 = DATA_RAW / "era5"
DATA_RAW_ERA5_BOZKURT = DATA_RAW_ERA5 / "bozkurt"    # small reference-event downloads
DATA_RAW_ERA5_PRESSURE = DATA_RAW_ERA5 / "pressure"  # full 1979-2024 pressure-level files
DATA_RAW_ERA5_SINGLE = DATA_RAW_ERA5 / "single"      # full 1979-2024 single-level files
DATA_RAW_ESWD = DATA_RAW / "eswd"                     # European Severe Weather Database export(s) (login-gated, optional)
DATA_RAW_FLOODS = DATA_RAW / "flood_events"           # Dartmouth Flood Observatory archive (no login needed)
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_LABELS = PROJECT_ROOT / "data" / "labels"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# ----------------------------------------------------------------------
# Reference event: the Bozkurt flash flood
# ----------------------------------------------------------------------
BOZKURT_LAT = 41.95
BOZKURT_LON = 34.0
BOZKURT_EVENT_DATE = "2021-08-11"          # main flood day
BOZKURT_WINDOW_START = "2021-08-09"        # -72h lookback, per the roadmap
BOZKURT_WINDOW_END = "2021-08-12"

# ----------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------
# Analog search box: large-scale circulation pattern (Z500, MSLP, 850 hPa wind).
# (north, south, west, east), degrees. West is negative for degrees W.
ANALOG_AREA = {"north": 55, "south": 30, "west": 0, "east": 50}

# Thermodynamic box: local moisture/instability features (CAPE, CIN, TCWV, IVT).
THERMO_AREA = {"north": 46, "south": 40, "west": 27, "east": 36}

# Core flood-label region: Western Black Sea catchments (Bartin, Kastamonu,
# Sinop, Zonguldak, Duzce). Approximate bounding box for now -- a stand-in
# for the real watershed polygons, refine with a shapefile/GeoJSON later if
# the coarse box turns out to matter for which days get labeled.
CORE_LABEL_REGION = {"north": 42.3, "south": 40.6, "west": 30.7, "east": 35.6}

# Extended region: same core box widened to the Bulgarian/Romanian Black Sea
# coast, for a robustness check on whether restricting to Turkey only
# changes the flood-day labels much.
EXTENDED_LABEL_REGION = {"north": 45.5, "south": 40.6, "west": 27.0, "east": 35.6}

# ----------------------------------------------------------------------
# Time ranges
# ----------------------------------------------------------------------
CLIMATOLOGY_YEARS = (1991, 2020)
FULL_YEARS = (1979, 2024)
SEASON_MONTHS = (5, 9)  # May-September, with a wider download buffer (Apr-Oct)
DOWNLOAD_MONTHS = (4, 10)

HOURS = ["00:00", "06:00", "12:00", "18:00"]
PRECIP_HOURS = [f"{h:02d}:00" for h in range(24)]  # precip needs all 24h, see note below

# ----------------------------------------------------------------------
# ERA5 variables
# ----------------------------------------------------------------------
PRESSURE_LEVELS = ["500", "700", "850", "925"]
PRESSURE_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
]

SINGLE_LEVEL_VARS = [
    "mean_sea_level_pressure",
    "total_column_water_vapour",
    "convective_available_potential_energy",
    "convective_inhibition",
    "sea_surface_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
]

# Total precipitation is requested separately with all 24 hourly steps,
# because ERA5's accumulated fields reset each hour: sampling only
# 00/06/12/18 would give 1-hour totals, not 6-hour totals.
PRECIP_VAR = "total_precipitation"

# ----------------------------------------------------------------------
# Analog search (03_find_analogs.py)
# ----------------------------------------------------------------------
# Primary criterion: statistical significance against a null (random-pair)
# distribution of pattern correlations. ANALOG_SIGNIFICANCE_LEVEL = 0.95
# means "this pattern match is more similar than 95% of random day/window
# pairs" -- i.e. a chance-level false-positive rate below 5%.
ANALOG_N_BOOTSTRAP = 1000        # random window/day pairs used to build the null distribution
ANALOG_SIGNIFICANCE_LEVEL = 0.95  # percentile of the null distribution used as the threshold

# Secondary, descriptive sensitivity checks -- NOT the primary criterion,
# reported alongside it to show the significant set isn't an artifact of
# one arbitrary cutoff choice.
ANALOG_N = 30            # "top N" analog count
ANALOG_PERCENTILE = 1.0  # "top X%" analog cutoff, in percent

# Two candidate days/timesteps closer together than this are treated as the
# same synoptic event (e.g. consecutive 6-hourly steps of one slow-moving
# trough) and only the best-scoring one is kept.
DECLUSTER_WINDOW_DAYS = 3

# ----------------------------------------------------------------------
# Flood labels (04_build_flood_labels.py)
# ----------------------------------------------------------------------
# A reported ESWD event counts as matching a given day if it falls within
# this many days -- ESWD event timestamps are sometimes off by a day from
# the ERA5 date depending on report time zone / rounding, and heavy rain
# can straddle midnight.
LABEL_MATCH_WINDOW_DAYS = 1
FEATURE_LAG_HOURS = [0, 12, 24, 48, 72]
CONTROL_PER_POSITIVE = 3
CONTROL_BUFFER_DAYS = 2
FEATURE_RNG_SEED = 42
ANALOG_VARIANT = "uncentred"
_V = "" if ANALOG_VARIANT == "uncentred" else ANALOG_VARIANT
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / _V
RESULTS_DIR = PROJECT_ROOT / "results" / _V
FIGURES_DIR = PROJECT_ROOT / "figures" / _V