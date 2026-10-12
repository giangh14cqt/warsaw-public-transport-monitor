"""
Warsaw Transit Telemetry & Attribution System: Domain Constants.
Defines corridor boundaries, operational transit cohorts, and IMGW station identifiers.
"""

from typing import Any, Dict, List, Union

import pandas as pd

from src.core.polish_calendar import (
    calculate_easter,
    generate_polish_calendar_df,
    get_polish_public_holidays,
    is_polish_school_day,
)

# Core study corridors in Warsaw (GTFS street search pattern & spatial bounds)
DEFAULT_CORRIDORS: Dict[str, Dict[str, Any]] = {
    "al_jerozolimskie": {
        "name": "Aleje Jerozolimskie",
        "pattern": "Jerozolimsk",
        "bbox": (20.895, 52.180, 21.030, 52.235),  # (min_lon, min_lat, max_lon, max_lat)
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
    "pulawska": {
        "name": "Puławska",
        "pattern": "Puławsk",
        "bbox": (21.015, 52.095, 21.030, 52.215),
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
    "trasa_wz": {
        "name": "Trasa W-Z (Solidarności)",
        "pattern": "Solidarności",
        "bbox": (20.970, 52.235, 21.055, 52.265),
        "default_lanes": 2,
        "is_dedicated_row": True,  # Dedicated PAT (Bus+Tram) right-of-way
    },
    "towarowa_okopowa": {
        "name": "Towarowa / Okopowa",
        "pattern": "Towarow|Okopow",
        "bbox": (20.975, 52.225, 20.990, 52.255),
        "default_lanes": 3,
        "is_dedicated_row": True,
    },
}

# Operational Transit Cohorts
TRANSIT_COHORTS: List[str] = [
    "pooled",
    "urban_tram",
    "urban_bus",
    "suburban_bus",
]

# Official IMGW-PIB Weather Station IDs
IMGW_STATIONS: Dict[str, str] = {
    "WARSZAWA_OKECIE_WMO": "12375",
    "WARSZAWA_OKECIE_HISTORICAL": "352200375",
    "WARSZAWA_BIELANY_HISTORICAL": "352200380",
}


def classify_transit_cohort(route_id: Any, is_tram: Union[int, bool, float] = 0) -> str:
    """
    Classify a transit route into its operational cohort.

    Parameters
    ----------
    route_id : Any
        GTFS route ID string or numeric.
    is_tram : int or bool
        1 if tram, 0 if bus.

    Returns
    -------
    str
        One of 'urban_tram', 'urban_bus', 'suburban_bus', 'night_bus', 'other'.
    """
    if bool(is_tram):
        return "urban_tram"

    r_str = str(route_id).strip()
    cohort = "other"
    if r_str.isdigit():
        r_num = int(r_str)
        if 1 <= r_num <= 79:
            cohort = "urban_tram"
        elif 100 <= r_num <= 599:
            cohort = "urban_bus"
        elif 700 <= r_num <= 899:
            cohort = "suburban_bus"
    elif r_str.upper().startswith("L"):
        cohort = "suburban_bus"
    elif r_str.upper().startswith("N"):
        cohort = "night_bus"

    return cohort


def classify_cohort_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized series cohort classification on a DataFrame."""
    if "transit_cohort" in df.columns:
        return df["transit_cohort"]

    return df.apply(
        lambda row: classify_transit_cohort(row.get("route_id", ""), row.get("is_tram", 0)),
        axis=1,
    )
