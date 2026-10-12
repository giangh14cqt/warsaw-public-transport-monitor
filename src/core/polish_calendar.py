"""
Polish Public Holiday and Academic/School Calendar Utility.
Provides domain-specific calendar features for Warsaw transit telemetry modeling:
- is_public_holiday: Statutory Polish non-working day (Ustawa z dnia 18 stycznia 1951 r. o dniach wolnych od pracy)
- is_school_day: Primary/secondary school instructional day in Mazowieckie voivodeship
- is_holiday_eve: Day immediately preceding a public holiday (heightened evening congestion)
"""

import datetime
import logging
import os
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_easter(year: int) -> datetime.date:
    """Compute Gregorian Easter Sunday via the Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def get_polish_public_holidays(year: int) -> Set[datetime.date]:
    """
    Return all statutory Polish public holidays for a given year.
    Includes fixed dates and movable Christian feasts (Easter Monday, Corpus Christi).
    """
    easter = calculate_easter(year)
    easter_monday = easter + datetime.timedelta(days=1)
    corpus_christi = easter + datetime.timedelta(days=60)
    whit_sunday = easter + datetime.timedelta(days=49)  # Zielone Świątki

    holidays = {
        datetime.date(year, 1, 1),   # Nowy Rok
        datetime.date(year, 1, 6),   # Święto Trzech Króli
        easter,                      # Wielkanoc (Niedziela)
        easter_monday,               # Poniedziałek Wielkanocny
        datetime.date(year, 5, 1),   # Święto Państwowe (Święto Pracy)
        datetime.date(year, 5, 3),   # Święto Narodowe Trzeciego Maja
        whit_sunday,                 # Zielone Świątki
        corpus_christi,              # Boże Ciało
        datetime.date(year, 8, 15),  # Wniebowzięcie NMP / Święto Wojska Polskiego
        datetime.date(year, 11, 1),  # Wszystkich Świętych
        datetime.date(year, 11, 11), # Narodowe Święto Niepodległości
        datetime.date(year, 12, 25), # Boże Narodzenie (pierwszy dzień)
        datetime.date(year, 12, 26), # Boże Narodzenie (drugi dzień)
    }
    return holidays


# Mazowieckie voivodeship (Warsaw) winter break dates (Ferie zimowe) by school year
MAZOWIECKIE_WINTER_BREAKS = {
    2024: (datetime.date(2024, 1, 15), datetime.date(2024, 1, 28)),
    2025: (datetime.date(2025, 2, 3), datetime.date(2025, 2, 16)),
    2026: (datetime.date(2026, 1, 19), datetime.date(2026, 2, 1)),
    2027: (datetime.date(2027, 1, 18), datetime.date(2027, 1, 31)),
}


def is_polish_school_day(date_val: datetime.date) -> bool:
    """
    Determine if a given date is a primary/secondary school instructional day in Warsaw / Mazowieckie.

    Returns False for:
    - Saturdays and Sundays
    - Statutory public holidays
    - Summer vacations (late June through August 31)
    - Winter break (Ferie zimowe - Mazowieckie schedule)
    - Christmas / New Year school break (Dec 23 - Jan 1)
    - Easter school break (Holy Thursday through Easter Tuesday)
    - Teacher's Day (Oct 14)
    """
    # Weekends are not school days
    if date_val.weekday() >= 5:
        return False

    # Public holidays are not school days
    year = date_val.year
    holidays = get_polish_public_holidays(year)
    if date_val in holidays:
        return False

    # Summer vacation: July and August, or late June (after June 22)
    month = date_val.month
    day = date_val.day
    if month in (7, 8):
        return False
    if month == 6 and day >= 25:
        # Most Polish school years end around June 23-28
        return False

    # Christmas break: Dec 23 - Dec 31 (Jan 1 is already caught by New Year holiday)
    if month == 12 and day >= 23:
        return False

    # Winter break (Mazowieckie)
    if year in MAZOWIECKIE_WINTER_BREAKS:
        start_wb, end_wb = MAZOWIECKIE_WINTER_BREAKS[year]
        if start_wb <= date_val <= end_wb:
            return False

    # Easter spring break: Holy Thursday (Easter - 3) to Easter Tuesday (Easter + 2)
    easter = calculate_easter(year)
    easter_break_start = easter - datetime.timedelta(days=3)
    easter_break_end = easter + datetime.timedelta(days=2)
    if easter_break_start <= date_val <= easter_break_end:
        return False

    # Teacher's Day (Oct 14): statutory free from didactic classes
    if month == 10 and day == 14:
        return False

    return True


def generate_polish_calendar_df(
    start_date: str = "2024-01-01",
    end_date: str = "2027-12-31",
) -> pd.DataFrame:
    """
    Generate a calendar lookup DataFrame covering the given range.

    Columns:
    - calendar_date (str 'YYYY-MM-DD')
    - is_school_day (int: 1 or 0)
    - is_public_holiday (int: 1 or 0)
    - is_holiday_eve (int: 1 or 0)
    """
    s_dt = datetime.date.fromisoformat(str(start_date)[:10])
    e_dt = datetime.date.fromisoformat(str(end_date)[:10])

    curr = s_dt
    records = []
    all_holidays: Set[datetime.date] = set()
    for y in range(s_dt.year - 1, e_dt.year + 2):
        all_holidays.update(get_polish_public_holidays(y))

    while curr <= e_dt:
        is_holiday = int(curr in all_holidays)
        is_school = int(is_polish_school_day(curr))
        next_day = curr + datetime.timedelta(days=1)
        is_eve = int(next_day in all_holidays and curr.weekday() < 5)

        records.append(
            {
                "calendar_date": curr.isoformat(),
                "is_school_day": is_school,
                "is_public_holiday": is_holiday,
                "is_holiday_eve": is_eve,
            }
        )
        curr += datetime.timedelta(days=1)

    df = pd.DataFrame(records)
    return df


def generate_calendar_parquet(output_path: Optional[str] = None) -> str:
    """Export calendar lookup table to Parquet."""
    out = output_path or "data/processed/calendar_lookup.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = generate_polish_calendar_df()
    df.to_parquet(out, index=False, compression="snappy")
    logger.info(f"Generated Polish calendar lookup table at {out} ({len(df)} dates)")
    return out


if __name__ == "__main__":
    df_cal = generate_polish_calendar_df("2026-10-01", "2026-10-15")
    print(df_cal.to_string())
