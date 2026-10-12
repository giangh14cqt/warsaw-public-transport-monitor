"""Unit tests for Polish public holiday and school calendar module."""

import datetime
import unittest

from src.core.polish_calendar import (
    calculate_easter,
    generate_polish_calendar_df,
    get_polish_public_holidays,
    is_polish_school_day,
)


class TestPolishCalendar(unittest.TestCase):
    def test_easter_calculation(self):
        # Known Easter Sunday dates
        self.assertEqual(calculate_easter(2024), datetime.date(2024, 3, 31))
        self.assertEqual(calculate_easter(2025), datetime.date(2025, 4, 20))
        self.assertEqual(calculate_easter(2026), datetime.date(2026, 4, 5))
        self.assertEqual(calculate_easter(2027), datetime.date(2027, 3, 28))

    def test_polish_public_holidays(self):
        holidays_2026 = get_polish_public_holidays(2026)
        # Fixed holidays
        self.assertIn(datetime.date(2026, 1, 1), holidays_2026)   # Nowy Rok
        self.assertIn(datetime.date(2026, 1, 6), holidays_2026)   # Trzech Króli
        self.assertIn(datetime.date(2026, 5, 1), holidays_2026)   # Święto Pracy
        self.assertIn(datetime.date(2026, 5, 3), holidays_2026)   # Konstytucji 3 Maja
        self.assertIn(datetime.date(2026, 8, 15), holidays_2026)  # Wniebowzięcie NMP
        self.assertIn(datetime.date(2026, 11, 1), holidays_2026)  # Wszystkich Świętych
        self.assertIn(datetime.date(2026, 11, 11), holidays_2026) # Niepodległości
        self.assertIn(datetime.date(2026, 12, 25), holidays_2026) # Boże Narodzenie 1
        self.assertIn(datetime.date(2026, 12, 26), holidays_2026) # Boże Narodzenie 2
        # Movable holidays
        self.assertIn(datetime.date(2026, 4, 6), holidays_2026)   # Poniedziałek Wielkanocny
        self.assertIn(datetime.date(2026, 6, 4), holidays_2026)   # Boże Ciało

    def test_school_day_logic(self):
        # Normal school days in Oct 2026
        self.assertTrue(is_polish_school_day(datetime.date(2026, 10, 5)))  # Monday
        self.assertTrue(is_polish_school_day(datetime.date(2026, 10, 6)))  # Tuesday
        self.assertTrue(is_polish_school_day(datetime.date(2026, 10, 9)))  # Friday

        # Weekends
        self.assertFalse(is_polish_school_day(datetime.date(2026, 10, 4))) # Sunday
        self.assertFalse(is_polish_school_day(datetime.date(2026, 10, 10))) # Saturday

        # Summer vacation
        self.assertFalse(is_polish_school_day(datetime.date(2026, 7, 15))) # July
        self.assertFalse(is_polish_school_day(datetime.date(2026, 8, 10))) # August

        # Teacher's Day
        self.assertFalse(is_polish_school_day(datetime.date(2026, 10, 14))) # Oct 14

        # Christmas break
        self.assertFalse(is_polish_school_day(datetime.date(2026, 12, 24))) # Dec 24

        # Mazowieckie winter break 2026 (Jan 19 - Feb 1)
        self.assertFalse(is_polish_school_day(datetime.date(2026, 1, 20)))

    def test_generate_calendar_df(self):
        df = generate_polish_calendar_df("2026-10-01", "2026-10-10")
        self.assertEqual(len(df), 10)
        self.assertIn("calendar_date", df.columns)
        self.assertIn("is_school_day", df.columns)
        self.assertIn("is_public_holiday", df.columns)
        self.assertIn("is_holiday_eve", df.columns)


if __name__ == "__main__":
    unittest.main()
