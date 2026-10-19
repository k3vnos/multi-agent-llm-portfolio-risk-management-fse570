import pandas as pd

from riskcausal.data.calendar import add_bdays, nyse_sessions


def test_nyse_calendar_excludes_holidays_and_weekends():
    s = nyse_sessions("2020-12-20", "2021-01-10")
    assert pd.Timestamp("2020-12-25") not in s
    assert pd.Timestamp("2021-01-01") not in s
    assert pd.Timestamp("2020-12-26") not in s  # Saturday
    assert pd.Timestamp("2020-12-24") in s


def test_add_bdays_moves_friday_release_to_monday():
    s = nyse_sessions("2023-03-01", "2023-03-31")
    out = add_bdays(pd.DatetimeIndex(["2023-03-10"]), 1, s)  # Friday
    assert out[0] == pd.Timestamp("2023-03-13")


def test_add_bdays_from_non_session_date():
    s = nyse_sessions("2023-03-01", "2023-03-31")
    out = add_bdays(pd.DatetimeIndex(["2023-03-11"]), 1, s)  # Saturday -> next session
    assert out[0] == pd.Timestamp("2023-03-13")
