import pandas as pd
import numpy as np


def create_datetime(start="01-01-2023-00:00", end="12-31-2023-23:00", freq='H', override_leap=False):
    if override_leap == False:
        if pd.Timestamp(start).is_leap_year:
            print("Input year is a leap year, but Override leap is set to False.\n"
                  "Creating non-leap year timeseries with previous year. Will rename index to original year. \n"
                  "Pass 'override_leap=True' to create timeseries with leap year.")
            ts = start.split("-")
            ts[2] = str(int(ts[2]) - 1)
            start = "-".join(ts)
            ts = end.split("-")
            ts[2] = str(int(ts[2]) - 1)
            end = "-".join(ts)
            return pd.Series(pd.date_range(start, end, freq=freq)).map(lambda t: t.replace(year=2040))
        else:
            return pd.Series(pd.date_range(start, end, freq=freq))
    else:
        if pd.Timestamp(start).is_leap_year:
            print("Selected year is a leap year. Override leap is set to True. Creating non-leap year timeseries.\n"
                  "Pass 'override_leap=False' to create non-leap timeseries.")
            return pd.Series(pd.date_range(start, end, freq=freq))
        else:
            print(
                "Override leap is set to True, but selected year is not a leap year. Creating non-leap year timeseries.\n"
                "Pass a leap year into the start to create non-leap timeseries.")
            return pd.Series(pd.date_range(start, end, freq=freq))


def hoy_to_date(hoy, year=2023):
    return pd.Timestamp(f'{year}-01-01') + pd.to_timedelta(hoy, unit='H')


def get_hoy(timestamp):
    """
    :param timestamp: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :return: int hour difference
    """

    year = timestamp.split("-")[0]
    delta = np.datetime64(timestamp) - np.datetime64(f"{year}-01-01")
    return delta.astype('timedelta64[h]').astype(np.int32)


def create_analysis_period(start_date, end_date):
    """
    :param start_date: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :param end_date: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :return: a series index of Timestamps
    """
    return create_datetime(start=start_date, end=end_date, freq='H', override_leap=False)


def create_timestep_chunks(total_timesteps, ncpu):
    incre = int(total_timesteps / ncpu)
    incre_range = range(0, total_timesteps, incre)
    incre_chunks = [np.arange(i, i + incre) for i in incre_range]

    return incre_chunks
