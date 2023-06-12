import calendar
import time

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from workbench.utilities import io



def ts_8760(year=2022, tz=None):
    if tz is None:
        index = pd.date_range(start=f"01-01-{year} 00:00", end=f"12-31-{year} 23:00", freq="h")
    else:
        index = pd.date_range(start=f"01-01-2022 00:30", end=f"12-31-2022 23:30", freq="h", tz=tz)
    if calendar.isleap(year):
        index = index[~((index.month == 2) & (index.day == 29))]
    return index


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


def datetime_index_to_hoy_array(datetime_idx):
    hoy_list = []
    for timestamp in datetime_idx:
        hoy_list.append(get_hoy(timestamp))
    return np.array(hoy_list)

def get_hoy(timestamp):
    """
    :param timestamp: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :return: int hour difference
    """

    if type(timestamp) is str:
        year = timestamp.split("-")[0]
    else:
        year = timestamp.year
    delta = np.datetime64(timestamp) - np.datetime64(f"{year}-01-01")
    return delta.astype('timedelta64[h]').astype(np.int32)


def create_analysis_period(start_date, end_date):
    """
    :param start_date: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :param end_date: a string input should be in the form 'YYYY-MM-DD-HH:MM'
    :return : a series index of Timestamps
    """
    return create_datetime(start=start_date, end=end_date, freq='H', override_leap=False)


def create_timestep_chunks(total_timesteps, ncpu):
    incre = int(total_timesteps / ncpu)
    incre_range = range(0, total_timesteps, incre)
    incre_chunks = [np.arange(i, i + incre) for i in incre_range]

    return incre_chunks


def build_analysis_period(sunup_array,hourly_resolution):
    annual_hourly_timeseries = np.arange(0, 8760, 1)
    sunup_timeseries = annual_hourly_timeseries[sunup_array].flatten()
    analysis_timeseries = sunup_timeseries[::hourly_resolution]
    if analysis_timeseries[-1] == sunup_timeseries[-1]:
        pass
    else:
        analysis_timeseries = np.append(analysis_timeseries, sunup_timeseries[-1])

    return analysis_timeseries


def get_timezone(latitude, longitude):
    """

    :param latitude: type float
    :param longitude: type float
    :return: timezone name  type string
    """

    return TimezoneFinder().timezone_at(lat=latitude, lng=longitude)


def filter_wea(wea_file, year, analysis_period):
    """
    Filters a .wea file using a datetime index based on the input hour range
    :param wea_file: the .wea filepath
    :param year: the year is needed for the datetime operation, but can be anything
    :param analysis_period: a string of to hours of the year formatted with a dash separating them (0-8760)
    :return: None
    """
    df, header = io.read_wea(wea_file, year=year)

    # filter df on datetime index
    analysis_start = hoy_to_date(int(analysis_period.split("-")[0]), year=year)
    analysis_end = hoy_to_date(int(analysis_period.split("-")[1]), year=year)
    df = df.loc[analysis_start:analysis_end]

    # cut to the body values and turn into a string
    body = df[['month', 'day', 'hour', 3, 4]].reset_index(drop=True).astype(str).to_csv(path_or_buf=None, header=None,
                                                                                        index=None, sep=" ")

    # append body to the header string
    header += body

    # overwrite full wea with trimmed wea
    with open(wea_file, "w") as fp:
        fp.writelines(header)


def current_time():
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())