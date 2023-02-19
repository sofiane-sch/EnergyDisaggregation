import pandas as pd
from energydisaggregation import DATACONFIG, CONFIG_WEATHER, CONFIG_POWER
import numpy as np


def get_critval(df_tot):
    df_tot["years"] = list(df_tot.index.get_level_values(0).year.map(str))
    df_tot["regions"] = list(df_tot.index.get_level_values(1))
    crit_val_df = _create_critval_df(df_tot)

    df_tot = pd.merge(df_tot, crit_val_df, on=["years", "regions"])
    df_tot.drop(columns=["years", "regions"])
    return df_tot


def _compute_polynomial_fit(df_tot, region, year):
    df = df_tot.xs(region, level=DATACONFIG["Region"]).loc[year]

    temp = df[CONFIG_WEATHER["Temperature"]]
    conso = df[CONFIG_POWER["Power"]]
    fit_poly = np.polyfit(temp, conso, 4)

    points = np.linspace(temp.min() - 1, temp.max() + 1, 400)
    values = [np.polyval(fit_poly, i) for i in points]

    return points, values


def _compute_min_satur(poly_values_x, poly_values_y):

    saturation = [
        poly_values_x[i]
        for i in range(1, len(poly_values_y) - 1)
        if poly_values_y[i - 1] < poly_values_y[i] > poly_values_y[i + 1]
    ]
    minimum = poly_values_x[np.argmin(poly_values_y)]

    return minimum, saturation


def _create_critval_df(df_tot):
    years = list(df_tot.index.get_level_values(0).year.map(str))
    regions = list(df_tot.index.get_level_values(1))
    regions_years = pd.Series(zip(regions, years)).unique()
    df = pd.DataFrame()

    for x in regions_years:
        pt, val = _compute_polynomial_fit(df_tot, x[0], x[1])
        minimum, saturation = _compute_min_satur(pt, val)
        new_row = {
            "years": x[1],
            "regions": x[0],
            "minimum": minimum,
            "saturation": saturation,
        }
        df = df.append(new_row, ignore_index=True)
    return df


def _compute_variance(df_tot, window_size):
    df = df_tot.copy()
    df.reset_index(inplace=True)
    df.set_index(DATACONFIG["Temperature"], inplace=True)
    df.sort_index(inplace=True)

    k_rolling_std = df.rolling(window_size).std()
    k_rolling_std[CONFIG_POWER["Power"]].plot(kind="scatter")
    print(k_rolling_std)
    return k_rolling_std
