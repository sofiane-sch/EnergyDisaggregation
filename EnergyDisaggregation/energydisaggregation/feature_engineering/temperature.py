import pandas as pd
import numpy as np
from energydisaggregation import DATACONFIG, CONFIG_WEATHER, CONFIG_POWER


def get_critval(df_tot):
    df_out = df_tot.copy()
    df_out["years"] = list(df_out.index.get_level_values(0).year.map(str))
    df_out["regions"] = list(df_out.index.get_level_values(1))
    crit_val_df = create_critval_df(df_out)

    df_out = df_out.reset_index()
    df_out = pd.merge(df_out, crit_val_df, how="left", on=["years", "regions"])
    df_out = df_out.set_index([CONFIG_POWER["Date"], CONFIG_POWER["Region"]])
    df_out.drop(columns=["years", "regions"])

    df_out = compute_gradient(df_out)
    df_out["diff_seuil"] = (
        df_out["temperature_seuil"] - df_out[CONFIG_WEATHER["Temperature"]]
    )
    return df_out


def compute_polynomial_fit(df_tot, region, year):
    df = df_tot.xs(region, level=DATACONFIG["Region"]).loc[year]

    temp = df[CONFIG_WEATHER["Temperature"]]
    conso = df[CONFIG_POWER["Power"]]
    fit_poly = np.polyfit(temp, conso, 4)

    points = np.linspace(temp.min() - 1, temp.max() + 1, 400)
    values_poly = [np.polyval(fit_poly, i) for i in points]
    values_gradient = [
        np.polyval([fit_poly[0] * 4, fit_poly[1] * 3, fit_poly[2] * 2, fit_poly[1]], i)
        for i in temp
    ]
    return points, values_poly, values_gradient


def compute_min_satur(poly_values_x, poly_values_y):
    saturation = [
        poly_values_x[i]
        for i in range(1, len(poly_values_y) - 1)
        if poly_values_y[i - 1] < poly_values_y[i] > poly_values_y[i + 1]
    ]
    minimum = poly_values_x[np.argmin(poly_values_y)]
    return minimum, saturation


def create_critval_df(df_tot):
    years = list(df_tot.index.get_level_values(0).year.map(str))
    regions = list(df_tot.index.get_level_values(1))
    regions_years = pd.Series(zip(regions, years)).unique()
    df_out = pd.DataFrame()
    for x in regions_years:
        pt, val, values_gradient = compute_polynomial_fit(df_tot, x[0], x[1])
        minimum, saturation = compute_min_satur(pt, val)
        new_row = {
            "years": x[1],
            "regions": x[0],
            "temperature_seuil": minimum,
            "saturation": saturation,
        }
        df_out = df_out.append(new_row, ignore_index=True)
        # df_out['gradient'] = values_gradient
    return df_out


def compute_gradient(df_tot):
    years = list(df_tot.index.get_level_values(0).year.map(str))
    regions = list(df_tot.index.get_level_values(1))
    regions_years = pd.Series(zip(regions, years)).unique()
    df_out = pd.DataFrame()
    for x in regions_years:
        df_out = df_tot.xs(x[0], level=DATACONFIG["Region"]).loc[x[1]].reset_index()
        temp = df_out[CONFIG_WEATHER["Temperature"]]
        conso = df_out[CONFIG_POWER["Power"]]
        fit_poly = np.polyfit(temp, conso, 4)
        df_out["gradient"] = df_out[CONFIG_WEATHER["Temperature"]].apply(
            lambda x: np.polyval(
                [fit_poly[0] * 4, fit_poly[1] * 3, fit_poly[2] * 2, fit_poly[3]], x
            )
        )
        df_out[CONFIG_POWER["Region"]] = x[0]
        df_out = df_out.append(df_out)
    df_out = df_out.set_index([CONFIG_POWER["Date"], CONFIG_POWER["Region"]])
    return df_out


def temperature_ressentie(df_tot):
    c1 = -8.785
    c2 = 1.611
    c3 = 2.339
    c4 = -0.146
    c5 = -1.231 * 10 ** (-2)
    c6 = -1.642 * 10 ** (-2)
    c7 = 2.212 * 10 ** (-3)
    c8 = 7.255 * 10 ** (-4)
    c9 = -3.582 * 10 ** (-6)
    df_tot["Vitesse du vent en km/h"] = df_tot["Vitesse du vent moyen 10 mn"] * 3.6
    df_tot["Température ressentie"] = df_tot["Température (°C)"]
    df_tot.loc[
        (df_tot[CONFIG_WEATHER["Temperature"]] > 20) & (df_tot["Humidité"] > 40),
        "Température ressentie",
    ] = (
        c1
        + c2 * df_tot["Température (°C)"]
        + c3 * df_tot["Humidité"]
        + c4 * df_tot["Température (°C)"] * df_tot["Humidité"]
        + c5 * df_tot["Température (°C)"] ** 2
        + c6 * df_tot["Humidité"] ** 2
        + c7 * (df_tot["Température (°C)"] ** 2) * df_tot["Humidité"]
        + c8 * df_tot["Température (°C)"] * (df_tot["Humidité"] ** 2)
        + c9 * (df_tot["Température (°C)"] ** 2) * (df_tot["Température (°C)"] ** 2)
    )
    df_tot.loc[
        (df_tot[CONFIG_WEATHER["Temperature"]] < 10)
        & (df_tot["Vitesse du vent en km/h"] > 4.8),
        "Température ressentie",
    ] = (
        13.12
        + 0.6215 * df_tot[CONFIG_WEATHER["Temperature"]]
        + (0.3965 * df_tot[CONFIG_WEATHER["Temperature"]] - 11.37)
        * df_tot["Vitesse du vent en km/h"] ** 0.16
    )
    df_tot.loc[
        (df_tot[CONFIG_WEATHER["Temperature"]] < 10)
        & (df_tot["Vitesse du vent en km/h"] < 4.8),
        "Température ressentie",
    ] = (
        df_tot[CONFIG_WEATHER["Temperature"]]
        + 0.2
        * (0.1345 * df_tot[CONFIG_WEATHER["Temperature"]] - 1.59)
        * df_tot["Vitesse du vent en km/h"]
    )
    return df_tot


def test_temperature():
    print("Temperature module can be accessed")
