from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

holidays = SchoolHolidayDates()
feries = JoursFeries()


def get_calendar_features(df_tot):
    """
    Gets the calendar features of a time series
    """
    df_out = df_tot.copy()

    dates = df_out.index.get_level_values(0)
    df_out["date"] = df_out.index.get_level_values(0).date
    df_out["zone"] = df_out.index.get_level_values(1).map(get_zone)

    df_out["saison"] = dates.map(get_season)
    df_out["week_day"] = dates.dayofweek
    df_out["month"] = dates.month
    df_out["hour"] = dates.hour
    df_out["is_holiday"] = df_out[["date", "zone"]].apply(get_holiday_for_zone, axis=1)
    df_out["is_bank_holiday"] = df_out["date"].apply(feries.is_bank_holiday)
    df_out["day_of_year"] = df_out["date"].apply(day_of_year)

    df_out = df_out.drop(columns=["date", "zone"])
    df_out = df_out.drop(
        columns=[
            "Consommation brute électricité (MW) - RTE",
            "Température (°C)",
            "Nebulosité totale",
            "Vitesse du vent moyen 10 mn",
            "Humidité",
        ]
    )
    return df_out


def get_season(a_date):
    """
    Gets the season of a date
    """

    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else
    date_num = a_date.timetuple().tm_yday

    if date_num in spring:
        season = "spring"
    elif date_num in summer:
        season = "summer"
    elif date_num in fall:
        season = "fall"
    else:
        season = "winter"
    return season


def get_zone(a_region):
    """
    Gets the season of a date
    """
    zone_b = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Nouvelle-Aquitaine"]
    zone_c = ["Île-de-France", "Occitanie"]

    if a_region in zone_b:
        zone = "B"
    elif a_region in zone_c:
        zone = "C"
    else:
        zone = "A"
    return zone


def get_holiday_for_zone(row):
    return holidays.is_holiday_for_zone(row["date"], row["zone"])


def day_of_year(date_str):
    return date_str.timetuple().tm_yday


def test_calendar():
    print("Calendar module can be accessed")
