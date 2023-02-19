from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

vacances = SchoolHolidayDates()


def get_calendar_features(df):  # voir l'index, revoir les noms
    """
    Gets the calendar features of a time series
    """
    df_copy = df.copy()
    feries = JoursFeries()
    vacances = SchoolHolidayDates()
    dates = df_copy.index.get_level_values(0)
    df_copy["date"] = df_copy.index.get_level_values(0).date
    df_copy["zone"] = df_copy.index.get_level_values(1).map(get_zone)

    df_copy["saison"] = dates.map(get_season)
    df_copy["week_day"] = dates.dayofweek
    df_copy["month"] = dates.month
    print("ok")
    df_copy["is_holiday"] = df_copy[["date", "zone"]].apply(
        get_holiday_for_zone, axis=1
    )
    print("ok2")
    print(df_copy)
    df_copy["is_bank_holiday"] = df_copy["date"].apply(feries.is_bank_holiday)

    # keep relevant columns
    df_copy = df_copy.drop(columns=["date", "zone"])
    return df_copy


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
    """ """
    return vacances.is_holiday_for_zone(row["date"], row["zone"])
