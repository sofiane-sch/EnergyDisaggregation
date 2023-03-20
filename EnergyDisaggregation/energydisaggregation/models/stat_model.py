from EnergyDisaggregation.energydisaggregation.models.base import Base
import pandas as pd
import numpy as np

from EnergyDisaggregation.energydisaggregation.feature_engineering.calendar import (
    get_zone,
    get_holiday_for_zone,
)
from vacances_scolaires_france import SchoolHolidayDates
from jours_feries_france import JoursFeries

holidays = SchoolHolidayDates()
feries = JoursFeries()

col_to_keep = [
    "consommation brute électricité (mw) - rte",
    "température (°c)_mean_48",
    "nebulosité totale_mean_3",
    "month",
    "week_day",
    "hour",
    "is_holiday",
    "is_bank_holiday",
    "Région",
]


def preprocess(df):
    df["Date - Heure"] = pd.to_datetime(df["Date - Heure"])
    df.set_index(["Date - Heure"], inplace=True)
    df["hour"] = df["hour"].astype("category")
    df["week_day"] = df["week_day"].astype("category")
    df["month"] = df["month"].astype("category")
    df = df[col_to_keep]
    df1 = pd.get_dummies(df, columns=["hour", "week_day", "month"])
    return df1


def split_year(df, test_year):
    test = df[(pd.to_datetime(df.index).year == test_year)]
    train = df[(pd.to_datetime(df.index).year != test_year)]
    print(len(test), len(train))
    X_train_year = train.drop(["consommation brute électricité (mw) - rte"], axis=1)
    y_train_year = train["consommation brute électricité (mw) - rte"]
    X_test_year = test.drop(["consommation brute électricité (mw) - rte"], axis=1)
    y_test_year = test["consommation brute électricité (mw) - rte"]
    return X_train_year, X_test_year, y_train_year, y_test_year


def generate_annee(annee, data_train, region):
    dates = pd.date_range(
        start=str(annee) + "/01/01",
        end=str(annee + 1) + "/01/01",
        freq="h",
        tz="Europe/Paris",
    )
    month_list = list(dates.month)
    day_list = list(dates.weekday)
    hour_list = list(dates.hour)
    df_tot = pd.DataFrame(
        {"month": month_list, "week_day": day_list, "hour": hour_list}, index=dates
    )[:-1]

    nebu_mean = list(
        data_train.groupby(
            [
                pd.to_datetime(data_train.index).day_of_year,
                pd.to_datetime(data_train.index).hour,
            ]
        ).mean()["nebulosité totale_mean_3"]
    )
    print(len(df_tot), len(nebu_mean))
    if len(df_tot) < len(nebu_mean):
        del nebu_mean[59 * 24 : 60 * 24]
        print(len(nebu_mean))
    if len(df_tot) > len(nebu_mean):
        left = nebu_mean[: 59 * 24]
        right = nebu_mean[59 * 24 :]
        nebu_mean = left.append(right)
    assert len(df_tot) == len(nebu_mean)

    df_tot["nebulosité totale_mean_3"] = nebu_mean
    df_tot["is_bank_holiday"] = list(
        pd.Series(dates.date).apply(feries.is_bank_holiday)
    )[:-1]
    zone_reg = get_zone(region)
    df_tot["is_holiday"] = list(
        pd.DataFrame(
            {"date": dates.date, "zone": np.repeat(zone_reg, len(dates))}
        ).apply(get_holiday_for_zone, axis=1)
    )[:-1]
    df1 = pd.get_dummies(df_tot, columns=["hour", "week_day", "month"])
    df1["Région"] = region
    return df1


class Stats(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X_train, y_train):
        X_train = X_train.drop(["Région"], axis=1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = X_test.drop(["Région"], axis=1)
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        X_test = X_test.drop(["Région"], axis=1)
        return self.model.score(X_test, y_test)

    def predict_therm_reg(self, X_train, X_test, year_to_pred):
        df_year = generate_annee(year_to_pred, X_train, str(X_train["Région"][0]))
        conso_reg = []
        temp = np.arange(-6, 33, 0.5)
        for i in range(len(df_year)):
            val_fix = df_year.iloc[[i], :]
            new_df = pd.concat([val_fix] * len(temp))
            new_df["température (°c)_mean_48"] = temp
            new_df = new_df[list(X_train.columns)]
            pred = self.predict(new_df)
            conso_reg.append(pred.min())
            if i % 100 == 0:
                print("Progress : ....", round(i / len(df_year) * 100, 2), "%")
        conso_pred = self.predict(
            X_test[(pd.to_datetime(X_test.index).year == year_to_pred)]
        )

        conso_therm = conso_pred - conso_reg
        return conso_pred, conso_reg, conso_therm

    def eval_corr(
        self, X_train, X_test, year_to_pred, conso_pred=[], conso_reg=[], conso_therm=[]
    ):
        if conso_pred == [] or conso_reg == [] or conso_therm == []:
            conso_pred, conso_reg, conso_therm = self.predict_therm_reg(
                X_train, X_test, year_to_pred
            )
        df_res = X_test[(pd.to_datetime(X_test.index).year == 2018)]
        df_res["conso_therm"] = conso_therm
        df_res["conso_reg"] = conso_reg
        df_res["conso_pred"] = conso_pred

        # heure
        corr_h_reg = df_res["température (°c)_mean_48"].corr(df_res["conso_reg"])
        corr_h_therm = df_res["température (°c)_mean_48"].corr(df_res["conso_therm"])

        # jour
        temp_d = df_res["température (°c)_mean_48"].resample("d").mean()
        conso_reg_d = df_res["conso_reg"].resample("d").mean()
        conso_therm_d = df_res["conso_therm"].resample("d").mean()

        corr_d_reg = temp_d.corr(conso_reg_d)
        corr_d_therm = temp_d.corr(conso_therm_d)

        df_corr = pd.DataFrame(
            {
                "corr_therm": corr_h_therm,
                "corr_reg": corr_h_reg,
                "corr_therm_day": corr_d_therm,
                "corr_reg_day": corr_d_reg,
            },
            index=[year_to_pred],
        )
        return df_corr

    def eval_temp_var(
        self,
        X_train,
        X_test,
        year_to_pred,
        n=10,
        conso_pred=[],
        conso_reg=[],
        conso_therm=[],
    ):
        if conso_pred == [] or conso_reg == [] or conso_therm == []:
            conso_pred, conso_reg, conso_therm = self.predict_therm_reg(
                X_train, X_test, year_to_pred
            )
        df_res = X_test[(pd.to_datetime(X_test.index).year == 2018)]

        thermo_sens = {}
        ind = np.random.randint(low=0, high=len(df_res), size=n)
        for i in ind:
            conso_reg_pred_ind = conso_reg[i]
            conso_therm_pred_ind = conso_therm[i]

            temp_real = df_res["température (°c)_mean_48"].iloc[i]
            vec_temp = np.linspace(temp_real - 4, temp_real + 4, 15)
            val_fix = df_res.iloc[[i], :].drop(["température (°c)_mean_48"], axis=1)

            new_df = pd.DataFrame({"température (°c)_mean_48": vec_temp})
            for x in val_fix.columns:
                new_df[x] = np.repeat(val_fix[x][0], len(vec_temp))
            new_df = new_df[list(X_train.columns)]
            pred = self.predict(new_df)
            conso_therm_sim = pred - conso_reg_pred_ind
            thermo_sens[list(X_test.index)[i]] = (
                conso_therm_pred_ind,
                vec_temp,
                conso_therm_sim,
            )

        return thermo_sens
