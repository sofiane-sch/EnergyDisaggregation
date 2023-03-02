# Import external modules
from EnergyDisaggregation.energydisaggregation.feature_engineering.temperature import get_critval, temperature_ressentie, temperature_lagh, nebul_features
from EnergyDisaggregation.energydisaggregation.feature_engineering.cyclical_transformation import cyclical_encoder
from EnergyDisaggregation.energydisaggregation.feature_engineering.calendar import get_calendar_features
from EnergyDisaggregation.energydisaggregation.feature_engineering.historical import timeseries_dataframe
from EnergyDisaggregation.energydisaggregation.dataloader.config import DATACONFIG, CONFIG_WEATHER, CONFIG_POWER
from EnergyDisaggregation.energydisaggregation.dataloader.dataloader import Dataloader
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

le = LabelEncoder()


raw_data_path = "Data/"
raw_power_filename = os.path.join(raw_data_path, "consommation-quotidienne-brute-regionale.csv")
raw_weather_filename = os.path.join(raw_data_path, "donnees-synop-essentielles-omm.csv")
df_tot = Dataloader.load_data(path_power=raw_power_filename, path_weather=raw_weather_filename,frequency="h", resampling_mode_power="sum", resampling_mode_weather="mean")


def get_dataframe(temp_lagh=48, neb_lagh=3, time_lagh=7):
    
    df_calendar = get_calendar_features(df_tot)
    df_cycl_m = cyclical_encoder(df_calendar, "month", 12)
    df_cycl_w = cyclical_encoder(df_calendar, "week_day", 7)
    df_critval = get_critval(df_tot)
    df_temp = temperature_ressentie(df_tot)
    df_lagh = temperature_lagh(df_tot, lagh=temp_lagh)
    df_nebul = nebul_features(df_tot, lagh=neb_lagh)
    df_lag = timeseries_dataframe(df_tot, time_lagh, CONFIG_WEATHER["Temperature"])
    df_lag1 = timeseries_dataframe(df_tot, time_lagh, CONFIG_POWER["Power"])
    new_df = pd.concat([df_tot, df_calendar, df_critval, df_temp,df_lagh, df_nebul, df_cycl_m, df_cycl_w, df_lag], axis=1)
    
    new_df.columns = new_df.columns.str.lower()
    new_df['saison'] = le.fit_transform(new_df['saison'])
    new_df.drop(columns=['week_day', 'month','humidité','years','regions'], inplace=True)
    return new_df
