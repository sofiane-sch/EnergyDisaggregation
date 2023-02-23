# Import external modules
import os
import pandas as pd
pd.options.plotting.backend = "plotly"
# Import initial internal ressources (class, functions, variables, etc..)
from EnergyDisaggregation.energydisaggregation.dataloader.dataloader import Dataloader
from EnergyDisaggregation.energydisaggregation.dataloader.config import DATACONFIG, CONFIG_WEATHER, CONFIG_POWER
# Import new internal ressources (class, functions, variables, etc..)
from EnergyDisaggregation.energydisaggregation.feature_engineering.historical import timeseries_dataframe
from EnergyDisaggregation.energydisaggregation.feature_engineering.calendar import get_calendar_features
from EnergyDisaggregation.energydisaggregation.feature_engineering.cyclical_transformation import cyclical_encoder
from EnergyDisaggregation.energydisaggregation.feature_engineering.temperature import get_critval,temperature_ressentie,temperature_lagh,nebul_features

# Define used paths & filenames
raw_data_path = "Data/"
raw_power_filename = os.path.join(raw_data_path, "consommation-quotidienne-brute-regionale.csv")
raw_weather_filename = os.path.join(raw_data_path, "donnees-synop-essentielles-omm.csv")
df_tot = Dataloader.load_data(path_power=raw_power_filename, path_weather=raw_weather_filename, frequency="d", resampling_mode_power="sum", resampling_mode_weather="mean")

def get_dataframe() : 
    df_calendar = get_calendar_features(df_tot)
    df_cycl_m = cyclical_encoder(df_calendar,"month",12)
    df_cycl_w = cyclical_encoder(df_calendar,"week_day",7)
    df_critval = get_critval(df_tot)
    df_temp = temperature_ressentie(df_tot)
    df_lag = temperature_lagh(df_tot,lagh=48)
    df_nebul = nebul_features(df_tot,lagh=3)
    new_df = pd.concat([df_tot,df_calendar,df_critval,df_temp,df_lag,df_nebul,df_cycl_m,df_cycl_w],axis=1)
    return new_df