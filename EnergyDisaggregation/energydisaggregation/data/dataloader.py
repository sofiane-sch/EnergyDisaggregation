import pandas as pd
from .config import CONFIG_POWER, CONFIG_WEATHER, DATACONFIG


class Dataloader:
    def __init__(self) -> None:
        pass

    @classmethod
    def pre_treatment(
        cls, data, config, frequency: str = "h", resampling_mode: str = "sum"
    ):
        data = data[list(config.values())]
        data[config["Date"]] = pd.to_datetime(
            data[config["Date"]].str.replace("T", " ").str.replace("+", " +")
        )
        data = (
            data.groupby([config["Date"], config["Region"]])
            .first()
            .sort_index(level=config["Date"])
        )
        data = data.unstack(level=config["Region"]).asfreq("30T").interpolate()
        data = data.resample(frequency).agg(resampling_mode)
        data = data.stack(level=config["Region"])
        return data

    @classmethod
    def load_power(cls, path: str, frequency: str = "h", resampling_mode: str = "sum"):
        df = pd.read_csv(path, sep=";")
        df = df[df[CONFIG_POWER["Status"]] == "Définitif"].drop(
            CONFIG_POWER["Status"], axis=1
        )
        config = {x: CONFIG_POWER[x] for x in CONFIG_POWER.keys()}
        config.pop("Status")
        df = cls.pre_treatment(
            data=df,
            config=config,
            frequency=frequency,
            resampling_mode=resampling_mode,
        )
        return df

    @classmethod
    def load_weather(
        cls, path: str, frequency: str = "h", resampling_mode: str = "sum"
    ):
        df = pd.read_csv(path, sep=";")
        df = cls.pre_treatment(
            data=df,
            config=CONFIG_WEATHER,
            frequency=frequency,
            resampling_mode=resampling_mode,
        )
        return df

    @classmethod
    def load_data(
        cls,
        path_power: str,
        path_weather: str,
        frequency: str = "h",
        resampling_mode: str = "sum",
    ):
        df_power = cls.load_power(
            path_power, frequency=frequency, resampling_mode=resampling_mode
        )
        df_weather = cls.load_weather(
            path_weather, frequency=frequency, resampling_mode=resampling_mode
        )
        df_tot = (
            df_power.reset_index()
            .merge(
                df_weather,
                left_on=[CONFIG_POWER["Date"], CONFIG_POWER["Region"]],
                right_on=[CONFIG_WEATHER["Date"], CONFIG_WEATHER["Region"]],
            )
            .set_index([DATACONFIG["Date"], DATACONFIG["Region"]])
        )
        del df_power
        del df_weather
        return df_tot
