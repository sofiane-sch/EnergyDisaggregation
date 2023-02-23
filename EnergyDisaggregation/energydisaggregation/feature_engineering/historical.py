def timeseries_dataframe(df, timelag, column):
    """Prend en entrée un dataframe avec un seul index (pour une région en particulier),
    ajoute autant de colonnes que l'on veut avoir de lag"""
    n = len(df.index)
    lag_dict = {}
    for i in range(timelag):
        data = [float("nan") for a in range(i + 1)]
        for j in range(n - (i + 1)):
            value = df.iloc[j + i + 1].loc[column] - df.iloc[j].loc[column]
            data.append(value)
        name = f"{column}_lag_{i + 1}"
        lag_dict[name] = data
    lag_df = pd.DataFrame(lag_dict)
    return lag_df


def test_historical():
    print("Historical module can be accessed")
