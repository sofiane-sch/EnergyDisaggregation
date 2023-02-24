import pandas as pd

def timeseries_dataframe(df, timelag, column):
    index = df.index
    n = len(index)
    lag_dict = {}
    for i in range(timelag):
        data = [float('nan') for a in range(12 * (i + 1))]
        for j in range(n - 12 * (i + 1)):
            value = df.iloc[j + 12 * (i + 1)].loc[column] - df.iloc[j].loc[column]
            data.append(value)
        name = f'{column}_lag_{i + 1}'
        lag_dict[name] = data
    lag_df = pd.DataFrame(lag_dict)
    lag_df.set_index(index, inplace=True)
    return lag_df


def test_historical():
    print("Historical module can be accessed")
