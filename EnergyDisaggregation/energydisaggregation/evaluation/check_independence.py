import pandas as pd


def eval_corr(
    model, X_train, X_test, year_to_pred, conso_pred=[], conso_reg=[], conso_therm=[]
):
    if conso_pred == [] or conso_reg == [] or conso_therm == []:
        conso_pred, conso_reg, conso_therm = model.predict_therm_reg(
            X_train, X_test, year_to_pred
        )
    df_res = X_test[(pd.to_datetime(X_test.index).year == year_to_pred)]
    df_res["conso_therm"] = conso_therm
    df_res["conso_reg"] = conso_reg
    df_res["conso_pred"] = conso_pred

    s1 = df_res["température (°c)_mean_48"]
    s2 = df_res["conso_therm"]
    s3 = df_res["conso_reg"]
    corr_therm = s1.rolling(247).corr(s2)
    corr_reg = s1.rolling(247).corr(s3)
    df_corr = pd.DataFrame(
        {"corr_therm": list(corr_therm), "corr_reg": list(corr_reg)},
        index=list(df_res.index),
    )
    return df_corr
