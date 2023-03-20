import pandas as pd


def eval_corr(
    X_train, X_test, year_to_pred, conso_pred=[], conso_reg=[], conso_therm=[]
):
    """
    if conso_pred == [] or conso_reg == [] or conso_therm == []:
        conso_pred, conso_reg, conso_therm = predict_therm_reg(
            X_train, X_test, year_to_pred
        )
    """
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
