import numpy as np
import pandas as pd


def eval_temp_var(
    model,
    X_train,
    X_test,
    year_to_pred,
    n=10,
    conso_pred=[],
    conso_reg=[],
    conso_therm=[],
):
    """
    if conso_pred == [] or conso_reg == [] or conso_therm == []:
        conso_pred, conso_reg, conso_therm = self.predict_therm_reg(
            X_train, X_test, year_to_pred
        )
    """
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
        pred = model.predict(new_df)
        conso_therm_sim = pred - conso_reg_pred_ind
        thermo_sens[list(X_test.index)[i]] = (
            conso_therm_pred_ind,
            vec_temp,
            conso_therm_sim,
        )

    return thermo_sens
