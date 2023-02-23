def cyclical_encoder(df, col, max_val):
    df = df.copy()
    # On encode les paramètres cycliques (heure, mois, etc) à l'aide de deux colonnes sin et cos
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df.loc[:, [col + "_sin", col + "_cos"]]


def test_cyclical_transformation():
    print("Cyclical transformation module can be accessed")
