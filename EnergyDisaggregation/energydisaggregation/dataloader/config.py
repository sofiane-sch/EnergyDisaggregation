CONFIG_POWER = {
    "Date": "Date - Heure",
    "Region": "Région",
    "Power": "Consommation brute électricité (MW) - RTE",
    "Status": "Statut - RTE",
}

CONFIG_WEATHER = {
    "Date": "Date",
    "Region": "region (name)",
    "Temperature": "Température (°C)",
    "Nebulosite": "Nebulosité totale",
    "Vitesse du vent moyen 10 mn": "Vitesse du vent moyen 10 mn",
    "Humidite": "Humidité",
}

dict_diff = {
    x: CONFIG_WEATHER[x] for x in CONFIG_WEATHER.keys() if x not in CONFIG_POWER.keys()
}

DATACONFIG = {x: CONFIG_POWER[x] for x in CONFIG_POWER.keys() if x != "Status"}
DATACONFIG.update(dict_diff)
