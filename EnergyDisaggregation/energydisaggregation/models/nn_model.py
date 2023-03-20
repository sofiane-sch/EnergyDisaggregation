from EnergyDisaggregation.energydisaggregation.models.base import Base
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.metrics import mean_squared_error

list_temp = [
    "température (°c)",
    "nebulosité totale",
    "vitesse du vent moyen 10 mn",
    "humidité",
    "vitesse du vent en km/h",
    "température ressentie",
    "temperature_seuil",
    "diff_seuil",
    "température ressentie.1",
    "température (°c)_mean_48",
    "température (°c)_min_48",
    "température (°c)_std_48",
    "température (°c)_max_48",
    "nebulosité totale_mean_3",
    "nebulosité totale_std_3",
    "température (°c)_lag_1",
    "température (°c)_lag_2",
    "température (°c)_lag_3",
    "température (°c)_lag_4",
    "température (°c)_lag_5",
    "température (°c)_lag_6",
    "température (°c)_lag_7",
]
list_calendar = [
    "saison",
    "hour_sin",
    "hour_cos",
    "week_day_sin",
    "week_day_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "is_bank_holiday",
    "day_of_year",
]


def split_year(df, test_year):
    test = df[(pd.to_datetime(df.index).year == test_year)]
    train = df[(pd.to_datetime(df.index).year != test_year)]
    print(len(test), len(train))
    X_train_year = train.drop(["consommation brute électricité (mw) - rte"], axis=1)
    y_train_year = train["consommation brute électricité (mw) - rte"]
    X_test_year = test.drop(["consommation brute électricité (mw) - rte"], axis=1)
    y_test_year = test["consommation brute électricité (mw) - rte"]
    return X_train_year, X_test_year, y_train_year, y_test_year


class FeaturesDataset(Dataset):
    """Calendar and Temperatures features dataset."""

    def __init__(self, df, list_temp, list_calendar):
        self.df = df
        self.temp = list_temp
        self.calendar = list_calendar

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        calendar_features = self.df.iloc[
            idx, self.df.columns.isin(self.calendar)
        ].tolist()
        temperature_features = self.df.iloc[
            idx, self.df.columns.isin(self.temp)
        ].tolist()
        target = self.df.iloc[idx]["consommation brute électricité (mw) - rte"]

        sample = {
            "temp": torch.tensor(temperature_features),
            "calendar": torch.tensor(calendar_features),
            "target": torch.tensor(target),
        }

        return sample


class NN(torch.nn.Module):
    def __init__(self, temp_dim, calendar_dim, hidden_dim, output_dim, num_layers):

        super(NN, self).__init__()

        # A list of layers for the temperature model
        self.temp = None

        # A list of layers for the calendar model
        self.calendar = None

        self.temp = torch.nn.ModuleList(
            [Linear(temp_dim, hidden_dim)]
            + [Linear(hidden_dim, hidden_dim) for i in range(num_layers - 2)]
            + [Linear(hidden_dim, output_dim)]
        )
        self.calendar = torch.nn.ModuleList(
            [Linear(calendar_dim, hidden_dim)]
            + [Linear(hidden_dim, hidden_dim) for i in range(num_layers - 2)]
            + [Linear(hidden_dim, output_dim)]
        )

    def reset_parameters(self):
        for layer in self.temp:
            layer.reset_parameters()
        for layer in self.calendar:
            layer.reset_parameters()

    def forward(self, x_temp, x_calendar):
        x = None

        for i in range(len(self.temp)):
            x_temp = self.temp[i](x_temp)
            x_temp = F.relu(x_temp)
            x_calendar = self.calendar[i](x_calendar)
            x_calendar = F.relu(x_calendar)
        x = x_temp + x_calendar
        return x, x_temp, x_calendar


def preprocess(df):
    df_region = df.bfill().ffill()
    df_region["Date - Heure"] = pd.to_datetime(df_region["Date - Heure"])
    df_region.set_index(["Date - Heure"], inplace=True)
    return df_region


class NeuralNetworkModel(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        temp_dim = len(list_temp)
        calendar_dim = len(list_calendar)
        self.model = NN(
            temp_dim,
            calendar_dim,
            kwargs["hidden_dim"],
            kwargs["output_dim"],
            kwargs["num_layers"],
        )

    def fit(self, df):
        features = FeaturesDataset(df, list_temp, list_calendar)
        train_loader = DataLoader(
            features, batch_size=1024, shuffle=True, num_workers=0
        )

        temp_dim = len(list_temp)
        calendar_dim = len(list_calendar)
        hidden_dim = 25
        output_dim = 1
        num_layers = 5
        model = NN(temp_dim, calendar_dim, hidden_dim, output_dim, num_layers)
        EPOCHS = 200
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function = torch.nn.MSELoss()

        # Run the training loop
        for epoch in range(0, EPOCHS):

            # Print epoch
            print(f"Starting epoch {epoch+1}")

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):

                # Get and prepare inputs
                inputs_temp = data["temp"].float()
                inputs_calendar = data["calendar"].float()
                targets = data["target"].float()

                targets = targets.reshape((targets.shape[0], 1))

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                out, out_temp, out_calendar = model(inputs_temp, inputs_calendar)

                # Compute loss
                loss = loss_function(out, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 10 == 0:
                    print(
                        "Loss after mini-batch %5d: %.3f" % (i + 1, current_loss / 500)
                    )
                    current_loss = 0.0

        # Process is complete.
        print("Training process has finished.")
        self.model = model

    def predict(self, X_test):
        features_test = FeaturesDataset(X_test, list_temp, list_calendar)
        predicted_conso = []
        conso_temp = []
        conso_calendar = []
        for i in range(len(features_test)):
            data = features_test[i]
            inputs_temp = data["temp"].float()
            inputs_calendar = data["calendar"].float()

            self.model.eval()
            # Make the prediction
            with torch.no_grad():
                out, out_temp, out_calendar = self.model(inputs_temp, inputs_calendar)
            predicted_value = out.item()
            predicted_value_temp = out_temp.item()
            predicted_value_calendar = out_calendar.item()

            predicted_conso.append(predicted_value)
            conso_temp.append(predicted_value_temp)
            conso_calendar.append(predicted_value_calendar)
        return predicted_conso, conso_temp, conso_calendar

    def score(self, X_test, y_test):
        return mean_squared_error(X_test, y_test)
