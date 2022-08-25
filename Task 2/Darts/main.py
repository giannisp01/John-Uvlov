import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt
import numpy as np
import darts.models as models
from darts.utils.statistics import plot_acf, check_seasonality
from darts.metrics import mape
import time


def Naive_Seasonal(data, train, val, K=1):
    naive_model = models.NaiveSeasonal(K=K)
    naive_model.fit(train)
    naive_forecast = naive_model.predict(36)

    data.plot(label="actual")
    message = "naive forecast (K=" + str(K) + ")"
    naive_forecast.plot(label=message)
    return naive_model, naive_forecast


def Inspect_Seasonality(train):
    plot_acf(train, m=12, alpha=0.05)
    for m in range(2, 25):
        is_seasonal, period = check_seasonality(train, m=m, alpha=0.05)
        if is_seasonal:
            print("There is seasonality of order {}.".format(period))
            return m


def Naive_Drift(data, train, val):
    drift_model = models.NaiveDrift()
    drift_model.fit(train)
    drift_forecast = drift_model.predict(36)

    data.plot(label="actual")
    drift_forecast.plot(label="drift")
    return drift_model, drift_forecast


def CombinedSeasonalDrift(drift_forecast, seasonal_forecast, data):
    data.plot()
    combined_forecast = drift_forecast + seasonal_forecast - train.last_value()
    combined_forecast.plot(label="combined")
    return combined_forecast


def theta_model(train, val):
    # Search for the best theta parameter, by trying 50 different values
    thetas = 2 - np.linspace(-10, 10, 50)

    best_mape = float("inf")
    best_theta = 0

    for theta in thetas:
        model = models.Theta(theta)
        model.fit(train)
        pred_theta = model.predict(len(val))
        res = mape(val, pred_theta)

        if res < best_mape:
            best_mape = res
            best_theta = theta

    best_theta_model = models.Theta(best_theta)
    best_theta_model.fit(train)
    pred_best_theta = best_theta_model.predict(len(val))

    return best_theta_model, pred_best_theta


def theta_statistics(data, train, val, predicted_theta_model, model_theta):
    train.plot(label="train")
    val.plot(label="true")
    predicted_theta_model.plot(label="prediction")
    plt.show()

    historical_fcast_theta = model_theta.historical_forecasts(
        data, start=0.6, forecast_horizon=3, last_points_only=True, stride=12
    )

    data.plot(label="data")
    historical_fcast_theta.plot(label="backtest 3-months ahead forecast (Theta)")
    eval_forecast(data, historical_fcast_theta)
    plt.show()

    historical_fcast_theta = model_theta.historical_forecasts(
        data, start=0.5, forecast_horizon=12, last_points_only=False, stride=12)
    data.plot(label="data")
    for forecast in historical_fcast_theta:
        forecast.plot()
    plt.show()

    raw_errors = model_theta.backtest(
        data, start=0.6, forecast_horizon=3, metric=mape, reduction=None, verbose=True
    )
    from darts.utils.statistics import plot_hist

    plot_hist(
        raw_errors,
        bins=np.arange(0, max(raw_errors), 1),
        title="Individual backtest error scores (histogram)",
    )
    plt.show()

    # [errors for errors in raw_errors if 2.0<=errors<=4.0]
    from darts.utils.statistics import plot_residuals_analysis

    plot_residuals_analysis(model_theta.residuals(data))
    plt.show()


def eval_model(model, train, val):
    model.fit(train)
    forecast = model.predict(len(val))
    print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, forecast)))


def eval_forecast(data, forecast):
    print("Mean absolute percentage error: {:.2f}%.".format(mape(data, forecast)))


# Read a pandas DataFrame
data = pd.read_csv('AirPassengers.csv')
# Create a TimeSeries object with datetime monthly index and number of passengers
data = TimeSeries.from_dataframe(data, 'Month', '#Passengers')
# Split Dataset Train
# Train -> Begin : 1957-12-01
# Test -> 1958-01-01 : End
train, val = data.split_before(pd.Timestamp("19580101"))
"""
seasonal_model, seasonal_forecast = Naive_Seasonal(data, train, val, K=Inspect_Seasonality(train))
drift_model, drift_forecast = Naive_Drift(data, train, val)
combined_forecast = CombinedSeasonalDrift(drift_forecast, seasonal_forecast, data)
plt.show()

eval_model(seasonal_model, train, val)
eval_model(drift_model, train, val)
eval_forecast(data, combined_forecast)

eval_model(models.ExponentialSmoothing(), train, val)
# time.sleep(5)
# eval_model(models.TBATS(), train, val)
# time.sleep(5)
eval_model(models.AutoARIMA(), train, val)
# time.sleep(5)
eval_model(models.Theta(), train, val)

model_theta, pred_theta = theta_model(train, val)
eval_forecast(data, pred_theta)
theta_statistics(data, train, val, pred_theta, model_theta)

"""

from darts.datasets import AirPassengersDataset, MonthlyMilkDataset

# Convert to float to make faster the training of the model
series_air = AirPassengersDataset().load().astype(np.float32)
series_milk = MonthlyMilkDataset().load().astype(np.float32)

# Train Dataset -> Except last 36 months (:-36)
# Validation/Test Dataset -> last 36 months of each series
train_air, val_air = series_air[:-36], series_air[-36:]
train_milk, val_milk = series_milk[:-36], series_milk[-36:]

train_air.plot(label="air_train")
val_air.plot(label="air_val")
train_milk.plot(label="milk_train")
val_milk.plot(label="milk_val")
plt.show()

from darts.dataprocessing.transformers import Scaler

# Define Scaler with default values (feature_range -> (0,1))
scaler = Scaler()
# fit_transform will transform the data of train_air dataset from 0 to 1 and return train_air_scaled Timeseries
train_air_scaled, train_milk_scaled = scaler.fit_transform([train_air, train_milk])
# min(train_air.values()) -> train_air[-5:] , 1949-11-01,  104 passengers
# max(train_air.values()) -> train_air[9:12] , 1957-08-01 , 467 passengers
train_air_scaled.plot(label="air_train")
train_milk_scaled.plot(label="milk_train")

plt.show()
"""
model = models.NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)

model.fit([train_air_scaled, train_milk_scaled], epochs=50, verbose=True)
pred_air = model.predict(series=train_air_scaled, n=36)
pred_milk = model.predict(series=train_milk_scaled, n=36)

# scale back:
pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

plt.figure(figsize=(10, 6))
series_air.plot(label="actual (air)")
series_milk.plot(label="actual (milk)")
pred_air.plot(label="forecast (air)")
pred_milk.plot(label="forecast (milk)")
plt.show()
"""
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

# Build some external covariates of type 2D Timeseries  containing monthly and yearly values
# Use datetime_attribute_timeseries function to generate Timeseries objects
# First Timeseries object is with scaled from 0 to 1 month values and DateTime Index
# Second Timeseries object is with scaled from 0 to 1 year values and DateTime Index
# Those timeseries are returned using dt_attr which returns Timeseries objects
# Using concatenate we make those timeseries along a given axis which can be time/*component*/sample
air_covs = concatenate(
    [
        # Returns a new TimeSeries with index time_index and values are the derived information about time_index
        # Create Features by deriving info of time_index
        # Required fields for dt_attr
        # time_index -> series_Air.time_index which is of type pd.DateTimeIndex of Timeseries Object
        # attribute -> an attribute of pd.DatetimeIndex , dtype
        dt_attr(series_air.time_index, "month", dtype=np.float32) / 12,
        (dt_attr(series_air.time_index, "year", dtype=np.float32) - 1948) / 12,
    ],
    axis="component",
)

milk_covs = concatenate(
    [
        dt_attr(series_milk.time_index, "month", dtype=np.float32) / 12,
        (dt_attr(series_milk.time_index, "year", dtype=np.float32) - 1962) / 13,
    ],
    axis="component",
)

air_covs.plot()
plt.title(
    "one multivariate time series of 2 dimensions, containing covariates for the air series:"
)
