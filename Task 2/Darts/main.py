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


def eval_model(model, train, val):
    model.fit(train)
    forecast = model.predict(len(val))
    print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, forecast)))

def eval_forecast(data, forecast):
    print("Mean absolute percentage error: {:.2f}%.".format(mape(data, forecast)))



data = pd.read_csv('AirPassengers.csv')
data = TimeSeries.from_dataframe(data, 'Month', '#Passengers')


train, val = data.split_before(pd.Timestamp("19580101"))

seasonal_model, seasonal_forecast = Naive_Seasonal(data, train, val, K=Inspect_Seasonality(train))
drift_model, drift_forecast = Naive_Drift(data, train, val)
combined_forecast = CombinedSeasonalDrift(drift_forecast, seasonal_forecast, data)
plt.show()

eval_model(seasonal_model, train, val)
eval_model(drift_model, train, val)
eval_forecast(data, combined_forecast)

eval_model(models.ExponentialSmoothing(), train, val)
#time.sleep(5)
#eval_model(models.TBATS(), train, val)
#time.sleep(5)
eval_model(models.AutoARIMA(), train, val)
#time.sleep(5)
eval_model(models.Theta(), train, val)

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

print(
    "The MAPE is: {:.2f}, with theta = {}.".format(
        mape(val, pred_best_theta), best_theta
    )
)

train.plot(label="train")
val.plot(label="true")
pred_best_theta.plot(label="prediction")
plt.show()