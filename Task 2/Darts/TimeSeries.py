import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('AirPassengers.csv')
AirPassengerSeries = TimeSeries.from_dataframe(data, 'Month', '#Passengers')

values = np.array([10, 20, 30])
times = pd.DatetimeIndex(['20190101', '20190301', '20190701'])
test = TimeSeries.from_times_and_values(times=times, values=values, freq='2MS', fill_missing_dates=True, fillna_value=0)

series1, series2 = AirPassengerSeries.split_after(pd.Timestamp('19561001'))


series1, series2 = AirPassengerSeries[:-10], AirPassengerSeries[-10:]

#list = [0,1,2,3,4,5,6,7,8,9]
# [1,2,3,4,5,6,7,8,9] = [-9,-8,-7,-6,-5,-4,-3,-2,-1]

series1.plot()
series2.plot()
plt.show()

AirPassengerSeries.diff().add_datetime_attribute("month").plot()

plt.show()