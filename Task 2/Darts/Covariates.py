# First, some imports:
import numpy as np
import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt
import darts.models as models
from darts.utils import timeseries_generation as tg
from darts.models import BlockRNNModel, RegressionModel, RNNModel
from darts.metrics import rmse


def eval_model(model, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=flow,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8,
                                          retrain=False,
                                          verbose=True,
                                          forecast_horizon=10)

    flow[-len(backtest) - 100:].plot()
    backtest.plot(label='backtest (n=10)')
    plt.show()
    print('Backtest RMSE = {}'.format(rmse(flow, backtest)))


def BRNN_NoCovariates(train_flow):
    brnn_no_cov = BlockRNNModel(input_chunk_length=30,
                                output_chunk_length=10,
                                n_rnn_layers=2)

    brnn_no_cov.fit(train_flow,
                    epochs=100,
                    verbose=True)
    return brnn_no_cov


def BRNN_PastMeltingCovariates(trainflow, melting):
    brnn_melting = BlockRNNModel(input_chunk_length=30,
                                 output_chunk_length=10,
                                 n_rnn_layers=2)

    brnn_melting.fit(trainflow,
                     past_covariates=melting,
                     epochs=100,
                     verbose=True)
    return brnn_melting


def BRNN_PastMeltingRainCovariates(train, melting, rainfalls):
    brnn_melting_and_rain = BlockRNNModel(input_chunk_length=30,
                                          output_chunk_length=10,
                                          n_rnn_layers=2)

    brnn_melting_and_rain.fit(train,
                              past_covariates=melting.stack(rainfalls),
                              epochs=100,
                              verbose=True)
    return brnn_melting_and_rain


def RNN_FutureRain(train, melting, rainfalls):
    rnn_rain = RNNModel(input_chunk_length=30,
                        training_length=40,
                        n_rnn_layers=2)

    rnn_rain.fit(train,
                 future_covariates=rainfalls,
                 epochs=100,
                 verbose=True)
    return rnn_rain


def Regression_PastMeltingFutureRain(train, melting, rainfalls):
    # Lags of past covariates must be negative which means in the past
    # Lag value -5 means that value at time t-5 is used to predict the target at time t
    # Past covariate lags [-5,-4,-3,-2,-1] means that model will look the last 5 past_Covariate values
    # Same as lags_past_covariates=5
    # Lags of future covariates must be positive which means in the future
    # We dont specify any lags because we dont want the model to look at past values of target (river flow)
    # but only look at covariates
    regr_model = RegressionModel(lags=None,
                                 lags_past_covariates=[-5],
                                 lags_future_covariates=[-4, -3, -2, -1, 0]
                                 )

    regr_model.fit(train,
                   past_covariates=melting,
                   future_covariates=rainfalls
    )
    return regr_model


np.random.seed(42)
LENGTH = 3 * 365  # 3 years of daily data

# Melting: a sine with yearly periodicity and additive white noise
melting = (tg.sine_timeseries(length=LENGTH,
                              value_frequency=(1 / 365),
                              freq='D',
                              # Default -> start=pd.Timestamp('2000-01-01 00:00:00'),
                              column_name='melting')
           + 0.15 * tg.gaussian_timeseries(length=LENGTH, freq='D'))  # Gaussian Noise
"""
# Plot sin graphical for the first two years
a = tg.sine_timeseries(length=LENGTH,
                       value_frequency=(1/365),
                       freq='D',
                       column_name='melting')
a[:730].plot()
plt.show()
"""
# Rainfalls: a sine with every 2 weeks periodicity and additive white noise
rainfalls = (tg.sine_timeseries(length=LENGTH,
                                value_frequency=(1 / 14),
                                freq='D',
                                column_name='rainfall')
             + 0.3 * tg.gaussian_timeseries(length=LENGTH, freq='D'))  # Gaussian Noise
"""
# Plot sin graphical for the first two weeks
a = tg.sine_timeseries(length=LENGTH,
                       value_frequency=(1/14),
                       freq='D',
                       column_name='rainfall')
a[:30].plot()
plt.show()
"""
# Calculate melting contribution
# shift(n=5) -> Shifts the time axis of Timeseries melting  by n=5 time steps
# Scale by 1/2 in order to take the melting contribution to the river flow
melting_contribution = 0.5 * melting.shift(5)

"""
# Shift by 5 days and plot to see the difference after shift and before
melt = melting.shift(5)
melting[:30].plot()
melt[:30].plot()
plt.show()

# Scale by 1.2 and plot to see the difference after scale and before
melt2= 0.5*melt
melt2.plot()
melt.plot()
plt.show()
"""
# We compute similar contribution from the rainfalls
all_contributions = [melting_contribution] + [0.1 * rainfalls.shift(lag) for lag in range(5)]

# We compute the final flow as the sum of everything,
# trimming series so they all have the same start time
flow = sum([series[melting_contribution.start_time():][:melting.end_time()]
            for series in all_contributions]).with_columns_renamed('melting', 'flow')

# add some white noise
flow += 0.1 * tg.gaussian_timeseries(length=len(flow))

melting.plot()
rainfalls.plot()
flow.plot(lw=4)
plt.show()
"""
Test1
rain = rainfalls.values()[5] + rainfalls.values()[1] + rainfalls.values()[2] + rainfalls.values()[3] + rainfalls.values()[4]
total = (rain*0.1)+(melting.values()[0]*0.5)
flow[:30].plot(label="River Flow")
rainfalls[:30].plot(label="Rainfall")
melting[:30].plot(label="Melting")
plt.show()

Test2
# Returns a Timeseries Object with melting_contribution from 2000-01-06 to 2002-12-31  
# melting_contribution.start_time() -> 2000-01-06 Because we need melting rate from 5 days ago to calculate river flow
# melting.end_time() -> 2002-12-31  Because our data for rainfall need the all of the past 5 days and for melting rate the 5th previous day
melting_contribution=all_contributions[0][melting_contribution.start_time():][:melting.end_time()]
# Returns a Timeseries object with rainfall contribution from 2000-01-06 to 2002-12-31 
# We need rainfall for each of the last five days including excluding t-5 
rainfall_contribution_t4=all_contributions[1][melting_contribution.start_time():][:melting.end_time()]    
rainfall_contribution_t3=all_contributions[2][melting_contribution.start_time():][:melting.end_time()]    
rainfall_contribution_t2=all_contributions[3][melting_contribution.start_time():][:melting.end_time()]    
rainfall_contribution_t1=all_contributions[4][melting_contribution.start_time():][:melting.end_time()]    
rainfall_contribution_t=all_contributions[5][melting_contribution.start_time():][:melting.end_time()]

print(f"{all_contributions[1][pd.Timestamp('2000-01-06')].values()} {all_contributions[2][pd.Timestamp('2000-01-06')].values()}
        {all_contributions[3][pd.Timestamp('2000-01-06')].values()} {all_contributions[4][pd.Timestamp('2000-01-06')].values()}
        {all_contributions[5][pd.Timestamp('2000-01-06')].values()}")    
rainfall_first5days= rain[1:6].values()*0.1
flow_2000-01-06=sum(rainfall_first5days)+all_contributions[0][pd.Timestamp('2000-01-06')]
flow[0]
"""

# We first set aside the first 80% as training series:
flow_train, _ = flow.split_before(0.8)

rnn_no_cov = BRNN_NoCovariates(flow_train)
eval_model(rnn_no_cov)

rnn_melting = BRNN_PastMeltingCovariates(flow_train, melting)
eval_model(rnn_melting, past_covariates=melting)

rnn_melting_rain = BRNN_PastMeltingRainCovariates(flow_train, melting, rainfalls)
eval_model(rnn_melting_rain, past_covariates = melting.stack(rainfalls))

rnn_futureRain = RNN_FutureRain(flow_train, melting, rainfalls)
eval_model(rnn_futureRain,future_covariates = rainfalls)

regression_model = Regression_PastMeltingFutureRain(flow_train, melting, rainfalls)

eval_model(regression_model)

print()
