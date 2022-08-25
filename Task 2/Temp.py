# First, some imports:
import numpy as np
from darts.utils import timeseries_generation as tg
import matplotlib.pyplot as plt

np.random.seed(42)

LENGTH = 3 * 365  # 3 years of daily data

# Melting: a sine with yearly periodicity and additive white noise
melting = (tg.sine_timeseries(length=LENGTH,
                              value_frequency=(1/365),
                              freq='D',
                              column_name='melting')
           + 0.15 * tg.gaussian_timeseries(length=LENGTH, freq='D'))

# Rainfalls: a sine with bi-weekly periodicity and additive white noise
rainfalls = (tg.sine_timeseries(length=LENGTH,
                                value_frequency=(1/14),
                                freq='D',
                                column_name='rainfall')
             + 0.3 * tg.gaussian_timeseries(length=LENGTH, freq='D'))

# We scale and shift the melting by 5 days; giving us the melting contribution
melting_contribution = 0.5 * melting.shift(5)

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

from darts.metrics import rmse

# We first set aside the first 80% as training series:
flow_train, _ = flow.split_before(0.8)


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
    print('Backtest RMSE = {}'.format(rmse(flow, backtest)))


plt.show()
from darts.models import BlockRNNModel

brnn_no_cov = BlockRNNModel(input_chunk_length=30,
                            output_chunk_length=10,
                            n_rnn_layers=2)

brnn_no_cov.fit(flow_train,
                epochs=100,
                verbose=True)

eval_model(brnn_no_cov)
plt.show()
