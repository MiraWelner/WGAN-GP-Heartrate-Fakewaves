"""
Mira Welner
August 4

"""
import logging
logging.getLogger('neuralprophet').setLevel(logging.WARNING)

from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

snip_len=100
m = NeuralProphet()
m.set_plotting_backend("plotly")

patient_names = '06-31-24', '09-40-14', '10-48-45', '11-03-38', '13-22-23', '14-17-50'
patient_datasets = [np.loadtxt(f"processed_data/heartrate_{name}_unscaled.csv") for name in patient_names]

for d in patient_datasets:
    d = d[:snip_len].copy()
    start_time = pd.to_datetime("2020-01-01 00:00:00")
    time_index = pd.date_range(start=start_time, periods=len(d), freq='S')
    df = pd.DataFrame({'ds': time_index, 'y': d})

    # Split data in half
    halfway = snip_len//2
    df_train = df.iloc[:halfway].copy()
    df_actual_future = df.iloc[halfway:].copy()

    # Define model
    m = NeuralProphet(
        n_changepoints=20,
        daily_seasonality=True,
        n_lags=halfway//2,
        n_forecasts=halfway//2
    )
    _ = m.fit(df_train)
    future_it1 = m.make_future_dataframe(df_train, periods=1, n_historic_predictions=True)
    forecast = m.predict(future_it1)
    forecast_only = forecast[f'yhat{halfway//2}'].iloc[halfway:].copy()
    future_it1['y'].iloc[halfway:]  = forecast_only

    m2 = NeuralProphet(
        n_changepoints=20,
        daily_seasonality=True,
        n_lags=halfway//2,
        n_forecasts=halfway//2
    )
    _ = m2.fit(future_it1.iloc[halfway//2:])
    future_it2 = m2.make_future_dataframe(future_it1, periods=1, n_historic_predictions=True)
    print(future_it2.shape)
    break
    forecast = m.predict(future_it2)
    forecast_only = forecast[f'yhat{halfway//2}'].iloc[int(halfway*1.5):].copy()
    future_it2['y'].iloc[int(halfway*1.5):]  = forecast_only
    train_and_firstcast = pd.concat([df_train, forecast_only], ignore_index=True)
    # Plot

    print(future_it2.shape)
    print(df_actual_future.shape)

    plt.figure(figsize=(12, 4))
    plt.plot(future_it2['y'], label='predicted', alpha=0.5)
    plt.plot(df_actual_future['y'], label='actual', alpha=0.5)
    plt.title("Recursive Forecast on Future Half")
    plt.legend()
    plt.tight_layout()
    plt.show()

    break  # only for one patient
