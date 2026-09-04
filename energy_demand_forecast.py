import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset


POWER_PATH = '한국전력거래소_시간별 전국 전력수요량_20251231.csv'
WEATHER_PATH = 'OBS_ASOS_TIM_20260904122533.csv'
LOOKBACK = 24
TARGET = 'power_demand_mwh'
GRAPH_TITLE = 'National Power Demand and Regional Weather in 2025'


def mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = np.abs(actual) > 1e-8
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def load_data(power_path=POWER_PATH, weather_path=WEATHER_PATH):
    power = pd.read_csv(power_path)
    hour_columns = [f'{hour}시' for hour in range(1, 25)]
    power_long = power.melt(id_vars=['날짜'], value_vars=hour_columns,
                            var_name='hour_label', value_name=TARGET)
    power_long['datetime'] = pd.to_datetime(power_long['날짜']) + pd.to_timedelta(
        power_long['hour_label'].str.rstrip('시').astype(int), unit='h')
    power_long = power_long[['datetime', TARGET]]

    weather = pd.read_csv(weather_path)
    weather = weather.rename(columns={
        '일시': 'datetime', '기온(°C)': 'temperature_c', '강수량(mm)': 'rainfall_mm',
        '풍속(m/s)': 'wind_speed_ms', '습도(%)': 'humidity_pct',
        '일조(hr)': 'sunshine_hr', '일사(MJ/m2)': 'solar_radiation_mj_m2'
    })
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    weather = weather[['지점명', 'datetime', 'temperature_c', 'rainfall_mm', 'wind_speed_ms',
                       'humidity_pct', 'sunshine_hr', 'solar_radiation_mj_m2']]
    zero_when_missing = ['rainfall_mm', 'sunshine_hr', 'solar_radiation_mj_m2']
    weather[zero_when_missing] = weather[zero_when_missing].fillna(0)
    continuous_weather = ['temperature_c', 'wind_speed_ms', 'humidity_pct']
    weather[continuous_weather] = weather.groupby('지점명')[continuous_weather].transform(
        lambda values: values.interpolate(limit_direction='both'))
    regional_frames = []
    for region, region_frame in weather.groupby('지점명'):
        region_frame = region_frame.drop(columns='지점명').set_index('datetime')
        region_frame = region_frame[~region_frame.index.duplicated(keep='first')]
        region_frame = region_frame.reindex(pd.date_range('2025-01-01 01:00', periods=8760, freq='h'))
        region_frame[continuous_weather] = region_frame[continuous_weather].interpolate(limit_direction='both')
        region_frame[zero_when_missing] = region_frame[zero_when_missing].fillna(0)
        region_frame.columns = [f'{region}_{column}' for column in region_frame.columns]
        regional_frames.append(region_frame)
    regional_weather = pd.concat(regional_frames, axis=1).reset_index(names='datetime')
    data = power_long.merge(regional_weather, on='datetime', how='inner').sort_values('datetime')
    data = data.drop_duplicates('datetime').set_index('datetime').asfreq('h')
    regional_continuous = [column for column in data.columns
                           if any(column.endswith(f'_{weather_column}')
                                  for weather_column in continuous_weather)]
    regional_zero = [column for column in data.columns
                     if any(column.endswith(f'_{weather_column}')
                            for weather_column in zero_when_missing)]
    data[regional_continuous] = data[regional_continuous].interpolate(limit_direction='both')
    data[regional_zero] = data[regional_zero].fillna(0)
    data[TARGET] = data[TARGET].astype(float)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    data['month_sin'] = np.sin(2 * np.pi * (data.index.month - 1) / 12)
    data['month_cos'] = np.cos(2 * np.pi * (data.index.month - 1) / 12)
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(float)
    return data.reset_index()


def feature_columns(data):
    time_columns = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                    'is_weekend']
    weather_columns = [column for column in data.columns
                       if column not in {'datetime', TARGET} and column not in time_columns]
    return weather_columns + time_columns + [TARGET]


def split_by_time(data):
    train_end = int(len(data) * 0.70)
    val_end = train_end + int(len(data) * 0.15)
    return data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy()


def make_sequences(data, scaler, target_scaler, start_index, end_index):
    values = scaler.transform(data[feature_columns(data)])
    targets = target_scaler.transform(data[[TARGET]]).ravel()
    X, y, timestamps = [], [], []
    for index in range(max(LOOKBACK, start_index), end_index):
        X.append(values[index - LOOKBACK:index])
        y.append(targets[index])
        timestamps.append(data.iloc[index]['datetime'])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), timestamps


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, 64)
        self.position = nn.Parameter(torch.zeros(1, LOOKBACK, 64))
        layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0.1,
                                           batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        encoded = self.encoder(self.projection(x) + self.position)
        return self.head(encoded[:, -1]).squeeze(-1)


def fit_deep_model(model, X_train, y_train, X_val, y_val, epochs=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=64, shuffle=False)
    val_x = torch.tensor(X_val, device=device)
    val_y = torch.tensor(y_val, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_loss, best_state = np.inf, None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(val_x), val_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()


def arima_one_step_forecast(train_values, validation_values, test_values):
    fitted = SARIMAX(train_values, order=(2, 1, 2), seasonal_order=(1, 0, 1, 24),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    predictions = []
    for actual in np.concatenate([validation_values, test_values]):
        predictions.append(float(np.asarray(fitted.forecast(steps=1))[0]))
        fitted = fitted.extend([actual])
    split = len(validation_values)
    return np.asarray(predictions[:split]), np.asarray(predictions[split:])


@dataclass
class Result:
    model: str
    actual: np.ndarray
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def build_result(name, actual, predicted, residual_std):
    margin = 1.96 * residual_std
    return Result(name, actual, predicted, predicted - margin, predicted + margin)


def metrics(result):
    actual, predicted = result.actual, result.predicted
    actual_std = np.std(actual, ddof=1)
    prediction_std = np.std(predicted, ddof=1)
    return {
        'model': result.model,
        'MAE': mean_absolute_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE': mape(actual, predicted),
        'actual_std': actual_std,
        'prediction_std': prediction_std,
        'std_ratio': prediction_std / actual_std,
        'interval_coverage_95': np.mean((actual >= result.lower) & (actual <= result.upper)) * 100,
    }


def save_prediction_csv(results, timestamps):
    output = pd.DataFrame({'datetime': timestamps})
    for result in results:
        prefix = result.model.lower()
        output[f'{prefix}_actual'] = result.actual
        output[f'{prefix}_prediction'] = result.predicted
        output[f'{prefix}_lower_95'] = result.lower
        output[f'{prefix}_upper_95'] = result.upper
    output.to_csv('forecast_predictions.csv', index=False)


def save_graphs(data, results):
    sns.set_theme(style='whitegrid')
    figure, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 2])
    test_start = len(data) - len(results[0].actual)
    dates = data['datetime'].iloc[test_start:test_start + 200]
    axes[0].plot(dates, results[0].actual[:200], label='Actual national demand', linewidth=2)
    for result in results:
        axes[0].plot(dates, result.predicted[:200], label=f'{result.model} prediction', linewidth=1.5)
    axes[0].set_title(GRAPH_TITLE)
    axes[0].set_xlabel('Datetime')
    axes[0].set_ylabel('Power demand (MWh)')
    axes[0].tick_params(axis='x', rotation=20)
    demand_handles, demand_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(demand_handles, demand_labels)
    metric_frame = pd.DataFrame([metrics(result) for result in results])
    sns.barplot(data=metric_frame.melt(id_vars='model', value_vars=['MAE', 'RMSE']),
                x='model', y='value', hue='variable', ax=axes[1])
    axes[1].set_title('Model error comparison')
    axes[1].set_ylabel('Error')
    figure.tight_layout()
    figure.savefig('forecast_comparison.png', dpi=180)
    plt.close(figure)
    for result in results:
        figure, axis = plt.subplots(figsize=(18, 6))
        x = np.arange(min(200, len(result.actual)))
        axis.plot(x, result.actual[:len(x)], label='Actual')
        axis.plot(x, result.predicted[:len(x)], label='Prediction')
        axis.fill_between(x, result.lower[:len(x)], result.upper[:len(x)], alpha=0.25, label='95% interval')
        axis.set_title(f'{result.model} 95% Prediction Interval')
        axis.set_xlabel('Test time index')
        axis.set_ylabel('Power demand (MWh)')
        axis.legend()
        figure.tight_layout()
        figure.savefig(f'{result.model.lower()}_interval.png', dpi=180)
        plt.close(figure)


def main():
    torch.manual_seed(42)
    data = load_data()
    data.to_csv('national_power_regional_weather_2025.csv', index=False)
    train, validation, test = split_by_time(data)
    train_end, val_end = len(train), len(train) + len(validation)
    input_scaler = StandardScaler().fit(train[feature_columns(data)])
    target_scaler = StandardScaler().fit(train[[TARGET]])
    X_train, y_train, _ = make_sequences(data, input_scaler, target_scaler, 0, train_end)
    X_val, y_val, _ = make_sequences(data, input_scaler, target_scaler, train_end, val_end)
    X_test, y_test, timestamps = make_sequences(data, input_scaler, target_scaler, val_end, len(data))
    actual_validation = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    actual_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    lstm = fit_deep_model(LSTMForecaster(X_train.shape[-1]), X_train, y_train, X_val, y_val)
    transformer = fit_deep_model(TransformerForecaster(X_train.shape[-1]), X_train, y_train, X_val, y_val)
    lstm_validation_prediction = target_scaler.inverse_transform(predict(lstm, X_val).reshape(-1, 1)).ravel()
    transformer_validation_prediction = target_scaler.inverse_transform(predict(transformer, X_val).reshape(-1, 1)).ravel()
    lstm_prediction = target_scaler.inverse_transform(predict(lstm, X_test).reshape(-1, 1)).ravel()
    transformer_prediction = target_scaler.inverse_transform(predict(transformer, X_test).reshape(-1, 1)).ravel()
    arima_validation_prediction, arima_prediction = arima_one_step_forecast(
        train[TARGET].to_numpy(), actual_validation, actual_test)
    results = [
        build_result('ARIMA', actual_test, arima_prediction,
                     np.std(actual_validation - arima_validation_prediction, ddof=1)),
        build_result('LSTM', actual_test, lstm_prediction,
                     np.std(actual_validation - lstm_validation_prediction, ddof=1)),
        build_result('Transformer', actual_test, transformer_prediction,
                     np.std(actual_validation - transformer_validation_prediction, ddof=1)),
    ]
    metrics_frame = pd.DataFrame([metrics(result) for result in results])
    metrics_frame.to_csv('forecast_metrics.csv', index=False)
    save_prediction_csv(results, timestamps)
    save_graphs(data, results)
    print(metrics_frame.to_string(index=False))
    print(f'Rows: total={len(data)}, train={len(train)}, validation={len(validation)}, test={len(test)}')
    print('Saved: national_power_regional_weather_2025.csv, forecast_metrics.csv, forecast_predictions.csv, forecast_comparison.png, *_interval.png')


if __name__ == '__main__':
    main()
