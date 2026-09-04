import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_PATH = 'power_weather_seoul_2025.csv'
OUTPUT_DIR = '.'
LOOKBACK = 24
TARGET_COL = 'power_demand_mwh'


def configure_korean_font():
    font_paths = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
    ]
    for path in font_paths:
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass

    preferred_fonts = [
        'Apple SD Gothic Neo',
        'Hiragino Sans GB',
        'NanumGothic',
        'Malgun Gothic',
        'Arial Unicode MS',
        'sans-serif',
    ]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = preferred_fonts
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='whitegrid', rc={'font.family': 'sans-serif', 'font.sans-serif': preferred_fonts})


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-8
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


@dataclass
class ForecastResult:
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    pred: np.ndarray
    actual: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL]).copy()
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df


def build_feature_columns() -> list[str]:
    return [
        'temperature_c', 'rainfall_mm', 'wind_speed_ms', 'humidity_pct',
        'sunshine_hr', 'solar_radiation_mj_m2', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend',
        'power_demand_mwh'
    ]


def make_sequences(df: pd.DataFrame, lookback: int = LOOKBACK):
    feature_cols = build_feature_columns()
    X, y = [], []
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i][feature_cols].values
        X.append(window)
        y.append(df.iloc[i][TARGET_COL])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, LOOKBACK, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def fit_deep_model(model, X_train, y_train, X_val, y_val, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(yb)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_deep_model(model, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return pred.cpu().numpy()


def scale_inputs(X_train, X_val, X_test):
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    flat_val = X_val.reshape(-1, X_val.shape[-1])
    flat_test = X_test.reshape(-1, X_test.shape[-1])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(flat_train)
    X_train_s = scaler.transform(flat_train).reshape(X_train.shape)
    X_val_s = scaler.transform(flat_val).reshape(X_val.shape)
    X_test_s = scaler.transform(flat_test).reshape(X_test.shape)
    return X_train_s, X_val_s, X_test_s, scaler


def train_arima(train_series, test_values):
    model = SARIMAX(
        train_series,
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test_values))
    return forecast, fitted


def evaluate_predictions(actual, pred):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred)
    r2 = 1.0 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    return mae, rmse, mape, r2


def make_result(model_name, actual, pred):
    mae, rmse, mape, r2 = evaluate_predictions(actual, pred)
    residual = actual - pred
    std = np.std(residual)
    lower = pred - 1.96 * std
    upper = pred + 1.96 * std
    return ForecastResult(model_name, mae, rmse, mape, r2, pred, actual, lower, upper)


def plot_comparison(results, save_path):
    configure_korean_font()
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 2])
    ax1 = axes[0]
    for result in results:
        actual = result.actual[:200]
        pred = result.pred[:200]
        ax1.plot(actual, label=f'{result.model_name} actual', alpha=0.6, linewidth=2)
        ax1.plot(pred, label=f'{result.model_name} pred', linewidth=2)
    ax1.set_title('Forecast Comparison (first 200 test points)')
    ax1.set_xlabel('Time index')
    ax1.set_ylabel('Power demand (MWh)')
    ax1.legend(loc='best', fontsize=10)

    ax2 = axes[1]
    metrics = pd.DataFrame([
        {'Model': r.model_name, 'MAE': r.mae, 'RMSE': r.rmse, 'MAPE': r.mape, 'R2': r.r2}
        for r in results
    ]).sort_values('RMSE')
    sns.barplot(data=metrics.melt(id_vars=['Model'], value_vars=['MAE', 'RMSE', 'MAPE']), x='Model', y='value', hue='variable', ax=ax2)
    ax2.set_title('Model Performance Comparison')
    ax2.set_ylabel('Error metric')
    ax2.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_interval_example(result, save_path):
    configure_korean_font()
    sns.set_theme(style='whitegrid')
    idx = min(200, len(result.actual))
    x = np.arange(idx)
    plt.figure(figsize=(18, 6))
    plt.plot(x, result.actual[:idx], label='Actual', linewidth=2)
    plt.plot(x, result.pred[:idx], label='Prediction', linewidth=2)
    plt.fill_between(x, result.lower[:idx], result.upper[:idx], alpha=0.25, label='95% interval')
    plt.title(f'{result.model_name} 95% Prediction Interval')
    plt.xlabel('Time index')
    plt.ylabel('Power demand (MWh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    train_df, val_df, test_df = split_by_time(df)

    train_X, train_y = make_sequences(train_df)
    val_X, val_y = make_sequences(val_df)
    test_X, test_y = make_sequences(test_df)

    test_actual = test_df[TARGET_COL].iloc[LOOKBACK:].astype(float).to_numpy()
    if len(test_actual) != len(test_y):
        raise ValueError(f'Test alignment mismatch: actual={len(test_actual)}, sequence={len(test_y)}')

    X_train_s, X_val_s, X_test_s, _ = scale_inputs(train_X, val_X, test_X)

    model_lstm = LSTMForecaster(input_dim=X_train_s.shape[-1], hidden_dim=64, num_layers=2)
    model_lstm = fit_deep_model(model_lstm, X_train_s, train_y, X_val_s, val_y, epochs=12, batch_size=64, lr=1e-3)
    pred_lstm = predict_deep_model(model_lstm, X_test_s)

    model_transformer = TransformerForecaster(input_dim=X_train_s.shape[-1], d_model=64, nhead=4, num_layers=2)
    model_transformer = fit_deep_model(model_transformer, X_train_s, train_y, X_val_s, val_y, epochs=12, batch_size=64, lr=1e-3)
    pred_transformer = predict_deep_model(model_transformer, X_test_s)

    train_series = train_df[TARGET_COL].astype(float)
    pred_arima, _ = train_arima(train_series, test_actual)
    pred_arima = np.asarray(pred_arima, dtype=np.float32)

    results = [
        make_result('ARIMA', test_actual, pred_arima),
        make_result('LSTM', test_actual, pred_lstm),
        make_result('Transformer', test_actual, pred_transformer),
    ]

    metrics_df = pd.DataFrame([
        {
            'model': r.model_name,
            'MAE': r.mae,
            'RMSE': r.rmse,
            'MAPE': r.mape,
            'R2': r.r2,
        }
        for r in results
    ])
    metrics_df.to_csv('forecast_metrics.csv', index=False)

    plot_comparison(results, 'forecast_comparison.png')
    for result in results:
        plot_interval_example(result, f'{result.model_name.lower()}_interval.png')

    print('\n=== Forecast Model Comparison ===')
    print(metrics_df.to_string(index=False))
    print('\nSaved files:')
    print('- forecast_metrics.csv')
    print('- forecast_comparison.png')
    print('- arima_interval.png')
    print('- lstm_interval.png')
    print('- transformer_interval.png')


if __name__ == '__main__':
    main()
