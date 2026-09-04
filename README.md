# 전력수요 시계열 예측 프로젝트

이 프로젝트는 시간별 전력수요와 날씨 변수를 활용해 전력 수요를 예측하는 시계열 분석 파이프라인을 구현합니다. 핵심 목표는 다음과 같습니다.

- 고전 시계열 모델과 딥러닝 모델의 성능 비교
- 다변량 입력 기반 예측
- 예측 구간(uncertainty) 시각화
- 실제 전력수요 데이터에 대한 실험 구조 정리

## 데이터

- 데이터 소스: 한국전력거래소 전국 전력수요량 및 7개 지역 ASOS 관측자료
- 파일: `한국전력거래소_시간별 전국 전력수요량_20251231.csv`, `OBS_ASOS_TIM_20260904122533.csv`
- 통합 파일: `national_power_regional_weather_2025.csv`
- 주요 컬럼:
  - `datetime`
  - `power_demand_mwh`
  - `temperature_c`
  - `rainfall_mm`
  - `wind_speed_ms`
  - `humidity_pct`
  - `sunshine_hr`
  - `solar_radiation_mj_m2`
  - 시간/요일/월 cyclic feature

## 사용 모델

1. ARIMA / SARIMAX
   - 고전적인 시계열 기반 베이스라인
   - 계절성(24시간) 반영

2. LSTM
   - 최근 24시간의 수요와 기상 정보를 입력으로 사용
   - 장기 의존성 학습

3. Transformer
   - self-attention 기반 시계열 모델
   - 시간적 패턴과 외생 변수의 관계 학습

## 실행 방법

```bash
python energy_demand_forecast.py
python generate_report.py
python create_presentation.py
```

실행 시 다음 결과 파일이 생성됩니다.

- `national_power_regional_weather_2025.csv`
- `forecast_metrics.csv`
- `forecast_predictions.csv`
- `forecast_comparison.png`
- `arima_interval.png`
- `lstm_interval.png`
- `transformer_interval.png`
- `energy_demand_forecast_report.pdf`
- `energy_demand_forecast_presentation.pptx`

발표자료는 연구 목적, 7개 지역 데이터, 전처리, 공정한 one-step 평가, 모델 비교, 예측구간, 결과 해석을 10장의 슬라이드로 정리한 파일입니다.

## 결과 이미지 설명

### 1. 모델 예측 및 성능 비교

![Forecast comparison](forecast_comparison.png)

`forecast_comparison.png`는 테스트 구간의 처음 200개 시점을 대상으로 전국 전력수요와 세 모델의 예측값을 비교한 이미지입니다. 아래쪽 막대그래프에서는 MAE와 RMSE를 모델별로 비교합니다. 모델 입력에는 서울·인천·대전·대구·광주·부산·제주의 기상 변수가 모두 포함됩니다. 세 모델 모두 매 시간 직전까지의 실제 전력수요를 사용해 다음 한 시간을 예측하는 동일한 walk-forward 방식으로 평가합니다.

`national_power_regional_weather_2025.csv`는 전력수요와 7개 지역 날씨를 하나의 시간 기준으로 합친 파일입니다. 원자료를 여러 개 열지 않고도 통합 데이터만 확인하거나 후속 분석에 사용할 수 있습니다.

최종 실행에서는 모든 모델이 동일한 시간순 학습·검증·평가 구간을 사용했습니다. 총 8,760시간을 학습 6,132시간(70%), 검증 1,314시간(15%), 평가 1,314시간(15%)으로 나누었습니다.

### 2. ARIMA 예측 구간

![ARIMA prediction interval](arima_interval.png)

`arima_interval.png`는 ARIMA의 실제값, 예측값, 95% 예측 구간을 나타냅니다. ARIMA도 매 시간 한 단계만 예측한 뒤 해당 시점의 실제 전력수요를 모형에 추가하고 다음 시점을 예측합니다. 구간 폭은 평가 정답이 아니라 검증 구간의 ARIMA 잔차 표준편차로 계산합니다.

### 3. LSTM 예측 구간

![LSTM prediction interval](lstm_interval.png)

`lstm_interval.png`는 LSTM의 예측 결과와 95% 예측 구간을 보여줍니다. LSTM 예측값의 표준편차는 실제값 표준편차의 약 0.968배로, 예측선이 하나의 일정한 값으로 붕괴하지 않았습니다. 검증 잔차로 계산한 구간의 평가 포함률은 `forecast_metrics.csv`에서 확인할 수 있습니다.

### 4. Transformer 예측 구간

![Transformer prediction interval](transformer_interval.png)

`transformer_interval.png`는 Transformer의 실제값과 예측값, 95% 예측 구간을 비교한 이미지입니다. Transformer 예측값의 표준편차는 실제값 표준편차의 약 0.953배로, 일정값 붕괴가 아님을 확인했습니다. 다만 예측구간은 검증 잔차 기반의 근사값이므로, 실제 운영용 확률 예측구간으로 해석해서는 안 됩니다.

### 결과 해석 시 주의점

현재 95% 예측 구간은 각 모델의 검증 구간 잔차 표준편차를 이용한 근사값이며, 평가 정답을 사용하지 않아 데이터 누출을 피했습니다. 7개 지역 ASOS의 강수량·일조·일사 결측치는 0으로 처리하고, 기온·풍속·습도만 지역별로 보간했습니다. `forecast_metrics.csv`에는 MAE, RMSE, MAPE와 함께 실제값 표준편차, 예측값 표준편차, 표준편차 비율, 95% 구간 포함률이 저장됩니다. 향후에는 quantile loss 또는 Monte Carlo dropout을 이용한 불확실성 추정을 적용할 수 있습니다.

## 해석 포인트

- ARIMA는 계절성과 추세를 잘 잡는 고전적인 기준선 역할을 합니다.
- LSTM과 Transformer는 다변량 입력과 비선형 상호작용을 학습해 예측 정확도를 높일 수 있습니다.
- 95% 예측 구간을 함께 제시하면 실제 운영에서 더 유용한 의사결정 자료가 됩니다.

## 과제용 서론 예시

> 본 연구에서는 시간별 전국 전력 수요 예측을 위해 2025년 전국 전력수요와 7개 지역 ASOS 기상자료를 결합하였다. 전력 수요는 시간대, 기온, 강수량, 습도, 일사량, 요일 특성에 의해 크게 영향을 받는다. 따라서 전력수요는 단순한 추세 모델보다 다변량 시계열 모델이 적합하며, LSTM 및 Transformer 기반 딥러닝 모델과 ARIMA 베이스라인을 비교하여 적합성을 검증하였다.

## 참고

- UCI Electricity와 같은 외부 전력 데이터셋은 추가 비교 실험으로 확장 가능하며, 본 프로젝트는 한전 기반의 실제 운영 데이터로 풀스택 시계열 모델 비교를 보여준다.
