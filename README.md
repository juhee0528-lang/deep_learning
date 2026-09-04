# 전력수요 시계열 예측 프로젝트

이 프로젝트는 시간별 전력수요와 날씨 변수를 활용해 전력 수요를 예측하는 시계열 분석 파이프라인을 구현합니다. 핵심 목표는 다음과 같습니다.

- 고전 시계열 모델과 딥러닝 모델의 성능 비교
- 다변량 입력 기반 예측
- 예측 구간(uncertainty) 시각화
- 실제 전력수요 데이터에 대한 실험 구조 정리

## 데이터

- 데이터 소스: 한전/기상연계 전력수요 데이터
- 파일: `power_weather_seoul_2025.csv`
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
   - 고전적인 시계열 기반 baseline
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
```

실행 시 다음 결과 파일이 생성됩니다.

- `forecast_metrics.csv`
- `forecast_comparison.png`
- `arima_interval.png`
- `lstm_interval.png`
- `transformer_interval.png`

## 해석 포인트

- ARIMA는 계절성과 추세를 잘 잡는 고전적인 기준선 역할을 합니다.
- LSTM과 Transformer는 다변량 입력과 비선형 상호작용을 학습해 예측 정확도를 높일 수 있습니다.
- 95% 예측 구간을 함께 제시하면 실제 운영에서 더 유용한 의사결정 자료가 됩니다.

## 과제용 서론 예시

> 본 연구에서는 시간별 전력 수요 예측을 위해 2025년 서울 기상-전력 결합 데이터를 활용하였다. 전력 수요는 시간대, 기온, 강수량, 습도, 일사량, 요일 특성에 의해 크게 영향을 받는다. 따라서 전력수요는 단순한 추세 모델보다 다변량 시계열 모델이 적합하며, LSTM 및 Transformer 기반 딥러닝 모델과 ARIMA baseline을 비교하여 적합성을 검증하였다.

## 참고

- UCI Electricity와 같은 외부 전력 데이터셋은 추가 비교 실험으로 확장 가능하며, 본 프로젝트는 한전 기반의 실제 운영 데이터로 풀스택 시계열 모델 비교를 보여준다.
