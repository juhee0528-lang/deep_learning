# 전국 전력수요 1시간 앞 예측

2025년 전력거래소 전국 전력수요와 7개 도시 ASOS 자료를 사용한 딥러닝 기말 프로젝트입니다.
원본에서 전처리, EDA, 모델 학습, 평가까지 다시 실행할 수 있습니다.

## 먼저 볼 파일

- `REPORT.md`: 실험 설계, 분석, 결과 해석과 한계
- `EDA.ipynb`: 실행 결과가 포함된 EDA 노트북
- `presentation/전력수요_발표자료.pptx`: 편집 가능한 발표자료
- `presentation/전력수요_발표자료.pdf`: 과제 제출용 발표 PDF
- `presentation/발표노트.md`: 20~25분 발표 흐름과 예상 질문
- `results/metrics.csv`: 최종 테스트 성능표
- `results/test_predictions.csv`: 시간별 실제값, 예측값과 보정 구간

## 실행

Python 3.13에서 실행했습니다. 새 가상환경을 만든 뒤 아래 순서로 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/pipeline.py --prepare-only
python src/pipeline.py
python src/verify.py
```

Windows에서는 가상환경 활성화 명령을 `.venv\Scripts\activate`로 바꿉니다.
학습은 CPU에서도 실행됩니다. 실행시간은 장비에 따라 달라집니다.
`pipeline.py`를 다시 실행하면 이 프로젝트의 전처리 파일·학습 모델·수치 결과·그래프를 갱신합니다.
발표자료는 해당 실험 결과의 스냅샷이므로 재학습 후에는 수치와 결론을 함께 갱신해야 합니다.
원본 CSV는 덮어쓰지 않습니다. 결과의 정확한 실행 환경과 설정은 `results/experiment_config.json`에 기록했습니다.

## 예측 조건

- 예측 시점에 t-1까지의 실제 전력수요·기상 관측을 알고 있다고 가정합니다.
- 최근 24시간의 입력으로 t의 수요 1개를 예측합니다. 하루 24시간을 한 번에 예측하는 실험이 아닙니다.
- 관측 데이터의 실제 제공 지연은 모형에 반영하지 않았습니다.
- 딥러닝은 직전 수요에 변화량을 더하는 잔차 구조입니다.
- 학습 6,132시간 / 검증 876시간 / 구간 보정 438시간 / 테스트 1,314시간을 시간순으로 분리했습니다.
- 표준화는 학습 구간만, epoch·seed 선택은 검증 구간만 사용합니다.
- 구간 보정과 테스트 자료로 모델의 가중치를 학습하지 않습니다.
- 강수·일조·일사 관측 행의 빈칸은 0으로 가정하고 결측 표시를 보존합니다. 관측 행 전체 누락과 연속형 변수의 결측은 과거 값으로 채웁니다.
- 이 가정은 모든 빈칸이 무현상을 의미한다는 증명이 아닙니다. QC 정보 부재를 한계로 남깁니다.

## 폴더

- `data/raw`: 제공받은 원본 2개, SHA-256은 data_audit.json 참조
- `data/processed`: 통합 데이터와 시간순 분할 CSV
- `src`: 전처리·EDA·학습·검증 코드
- `models`: 학습 가중치, 표준화 계수와 SARIMA 모델
- `results`: 결측치 감사, EDA 통계, 학습 이력, seed 비교와 평가표
- `figures`: 데이터에서 생성한 분석 그래프
- `presentation`: 발표자료와 발표노트

## 제출 전 사용자가 확인할 항목

안내문 기준 발표일은 2026년 9월 22일, 발표 20~25분과 질의응답 5~10분입니다.
주제 등록은 9월 8일 오후 10시까지 지정 Slack 채널에 조·주제·데이터 링크를 올립니다.
현재 프로젝트 주제가 다른 조와 겹치는지는 Slack에서 확인해야 합니다.
표지의 조·발표자 정보는 확정 후 넣고, ZIP 이름 앞에 실제 조 번호를 붙입니다.
발표 후 피드백을 반영한 전처리 데이터, 코드, 결과 CSV, 발표 PDF를 ZIP으로 제출합니다.
수신 주소는 `slcf.snu@gmail.com`입니다. 메일·Slack 전송은 수행하지 않았습니다.
상호평가와 조원 내 평가는 별도로 수행해야 합니다. 발표 후 최종 제출 시각은 안내문에 명시되어 있지 않습니다.

## 데이터와 방법론 출처

- [전력거래소 시간별 전국 전력수요량](https://www.data.go.kr/tcs/dss/selectFileDataDetailView.do?publicDataPk=15065266): MWh, 육지·제주 포함 발전단 수요의 잠정자료
- [기상청 ASOS](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36): 서울·인천·대전·대구·광주·부산·제주
- [Hochreiter & Schmidhuber, Long Short-Term Memory, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Vaswani et al., Attention Is All You Need, 2017](https://arxiv.org/abs/1706.03762)
- [statsmodels SARIMAX 문서](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

과거 버전에서 같은 테스트 기간의 결과를 이미 확인한 뒤 실험을 보완했습니다.
따라서 새로운 미관측 기간에 대한 확증 실험으로 주장하지 않으며, 별도 연도에서의 추가 검증이 필요합니다.
