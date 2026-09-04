from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).parent
PDF_PATH = ROOT / 'energy_demand_forecast_report.pdf'
FONT_PATH = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/7a0b5c0f3c1d41c4c52a33343496c9c65ad52c50.asset/AssetData/NanumGothic.ttc'
pdfmetrics.registerFont(TTFont('KoreanFont', FONT_PATH, subfontIndex=0))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyK', parent=styles['BodyText'], fontName='KoreanFont', fontSize=8.5, leading=12, spaceAfter=4))
styles.add(ParagraphStyle(name='TitleK', parent=styles['Title'], fontName='KoreanFont', fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle(name='HeadK', parent=styles['Heading2'], fontName='KoreanFont', fontSize=12, leading=15, spaceBefore=6, spaceAfter=4))


def p(text, style='BodyK'):
    return Paragraph(text, styles[style])


def figure(name, width=16.8 * cm):
    item = Image(str(ROOT / name))
    scale = width / item.imageWidth
    item.drawWidth = width
    item.drawHeight = item.imageHeight * scale
    return item


def metric_table(frame):
    headings = ['모델', 'MAE', 'RMSE', 'MAPE', '예측/실제 표준편차', '95% 포함률']
    rows = [headings]
    for _, row in frame.iterrows():
        rows.append([row['model'], f"{row['MAE']:,.2f}", f"{row['RMSE']:,.2f}", f"{row['MAPE']:.2f}%",
                     f"{row['std_ratio']:.3f}", f"{row['interval_coverage_95']:.2f}%"])
    table = Table(rows, colWidths=[3*cm, 3*cm, 3*cm, 2.4*cm, 4*cm, 3*cm], repeatRows=1)
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'KoreanFont'), ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#24527A')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), .3, colors.HexColor('#AAB7C4')), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F6F8')),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return table


def main():
    metrics = pd.read_csv(ROOT / 'forecast_metrics.csv')
    predictions = pd.read_csv(ROOT / 'forecast_predictions.csv')
    regional_text = '서울, 인천, 대전, 대구, 광주, 부산, 제주'
    story = [
        p('2025년 전국 전력수요 시계열 예측 보고서', 'TitleK'),
        p('National Power Demand and Regional Weather in 2025', 'HeadK'),
        p('이 보고서는 한국전력거래소의 2025년 시간별 전국 전력수요와 7개 지역 ASOS 관측자료를 결합해 ARIMA, LSTM, Transformer를 비교한 결과입니다. 목표는 시간대와 지역별 날씨를 이용해 다음 1시간의 전국 수요를 예측하고, 예측 오차와 불확실성을 함께 평가하는 것입니다.'),
        p('1. 데이터와 전처리', 'HeadK'),
        p(f'사용 지역은 {regional_text}입니다. 전력 수요 데이터의 1시~24시 열을 시간별 행으로 변환하고 ASOS 일시와 정확히 일치시켜 지역별 기온, 강수량, 풍속, 습도, 일조, 일사량을 구성했습니다. 강수량·일조·일사량 결측치는 0으로 처리하고, 기온·풍속·습도 결측치는 지역별 선형 보간했습니다. 시간대·요일·월은 sin/cos 주기 특성으로 변환했습니다.'),
        p('전체 8,760시간을 셔플 없이 시간순으로 학습 6,132시간(70%), 검증 1,314시간(15%), 평가 1,314시간(15%)으로 나누었습니다. 모든 입력 스케일러와 타깃 스케일러는 학습 데이터에만 적합했습니다. 통합 전처리 데이터는 national_power_regional_weather_2025.csv에 저장했습니다.'),
        p('3. 탐색적 데이터 분석(EDA)', 'HeadK'), figure('eda_overview.png', 15.8 * cm),
        p('시간대별 평균 수요에서 반복되는 일중 패턴을 확인했고, 지역별 평균 기온 차이를 확인했습니다. 이 결과를 바탕으로 시간 주기 특성과 7개 지역 날씨 변수를 모델 입력에 포함했습니다.'),
        p('4. 동일한 평가 방식', 'HeadK'),
        p('세 모델 모두 직전까지의 실제 수요를 사용해 다음 1시간을 예측하는 walk-forward 방식입니다. LSTM과 Transformer는 최근 24시간 입력창을 사용하고, ARIMA도 1시간 예측 후 실제 관측값을 추가한 다음 시점을 예측합니다. 따라서 ARIMA만 여러 달을 한 번에 예측하는 불공정한 조건을 제거했습니다.'),
        p('5. 학습 설정과 성능 평가', 'HeadK'),
        p('LSTM과 Transformer는 최근 24시간을 입력으로 사용했습니다. 두 모델 모두 MSELoss와 Adam optimizer를 사용하고 learning rate는 0.001입니다. LSTM은 hidden size 64, 2개 층, Transformer는 d_model 64, 4개 attention head, 2개 encoder 층으로 구성했습니다. MAE, RMSE, MAPE와 표준편차 비율로 평가했습니다.'),
        metric_table(metrics), Spacer(1, 5),
        p(f'평가 예측 행 수는 {len(predictions):,}개입니다. 표준편차 비율은 예측값 표준편차를 실제값 표준편차로 나눈 값이며 0에 가까울수록 상수 예측을 의미합니다. LSTM과 Transformer의 비율은 각각 {metrics.loc[metrics.model == "LSTM", "std_ratio"].iloc[0]:.3f}, {metrics.loc[metrics.model == "Transformer", "std_ratio"].iloc[0]:.3f}으로 실제 변동성을 보존했습니다.'),
        p('6. 전체 모델 비교', 'HeadK'), figure('forecast_comparison.png'),
        p('전국 실제 전력수요와 세 모델의 예측선을 같은 평가 구간에 표시했습니다. 아래 막대그래프는 MAE와 RMSE를 비교합니다. 입력에는 7개 지역 ASOS 기상 변수가 모두 포함됩니다.'),
        p('7. ARIMA 예측 구간', 'HeadK'), figure('arima_interval.png'),
        p('ARIMA는 매 시점 실제값을 반영하는 1시간 예측입니다. 예측 구간 폭은 평가 정답이 아니라 검증 구간 잔차 표준편차로 계산했습니다. 평가 포함률은 표의 95% 포함률 열에서 확인할 수 있습니다.'),
        p('8. LSTM 예측 구간', 'HeadK'), figure('lstm_interval.png'),
        p('LSTM은 최근 24시간의 수요와 7개 지역 날씨를 사용합니다. 예측 표준편차가 실제 표준편차와 비슷하므로 일정한 값으로 붕괴하지 않았습니다.'),
        p('9. Transformer 예측 구간', 'HeadK'), figure('transformer_interval.png'),
        p('Transformer는 self-attention으로 최근 시계열의 관계를 학습합니다. 표준편차 비율과 95% 포함률을 함께 확인하면 예측 변동성과 구간의 유효성을 평가할 수 있습니다.'),
        p('10. 결과 해석과 한계', 'HeadK'),
        p('이번 실험은 전국 전력수요와 7개 주요 지역의 ASOS 날씨를 함께 사용했습니다. 최종 평가에서는 ARIMA의 RMSE가 가장 낮았고, LSTM과 Transformer도 실제 수요의 변동성을 보존했습니다. 모든 모델의 시험 조건은 동일한 one-step-ahead 방식으로 맞췄고, 예측 구간은 검증 잔차만 사용해 데이터 누출을 피했습니다. 현재 구간은 잔차 기반 근사이므로 엄밀한 확률 구간은 아닙니다. 향후 quantile loss, conformal prediction, Monte Carlo dropout, ARIMA 차수 탐색을 적용할 수 있습니다.'),
        p('생성 파일: energy_demand_forecast.py, national_power_regional_weather_2025.csv, forecast_metrics.csv, forecast_predictions.csv, forecast_comparison.png, arima_interval.png, lstm_interval.png, transformer_interval.png'),
    ]
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=1.1*cm, leftMargin=1.1*cm, topMargin=1*cm, bottomMargin=1*cm)
    doc.build(story)
    print(PDF_PATH)


if __name__ == '__main__':
    main()
