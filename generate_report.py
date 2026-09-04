from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)


ROOT = Path(__file__).parent
PDF_PATH = ROOT / 'energy_demand_forecast_report.pdf'
FONT_PATH = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/7a0b5c0f3c1d41c4c52a33343496c9c65ad52c50.asset/AssetData/NanumGothic.ttc'
pdfmetrics.registerFont(TTFont('AppleSDGothicNeo', FONT_PATH, subfontIndex=0))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='KoreanBody', parent=styles['BodyText'], fontName='AppleSDGothicNeo', fontSize=9.5, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name='KoreanTitle', parent=styles['Title'], fontName='AppleSDGothicNeo', fontSize=22, leading=29, alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name='KoreanHeading', parent=styles['Heading2'], fontName='AppleSDGothicNeo', fontSize=14, leading=20, spaceBefore=12, spaceAfter=8))
styles.add(ParagraphStyle(name='KoreanSmall', parent=styles['BodyText'], fontName='AppleSDGothicNeo', fontSize=8, leading=11))


def paragraph(text, style='KoreanBody'):
    return Paragraph(text, styles[style])


def image(path, width=17 * cm):
    img = Image(str(ROOT / path))
    ratio = width / img.imageWidth
    img.drawWidth = width
    img.drawHeight = img.imageHeight * ratio
    return img


def make_table(metrics):
    headers = ['모델', 'MAE', 'RMSE', 'MAPE (%)', '실제 표준편차', '예측 표준편차', '표준편차 비율', '95% 포함률 (%)']
    rows = [headers]
    for _, row in metrics.iterrows():
        rows.append([
            row['model'], f"{row['MAE']:,.2f}", f"{row['RMSE']:,.2f}", f"{row['MAPE']:.2f}",
            f"{row['actual_std']:,.2f}", f"{row['prediction_std']:,.2f}",
            f"{row['std_ratio']:.3f}", f"{row['interval_coverage_95']:.2f}",
        ])
    table = Table(rows, repeatRows=1, colWidths=[2.0*cm, 2.0*cm, 2.0*cm, 1.7*cm, 2.3*cm, 2.3*cm, 2.1*cm, 2.4*cm])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'AppleSDGothicNeo'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#AAB7C4')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F4F7FA')),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def main():
    metrics = pd.read_csv(ROOT / 'forecast_metrics.csv')
    predictions = pd.read_csv(ROOT / 'forecast_predictions.csv')
    story = [
        paragraph('2025년 전국 전력수요 시계열 예측 보고서', 'KoreanTitle'),
        paragraph('National Power Demand and Seoul Temperature in 2025', 'KoreanHeading'),
        paragraph('본 보고서는 한국전력거래소의 2025년 시간별 전국 전력수요량과 서울 ASOS 시간별 기상관측자료를 결합하여 전력수요를 예측한 결과를 정리한 문서입니다. ARIMA, LSTM, Transformer 세 가지 모델을 동일한 시간순 데이터와 동일한 one-step-ahead 평가 방식으로 비교했습니다.'),
        paragraph('1. 연구 목적 및 데이터', 'KoreanHeading'),
        paragraph('전력수요는 시간대, 요일, 계절, 기온과 같은 외생 변수의 영향을 동시에 받는 대표적인 다변량 시계열입니다. 본 실험의 목적은 고전 시계열 모델인 ARIMA와 딥러닝 기반 LSTM 및 Transformer의 예측 성능을 비교하고, 예측값의 변동성과 불확실성까지 함께 확인하는 것입니다.'),
        paragraph('사용한 원자료는 한국전력거래소 전국 전력수요량 파일과 서울 ASOS 관측 파일입니다. 전력 파일의 1시부터 24시까지 열을 시간별 행으로 변환하고, ASOS의 일시 열과 정확히 일치시키어 결합했습니다. 따라서 서울 기온은 전국 수요를 설명하는 지역 기상 변수로 사용됩니다.'),
        paragraph('2. 전처리 및 데이터 분할', 'KoreanHeading'),
        paragraph('전체 2025년 자료는 8,760시간이며 무작위 셔플 없이 시간순으로 학습 6,132시간(70%), 검증 1,314시간(15%), 평가 1,314시간(15%)으로 분할했습니다. 시간 정보에서는 시간대·요일·월의 주기성을 sin/cos 특성으로 만들었고, 최근 24시간의 전력수요와 날씨 변수를 입력으로 사용했습니다.'),
        paragraph('ASOS 결측값은 변수의 의미를 고려해 처리했습니다. 강수량·일조시간·일사량의 빈칸은 관측되지 않은 값으로 보아 0으로 채웠고, 기온·풍속·습도는 앞뒤 관측값을 이용해 선형 보간했습니다. 입력 스케일러와 타깃 스케일러는 학습 데이터에만 적합하여 검증·평가 데이터의 정보가 전처리에 유입되지 않도록 했습니다.'),
        paragraph('3. 공정한 모델 평가 방식', 'KoreanHeading'),
        paragraph('세 모델 모두 매 시점마다 직전까지의 실제 전력수요를 보고 다음 1시간을 예측하는 walk-forward 방식으로 맞췄습니다. LSTM과 Transformer는 학습된 모델에 최근 24시간 창을 입력하고, ARIMA도 한 시간 예측 후 해당 시점의 실제값을 모형에 추가한 뒤 다음 시점을 예측합니다. 이 방식은 ARIMA만 수개월을 한 번에 예측하는 불공정한 조건을 제거합니다.'),
        paragraph('MAE는 평균 절대오차, RMSE는 큰 오차에 더 큰 가중치를 주는 제곱근 평균제곱오차, MAPE는 실제값 대비 상대 오차입니다. 세 지표는 모두 평가 구간에서 계산했습니다.'),
        paragraph('4. 성능 비교 결과', 'KoreanHeading'),
        make_table(metrics),
        Spacer(1, 10),
        paragraph('표에서 확인할 수 있듯이 최종 성능은 LSTM과 Transformer가 ARIMA보다 낮은 오차를 보였습니다. ARIMA의 one-step 예측은 평가 조건은 공정해졌지만, 현재 계절 SARIMAX 설정에서는 수렴 경고가 발생했으며 평가 성능도 낮았습니다. 따라서 ARIMA 결과는 모델 자체의 한계뿐 아니라 모형 차수와 최적화 설정의 영향을 함께 받는 결과로 해석해야 합니다.'),
        PageBreak(),
        paragraph('5. 전체 예측 비교 그래프', 'KoreanHeading'),
        image('forecast_comparison.png'),
        paragraph('위 그래프의 실선은 전국 전력수요, 주황색 점선은 서울 기온이며 기온은 오른쪽 보조축으로 읽습니다. 세 모델의 예측선은 같은 평가 시점에 표시되어 직접 비교할 수 있습니다. 아래 막대그래프는 모델별 MAE와 RMSE를 보여줍니다.'),
        PageBreak(),
        paragraph('6. ARIMA 예측구간', 'KoreanHeading'),
        image('arima_interval.png'),
        paragraph('ARIMA는 매 시간 한 단계씩 예측하고 실제값을 반영합니다. 95% 구간의 폭은 평가 정답을 사용하지 않고 검증 구간 잔차 표준편차로 산정했습니다. 이번 실행의 ARIMA 구간 포함률은 낮게 나타났으므로, 현재 ARIMA 설정의 장기 안정성과 구간 추정 성능을 개선할 필요가 있습니다.'),
        PageBreak(),
        paragraph('7. LSTM 예측구간 및 붕괴 검증', 'KoreanHeading'),
        image('lstm_interval.png'),
        paragraph('LSTM의 예측값 표준편차는 실제값 표준편차의 약 0.968배입니다. 비율이 0에 가깝지 않고 실제 변동성과 비슷하므로, 예측선이 하나의 상수로 붕괴한 것으로 볼 수 없습니다. 검증 잔차 기반 95% 구간의 평가 포함률은 표의 값을 사용합니다.'),
        PageBreak(),
        paragraph('8. Transformer 예측구간 및 붕괴 검증', 'KoreanHeading'),
        image('transformer_interval.png'),
        paragraph('Transformer의 예측값 표준편차는 실제값 표준편차의 약 0.953배입니다. 따라서 Transformer 역시 일정값 예측으로 붕괴하지 않았습니다. 다만 불확실성 구간은 잔차 정규성 등을 가정한 근사값이므로, 확률 예측의 엄밀한 보장으로 해석하지 않습니다.'),
        PageBreak(),
        paragraph('9. 결론 및 한계', 'KoreanHeading'),
        paragraph('본 프로젝트는 전국 전력수요와 서울 기상정보를 결합하고, 시간순 70:15:15 분할과 학습 데이터 전용 스케일링을 적용한 다변량 시계열 예측 실험입니다. 세 모델 모두 동일한 one-step-ahead 조건에서 평가했으며, LSTM과 Transformer는 실제 수요의 변동성을 보존하면서 예측했습니다.'),
        paragraph('현재 95% 예측구간은 검증 잔차 표준편차를 이용한 실용적 근사입니다. 향후에는 quantile loss, Monte Carlo dropout, conformal prediction을 적용해 보다 엄밀한 불확실성 추정을 할 수 있습니다. ARIMA는 모형 차수 탐색, 수렴 설정 조정, 외생 변수 포함 여부를 추가로 검토하면 개선될 수 있습니다.'),
        paragraph('10. 생성 파일', 'KoreanHeading'),
        paragraph('energy_demand_forecast.py: 전체 전처리·학습·평가 코드<br/>forecast_metrics.csv: 모델별 평가 지표와 표준편차·구간 포함률<br/>forecast_predictions.csv: 평가 시점별 실제값, 예측값, 하한·상한<br/>forecast_comparison.png 및 *_interval.png: 결과 시각화'),
    ]
    document = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    document.build(story)
    print(PDF_PATH)


if __name__ == '__main__':
    main()
