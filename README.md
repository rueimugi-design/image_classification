# 수위 감지 시스템 — LightGBM & 1D-CNN+GRU

## 프로젝트 소개
여수·통영 남해안 지역의 기상 관측 데이터를 활용하여  
누적강수량을 예측하는 머신러닝 및 딥러닝 모델을 개발하였다.

---

## 팀원
| 학번 | 이름 |
|------|------|
| 22619008 | 김진우 |
| 24619023 | 윤서연 |
| 24619045 | 오랑거 |

---

## 사용 모델
- **LightGBM** — 머신러닝 기반 그래디언트 부스팅 모델
- **1D-CNN + GRU** — 딥러닝 기반 시계열 예측 모델

---

## 데이터
- 출처 : [기상자료개방포털 (기상청)](https://data.kma.go.kr)
- 지역 : 여수, 통영 관측소
- 기간 : 2023 ~ 2025년
- 전처리 : 30분 간격 원시 데이터를 일별 최대값으로 집계

---

## 입력 피처
| 피처 | 설명 |
|------|------|
| 누적강수량 | 수위의 직접적인 원인 |
| 풍향 | 비 이동 방향 |
| 풍속 | 강도 영향 |
| 현지기압 / 해면기압 | 급격한 기압 변화 |
| 습도 | 공기 내 수분량 |

---

## 모델 구조

### LightGBM
- n_estimators : 500
- learning_rate : 0.05
- max_depth : 6
- 입력 : 과거 14일 누적강수량 (lag feature)

### 1D-CNN + GRU
- Conv1D(64) × 2 + BatchNormalization
- MaxPooling1D → Dropout(0.2)
- Bidirectional GRU(64) → Dropout(0.2)
- Dense(32) → Dense(1, activation='sigmoid')
- Loss : MSE / Epochs : 1000 (EarlyStopping 적용)

---

## 평가 지표
- MSE, RMSE, MAE, R², NSE
- Precision, Recall, F1-score (Threshold 50mm)

---

## 주요 결과 [Yeosu]

| 모델 | MSE | RMSE | R² |
|------|-----|------|----|
| LightGBM | 39.07 | 6.25 | 0.9676 |
| 1D-CNN+GRU | 344.06 | 18.55 | 0.7145 |

> LightGBM이 전반적으로 우수한 성능을 보임

---

## 실행 방법

```bash
pip install lightgbm tensorflow scikit-learn pandas matplotlib
python yeosu_final.py
```

---

## 출처
- 데이터 : [기상자료개방포털](https://data.kma.go.kr)
- 베이스라인 참고 : [Rainfall-Prediction-LSTM (GitHub)](https://github.com/Suchetha21/Rainfall-Prediction-LSTM)
