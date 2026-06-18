
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score

n_past = 14
threshold = 50
LGB_COLOR = '#657af0'
CNN_COLOR = '#f57b52'

# %%
# 여수 데이터
df1 = pd.read_csv('C:/Users/Urangoo/OneDrive/Desktop/yeosu3/yeosu23.csv', encoding='utf-8-sig')
df2 = pd.read_csv('C:/Users/Urangoo/OneDrive/Desktop/yeosu3/yeosu24.csv', encoding='utf-8-sig')
df3 = pd.read_csv('C:/Users/Urangoo/OneDrive/Desktop/yeosu3/yeosu25.csv', encoding='utf-8-sig')
df = pd.concat([df1, df2, df3], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
df['날짜'] = df['일시'].dt.date
df['누적강수량(mm)'] = pd.to_numeric(df['누적강수량(mm)'], errors='coerce')
daily_df = df.groupby('날짜')['누적강수량(mm)'].max().reset_index().dropna()

scaler= MinMaxScaler()
scaled_data = scaler.fit_transform(daily_df[['누적강수량(mm)']])
dates = daily_df['날짜'].astype(str).values

X, y = [], []
for i in range(n_past, len(scaled_data)):
    X.append(scaled_data[i-n_past:i].flatten())
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)
dates = dates[n_past:]
step  = max(1, len(dates) // 25)

# %%
# LightGBM
feature_names = [f'lag_{i+1}' for i in range(n_past)]
X_df = pd.DataFrame(X, columns=feature_names)

lgb_model = lgb.LGBMRegressor(n_estimators=2000, 
                              learning_rate=0.05, 
                              random_state=42, 
                              verbose=-1)
lgb_model.fit(X_df, y)

lgb_pred = scaler.inverse_transform(lgb_model.predict(X_df).reshape(-1, 1))
real = scaler.inverse_transform(y.reshape(-1, 1))

lgb_mse  = mean_squared_error(real, lgb_pred)
lgb_rmse = np.sqrt(lgb_mse)
lgb_mae = float(np.mean(np.abs(real - lgb_pred)))
lgb_r2 = float(1 - np.sum((real - lgb_pred)**2) / np.sum((real - np.mean(real))**2))
rb = (real >= threshold).astype(int)
pb = (lgb_pred >= threshold).astype(int)

print("LightGBM Yeosu")
print(f"MSE: {round(lgb_mse,3)}  RMSE: {round(lgb_rmse,3)}")
print(f"Precision: {round(precision_score(rb,pb,zero_division=0),3)}  "
      f"Recall: {round(recall_score(rb,pb,zero_division=0),3)}  "
      f"F1: {round(f1_score(rb,pb,zero_division=0),3)}")

lgb_result = {'MSE': round(lgb_mse,4), 'RMSE': round(lgb_rmse,4),
              'MAE': round(lgb_mae,4),  'R2': round(lgb_r2,4), 'NSE': round(lgb_r2,4)}

plt.figure(figsize=(18, 6))
plt.plot(dates, lgb_pred, color='tab:orange', lw=1.0, label='LightGBM Prediction', zorder=1, alpha=0.9)
plt.plot(dates, real,     color='tab:blue',   lw=1.5, label='Real Rainfall',        zorder=2, alpha=0.8)
plt.xticks(np.arange(0, len(dates), step), dates[::step], rotation=45)
plt.xlabel('Date'); plt.ylabel('Rainfall (mm)')
plt.title('LightGBM Rainfall Prediction [Yeosu]')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()


# %%
# 1D-CNN+GRU
scaler= MinMaxScaler()
scaled_data = scaler.fit_transform(daily_df[['누적강수량(mm)']])

X, y = [], []
for i in range(n_past, len(scaled_data)):
    X.append(scaled_data[i-n_past:i].flatten())
    y.append(scaled_data[i, 0])
X = np.array(X)
y = np.array(y)
X_seq = X.reshape(X.shape[0], n_past, 1)

tf.keras.backend.clear_session()
inp = Input(shape=(n_past, 1))
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
x = BatchNormalization()(x)
x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)
x = Bidirectional(GRU(64, return_sequences=True))(x)   # return_sequences=True
x = Dropout(0.2)(x)
x = GRU(32, return_sequences=False)(x)                 # нэмэлт давхарга
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

model.fit(
    X_seq, y,
    epochs=1000,
    batch_size=8,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=50, 
                      restore_best_weights=True, 
                      verbose=0),
        ReduceLROnPlateau(patience=20, 
                          factor=0.5, 
                          min_lr=1e-6, 
                          verbose=0),
    ],
    verbose=1,
)

cnn_pred = scaler.inverse_transform(model.predict(X_seq, verbose=0))
real     = scaler.inverse_transform(y.reshape(-1, 1))

cnn_mse = mean_squared_error(real, cnn_pred)
cnn_rmse = np.sqrt(cnn_mse)
cnn_mae = float(np.mean(np.abs(real - cnn_pred)))
cnn_r2 = float(1 - np.sum((real - cnn_pred)**2) / np.sum((real - np.mean(real))**2))
rb = (real >= threshold).astype(int)
pb = (cnn_pred >= threshold).astype(int)

print(f"\npred max: {cnn_pred.max():.2f}  real max: {real.max():.2f}")
print("1D-CNN+GRU Yeosu")
print(f"MSE: {round(cnn_mse,3)}  RMSE: {round(cnn_rmse,3)}")
print(f"Precision: {round(precision_score(rb,pb,zero_division=0),3)}  "
      f"Recall: {round(recall_score(rb,pb,zero_division=0),3)}  "
      f"F1: {round(f1_score(rb,pb,zero_division=0),3)}")

cnn_result = {'MSE': round(cnn_mse,4), 'RMSE': round(cnn_rmse,4),
              'MAE': round(cnn_mae,4),  'R2': round(cnn_r2,4), 'NSE': round(cnn_r2,4)}

plt.figure(figsize=(18, 6))
plt.plot(dates, cnn_pred, color='tab:orange', lw=1.0, label='1D-CNN+GRU Prediction', zorder=1, alpha=0.9)
plt.plot(dates, real,     color='tab:blue',   lw=1.5, label='Real Rainfall',           zorder=2, alpha=0.8)
plt.xticks(np.arange(0, len(dates), step), dates[::step], rotation=45)
plt.xlabel('Date'); plt.ylabel('Rainfall (mm)')
plt.title('1D-CNN+GRU Rainfall Prediction [Yeosu]')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()


# %%
# 최종 비교 bar chart
summary = pd.DataFrame([lgb_result, cnn_result],
                       index=['LightGBM [Yeosu]', 'CNN+GRU [Yeosu]'])
metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'NSE']
colors = [LGB_COLOR, CNN_COLOR]

fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
for ax, m in zip(axes, metrics):
    vals = [summary.loc[mdl, m] for mdl in summary.index]
    bars = ax.bar(range(len(summary)), vals, color=colors, width=0.6, edgecolor='white')
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=15, ha='right', fontsize=7)
    ax.set_title(m, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
fig.legend(handles=[
    Patch(facecolor=LGB_COLOR, label='LightGBM'),
    Patch(facecolor=CNN_COLOR, label='CNN+GRU'),
], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=10)
fig.suptitle('모델 성능 비교 [Yeosu]', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n완료!")
# %%
