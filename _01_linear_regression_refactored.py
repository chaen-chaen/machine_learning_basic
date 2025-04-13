# House Rent 금액 예측 모델 만들기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 데이터 불러오기
rent_df = pd.read_csv("https://raw.githubusercontent.com/DSNote/fastcampus/refs/heads/main/rent.csv")

# 결측치 처리하기
rent_df_dropna = rent_df.dropna(subset=['Size', 'BHK'])

# 아웃라이어 제거하기
outliers_index = rent_df['Rent'].sort_values(ascending=False).head(4).index
rent_df_outlier = rent_df_dropna.drop(index = outliers_index)

# 카테고리 변수 처리하기
rent_df_new = pd.get_dummies(rent_df_outlier, columns = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'], drop_first=True)
rent_df_new = rent_df_new.drop(['Posted On', 'Floor', 'Area Locality'], axis=1)

# 학습 데이터셋 / 검증 데이터셋 나누기
X = rent_df_new.drop('Rent', axis=1)
y = rent_df_new['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 평가 함수
def evaluate_model(model, X_train, y_train, X_test, y_test, log=False):
    if log:
        y_train = np.log(y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if log:
        y_pred = np.exp(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2

mae, mse, rmse, r2 = evaluate_model(LinearRegression(), X_train, y_train, X_test, y_test)
mae_log, mse_log, rmse_log, r2_log = evaluate_model(LinearRegression(), X_train, y_train, X_test, y_test, log=True)

print("[원본]")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
print("[로그 변환]")
print(f"MAE: {mae_log:.2f}, MSE: {mse_log:.2f}, RMSE: {rmse_log:.2f}, R²: {r2_log:.4f}")
