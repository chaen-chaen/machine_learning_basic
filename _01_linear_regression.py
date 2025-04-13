# House Rent 금액 예측 모델 만들기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 불러오기
rent_df = pd.read_csv("https://raw.githubusercontent.com/DSNote/fastcampus/refs/heads/main/rent.csv")

# 데이터 파악하기
# print(rent_df.info())
# print(rent_df.describe()) 
# print(rent_df.columns)
# print(rent_df['Rent'].sort_values)
# sns.displot(rent_df[['BHK']])
# sns.boxplot(y=rent_df['Size'])
# plt.show()

# 아웃라이어 제거하기
rent_df_outlier = rent_df.drop(1837)

# 결측치 처리하기
# print(rent_df.isna().sum())
# rent_df.drop(1)
# na_index = rent_df[rent_df[['Size','BHK']].isna().any(axis=1)].index
# print(na_index)
rent_df_dropna = rent_df_outlier.dropna(subset=['Size', 'BHK'])
# print(rent_df_dropna.info())

# 카테고리 변수 처리하기
# print(rent_df_dropna['Area Type'].value_counts())
# for i in ['Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']:
#   print(i, rent_df_dropna[i].nunique())
rent_df_new = pd.get_dummies(rent_df_dropna, columns = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
rent_df_new = rent_df_new.drop(['Posted On', 'Floor', 'Area Locality'], axis=1)

# 학습 데이터셋 / 검증 데이터셋 나누기
X = rent_df_new.drop('Rent', axis=1)
y = rent_df_new['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

# 평가 지표 확인하기
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
print(mae, mse)

# 로그 활용하기
y_train_log = np.log(y_train)
lr_log = LinearRegression()
lr_log.fit(X_train, y_train_log)
pred_log = lr_log.predict(X_test)
pred_exp = np.exp(pred_log)

mae_log = mean_absolute_error(y_test, pred_exp)
mse_log = mean_squared_error(y_test, pred_exp)
print(mae_log, mse_log)