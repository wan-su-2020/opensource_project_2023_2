import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error

# 데이터셋 로드 
food_production_data = pd.read_csv(r"C:\Users\82102\Desktop\Food_Production.csv")


# 1) Groupby를 사용한 데이터 그룹화 후 간단한 통계 분석

# 둥물 사료 사용 온실 배출 여부에 따른 총 온실 배출량 평균
#이를 통해 동물 사료를 사용 했을 시 온실 배출 평균 값이 현저히 높은 것을 파악할 수 있다.

food_production_data['Animal Feed Used'] = food_production_data['Animal Feed']>0

grouped_animal_emissions = food_production_data.groupby('Animal Feed Used')['Total_emissions']

# 그룹별 최소 및 최대 배출량 계산
grouped_min_max = food_production_data.groupby('Animal Feed Used')['Total_emissions'].agg(['min', 'max'])

# 최소 및 최대 배출량에 해당하는 제품 찾기
mean_total_emissions = food_production_data['Total_emissions'].mean()
min_emissions_product = food_production_data[food_production_data['Total_emissions'] == grouped_min_max.loc[False, 'max']]['Food product'].iloc[0]
max_emissions_product = food_production_data[food_production_data['Total_emissions'] == grouped_min_max.loc[True, 'min']]['Food product'].iloc[0]
animal_feed_used_mean = food_production_data[food_production_data['Animal Feed Used'] == True]['Total_emissions'].mean()
animal_feed_not_used_mean = food_production_data[food_production_data['Animal Feed Used'] == False]['Total_emissions'].mean()


# 결과 출력
print(grouped_animal_emissions)
print(f"Total_emissions의 평균값: {mean_total_emissions}")
print(f"Animal Feed가 0인 경우 최대 배출량을 가진 제품: {min_emissions_product}, 배출량: {grouped_min_max.loc[False, 'max']}")
print(f"Animal Feed가 0인 경우 최소 배출량을 가진 제품: {min_emissions_product}, 배출량: {grouped_min_max.loc[False, 'min']}")
print(f"Animal Feed를 사용하는 경우 최소 배출량을 가진 제품: {max_emissions_product}, 배출량: {grouped_min_max.loc[True, 'min']}")
print(f"Animal Feed를 사용하는 경우 최대 배출량을 가진 제품: {max_emissions_product}, 배출량: {grouped_min_max.loc[True, 'max']}")
print(f"'Animal Feed Used'가 True인 경우의 평균 Total Emissions: {animal_feed_used_mean}")
print(f"'Animal Feed Used'가 0인 경우의 평균 Total Emissions: {animal_feed_not_used_mean}")



# 2) 데이터 시각화
# a) Aninmal Feed와 Total emissions의 관계 그래프
plt.figure(figsize=(10, 6))
sns.barplot(x='Animal Feed', y='Total_emissions', data=food_production_data)
plt.title('Average Total Emissions by Animal Feed Usage')
plt.xlabel('Animal Feed Used')
plt.ylabel('Average Total Emissions')
plt.show()

# b) 각 Food product별 Total emissions 막대 그래프
plt.figure(figsize=(12, 8))
sns.barplot(x='Total_emissions', y='Food product', data=food_production_data)
plt.title('Total Emissions by Food Product')
plt.xlabel('Total Emissions')
plt.ylabel('Food Product')
plt.show()

# c) Total emissions과 Land use change의 관계를 보여주는 산점도
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Land use change', y='Total_emissions', data=food_production_data)
plt.title('Total Emissions vs Land Use Change')
plt.xlabel('Land Use Change')
plt.ylabel('Total Emissions')
plt.show()

# 수치형 데이터만 선택
numerical_data = food_production_data.select_dtypes(include=['float64', 'int64'])

# d) 전체 데이터셋에 대한 상관관계 매트릭스 계산
correlation_matrix = numerical_data.corr()

# Heatmap 생성
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations for the Entire Dataset')
plt.show()

# 3) 머신러닝 모델 학습 및 평가
# 결측치 제거 및 필요한 열 선택
ml_data = food_production_data.dropna(subset=['Total_emissions', 'Land use change', 'Animal Feed', 'Farm','Processing','Transport','Packging','Retail','Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)'])
X = ml_data[['Land use change', 'Animal Feed', 'Farm','Processing','Transport','Packging','Retail','Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)']]
y = ml_data['Total_emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R^2 Score: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Emissions')
plt.ylabel('Predicted Total Emissions')
plt.title('Actual vs Predicted Total Emissions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
