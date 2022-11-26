import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor

train_pd = pd.read_csv("train.csv", sep=' ')
test_pd = pd.read_csv("test.csv", sep=' ')

continuous_variable = ['power', 'kilometer', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8',
                       'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']

for i in continuous_variable:
    fig = sns.distplot(np.log1p(train_pd[i]))
    fig_save = fig.get_figure()
    fig_save.savefig('{}.png'.format(i), dpi=300)
    fig_save.clear()

corr = train_pd[continuous_variable].corr()
ax = plt.subplots(figsize=(20, 16))  # 调整画布大小
fig = sns.heatmap(corr, vmax=.8, square=True, annot=True)  # 画热力图   annot=True 表示显示系数
fig_save = fig.get_figure()
fig_save.savefig('heatmap.png', dpi=300)
fig_save.clear()
# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

df = pd.concat([train_pd, test_pd], axis=0)

df = df.replace('-', np.nan)

print(df.isnull().sum())

df['fuelType'] = df['fuelType'].fillna(0).astype(np.float64)
df['gearbox'] = df['gearbox'].fillna(0).astype(np.float64)
df['bodyType'] = df['bodyType'].fillna(0).astype(np.float64)
df['model'] = df['model'].fillna(0).astype(np.float64)
df['notRepairedDamage'] = df['notRepairedDamage'].fillna(0).astype(np.float64)

df['creatDate_year'] = df['creatDate'].apply(lambda x: str(x)[:4]).astype(np.int32)
df['creatData_month'] = df['creatDate'].apply(lambda x: str(x)[4:6]).astype(np.int32)
df['creatDate_day'] = df['creatDate'].apply(lambda x: str(x)[6:8]).astype(np.int32)
df['regDate_year'] = df['regDate'].apply(lambda x: str(x)[:4]).astype(np.int32)
df['regData_month'] = df['regDate'].apply(lambda x: str(x)[4:6]).astype(np.int32)
df['regDate_day'] = df['regDate'].apply(lambda x: str(x)[6:8]).astype(np.int32)

df['power'] = df['power'].apply(lambda x: 600 if x > 600 else x)

bin = [i*10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)
bin = [i*10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)

print(df['power_bin'])

v_high_correspondence = ['v_0', 'v_3', 'v_8', 'v_12']

for i in v_high_correspondence:
    for j in v_high_correspondence:
        df[i + "+" + j] = df[i] + df[j]

for i in v_high_correspondence:
    for j in v_high_correspondence:
        df[i + '-' + j] = df[i] - df[j]

for i in v_high_correspondence:
    for j in v_high_correspondence:
        df[i + '*' + j] = df[i] * df[j]

y_id = test_pd["SaleID"]  # 获取测试集的id，以便后面保存为csv文件

df.drop(['SaleID'], axis=1, inplace=True)
df.drop(['regDate'], axis=1, inplace=True)
df.drop(['creatDate'], axis=1, inplace=True)
df.drop(['regionCode'], axis=1, inplace=True)
df.drop(['name'], axis=1, inplace=True)

train_pd = df[df['price'].notnull()]
test_pd = df[df['price'].isnull()]

# model = RandomForestRegressor()

model = LGBMRegressor(
    n_estimators=300000,
    learning_rate=0.02,
    boosting_type='gbdt',
    objective='regression_l1',
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_freq=1,
    bagging_fraction=0.8,
    lambda_l2=2,
    random_state=2022,
    metric='mae',
)

# model = RandomForestRegressor(n_estimators=25, random_state=42, oob_score=True)
model.fit(train_pd.drop('price', axis=1), np.log1p(train_pd['price']))

y_pred = model.predict(test_pd.drop('price', axis=1))
y_pred = np.expm1(y_pred)

result = pd.DataFrame({'SaleID': y_id, 'price': y_pred.astype(np.int32)})
result.to_csv("submission.csv", index=False)
