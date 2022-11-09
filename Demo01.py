
import warnings

warnings.simplefilter('ignore')

import os
import re
import gc
import json
import numpy as np
import pandas as pd

pd.options.display.max_columns=None
pd.options.display.max_rows=200
pd.options.display.float_format=lambda x: '%.3f' % x
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib as plt
import os

os.chdir(r"C:\Users\唐子龙\Desktop\data\data")

# 读取数据
train = pd.read_csv(r"C:\Users\唐子龙\Desktop\data\train_dataset.csv", sep='\t')
test = pd.read_csv(r"C:\Users\唐子龙\Desktop\data\test_dataset.csv", sep='\t')
data=pd.concat([train,test],axis=0)
# location列转成多列

data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

# 删除值相同的列
data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
# 删除太多空值的列
data.drop(columns=['auth_type'],axis=1,inplace=True)

# 日期数据处理 按时间升序把进行操作的用户排列出来
data['op_date'] = pd.to_datetime(data['op_date'])
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)# 按值排序
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
data['ts_diff1'] = data['op_ts'] - data['last_ts']


data['op_date_month'] = data['op_date'].dt.month
data['op_date_year'] = data['op_date'].dt.year
data['op_date_day'] = data['op_date'].dt.day
data['op_date_dayofweek'] = data['op_date'].dt.dayofweek
data['op_date_ymd'] = data['op_date'].dt.year * 100 + data['op_date'].dt.month
data['op_date_hour'] = data['op_date'].dt.hour
# 一天中处于哪个时候构造字典
period_dict = {
    0: 0, 1: 0, 2: 1,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 2, 10:3, 11:3,
    12:3, 13:3, 14:4,
    15:4, 16:4, 17:4,
    18:5, 19:5, 20:5,
    21:5, 22:0, 23:0
}
data['hour_cut'] = data['op_date_hour'].map(period_dict)
# 一年中的哪个季度
season_dict = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
    10: 4, 11: 4, 12: 4,

}
data['month_cut'] = data['op_date_month'].map(season_dict)
data['dayofyear'] = data['op_date'].apply(lambda x: x.dayofyear)  # 一年中的第几天
data['weekofyear'] = data['op_date'].apply(lambda x: x.week)  # 一年中的第几周
data['是否周末'] = data['op_date'].apply(lambda x: True if x.dayofweek in [4, 5, 6] else False)  # 是否周末
data.loc[((data['op_date_hour'] >= 7) & (data['op_date_hour'] <=23)), 'isworktime'] = 1

# 特征编码
data['ip_risk_level'] = data['ip_risk_level'].map({'1级': 1, '2级': 2, '3级': 3})
#根据五个特征来计算有几个独特的用户
for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')
for i in ['os_type']:
    data[i + '_n'] = data.groupby(['user_name', 'op_date_ymd', 'op_date_hour'])[i].transform('nunique')
lis = ['user_name', 'action',
       'ip',
       'ip_location_type_keyword', 'device_model',
       'os_type', 'os_version', 'browser_type', 'browser_version',
       'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
       'location_third_lvl']
# one_hot编码
data_re = data[lis]
df_processed = pd.get_dummies(data_re, prefix_sep="_", columns=data_re.columns)
lis_sx = [i for i in data.columns if i not in lis]
data = pd.concat([data[lis_sx], df_processed], axis=1)
#
train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]

ycol = 'risk_label'
feature_names = list(
    filter(lambda x: x not in [ycol, 'session_id','op_date', 'location'], train.columns))
model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',# 提升树的类型
                           tree_learner='serial',
                           num_leaves=29,
                           max_depth=7,# 最大树的深度
                           learning_rate=0.7,
                           n_estimators=1590,
                           subsample=0.7,
                           feature_fraction=0.95,
                           reg_alpha=0.,
                           reg_lambda=0.,
                           random_state=1973,
                           is_unbalance=True,
                           metric='auc')# 模型度量标准

oof = []
prediction= test[['session_id']]
prediction[ycol] = 0
df_importance_list = []
#交叉验证防止train集过拟合
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1950)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]
    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id + 1))
# 模型训练
    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=250,
                          eval_metric='auc',
                          early_stopping_rounds=800)
    #返回验证集的概率
    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['session_id', ycol]].copy()
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)
    #返回测试集的概率
    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction[ycol] += pred_test[:, 1] / kfold.n_splits


print('roc_auc_score', roc_auc_score(df_oof[ycol], df_oof['pred']))

prediction['id'] = range(len(prediction))
prediction['id'] = prediction['id'] + 1
prediction = prediction[['id', 'risk_label']].copy()
prediction.columns = ['id', 'ret']
prediction.to_csv(r"C:\Users\唐子龙\Desktop\data\result01.csv", index=False)
