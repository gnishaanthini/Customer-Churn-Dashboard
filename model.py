import numpy as np
import pandas as pd

train=pd.read_csv("train.csv")
# test_df=pd.read_csv("test.csv")

train['total_charge']=train['total_day_charge']+train['total_eve_charge']+train['total_night_charge']+train['total_intl_charge']

train['total_minutes']=train['total_day_min']+train['total_eve_min']+train['total_night_minutes']+train['total_intl_minutes']

train['day']=train['total_day_calls']+train['total_day_charge']+train['total_day_min']



X=train.drop('Churn',axis=1)
y=train['Churn']

from imblearn.over_sampling import SMOTE

smote = SMOTE()


x_smote, y_smote = smote.fit_resample(X, y)

x_smote=x_smote[['customer_service_calls','intertiol_plan','voice_mail_plan','total_day_min','total_day_charge','location_code','total_charge','total_minutes','total_intl_calls','total_intl_charge','number_vm_messages','day']]

import xgboost as xgb

model=xgb.XGBClassifier(n_estimators=1000,learning_rate=0.05)

model.fit(x_smote,y_smote)

# print(model.predict(test_df))

import joblib

joblib.dump(model, "clf.pkl")