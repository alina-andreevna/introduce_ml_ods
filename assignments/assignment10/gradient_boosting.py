import pandas as pd
from xgboost import XGBClassifier
from package import MakeFeatures
from joblib import dump

R_STATE = 17


train = pd.read_csv('../../data/flight_delays_train.csv')
test = pd.read_csv('../../data/flight_delays_test.csv')

X_train, y_train = train.drop(['dep_delayed_15min'], axis=1), train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

mf = MakeFeatures(is_weekend=1,
                  seasons=1,
                  day_of_month=1,
                  dep_capital=1,
                  arr_capital=1,
                  route=1,
                  log_dist=1,
                  uc=1,
                  time=1,
                  new_dep_time=1,
                  hour_imp=1)
mf.fit(X_train)

X_train_new = mf.fit_transform(X_train)

xgb = XGBClassifier(learning_rate=0.1,
                    max_depth=4,
                    min_child_weight=2,
                    n_estimators=1000,
                    gamma=0.1,
                    subsamle=0.9,
                    colsample_bytree=0.5,
                    reg_alpha=1.4,
                    objective='binary:logistic'
                    )

print('XGBClassifier training')
xgb.fit(X_train_new.values, y_train)

dump(clf, 'xgb.joblib')
