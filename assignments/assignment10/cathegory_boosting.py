import pandas as pd
from catboost import CatBoostClassifier, Pool
from package import MakeFeatures

R_STATE = 17

train = pd.read_csv('../../data/flight_delays_train.csv')
test = pd.read_csv('../../data/flight_delays_test.csv')

X_train, y_train = train.drop(['dep_delayed_15min'], axis=1), train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

print(X_train.head())

mf = MakeFeatures(is_weekend=1,
                  seasons=1,
                  day_of_month=1,
                  dep_capital=1,
                  arr_capital=1,
                  route=0,
                  log_dist=1,
                  uc=1,
                  time=1,
                  new_dep_time=1,
                  hour_imp=1)
mf.fit(X_train)

X_train_new = mf.fit_transform(X_train)

X = pd.concat([X_train_new, X_train.Dest, X_train.Origin], axis=1, sort=False)

X['UQ_DEST'] = X_train.UniqueCarrier+'_'+X_train.Dest
X['UQ_ORIG'] = X_train.UniqueCarrier+'_'+X_train.Origin

train = Pool(X, y_train, cat_features=[37, 38, 39, 40])

cbc_model = CatBoostClassifier(random_seed=R_STATE, verbose=True)
cbc_model.fit(train)

dump(cbc_model, 'catb.joblib')
