import pandas as pd
from package import MakeFeatures


# import pickle

df_train = pd.read_csv('../../data/flight_delays_train.csv').drop(['dep_delayed_15min'], axis=1)
user_df = df_train[df_train.Month == 'c-8']
print(user_df)

import streamlit as st
# from catboost import CatBoostClassifier
from xgboost import Booster, XGBClassifier


clf = XGBClassifier()
booster = Booster()
booster.load_model('models/xgbclf_save_model.model')
clf._Booster = booster


print('load success!')
st.write("""
    ### Kaggle InClass cometition: https://www.kaggle.com/c/flight-delays-2017
    ### load success!
    """)

def prepare_user_data(df: pd.DataFrame, predicator: str):
    if predicator == 'XGBoost':
        with open('train_columns/col_train_xgb.txt', 'r') as fid:
            columns = fid.read().split(sep='\n')
    else:
        with open('train_columns/col_train_catb.txt', 'r') as fid:
            columns = fid.read().split(sep='\n')
    columns.pop(-1)
    diff_features = set(columns) - set(df.columns.tolist())
    new_data = dict.fromkeys(list(diff_features), [0])
    dataframe = pd.concat([df, pd.DataFrame.from_dict(new_data)], axis=1, sort=False)

    return dataframe.reindex(columns=columns)

def feature_construct(df: pd.DataFrame, predicator: str):
    if predicator == 'XGBoost':
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
    else:
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

    mf.fit(df)
    df_modified = mf.fit_transform(df)
    return df_modified

modified_df = feature_construct(user_df, 'XGBoost')

dataframe_for_predict = prepare_user_data(modified_df, 'XGBoost')

print(dataframe_for_predict)

prediction = clf.predict(dataframe_for_predict)
probability = clf.predict_proba(dataframe_for_predict)

print(prediction)
print(probability)

st.write(prediction)
st.write(probability)
