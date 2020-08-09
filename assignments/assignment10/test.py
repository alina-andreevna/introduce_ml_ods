import pandas as pd
from package import MakeFeatures


# import pickle

df_train = pd.read_csv('../../data/flight_delays_train.csv').drop(['dep_delayed_15min'], axis=1)
user_df = df_train[df_train.Month == 'c-8']
print(user_df)

import streamlit as st
from catboost import CatBoostClassifier
# from xgboost import Booster, XGBClassifier


clf = CatBoostClassifier()
clf.load_model('models/cbcclf_model')


print('load success!')
st.write("""
    ### Kaggle InClass cometition: https://www.kaggle.com/c/flight-delays-2017
    ### load success!
    """)
#
def prepare_user_data(df: pd.DataFrame, df_first: pd.DataFrame):
    X = pd.concat([df, df_first.Dest, df_first.Origin], axis=1, sort=False)
    X['UQ_DEST'] = df_first.UniqueCarrier + '_' + df_first.Dest
    X['UQ_ORIG'] = df_first.UniqueCarrier + '_' + df_first.Origin
    X.dropna(inplace=True)
    return X

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

modified_df = feature_construct(user_df, 'CatBoost')

print(modified_df)

dataframe_for_predict = prepare_user_data(modified_df, user_df)

print(dataframe_for_predict, dataframe_for_predict.columns)

prediction = clf.predict(dataframe_for_predict)
probability = clf.predict_proba(dataframe_for_predict)

print(prediction)
print(probability)

st.write(prediction)
st.write(probability)
