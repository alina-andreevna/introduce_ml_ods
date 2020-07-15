#
# To launch use 'streamlit run main.py' in Terminal
#

import streamlit as st
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from catboost import CatBoostClassifier, Pool
from package import MakeFeatures
from joblib import load


def user_features(df: pd.DataFrame):

    option_month = st.sidebar.selectbox(
        'Choose month (1 - January, 12 - December)',
        range(1, 13))

    option_day_of_month = st.sidebar.selectbox(
        'Choose day_of_month',
        range(1, 31))

    option_day_of_week = st.sidebar.selectbox(
        'Choose day_of_week (1 - Monday, 7 - Saturday)',
        range(1, 8))

    option_dep_time = st.sidebar.text_input(
        'Choose departue time. HHMM where HH - hour (0-23), MM - minute (0-59) ',
        2359)

    option_unique_carrier = st.sidebar.selectbox(
        'Choose unique carrier',
        df['UniqueCarrier'].unique())

    option_origin = st.sidebar.selectbox(
        'Choose origin point',
        df['Origin'].unique())

    option_dest = st.sidebar.selectbox(
        'Choose destination point',
        df['Dest'].unique())

    option_distance = st.sidebar.text_input(
        'Choose distance between origin and distance',
        100)

    features = {'Month': ['c-' + str(option_month)],
                'DayOfMonth': ['c-' + str(option_day_of_month)],
                'DayOfWeek': ['c-' + str(option_day_of_week)],
                'DepTime': [int(option_dep_time)],
                'UniqueCarrier': [option_unique_carrier],
                'Origin': [option_origin],
                'Dest': [option_dest],
                'Distance': [int(option_distance)]}
    df_features = pd.DataFrame(features)

    return df_features


def feature_construct(df: pd.DataFrame):
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
    return mf.fit_transform(df)


def make_predict(df: pd.DataFrame, classifier: str):
    if classifier == "XGBoost":
        clf = load('xgb.joblib')
    else:
        clf = load('catb.joblib')

    prediction = clf.predict(df)
    probability = clf.predict_proba(df)

    return prediction, probability



df = pd.read_csv('../../data/flight_delays_train.csv')


st.write("""
# Simple App for Final task of ODS.ai machine learning course
""")

st.write("""
### This app predicts the 15 minutes delay of flights for user features. 
### Please, enter parameters to predict in sidebar on the left side, choose model for predict and push "Predict" button. You can see result (depayed your flight or no delayed) and result probability.
""")

st.write("")
st.write("")

user_df = user_features(df)

st.write('User Input parameters (features modified for correct preprocessing)')
st.write(user_df)

modified_df = feature_construct(user_df)

predicator = st.selectbox(
        'Choose model',
        ['XGBoost', 'CatBoost'])

if st.button('Predict delay'):
    st.write('prediction: %s' % make_predict(modified_df, predicator)[0])
    st.write('probability: %s' % make_predict(modified_df, predicator)[1])
