#
# To launch use 'streamlit run main.py' in Terminal
#

import streamlit as st
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import datasets
# from catboost import CatBoostClassifier, Pool
# from package import MakeFeatures
# from joblib import load


def user_features(df: pd.DataFrame):

    option_month = st.sidebar.selectbox(
        'Choose month',
        range(0, 13))

    option_day_of_month = st.sidebar.selectbox(
        'Choose day_of_month',
        range(1, 31))

    option_day_of_week = st.sidebar.selectbox(
        'Choose day_of_week',
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    option_dep_time = st.sidebar.text_input(
        'Choose departue time. HHMM where HH - hour (0-23), MM - minute (0-59) ',
        0000)

    option_unique_carrier = st.sidebar.selectbox(
        'Choose unique carrier',
        df['UniqueCarrier'].unique())

    option_origin = st.sidebar.selectbox(
        'Choose origin point',
        df['Origin'].unique())

    option_dest = st.sidebar.selectbox(
        'Choose desination point',
        df['Dest'].unique())

    option_distance = st.sidebar.text_input(
        'Choose distance between origin and distance',
        100)

    features = {'Month': [option_month],
                'DayOfMonth': [option_day_of_month],
                'DayOfWeek': [option_day_of_week],
                'DepTime': [option_dep_time],
                'UniqueCarrier': [option_unique_carrier],
                'Origin': [option_origin],
                'Dest': [option_dest],
                'Distance': [option_distance]}
    df_features = pd.DataFrame(features)

    return df_features

def make_predict(df: pd.DataFrame):
    return 1+2


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

st.write('User Input parameters')
st.write(user_df.head())

predicator = st.selectbox(
        'Choose model',
        ['XGBoost', 'CatBoost'])

if st.button('Predict delay'):
    st.write('result: %s' % make_predict(user_df))

# if classifier == 'CatBoost':
#     clf = load('catb.joblib')
#
#     mf = MakeFeatures(is_weekend=1,
#                       seasons=1,
#                       day_of_month=1,
#                       dep_capital=1,
#                       arr_capital=1,
#                       route=0,
#                       log_dist=1,
#                       uc=1,
#                       time=1,
#                       new_dep_time=1,
#                       hour_imp=1)
#
#     mf.fit(df)
#
#     df_to_predict = mf.fit_transform(df)
#
#     df_to_predict = pd.concat([df_to_predict, df.Dest, df.Origin], axis=1, sort=False)
#
#     df_to_predict['UQ_DEST'] = df.UniqueCarrier + '_' + df.Dest
#     df_to_predict['UQ_ORIG'] = df.UniqueCarrier + '_' + df.Origin
#
#     df_ready = Pool(df_to_predict, cat_features=[37, 38, 39, 40])
#
# else:
#     clf = load('xgb.joblib')
#
#     mf = MakeFeatures(is_weekend=1,
#                       seasons=1,
#                       day_of_month=1,
#                       dep_capital=1,
#                       arr_capital=1,
#                       route=1,
#                       log_dist=1,
#                       uc=1,
#                       time=1,
#                       new_dep_time=1,
#                       hour_imp=1)
#
#     mf.fit(train)
#     df_ready = mf.fit_transform(train)

# X_train, y_train = train.drop(['dep_delayed_15min'], axis=1), train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
#
# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
#
# clf = RandomForestClassifier()
# clf.fit(X, Y)
# # clf = RandomForestClassifier()
# # clf.fit(X_train, y_train)
#
# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)
#
# st.subheader('Prediction')
# st.write(prediction)
#
# st.subheader('Prediction Probability')
# st.write(prediction_proba)
