#
# To launch use 'streamlit run main.py' in Terminal
#

import streamlit as st
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from catboost import CatBoostClassifier, Pool
from xgboost import Booster, XGBClassifier
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


def make_predict(df: pd.DataFrame, classifier: str):
    if classifier == "XGBoost":
        clf = XGBClassifier()
        booster = Booster()
        booster.load_model('models/xgbclf_save_model.model')
        clf._Booster = booster

    else:
        clf = load('models/catb.joblib')

    prediction = clf.predict(df)
    probability = clf.predict_proba(df)

    return prediction, probability


def main():

    df_train = pd.read_csv('../../data/flight_delays_train.csv').drop(['dep_delayed_15min'], axis=1)

    st.write("""
    # Simple App for Final task of ODS.ai machine learning course
    """)

    st.write("""
    ### This app predicts the 15 minutes delay of flights for user features. 
    ### Please, enter parameters to predict in sidebar on the left side, choose model for predict and push "Predict" 
    ### button. You can see result (delayed your flight or no delayed) and result probability.
    """)

    st.write("""
        ### Kaggle InClass cometition: https://www.kaggle.com/c/flight-delays-2017
        """)

    st.write("")
    st.write("")

    user_df = user_features(df_train)

    st.write('User Input parameters (features modified for correct preprocessing)')
    st.write(user_df)

    predicator = st.selectbox(
            'Choose model',
            ['XGBoost', 'CatBoost'])

    modified_df = feature_construct(user_df, predicator)

    dataframe_for_predict = prepare_user_data(modified_df, predicator)

    if st.button('Predict delay'):
        predict, predict_proba = make_predict(dataframe_for_predict, predicator)
        st.write("Result prediction:", predict)
        st.write("Result probability of prediction:", predict_proba)


if __name__ == "__main__":
    main()