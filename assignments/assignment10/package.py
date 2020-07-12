import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

class MakeFeatures(BaseEstimator):
    def __init__(self, is_weekend=0, seasons=0, day_of_month=0, time=0, dep_capital=0, arr_capital=0, 
                 route=0, uc=0, log_dist=0, new_dep_time=0, hour_imp=0):
        self.is_weekend = is_weekend
        self.seasons = seasons
        self.day_of_month = day_of_month
        self.time = time
        self.dep_capital = dep_capital
        self.arr_capital = arr_capital
        self.route = route
        self.uc=uc
        self.log_dist = log_dist
        self.new_dep_time = new_dep_time
        self.hour_imp = hour_imp

                
    def make_harmonic_features_cos(self, value, period=12):       
        value *= 2 * np.pi / period
        return np.cos(value)
    
    def make_harmonic_features_sin(self, value, period=12):       
        value *= 2 * np.pi / period
        return np.sin(value)
    
    def make_time(self, t):
        if t<700:
            return 'night'
        elif t<1200:
            return 'morning'
        elif t<1700:
            return 'day'
        else:
            return 'evening'
    
    def make_part_of_month(self, t):
        if int(t[2:])<16:
            return 0
        else:
            return 1
    
    def is_capital(self, cap):
        capital_avia = ['DCA', 'IAD', 'BWI']
        if cap in capital_avia:
            return 1
        else:
            return 0
    
    def route_encoding (self, X: pd.DataFrame):
        encoder = OneHotEncoder(handle_unknown='ignore')
        cat_features = encoder.fit_transform(X.loc[:,'Origin':'Dest']) 

        orig = encoder.categories_[0] + '_' + 'Origin'
        dest = encoder.categories_[1] + '_' + 'Dest'
        cols = np.concatenate((orig,dest), axis=0)

        return pd.DataFrame.sparse.from_spmatrix(cat_features), cols, encoder
    
    def fit(self, X : pd.DataFrame):
        
        new_features={}
        
        if self.seasons == 1:
            try:
                s_cos = pd.Series(X.Month.map({'c-1': self.make_harmonic_features_cos(1), 
                                            'c-2': self.make_harmonic_features_cos(2), 
                                            'c-3': self.make_harmonic_features_cos(3), 
                                            'c-4': self.make_harmonic_features_cos(4),  
                                            'c-5': self.make_harmonic_features_cos(5),
                                            'c-6': self.make_harmonic_features_cos(6),
                                            'c-7': self.make_harmonic_features_cos(7),
                                            'c-8': self.make_harmonic_features_cos(8),
                                            'c-9': self.make_harmonic_features_cos(9),
                                            'c-10': self.make_harmonic_features_cos(10),
                                            'c-11': self.make_harmonic_features_cos(11),
                                            'c-12': self.make_harmonic_features_cos(12),}).values)
                
                s_sin = pd.Series(X.Month.map({'c-1': self.make_harmonic_features_sin(1), 
                                            'c-2': self.make_harmonic_features_sin(2), 
                                            'c-3': self.make_harmonic_features_sin(3), 
                                            'c-4': self.make_harmonic_features_sin(4),  
                                            'c-5': self.make_harmonic_features_sin(5),
                                            'c-6': self.make_harmonic_features_sin(6),
                                            'c-7': self.make_harmonic_features_sin(7),
                                            'c-8': self.make_harmonic_features_sin(8),
                                            'c-9': self.make_harmonic_features_sin(9),
                                            'c-10': self.make_harmonic_features_sin(10),
                                            'c-11': self.make_harmonic_features_sin(11),
                                            'c-12': self.make_harmonic_features_sin(12),}).values)
                new_features['Seasons_cos']=s_cos
                new_features['Seasons_sin']=s_sin
                print('Seasons complete!')
            except AttributeError:
                print('No Month feature')
                
        
        if self.is_weekend == 1:
            try:
                d = pd.Series(X.DayOfWeek.map({'c-7': 1, 'c-6':1, 'c-1': 0, 'c-2': 0, 'c-3': 0,
                                                      'c-4': 0, 'c-5': 0}).values)
                
                new_features['IsWeekend']=d
                print('IsWeekend complete!')
                
            except AttributeError:
                print('No DayOfWeek feature')

        
        if self.time == 1:           
            try:
                t = pd.get_dummies(pd.Series(X.DepTime.apply(self.make_time)))
                print('TimeOfDay complete!')
            except AttributeError:
                print('No DepTime feature')


        if self.day_of_month == 1:
            try:
                m = pd.Series(X.DayofMonth.str[2:])
                new_features['Day'] = m.map(int)
                print('Day complete!')
                
            except AttributeError:
                print('No DayofMonth feature')
        
        if self.dep_capital == 1:
            try:
                cd = pd.Series(X.Origin.apply(self.is_capital))
                new_features['DepFromCap']=cd
                print('DepFromCap complete!')
            except AttributeError:
                print('No Origin feature')

        
        if self.arr_capital == 1:
            try:
                ca = pd.Series(X.Dest.apply(self.is_capital))
                new_features['ArrForCap']=ca
                print('ArrivInCap complete!')
            except AttributeError:
                print('No Dest feature')

                
        if self.log_dist==1:
            try:
                ld = pd.Series(X.Distance.apply(np.log))
                new_features['LogDist']=ld
                print('LogDist complete!')
            except AttributeError:
                print('No Distance feature')

        
        if self.uc == 1:
            try:
                uc =  pd.get_dummies(X.UniqueCarrier)
                print('UniqueCarrier dummies complete!')
            except AttributeError:
                print('No UniqueCarrier feature')
        
        if self.new_dep_time == 1:
            try:
                hour = pd.Series(X['DepTime'] // 100)
                hour.loc[hour == 24] = 0
                hour.loc[hour == 25] = 1
                minute = pd.Series(X['DepTime'] % 100)
                new_features['Hour'] = hour
                new_features['Minute'] = minute
                print('Hour Minute feature complete!')
            except AttributeError:
                print('No DepTime feature')
        
        if self.hour_imp == 1:
            if self.new_dep_time == 0:
                print('No Hour feature, set new_dep_time=1')
            else:
                try:
                    new_features['Hour_sq'] = new_features['Hour'] ** 2
                    new_features['Hour_sq2'] = new_features['Hour'] ** 4
                    print('Hour_sq-s feature complete!') 
                except AttributeError:
                    print('No Hour feature')   
            
        self.new_frame = pd.DataFrame(new_features)

        if self.uc==1:
            self.new_frame=pd.concat([self.new_frame, uc], axis=1, sort=False)
        
        if self.route==1:
            try:
                data, cols, self.encoder = self.route_encoding(X)
                data.set_axis(list(cols), axis='columns', inplace=True)
                self.new_frame=pd.concat([self.new_frame, data.astype(int)], axis=1, sort=False)
                print('Route complete!')
            except AttributeError:
                print('No Dest or Origin feature')

        if self.time == 1:
            self.new_frame=pd.concat([self.new_frame, t], axis=1, sort=False)
            
        return self
    
    def fit_transform(self, X:pd.DataFrame):
        self.fit(X)
        
        if self.log_dist==0:
            return pd.concat([X[['Distance', 'DepTime']], self.new_frame], axis=1, sort=False) 
        elif self.new_dep_time == 1:
            return self.new_frame
        else:
            return pd.concat([X[['DepTime']], self.new_frame], axis=1, sort=False)
    
    def transform(self, X:pd.DataFrame):
        try:
            return self.new_frame
        except AttributeError:
            print('You should fit classifier MakeFeatures before transform')
            

def make_predict(prediction : pd.Series, file_name : str):
	PATH_ANS = 'csv'
	return pd.Series(prediction, name='dep_delayed_15min').to_csv(PATH_ANS + '/' + file_name+'.csv', index_label='id', header=True)