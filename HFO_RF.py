# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import pandas as pd
from HFO_RF_function import RF_feature_selection, RF_classifier_predict, RF_classifier_cv, RF_classifier_loocv

maindir = os.getcwd()

df_train = pd.read_csv(join(maindir,'event_training.csv'))
 
df_pred = pd.read_csv(join(maindir,'event_prediction.csv'))
    
df_train['area'] = df_train['area'].replace(['HFO','non-HFO'], [1, 0])

df_train['duration'] = df_train['duration'] * 1000

df_pred['area'] = df_pred['area'].replace(['HFO','non-HFO'], [1, 0])

df_pred['duration'] = df_pred['duration'] * 1000

feature = ['fr_ampl','r_ampl','fr_freq','r_freq','duration']
                    
ranked_feature, rank_stat = RF_feature_selection(df_train, feature)
        
prediction, event = RF_classifier_predict(df_train, df_pred, ranked_feature)
    
test_cv, event = RF_classifier_loocv(df_train, ranked_feature)

test_cv, feature_importance = RF_classifier_cv(df_train, ranked_feature)
    
pd.DataFrame(rank_stat).to_excel('RF_RS.xlsx', index = False, header = False)
    
pd.DataFrame(prediction).to_excel('RF_prediction.xlsx', index = False, header = False)

pd.DataFrame(test_cv).to_excel('RF_test_cv.xlsx', index = False, header = False)

pd.DataFrame(feature_importance).to_excel('RF_FI.xlsx', index = False, header = False)
    
event.to_excel('RF_event.xlsx', index = False, header = False)

    