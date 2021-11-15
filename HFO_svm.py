# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import pandas as pd
from HFO_svm_function import svm_feature_selection, svm_classifier_predict, svm_classifier_validation

maindir = os.getcwd()

df_train = pd.read_csv(join(maindir,'event_training.csv'))
 
df_pred = pd.read_csv(join(maindir,'event_prediction.csv'))
    
df_train['area'] = df_train['area'].replace(['RA','non-RA'], [1, 0])

df_pred['area'] = df_pred['area'].replace(['RA','non-RA'], [1, 0])

feature = ['fr_ampl','r_ampl','fr_freq','r_freq','duration']
                    
ranked_feature, rank_stat = svm_feature_selection(df_train, feature)
        
prediction, event = svm_classifier_predict(df_train, df_pred, ranked_feature)
    
test_cv, event = svm_classifier_validation(df_train, ranked_feature)
    
pd.DataFrame(rank_stat).to_excel('rank_stat.xls', index = False, header = False)
    
pd.DataFrame(prediction).to_excel('prediction.xls', index = False, header = False)
    
event.to_excel('event.xls', index = False, header = False)
    
test_cv.to_excel('test_cv.xls', index = False, header = False)


    