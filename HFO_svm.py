# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from os.path import join
import fnmatch
import pandas as pd
from HFO_svm_function import svm_feature_selection, svm_classifier_predict, svm_classifier_validation

maindir = os.getcwd()

file = [s for s in fnmatch.filter(os.listdir(maindir),'event_analysis.csv') if not s.startswith('._')]

df = pd.read_csv(join(maindir,file[0]))
    
df['area'] = df['area'].replace(['RA','non-RA'], [1, 0])

feature = ['fr_ampl','r_ampl','fr_freq','r_freq','duration']

region = [['MTL'],['Neocortex']]

for number, s in enumerate(region):
    
    globals()['df_%s' % number] = df[df['group'].isin(s)]
            
    ranked_feature, rank_stat = svm_feature_selection(globals()['df_%s' % number], feature)
        
    prediction, event = svm_classifier_predict(globals()['df_%s' % number], s, ranked_feature)
    
    test_cv = svm_classifier_validation(globals()['df_%s' % number], s, ranked_feature)
    
    pd.DataFrame(rank_stat).to_excel('rank_stat_%s.xls' %''.join(s), index = False, header = False)
    
    pd.DataFrame(prediction).to_excel('prediction_%s.xls' %''.join(s), index = False, header = False)
    
    event.to_excel('event_%s.xls' %''.join(s), index = False, header = False)
    
    test_cv.to_excel('test_cv_%s.xls' %''.join(s), index = False, header = False)


    