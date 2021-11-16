#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 7 15:14:21 2021

@author: victor
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

def svm_outlier_removal(df, feature):
    
    X_train = df[feature].to_numpy()
    
    y_train = df['area'].to_numpy()
    
    iso = IsolationForest(contamination = 0.1)
    
    yhat = iso.fit_predict(X_train)
    
    mask = yhat != -1
    
    X_train, y_train = X_train[mask, :], y_train[mask]
        
    X_train = pd.DataFrame(scale(X_train), columns = feature)
    
    return X_train, y_train

def svm_feature_selection(df, feature):
        
    X_train, y_train = svm_outlier_removal(df, feature)
    
    F, p = f_classif(X_train, y_train)
    
    rank = np.argsort(-F)
    
    ranked_feature = [feature[i] for i in rank]
    
    F = [round(F[i],3) for i in rank]
    
    p = [round(p[i],3) for i in rank]
    
    rank_stat = np.concatenate((np.array([ranked_feature]), np.array([F]), np.array([p])))    
                    
    return ranked_feature, rank_stat

def svm_classifier_validation(df, feature):
    
    print('Validation SVM with features - %s' %(', '.join(feature)))
    
    params = [{'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
               
               'model__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'model__kernel': ['rbf']}]
    
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
    
    pipeline = Pipeline(steps = [('sampling', SMOTEENN()), ('model', SVC(random_state = 1))])

    search = GridSearchCV(estimator = pipeline, param_grid = params, scoring = 'roc_auc', cv = cv, n_jobs = -1, verbose = 1)
    
    for number, patient in enumerate(df['ID'].unique()):
        
        print('Patient - %s' %(patient))
        
        X_train, y_train = svm_outlier_removal(df[~ df['ID'].isin([patient])], feature)
            
        X_train = X_train.to_numpy()
        
        X_test = df[df['ID'].isin([patient])]
        
        y_test = X_test['area'].to_numpy()
        
        X_test = X_test[feature].to_numpy()
        
        X_test = scale(X_test)
            
        clf = search.fit(X_train, y_train)
                
        y_pred = clf.predict(X_test)
        
        if number == 0:
                        
            auc_score = list([roc_auc_score(y_test, y_pred)])
            
            event = svm_classifier_rate(df[df['ID'].isin([patient])], feature, y_pred)
            
        else:
            
            auc_score.append(roc_auc_score(y_test, y_pred))
            
            event = pd.concat((event, svm_classifier_rate(df[df['ID'].isin([patient])], feature, y_pred)), axis = 0)
            
        del clf, X_test, y_test, X_train, y_train
            
    result = pd.concat((pd.DataFrame({'auc_score': [np.mean(auc_score)]}), pd.DataFrame({'SD': [np.std(auc_score)]}),
                            
                                        pd.DataFrame([''.join(str(feature))[2:-2]], columns = {'feature'})), axis = 1)
            
    return result, event

def svm_classifier_predict(df_train, df_pred, feature):
    
    print('Prediction with features - %s' %(', '.join(feature)))
    
    X_train, y_train = svm_outlier_removal(df_train, feature)
                                
    X_train = X_train.to_numpy()
                                                                                                                            
    params = [{'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'model__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'model__kernel': ['rbf']}]
                
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
    
    pipeline = Pipeline(steps = [('sampling', SMOTEENN()), ('model', SVC(random_state = 1))])

    search = GridSearchCV(estimator = pipeline, param_grid = params, scoring = 'roc_auc', cv = cv, n_jobs = -1, verbose = 1)
                                                
    clf = search.fit(X_train, y_train).best_estimator_
                        
    df_pred = df_pred.reset_index(drop = True)
            
    X = scale(df_pred[feature].to_numpy())
                             
    y = df_pred['area'].to_numpy()
                
    y_pred = np.array(clf.predict(X)).T
        
    prediction = np.empty((2,7), dtype = object)
                    
    for n, patient in enumerate(['TP', 'FP']):
                 
        if patient == 'TP':
                     
            idx = df_pred[df_pred['class'] == 'TP'].index.values
                    
        elif patient == 'FP':
                     
            idx = df_pred[df_pred['class'] == 'FP'].index.values
                    
        TN, FP, FN, TP = metrics.confusion_matrix(y[idx], y_pred[idx], labels = [1, 0]).ravel()
        
        accuracy = (TN + TP)/(TN + FP + FN + TP)
                            
        prediction[n, 0] = patient
                                                                                
        prediction[n, 1] = round(accuracy,3)
    
        prediction[n, 2] = TN
        
        prediction[n, 3] = FP
                
        prediction[n, 4] = FN
                
        prediction[n, 5] = TP
            
    event = svm_classifier_rate(df_pred, feature, y_pred)
        
    del clf, y_pred, TN, FP, FN, TP, accuracy
        
    return prediction, event

def svm_classifier_rate(df_pred, feature, y_pred):
                
    y_pred = pd.DataFrame({'TD':y_pred})

    event = pd.concat([df_pred, y_pred], axis = 1)
        
    event = event[event['TD'].isin(['1'])]
    
    event = event[['ID','channels','group','area','TD']].reset_index(drop = True)
    
    r = pd.DataFrame([''.join(str(feature))[2:-2] for i in range(len(event))], columns = ['feature'])
    
    event = pd.concat([event, r], axis = 1)
    
    return event
    
    
    
    
    