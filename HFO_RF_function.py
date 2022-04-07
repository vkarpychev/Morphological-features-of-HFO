#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 7 15:14:21 2021

@author: victor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def RF_outlier_removal(df, feature):
    
    for number, patient in enumerate(df['ID'].unique()):
        
        df_nhfo = df[df['area'] == 0]
        
        df_hfo = df[df['area'] == 1]
        
        X_nhfo = df_nhfo[df_nhfo['ID'].isin([patient])][feature].to_numpy()
                
        y_nhfo = df_nhfo[df_nhfo['ID'].isin([patient])]['area'].to_numpy()
                
        X_hfo = df_hfo[df_hfo['ID'].isin([patient])][feature].to_numpy()
        
        y_hfo = df_hfo[df_hfo['ID'].isin([patient])]['area'].to_numpy()
            
        iso = IsolationForest(contamination = 0.01)
    
        yhat_nhfo = iso.fit_predict(X_nhfo)
    
        mask = yhat_nhfo != -1
    
        X_nhfo, y_nhfo = X_nhfo[mask, :], y_nhfo[mask]
        
        yhat_hfo = iso.fit_predict(X_hfo)
    
        mask = yhat_hfo != -1
    
        X_hfo, y_hfo = X_hfo[mask, :], y_hfo[mask]
        
        X = np.concatenate((X_nhfo, X_hfo))
        
        y = np.concatenate((y_nhfo, y_hfo))
                
        if number == 0:
            
            X_train = scale(X)
            
            y_train = y
            
        else:
            
            X_train = np.concatenate((X_train, scale(X)))
            
            y_train = np.concatenate((y_train, y))
        
    X_train = pd.DataFrame(X_train, columns = feature)
    
    return X_train, y_train

def RF_feature_selection(df, feature):
        
    X_train, y_train = RF_outlier_removal(df, feature)
    
    F, p = f_classif(X_train, y_train)
    
    rank = np.argsort(-F)
    
    ranked_feature = [feature[i] for i in rank]
    
    F = [round(F[i],3) for i in rank]
    
    p = [round(p[i],3) for i in rank]
    
    rank_stat = np.concatenate((np.array([ranked_feature]), np.array([F]), np.array([p])))    
                    
    return ranked_feature, rank_stat

def RF_classifier_loocv(df, feature):
    
    print('Validation RF with features - %s' %(', '.join(feature)))
    
    params = [{'model__max_samples': list(np.arange(0.1, 0.8, 0.1)), 'model__max_features': ['auto', 'log2'], 
              
              'model__max_depth': [10, 20, 40, 60, 80, 100, None], 'model__n_estimators': [100, 200, 600, 1000, 1400, 2000]}]
                  
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
   
    pipeline = Pipeline(steps = [('sampling', SMOTE()),('model', RandomForestClassifier(bootstrap = True, criterion = 'gini', random_state = 1))])

    search = GridSearchCV(estimator = pipeline, param_grid = params, scoring = 'roc_auc', cv = cv, n_jobs = -1, verbose = 1)
   
    for number, patient in enumerate(df['ID'].unique()):
        
        print('Patient - %s' %(patient))
        
        X_train, y_train = RF_outlier_removal(df[~ df['ID'].isin([patient])], feature)
            
        X_train = X_train.to_numpy()
        
        X_test = df[df['ID'].isin([patient])]
        
        y_test = X_test['area'].to_numpy()
        
        X_test = X_test[feature].to_numpy()
        
        X_test = scale(X_test)
            
        clf = search.fit(X_train, y_train)
                
        y_pred = clf.predict(X_test)
                                
        if number == 0:
            
            auc_score = list([metrics.roc_auc_score(y_test, y_pred)])
                        
            event = RF_classifier_rate(df[df['ID'].isin([patient])], feature, y_pred)
            
        else:
            
            auc_score.append(metrics.roc_auc_score(y_test, y_pred))
                        
            event = pd.concat((event, RF_classifier_rate(df[df['ID'].isin([patient])], feature, y_pred)), axis = 0)
            
        del clf, X_test, y_test, X_train, y_train
            
    return auc_score, event

def RF_classifier_cv(df, feature):
    
    print('Validation RF with features - %s' %(', '.join(feature)))
    
    params = [{'model__max_samples': list(np.arange(0.1, 0.8, 0.1)), 'model__max_features': ['auto', 'log2'], 
             
             'model__max_depth': [10, 20, 40, 60, 80, 100, None], 'model__n_estimators': [100, 200, 600, 1000, 1400, 2000]}]
                  
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
   
    pipeline = Pipeline(steps = [('sampling', SMOTE()),('model', RandomForestClassifier(bootstrap = True, criterion = 'gini', random_state = 1))])

    search = GridSearchCV(estimator = pipeline, param_grid = params, scoring = 'roc_auc', cv = cv, n_jobs = -1, verbose = 1)
   
    X_train, y_train = RF_outlier_removal(df, feature)
            
    X_train = X_train.to_numpy()
        
    clf = search.fit(X_train, y_train)
    
    auc_score = list([clf.cv_results_['mean_test_score'][clf.best_index_]])
    
    sd = list([clf.cv_results_['std_test_score'][clf.best_index_]])
    
    result = pd.concat((pd.DataFrame({'auc_score': auc_score}), pd.DataFrame({'SD': sd}),
                            
                        pd.DataFrame([''.join(str(feature))[2:-2] for i in range(len(auc_score))], columns = {'feature'})), axis = 1)
            
    feature_importance_plot(clf, feature, ['TN'])
        
    importance = clf.best_estimator_.named_steps.model.feature_importances_
    
    sd = np.std([tree.feature_importances_ for tree in clf.best_estimator_.named_steps.model.estimators_], axis = 0)
                                        
    feature_importance = np.concatenate((np.array([feature]).T, np.array([importance]).T, np.array([sd]).T), axis = 1)
            
    del clf, X_train, y_train, importance
            
    return result, feature_importance

def feature_importance_plot(clf, feature, patient):

    feature_importance = clf.best_estimator_.named_steps.model.feature_importances_
    
    std = np.std([tree.feature_importances_ for tree in clf.best_estimator_.named_steps.model.estimators_], axis = 0)
    
    idx = feature_importance.argsort()

    y_ticks = np.arange(0, len(feature))

    fig, ax = plt.subplots()

    ax.barh(y_ticks, feature_importance[idx], xerr = std[idx], height = 0.5, color = (0, 0, 0, 0), 
            
            edgecolor = (0.2, 0.4, 0.6, 0.9), linewidth = 2.3)

    ax.set_yticks(y_ticks)

    ax.set_yticklabels(np.array(['Am-FR','Am-ripple','Fr-FR','Fr-ripple','D-HFO'])[idx])

    ax.set_title('Feature Importances')
    
    fig.tight_layout()
    
    plt.show()
    
    fig.savefig('Figures/FI/RF_TN.png', dpi = 1000, bbox_inches = 'tight', pad_inches = 0.1)
    
    return

def RF_classifier_predict(df_train, df_pred, feature):
    
    print('Prediction RF with features - %s' %(', '.join(feature)))
    
    X_train, y_train = RF_outlier_removal(df_train, feature)
                                
    X_train = X_train.to_numpy()
                                                                                                                                
    params = [{'model__max_samples': list(np.arange(0.1, 0.8, 0.1)), 'model__max_features': ['auto', 'log2'], 
              
              'model__max_depth': [10, 20, 40, 60, 80, 100, None], 'model__n_estimators': [100, 200, 600, 1000, 1400, 2000]}]
                  
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
   
    pipeline = Pipeline(steps = [('sampling', SMOTE()),('model', RandomForestClassifier(bootstrap = True, criterion = 'gini', random_state = 1))])

    search = GridSearchCV(estimator = pipeline, param_grid = params, scoring = 'roc_auc', cv = cv, n_jobs = -1, verbose = 1)
                                                
    clf = search.fit(X_train, y_train).best_estimator_
                                      
    X, y = scale(df_pred.reset_index(drop = True)[feature].to_numpy()), df_pred['area'].to_numpy()
                                             
    y_pred = np.array(clf.predict(X)).T
        
    prediction = np.empty((2,6), dtype = object)
                    
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
            
    event = RF_classifier_rate(df_pred, feature, y_pred)
        
    del clf, y_pred, TN, FP, FN, TP, accuracy
        
    return prediction, event

def RF_classifier_rate(df_pred, feature, y_pred):
                
    y_pred = pd.DataFrame({'TD':y_pred})

    event = pd.concat([df_pred, y_pred], axis = 1)
        
    event = event[event['TD'] == 1]
    
    event = event[['ID','channels','group','area','TD']].reset_index(drop = True)
    
    r = pd.DataFrame([''.join(str(feature))[2:-2] for i in range(len(event))], columns = ['feature'])
    
    event = pd.concat([event, r], axis = 1)
    
    return event
    
    
    
    