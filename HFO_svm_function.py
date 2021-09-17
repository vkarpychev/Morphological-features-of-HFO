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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.combine import SMOTEENN

def svm_outlier_removal(df, feature):
    
    X_train = df[df['class'].isin(['TN'])][feature].to_numpy()
    
    y_train = df[df['class'].isin(['TN'])]['area'].to_numpy()
    
    iso = IsolationForest(contamination = 0.1)
    
    yhat = iso.fit_predict(X_train)
    
    mask = yhat != -1
    
    X_train, y_train = X_train[mask, :], y_train[mask]
        
    X_train = pd.DataFrame(scale(X_train), columns = feature)
    
    return X_train, y_train

def svm_feature_selection(df, feature):
        
    X_train, y_train = svm_outlier_removal(df, feature)
    
    X_resampled, y_resampled = SMOTEENN().fit_resample(X_train, y_train)
    
    F, p = f_classif(X_resampled, y_resampled)
    
    rank = np.argsort(-F)
    
    ranked_feature = [feature[i] for i in rank]
    
    F = [round(F[i],3) for i in rank]
    
    p = [round(p[i],3) for i in rank]
    
    rank_stat = np.concatenate((np.array([ranked_feature]), np.array([F]), np.array([p])))    
                    
    return ranked_feature, rank_stat

def svm_classifier_train(X_resampled, y_resampled, s, ranked_feature):
    
    for n, element in enumerate(ranked_feature):
        
        if n == 0:
            
            all_ranked_feature = [[element]]
            
        else:
    
            all_ranked_feature.append(list(all_ranked_feature[n-1]) + [element])
            
    del element, n
        
    for number, rank in enumerate(all_ranked_feature):
        
        print('Training SVM in the %s with feature - %s' %(''.join(s),', '.join(rank)))
            
        X_resampled_rank = X_resampled[rank].to_numpy()
        
        params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf']}]
                
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
                        
        search = GridSearchCV(estimator = SVC(), param_grid = params, cv = cv, scoring = 'f1', n_jobs = -1)
        
        search = search.fit(X_resampled_rank, y_resampled)
        
        if number == 0:
        
            training = np.empty((6,7), dtype = object)
            
            training[0,0] = 'rank'
            
            training[0,1] = 'kernel'
            
            training[0,2] = 'C'
                            
            training[0,3] = 'gamma'
            
            training[0,4] = 'degree'
                                    
            training[0,5] = 'roc_auc'
            
            training[0,6] = 'std'
            
        training[number + 1,0] = ','.join(all_ranked_feature[number])
        
        training[number + 1,1] = search.best_estimator_.kernel
            
        training[number + 1,2] = search.best_estimator_.C
        
        training[number + 1,3] = search.best_estimator_.gamma
            
        training[number + 1,4] = search.best_estimator_.degree
            
        training[number + 1,5] = max(search.cv_results_.get('mean_test_score'))

        training[number + 1,6] = search.cv_results_.get('std_test_score')[np.where(search.cv_results_.get('mean_test_score') 
                                                                                 
                                                 == max(search.cv_results_.get('mean_test_score')))[0][0]]

    return training

def svm_classifier_validation(df, s, ranked_feature):
    
    for n, element in enumerate(ranked_feature):
        
        if n == 0:
            
            all_ranked_feature = [[element]]
            
        else:
    
            all_ranked_feature.append(list(all_ranked_feature[n-1]) + [element])
            
    del element, n
        
    for number, rank in enumerate(all_ranked_feature):
        
        if s == ['MTL']:
            
            patient = [3,4,6,11,15,18,19,34,37,40,42]
            
        elif s == ['Neocortex']:
            
            patient = [2,4,9,19,37]
        
        for n in patient:
            
            try:
            
                X_train = df[df['class'].isin(['TN'])]
            
                X_train = X_train[~ X_train['ID'].isin([str(n)])]
                
                X_train, y_train = svm_outlier_removal(df, ranked_feature)
            
                X_resampled, y_resampled = SMOTEENN().fit_resample(X_train, y_train)
                
                X_resampled_rank = X_resampled[rank].to_numpy()
                            
                print('Validation SVM in the %s with feature - %s, patient - %s' %(''.join(s),', '.join(rank), n))
                    
                params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf']}]
                
                cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
                        
                search = GridSearchCV(estimator = SVC(), param_grid = params, cv = cv, scoring = 'f1', n_jobs = -1)
                        
                clf = search.fit(X_resampled_rank, y_resampled).best_estimator_
                
                X_test = df[df['class'].isin(['TN'])]
            
                X_test = X_test[X_test['ID'].isin([str(n)])]
            
                y_test = X_test['area'].to_numpy()
            
                X_test = scale(X_test[rank].to_numpy())
                
                y_pred = clf.predict_proba(X_test)[:,1]
                                
                precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
                
                f = (2 * precision * recall)/(precision + recall)
                
                f = f[~ np.isnan(f)]
                                
                f = f[np.argmax(f)]
            
                if s == ['MTL'] and n == 3:
                                
                    fscore = list([f])
                                
                elif s == ['Neocortex'] and n == 2:
                    
                    fscore = list([f])
    
                else:
                                    
                    fscore.append(f)
                    
            except:
                
                continue
                      
        result = pd.concat((pd.DataFrame({'fscore': fscore}), pd.DataFrame([''.join(str(rank))[2:-2] 
                                                                            
                            for i in range(len(fscore))], columns = {'rank'})), axis = 1)
    
        if number == 0:
            
            result_tn = result
            
        else:
            
            result_tn = pd.concat([result_tn, result], axis = 0)
            
        del fscore, clf, X_test, X_resampled_rank, n
            
    return result_tn

def svm_classifier_test(df, s, ranked_feature, training):
    
    for n, element in enumerate(ranked_feature):
        
        if n == 0:
            
            all_ranked_feature = [[element]]
            
        else:
    
            all_ranked_feature.append(list(all_ranked_feature[n-1]) + [element])
            
    del element, n
    
    X_train = df[df['class'].isin(['TN'])]
    
    X_train, y_train = svm_outlier_removal(X_train, ranked_feature)

    X_resampled, y_resampled = SMOTEENN().fit_resample(X_train, y_train)
        
    for number, rank in enumerate(all_ranked_feature):
        
        X_resampled_rank = X_resampled[rank].to_numpy()
                                                
        params = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf']}]
                
        cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 1)
                        
        search = GridSearchCV(estimator = SVC(), param_grid = params, cv = cv, scoring = 'f1', n_jobs = -1)
                        
        clf = search.fit(X_resampled_rank, y_resampled).best_estimator_
                
        if s == ['MTL']:
            
            globals()['test_%s' % number] = np.empty((1,7), dtype = object)
                            
            X_test = df[df['class'].isin(['TP'])]
                
            X_test = scale(X_test[rank].to_numpy())
                             
            y_test = df[df['class'].isin(['TP'])]['area'].to_numpy()
                
            y_pred = np.array(clf.predict(X_test)).T
                        
            TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred, labels = [1, 0]).ravel()
        
            accuracy = (TN + TP)/(TN + FP + FN + TP)
                
            globals()['test_%s' % number][0, 0] = ','.join(all_ranked_feature[number])
            
            globals()['test_%s' % number][0, 1] = ''.join(str([2, 3, 9, 10, 20]))[1:-1]
                                                                    
            globals()['test_%s' % number][0, 2] = round(accuracy,3)
    
            globals()['test_%s' % number][0, 3] = TN
        
            globals()['test_%s' % number][0, 4] = FP
                
            globals()['test_%s' % number][0, 5] = FN
                
            globals()['test_%s' % number][0, 6] = TP
            
            if number == 0:
            
                event = svm_classifier_rate(df, s, rank, [], y_pred)
                
            else:
                
                v = svm_classifier_rate(df, s, rank, [], y_pred)
                                                    
                event = pd.concat([event, v], axis = 0)
                                    
        elif s == ['Neocortex']:
            
            globals()['test_%s' % number] = np.empty((2,7), dtype = object)
            
            for n, patient in enumerate([19, 42]):
                
                X_test = df[df['ID'].isin([str(patient)])]
                
                y_test = df[df['ID'].isin([str(patient)])]
                
                if patient == 19:
                                                        
                    X_test = X_test[X_test['class'].isin(['FP'])]
                
                    X_test = scale(X_test[rank].to_numpy())
                             
                    y_test = y_test[y_test['class'].isin(['FP'])]['area'].to_numpy()
                
                elif patient == 42:
                                        
                    X_test = X_test[X_test['class'].isin(['TP'])]
                
                    X_test = scale(X_test[rank].to_numpy())
                             
                    y_test = y_test[y_test['class'].isin(['TP'])]['area'].to_numpy()
                
                y_pred = np.array(clf.predict(X_test)).T
                    
                if n == 0 and number == 0:
                                
                    event = svm_classifier_rate(df, s, rank, patient, y_pred)
                        
                elif n != 0 or number != 0: 
                        
                    v = svm_classifier_rate(df, s, rank, patient, y_pred)
                        
                    event = pd.concat([event, v], axis = 0)
                                                            
                TN, FP, FN, TP = metrics.confusion_matrix(y_test, y_pred, labels = [1, 0]).ravel()
        
                accuracy = (TN + TP)/(TN + FP + FN + TP)
                
                globals()['test_%s' % number][n, 0] = ','.join(all_ranked_feature[number])
                
                globals()['test_%s' % number][n, 1] = patient
                                                                    
                globals()['test_%s' % number][n, 2] = round(accuracy,3)
        
                globals()['test_%s' % number][n, 3] = TN
        
                globals()['test_%s' % number][n, 4] = FP
                
                globals()['test_%s' % number][n, 5] = FN
                
                globals()['test_%s' % number][n, 6] = TP
                                       
        if number == 0:
                    
            test = np.empty((1,7), dtype = object)
            
            test[0,0] = 'rank'
            
            test[0,1] = 'patient'
                        
            test[0,2] = 'accuracy'
            
            test[0,3] = 'TN'
            
            test[0,4] = 'FP'
            
            test[0,5] = 'FN'
            
            test[0,6] = 'TP'
            
            test = np.concatenate((test, globals()['test_%s' % number]))
        
        else:
            
            test = np.concatenate((test, globals()['test_%s' % number]))
            
        del clf, X_resampled_rank, y_pred, TN, FP, FN, TP, accuracy, globals()['test_%s' % number],
        
    return test, event

def svm_classifier_rate(df, s, rank, patient, y_pred):
    
    if s == ['MTL']:
    
        X_test = df[df['class'].isin(['TP'])]
        
    elif s == ['Neocortex']:
        
        if patient == 19:
            
            X_test = df[df['class'].isin(['FP'])]
            
        elif patient == 42:
            
            X_test = df[df['class'].isin(['TP'])]

    X_test = X_test.reset_index(drop = True)
        
    y_pred = pd.DataFrame({'TD':y_pred})

    event = pd.concat([X_test, y_pred], axis = 1)
        
    event = event[event['TD'].isin(['1'])]
    
    event = event[['ID','channels','group','area','TD']].reset_index(drop = True)
    
    r = pd.DataFrame([''.join(str(rank))[2:-2] for i in range(len(event))], columns = ['rank'])
    
    event = pd.concat([event, r], axis = 1)
    
    return event
    
    
    
    
    