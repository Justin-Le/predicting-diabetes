import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features_pima import *
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, \
                             ExtraTreesClassifier

data = pd.read_csv('./diabetes.csv')
X = data.ix[:, :-1]
y = data.ix[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# XGBoost
n_estimators = 100
dtrain = xgb.DMatrix(X_train, y_train)

"""
params = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# bst = xgb.cv(params, dtrain, nfold=10)
clf = xgb.XGBClassifier(max_depth = 5,
                        n_estimators=n_estimators,
                        learning_rate=0.1, 
                        subsample=1.0,
                        colsample_bytree=0.5,
                        min_child_weight = 15,
                        reg_alpha=0.03,
                        seed=1)
params = clf.get_xgb_params()
cv = xgb.cv(params, dtrain, num_boost_round=n_estimators, nfold=10, seed=1)
clf.fit(X_train, y_train, eval_metric='logloss',
        eval_set=[(X_test, y_test)])
results = pd.DataFrame(clf.evals_result())
print '\n'
print results
print '\n'
"""

clf = XGBClassifier(n_estimators=100, #70
                    max_depth=3, 
                    min_child_weight=5, 
                    gamma=0.5, 
                    learning_rate=0.05, 
                    subsample=0.7, 
                    colsample_bytree=0.7, 
                    reg_alpha=0.001,
                    seed=1)
clf.fit(X_train, y_train, eval_metric='logloss')
        
pred = clf.predict(X_test)
print "\nlog-loss of XGB: " 
print log_loss(y_test, pred)
print "\nAUC of XGB: " 
print roc_auc_score(y_test, pred)
print "\nF1-score of XGB: " 
print f1_score(y_test, pred)
print "\nAccuracy of XGB: " 
print accuracy_score(y_test, pred)
print "\n"

"""
# xgb_param = clf.get_xgb_params()
# num_boost_round = clf.get_params()['n_estimators']
# cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, nfold=10)
# print cvresult

# Forest hyperparameters
max_features = [4, 5]
n_estimators = [n_estimators]


models = [RandomForestClassifier(),
          ExtraTreesClassifier()]

params = [{'max_features': max_features, 'n_estimators': n_estimators}]

for model in models:
    clf = GridSearchCV(model, param_grid=params, cv=10, scoring='neg_log_loss')
    clf.fit(X_train, y_train)
    results = pd.DataFrame(clf.cv_results_)
    print '\n'
    print results
    print '\n'
"""
