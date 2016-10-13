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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

# XGBoost
n_estimators = 100
dtrain = xgb.DMatrix(X_train, y_train)

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

clf = RandomForestClassifier(n_estimators=100, max_depth=4, max_features=5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print "\nlog-loss of Random Forest: " 
print log_loss(y_test, pred)
print "\nAUC of Random Forest: " 
print roc_auc_score(y_test, pred)
print "\nF1-score of Random Forest: " 
print f1_score(y_test, pred)
print "\nAccuracy of Random Forest: " 
print accuracy_score(y_test, pred)
print "\n"

clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, max_features=5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print "\nlog-loss of Extra Trees: " 
print log_loss(y_test, pred)
print "\nAUC of Extra Trees: " 
print roc_auc_score(y_test, pred)
print "\nF1-score of Extra Trees: " 
print f1_score(y_test, pred)
print "\nAccuracy of Extra Trees: " 
print accuracy_score(y_test, pred)
print "\n"

"""
# XGBoost CV
xgb_param = clf.get_xgb_params()
num_boost_round = clf.get_params()['n_estimators']
cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=num_boost_round, nfold=10, seed=1)
print cvresult

# Forest grid search
max_depth = [3, 4, 5]
max_features = [3, 4, 5]
n_estimators = [n_estimators]

models = [RandomForestClassifier(),
          ExtraTreesClassifier()]

params = [{'max_depth': max_depth, 'max_features': max_features, 'n_estimators': n_estimators}]

for model in models:
    clf = GridSearchCV(model, param_grid=params, cv=10, scoring='neg_log_loss')
    clf.fit(X_train, y_train)
    results = pd.DataFrame(clf.cv_results_)
    print '\n'
    print results
    print '\n'
"""
