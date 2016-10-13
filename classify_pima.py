import numpy as np
import pandas as pd
from features_pima import *
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             ExtraTreesClassifier

data = extract_features()
X = data.ix[:, :-1]
y = data.ix[:, -1]

# Isolate a training set for CV.
# Testing set won't be touched until CV is complete.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Note that the train/test split *must* be done before any imputation.
# If we impute values before isolating the training set,
# then values from the testing set might 
# influence the imputed values in the training set.
# See Abu-Mostafa's "Learning from Data" to read more about this issue,
# sometimes referred to as "data snooping".

########################################
# Impute meaningless zero values
########################################

# Impute each set in isolation to avoid snooping
X_train = impute_pima(X_train)
X_test = impute_pima(X_test)

########################################
# Cross-validate
########################################

# Instantiate logistic regression
clf = LogisticRegression()
print "="*50
print "\nLogistic regression results for 10-fold cross-validation:"
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print "Accuracies:\n %s\n" % str(scores)
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')
print "\nAUC:\n %s\n" % str(scores)

# Instantiate XGBoost
n_estimators = 100
dtrain = xgb.DMatrix(X_train, y_train)

# XGBoost was tuned on the raw data.
bst = XGBClassifier(n_estimators=100, #70
                    max_depth=3, 
                    min_child_weight=5, 
                    gamma=0.5, 
                    learning_rate=0.05, 
                    subsample=0.7, 
                    colsample_bytree=0.7, 
                    reg_alpha=0.001,
                    seed=1)

# Cross-validate XGBoost
params = bst.get_xgb_params() # Extract parameters from XGB instance to be used for CV
num_boost_round = bst.get_params()['n_estimators'] # XGB-CV has different names than sklearn

cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 
                  nfold=10, metrics=['logloss', 'auc'], seed=1)

print "="*50
print "\nXGBoost results for 10-fold cross-validation:"
print cvresult

# XGBoost summary
print "="*50
print "\nXGBoost summary for 100 rounds of 10-fold cross-validation:"
print "\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min()
print "\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max()
print "="*50

########################################
# Test
########################################

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print "="*50
print "\nLogistic regression performance on unseen data:"
print "\nlog-loss: %.4f" % log_loss(y_test, pred)
print "\nAUC: %.4f" % roc_auc_score(y_test, pred)
print "\nF1 score: %.4f" % f1_score(y_test, pred)
print "\nAccuracy: %.4f" % accuracy_score(y_test, pred)
print "="*50

bst.fit(X_train, y_train, eval_metric='logloss')
pred = bst.predict(X_test)

print "="*50
print "\nXGBoost performance on unseen data:"
print "\nlog-loss: %.4f" % log_loss(y_test, pred)
print "\nAUC: %.4f" % roc_auc_score(y_test, pred)
print "\nF1 score: %.4f" % f1_score(y_test, pred)
print "\nAccuracy: %.4f" % accuracy_score(y_test, pred)
print "="*50

"""
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
