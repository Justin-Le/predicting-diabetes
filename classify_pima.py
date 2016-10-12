import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features_pima import *
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, \
                             ExtraTreesClassifier

data = pd.read_csv('./diabetes.csv')
X = data.ix[:, :-1]
y = data.ix[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# XGBoost
dtrain = xgb.DMatrix(X_train, y_train)
params = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# bst = xgb.cv(params, dtrain, nfold=10)
clf = XGBClassifier()
clf.fit(X_train, y_train,
        eval_set=[(X_test, y_test)])
print '\n'
print clf.evals_result()
print '\n'

# Forest hyperparameters
max_features = [4, 5]
n_estimators = [100]

models = [RandomForestClassifier(),
          ExtraTreesClassifier()]

params = [{'max_features': max_features, 'n_estimators': n_estimators}]

for model in models:
    clf = GridSearchCV(model, param_grid=params, cv=10)
    clf.fit(X_train, y_train)
    results = pd.DataFrame(clf.cv_results_)
    print '\n'
    print results
    print '\n'
