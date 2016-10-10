import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from features_pima import *
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, \
                             ExtraTreesClassifier

data = pd.read_csv('./diabetes.csv')

estimators = [('rf', RandomForestClassifier()),
               ('extra', ExtraTreesClassifier())]
max_features = ["sqrt", "log2"]
params = dict(rf__max_features=max_features, extra__max_features=max_features)

pipe = Pipeline(estimators)
pipe.set_params()

grid_search = GridSearchCV(pipe, param_grid=params)
