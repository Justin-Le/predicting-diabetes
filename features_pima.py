import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_features(file='/diabetes.csv'):
    data = pd.read_csv('./diabetes.csv')
    X = data.ix[:, :-1]
    y = data.ix[:, -1]

    ######################################## 
    # Create binary features
    ######################################## 

    # Insulin >= 400
    X['insulin_geq_400'] = np.where(X['Insulin'] >= 400, 1, 0)

    # BMI >= 48
    X['bmi_geq_48'] = np.where(X['BMI'] >= 48, 1, 0)

    # Diabetes Pedigree Function >= 1
    X['pedigree_geq_1'] = np.where(X['DiabetesPedigreeFunction'] >= 1.0, 1, 0)

    # Glucose >= 170
    X['glucose_geq_170'] = np.where(X['Glucose'] >= 170, 1, 0)

    # Blood Pressure >= 92
    X['blood_geq_92'] = np.where(X['BloodPressure'] >= 92, 1, 0)

    # One or two pregnancies
    X['preg_1_or_2'] = np.where(X['Pregnancies'] == 1, 1, 0) +\
                          np.where(X['Pregnancies'] == 2, 1, 0)

    # Age <= 28
    X['age_leq_28'] = np.where(X['Age'] <= 28, 1, 0)

    # Age is 52 or 53
    X['age_52_or_53'] = np.where(X['Age'] == 52, 1, 0) +\
                           np.where(X['Age'] == 53, 1, 0)

    # 10 <= Skin Thickness >= 23
    X['skin_10_to_23'] = np.where(X['SkinThickness'] >= 10, 1, 0) -\
                            np.where(X['SkinThickness'] > 23, 1, 0)

    return X, y

def impute_pima(X):
    normals = [0]*3
    variables = ['Glucose', 'SkinThickness', 'BMI']

    # Generate imputation values with Gaussian randomness.
    for n, v in zip(range(len(normals)), variables):
        # Shift the mean up to account for skewness caused by zeros.
        v_mean = X[v].mean()*1.5

        # Use surrogate deviation.
        # (Sometimes I get strange values when using .std(). Why?)
        v_std = v_mean*0.1

        normals[n] = np.random.normal(loc = v_mean, scale = v_std)

    # Impute.
    X = X.replace(to_replace = {'Glucose': {0: normals[0]}, 
                                      'SkinThickness': {0: normals[1]}, 
                                      'BMI': {0: normals[2]}})

    return X

if __name__ == "__main__":
    extract_features()

