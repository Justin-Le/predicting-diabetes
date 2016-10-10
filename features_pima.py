import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def extract_features(file='/diabetes.csv'):
    data = pd.read_csv('./diabetes.csv')

    ######################################## 
    # Create binary features
    ######################################## 

    # Insulin >= 400
    data['insulin_geq_400'] = np.where(data['Insulin'] >= 400, 1, 0)

    # BMI >= 48
    data['bmi_geq_48'] = np.where(data['BMI'] >= 48, 1, 0)

    # Diabetes Pedigree Function >= 1
    data['pedigree_geq_1'] = np.where(data['DiabetesPedigreeFunction'] >= 1.0, 1, 0)

    # Glucose >= 170
    data['glucose_geq_170'] = np.where(data['Glucose'] >= 170, 1, 0)

    # Blood Pressure >= 92
    data['blood_geq_92'] = np.where(data['BloodPressure'] >= 92, 1, 0)

    # One or two pregnancies
    data['preg_1_or_2'] = np.where(data['Pregnancies'] == 1, 1, 0) +\
                          np.where(data['Pregnancies'] == 2, 1, 0)

    # Age <= 28
    data['age_leq_28'] = np.where(data['Age'] <= 28, 1, 0)

    # Age is 52 or 53
    data['age_52_or_53'] = np.where(data['Age'] == 52, 1, 0) +\
                           np.where(data['Age'] == 53, 1, 0)

    # 10 <= Skin Thickness >= 23
    data['skin_10_to_23'] = np.where(data['SkinThickness'] >= 10, 1, 0) -\
                            np.where(data['SkinThickness'] > 23, 1, 0)

    return data

if __name__ == "__main__":
    extract_features()
