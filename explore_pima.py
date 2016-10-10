import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Seaborn for plotting
# and ignore all warnings
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
plt.style.use('ggplot')

data = pd.read_csv('./diabetes.csv')
print data.head()
print data.shape

# y = data.Outcome
# X = data.drop('Outcome', 1)

########################################
# Abscissa: glucose
########################################

"""
grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'BloodPressure')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'SkinThickness')
grid.add_legend()
plt.show()
"""

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'Insulin')
grid.add_legend()
plt.show()

# High risk: insulin_geq_400

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'BMI')
grid.add_legend()
plt.show()

# High risk: BMI_geq_48

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'DiabetesPedigreeFunction')
grid.add_legend()
plt.show()

# High risk: pedigree_geq_1

# From all scatterplots
# High risk: glucose_geq_170

########################################
# Abscissa: blood pressure
########################################

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'BloodPressure', 'SkinThickness')
grid.add_legend()
plt.show()

# High risk: blood_geq_92

"""
# These scatterplots verify previous ones

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'BloodPressure', 'Insulin')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'BloodPressure', 'BMI')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'BloodPressure', 'DiabetesPedigreeFunction')
grid.add_legend()
plt.show()
"""

########################################
# Histograms
########################################

n_bins = data.Pregnancies.max() - data.Pregnancies.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'Pregnancies', bins=np.arange(0, n_bins)-0.5)
plt.show()

# Low risk: preg_1_or_2

n_bins = data.Age.max() - data.Age.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'Age', bins=np.arange(20, n_bins)-0.5)
plt.show()

# Low risk: age_leq_28
# High risk: age_52_or_53

data = data[data.SkinThickness != 0]
n_bins = data.SkinThickness.max() - data.SkinThickness.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'SkinThickness', bins=np.arange(n_bins)-0.5)
plt.show()

# Low risk: st_10_to_23
