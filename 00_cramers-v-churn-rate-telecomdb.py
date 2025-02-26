# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]



################ Cramer's V Test ################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Define file path
file_path = "C:/Users/hdemi/OneDrive/Desktop/'Excel for Data Analytics with CRM Metrics'/05 Modül Materyalleri Excel for Data Analytics with CRM Metrics/datatelecom01.xlsx"

# Load data
data = pd.read_excel(file_path)

print(data.columns)
# Define categorical variables to be transformed into dummy variables
categorical_columns = [
    'Churn1', 'Partner', 'Dependents', 'TenureGroup', 'InternetService', 'AddTechServ', 
    'AnyStreaming', 'Contract'
]

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

print(data.columns)
# Define variables for Cramer's V calculation
cramers_columns = [
    'Churn1_1', 'Partner_P', 'Dependents_ND', 'TenureGroup_T2', 
    'InternetService_FB', 'AddTechServ_1', 'AnyStreaming_1',  
    'Contract_TY'
]

# Function to calculate Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Initialize an empty DataFrame for storing Cramer's V values
cramers_results = pd.DataFrame(index=cramers_columns, columns=cramers_columns)

# Compute Cramer's V for each pair of categorical variables
for col1 in cramers_columns:
    for col2 in cramers_columns:
        if col1 != col2:
            cramers_results.loc[col1, col2] = cramers_v(data[col1], data[col2])
        else:
            cramers_results.loc[col1, col2] = 1  # Self-comparison

# Convert values to float for better heatmap visualization
cramers_results = cramers_results.astype(float)

# Display the Cramer's V matrix
print("Cramer's V Score Matrix:")
print(cramers_results)

# Visualize the Cramer's V scores using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cramers_results, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Cramer's V Heatmap")
plt.show()

## Tips to Interpret:
#Cramér's V takes a value between 0 and 1. 
#The closer to 0, the weaker the relationship between the two variables. 
#Closer to 1, the stronger the relationship between the two variables.
#If the value is above 0.20, we can consider the relationship to be statistically significant.