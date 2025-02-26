# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]



################ Mutual Information Test ################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

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
# Define variables for MI test
mi_columns = [
    'Churn1_1', 'Partner_P', 'Dependents_ND', 'TenureGroup_T2', 
    'InternetService_FB', 'AddTechServ_1', 'AnyStreaming_1',  
    'Contract_TY'
]

# Prepare DataFrame for MI results
mi_results = pd.DataFrame(index=mi_columns, columns=mi_columns)

# Compute Mutual Information for each pair of variables
for col1 in mi_columns:
    for col2 in mi_columns:
        if col1 != col2:
            mi_score = mutual_info_classif(
                data[[col1]], data[col2], discrete_features=True
            )[0]  # mutual_info_classif returns an array, so we take the first element
            mi_results.loc[col1, col2] = mi_score
        else:
            mi_results.loc[col1, col2] = 1  # Self-comparison

# Convert values to float for better heatmap visualization
mi_results = mi_results.astype(float)

# Display the Mutual Information matrix
print("Mutual Information Score Matrix:")
print(mi_results)

# Visualize the MI scores using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(mi_results, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Mutual Information (MI) Score Heatmap")
plt.show()

## Tips to Interpret:
#If MI score is between 0.00 - 0.05; No relationship or Too weak
#If MI score is between 0.05 - 0.10; Weak relationship (potential slight dependency)
#If MI score is between 0.10 - 0.30; moderate relationship 