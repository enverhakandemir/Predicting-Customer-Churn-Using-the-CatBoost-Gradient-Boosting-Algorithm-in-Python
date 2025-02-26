# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]



################ Chi-Square Test ################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Define file path
file_path = "C:/Users/hdemi/OneDrive/Desktop/'Excel for Data Analytics with CRM Metrics'/05 Modül Materyalleri Excel for Data Analytics with CRM Metrics/datatelecom01.xlsx"

# Load data
data = pd.read_excel(file_path)
print(data.columns) # Should select categorical variables from the data to write them into the list (categorical_columns). 

# Define categorical variables to be transformed into dummy variables
categorical_columns = [
    'Churn1', 'Partner', 'Dependents', 'TenureGroup', 'InternetService', 'AddTechServ', 
    'AnyStreaming', 'Contract'
]

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

print(data.columns) # In order to get the Chi-Square score, the dummy variables should be written into the list (chi_square_columns).
# Define variables for Chi-Square test
chi_square_columns = [
    'Churn1_1', 'Partner_P', 'Dependents_ND', 'TenureGroup_T2', 
    'InternetService_FB', 'AddTechServ_1', 'AnyStreaming_1',  
    'Contract_TY'
 ] 

# Initialize an empty DataFrame for storing p-values
chi_square_results = pd.DataFrame(index=chi_square_columns, columns=chi_square_columns)

# Perform Chi-Square Test for each pair of categorical variables
for col1 in chi_square_columns:
    for col2 in chi_square_columns:
        if col1 != col2:  # Avoid self-comparison
            contingency_table = pd.crosstab(data[col1], data[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            chi_square_results.loc[col1, col2] = p
        else:
            chi_square_results.loc[col1, col2] = 1  # Self-comparison

# Convert values to float for better heatmap visualization
chi_square_results = chi_square_results.astype(float)

# Display the Chi-Square p-value matrix
print("Chi-Square P-Value Matrix:")
print(chi_square_results)

# Visualize the p-values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(chi_square_results, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Chi-Square Test P-Value Heatmap")
plt.show()
