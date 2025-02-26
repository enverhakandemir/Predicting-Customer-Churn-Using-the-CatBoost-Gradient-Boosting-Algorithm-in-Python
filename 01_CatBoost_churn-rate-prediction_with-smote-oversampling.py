# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Import necessary libraries
import pandas as pd  # Data manipulation (reading Excel, handling DataFrames)
import numpy as np  # Mathematical operations (arrays, meshgrid for visualization)
import matplotlib.pyplot as plt  # Visualization (feature importance, decision boundaries)
from collections import Counter  # Count class distributions in the dataset

# Install CatBoost library if not installed (Run this in the terminal before execution)
# !pip install catboost

# Import CatBoost and necessary ML tools
from catboost import CatBoostClassifier, Pool  # CatBoost for gradient boosting on categorical data
from sklearn.model_selection import train_test_split  # Splitting data into training/testing sets
from sklearn.metrics import classification_report, accuracy_score  # Model evaluation (accuracy, classification report)
from imblearn.over_sampling import SMOTE  # SMOTE technique to handle imbalanced datasets

# Load dataset from the specified path
file_path = "C:/Users/hdemi/OneDrive/Desktop/'Excel for Data Analytics with CRM Metrics'/05 Modül Materyalleri Excel for Data Analytics with CRM Metrics/datatelecom01.xlsx"
data = pd.read_excel(file_path)

# Define independent (features) and dependent (target) variables
feature_columns = [
    'Partner', 'Dependents', 'TenureGroup', 'InternetService', 'AddTechServ', 'AnyStreaming', 'Contract'
]
target = 'Churn1'

X = data[feature_columns]  # Feature matrix
y = data[target]  # Target variable

## X = pd.get_dummies(X, drop_first=True)  # Uncomment if one-hot encoding is needed

# defining categorical features
cat_features = list(range(len(feature_columns)))  

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import LabelEncoder
# Encode categorical variables using Label Encoding
label_encoders = {}  # Dictionary to store LabelEncoders for each categorical column

for col in feature_columns:
    le = LabelEncoder()
    X_train.loc[:, col] = le.fit_transform(X_train[col])  # Safely modify training data
    X_test.loc[:, col] = le.transform(X_test[col])  # Transform test data using the same encoder
    label_encoders[col] = le  # Store encoder for potential inverse transformation

## X.to_excel("X_dataframe.xlsx", index=False)  # Save transformed data for inspection
## X_train.to_excel("X_training_df.xlsx", index=False)  # Save transformed training data

# Display the mapping of categorical values to numerical labels
for col in feature_columns:
    print(f"Label Encoding for '{col}':")
    for i, category in enumerate(label_encoders[col].classes_):
        print(f"  {category} → {i}")
    print("-" * 30)

# Handle class imbalance using SMOTE
print("Original class distribution:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution after oversampling:", Counter(y_train_resampled))

# Ensure categorical features remain integers after SMOTE (SMOTE might convert them to floats)
X_train_resampled.iloc[:, cat_features] = X_train_resampled.iloc[:, cat_features].astype(int)

# Convert datasets into CatBoost Pool format
train_pool = Pool(X_train_resampled, label=y_train_resampled, cat_features=cat_features)
test_pool = Pool(X_test, label=y_test, cat_features=cat_features)

# Train a CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,          # Number of boosting iterations
    learning_rate=0.03,      # Learning rate
    depth=6,                 # Depth of decision trees
    loss_function='Logloss', # Loss function for binary classification
    eval_metric='Accuracy',  # Evaluation metric
    cat_features=cat_features,  # Specify categorical features
    verbose=50,              # Print training logs every 50 iterations
    random_seed=42           # Ensure reproducibility
    #l2_leaf_reg=3  # Uncomment to add L2 regularization for preventing overfitting
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=40, use_best_model=True)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get feature importance scores
feature_importances = model.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print feature importance scores
print("\nFeature Importances:")
print(importance_df)

# ### Feature Importance Visualization
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (CatBoost)')
plt.gca().invert_yaxis()  # Place the most important feature at the top
plt.show()

# ### Decision Boundary Visualization (Using Two Features)
feature_x = 'TenureGroup'  
feature_y = 'Contract'

# Convert selected feature names to their index positions
x_idx = feature_columns.index(feature_x)
y_idx = feature_columns.index(feature_y)

# Extract relevant training data
X_train_subset = X_train_resampled[[feature_x, feature_y]]
y_train_classes = y_train_resampled.values

# Generate mesh grid for visualization
x_min, x_max = X_train_subset[feature_x].min() - 1, X_train_subset[feature_x].max() + 1
y_min, y_max = X_train_subset[feature_y].min() - 1, X_train_subset[feature_y].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

# Placeholder dataset for decision boundary visualization
X_decision_boundary = pd.DataFrame(np.zeros((xx.ravel().shape[0], X_train.shape[1])), columns=X_train.columns)

# Assign mesh grid values to the selected features
X_decision_boundary[feature_x] = xx.ravel()
X_decision_boundary[feature_y] = yy.ravel()

# Convert categorical features back to their appropriate types
X_decision_boundary.iloc[:, cat_features] = X_decision_boundary.iloc[:, cat_features].astype(str) 

# Predict using the trained model
Z = model.predict(X_decision_boundary)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train_subset[feature_x], X_train_subset[feature_y], c=y_train_classes, edgecolor='k', cmap='coolwarm')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("Decision Boundary - CatBoost")
plt.colorbar(label="Class")
plt.show()

##################### ROC AUC Curve #####################
from sklearn.metrics import roc_curve, auc

# Get probability predictions for the positive class (Churn = 1)
y_prob = model.predict_proba(X_test)[:, 1]  

# Compute the ROC curve values (False Positive Rate & True Positive Rate)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)  # Calculate the AUC score

# Print the AUC score for evaluation
print(f"AUC Score: {roc_auc:.4f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random model reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Interpretation tips:
# -AUC score close to 1 indicates a strong model performance.
# -AUC ~ 0.5 means the model is performing no better than random chance.

##################### SHAP Explanation #####################
# Install SHAP if not installed: (Run this in the terminal before execution)
# !pip install shap
import shap

# Create a SHAP explainer for the trained CatBoost model
explainer = shap.Explainer(model)

# Compute SHAP values for test data
shap_values = explainer(X_test)

# Summary plot (Overall feature impact on predictions)
shap.summary_plot(shap_values, X_test)
##Tips: 
#-The summary plot shows which features contribute the most to predictions.
#-Higher SHAP values indicate stronger influence on churn prediction.

#### Heatmap for SHAP values (shows feature interactions over instances)
shap.plots.heatmap(shap_values)
#Tip: -The heatmap visualizes how feature values influence the output across samples.


##################### Loss Curve (Training vs Validation) #####################
# Retrieve evaluation metrics from the trained model
evals_result = model.get_evals_result()

# Plot Training vs Validation Logloss
plt.figure(figsize=(8, 6))
plt.plot(evals_result['learn']['Logloss'], label='Train Logloss', color='blue')
plt.plot(evals_result['validation']['Logloss'], label='Validation Logloss', color='red')
plt.xlabel('Iterations')
plt.ylabel('Logloss')
plt.title('Training vs Validation Loss Curve')
plt.legend()
plt.grid()
plt.show()

# Tips for Interpretation of a Loss Curve:
# - If the validation loss stops decreasing while train loss keeps decreasing, the model might be overfitting.
# - Use `early_stopping_rounds` to prevent unnecessary training and improve generalization.
# - A gap between training and validation loss may indicate the need for regularization.


##################### Decision Tree Visualization #####################
from catboost import CatBoostClassifier  # (Already imported above, but kept for clarity)

# Select a specific tree index to visualize
tree_index = 18  # Choose a tree to visualize (tree indices start from 0)

# Generate a Graphviz object for the selected tree
graph = model.plot_tree(tree_idx=tree_index, pool=train_pool)

# Save the tree as a PDF file
graph.render("catboost_tree_18th", format="pdf", cleanup=True)