import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd
from app.models.notebook import NotebookInfo, ModelType, TaskType
from app.models.training import TrainingRequest, FeatureConfig

NOTEBOOKS_PATH = "./storage/notebooks"

def generate_notebook(df: pd.DataFrame, notebook_info: NotebookInfo, request: TrainingRequest) -> nbformat.NotebookNode:
    """
    Generate a Jupyter notebook for the specified model training configuration.
    """
    # Create the notebook with necessary cells
    notebook = new_notebook()
    notebook.cells = []
    
    # Add title and description
    notebook.cells.append(new_markdown_cell(f"""# Machine Learning Model: {request.model_type} for {request.task_type}
    
Generated on {notebook_info.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

This notebook demonstrates an end-to-end machine learning workflow for {request.task_type.lower()} using a {request.model_type.replace('_', ' ')} model.

**Dataset:** {request.dataset_id}
**Target Variable:** {request.target_column}
**Task Type:** {request.task_type}
"""))
    
    # Setup imports and dataset loading
    notebook.cells.append(new_code_cell("""# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
%matplotlib inline

# Display settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
"""))
    
    # Load the dataset
    notebook.cells.append(new_markdown_cell("## Data Loading and Overview"))
    notebook.cells.append(new_code_cell("""# Load the dataset
# In a production environment, this would load from a file path or database
# For this notebook, we're using the data that was uploaded
df = pd.read_csv('data.csv')  # Placeholder - this will be replaced with actual data

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\\nFirst few rows:")
display(df.head())

# Summary statistics
print(f"\\nSummary statistics:")
display(df.describe(include='all').T)

# Check for missing values
print(f"\\nMissing values per column:")
display(df.isnull().sum())
"""))
    
    # Exploratory data analysis
    notebook.cells.append(new_markdown_cell("## Exploratory Data Analysis"))
    
    # Add visualizations based on task type
    if request.task_type == TaskType.REGRESSION:
        notebook.cells.append(new_code_cell("""# Explore the target variable
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['%s'], bins=30, alpha=0.7)
plt.title('Distribution of Target Variable')
plt.xlabel('%s')

plt.subplot(1, 2, 2)
plt.boxplot(df['%s'])
plt.title('Boxplot of Target Variable')
plt.tight_layout()
plt.show()

# Correlation analysis
corr = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter plots for top correlated features
top_corr_features = corr['%s'].drop('%s').abs().sort_values(ascending=False).head(3).index
fig, axes = plt.subplots(1, len(top_corr_features), figsize=(18, 6))
for i, col_name in enumerate(top_corr_features):
    sns.regplot(x=col_name, y='%s', data=df, ax=axes[i])
    axes[i].set_title(f'{col_name} vs %s')
plt.tight_layout()
plt.show()
""" % (request.target_column, request.target_column, request.target_column, request.target_column, request.target_column, request.target_column, request.target_column)))
    
    elif request.task_type == TaskType.CLASSIFICATION:
        notebook.cells.append(new_code_cell("""# Explore the target variable
plt.figure(figsize=(10, 6))
target_counts = df['%s'].value_counts()
sns.barplot(x=target_counts.index, y=target_counts.values)
plt.title('Distribution of Target Classes')
plt.xlabel('%s')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation analysis for numerical features
numeric_df = df.select_dtypes(include=['number'])
if numeric_df.shape[1] > 1:  # Only if we have numeric features
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Feature distribution by class
numerical_features = df.select_dtypes(include=['number']).columns.tolist()
numerical_features = [f for f in numerical_features if f != '%s'][:3]  # Top 3 numerical features

if numerical_features:
    fig, axes = plt.subplots(len(numerical_features), 1, figsize=(12, 4*len(numerical_features)))
    if len(numerical_features) == 1:
        axes = [axes]  # Make axes iterable if only one feature
        
    for i, feature in enumerate(numerical_features):
        sns.boxplot(x='%s', y=feature, data=df, ax=axes[i])
        axes[i].set_title(f'{feature} by %s')
    
    plt.tight_layout()
    plt.show()
""" % (request.target_column, request.target_column, request.target_column, request.target_column, request.target_column)))
    
    # Model imports section
    model_import_code = """# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
"""
    
    if request.task_type == TaskType.REGRESSION:
        if request.model_type == ModelType.LINEAR_REGRESSION:
            model_import_code += "from sklearn.linear_model import LinearRegression\n"
        elif request.model_type == ModelType.DECISION_TREE:
            model_import_code += "from sklearn.tree import DecisionTreeRegressor\n"
        elif request.model_type == ModelType.RANDOM_FOREST:
            model_import_code += "from sklearn.ensemble import RandomForestRegressor\n"
        elif request.model_type == ModelType.GRADIENT_BOOSTING:
            model_import_code += "from sklearn.ensemble import GradientBoostingRegressor\n"
        elif request.model_type == ModelType.SVM:
            model_import_code += "from sklearn.svm import SVR\n"
        elif request.model_type == ModelType.KNN:
            model_import_code += "from sklearn.neighbors import KNeighborsRegressor\n"
    elif request.task_type == TaskType.CLASSIFICATION:
        if request.model_type == ModelType.LOGISTIC_REGRESSION:
            model_import_code += "from sklearn.linear_model import LogisticRegression\n"
        elif request.model_type == ModelType.DECISION_TREE:
            model_import_code += "from sklearn.tree import DecisionTreeClassifier\n"
        elif request.model_type == ModelType.RANDOM_FOREST:
            model_import_code += "from sklearn.ensemble import RandomForestClassifier\n"
        elif request.model_type == ModelType.GRADIENT_BOOSTING:
            model_import_code += "from sklearn.ensemble import GradientBoostingClassifier\n"
        elif request.model_type == ModelType.SVM:
            model_import_code += "from sklearn.svm import SVC\n"
        elif request.model_type == ModelType.KNN:
            model_import_code += "from sklearn.neighbors import KNeighborsClassifier\n"
    
    model_import_code += """from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
"""
    
    notebook.cells.append(new_code_cell(model_import_code))
    
    # Data preprocessing and split
    notebook.cells.append(new_markdown_cell("## Data Preprocessing and Split"))
    
    data_preprocessing_code = """# Split data into features and target
X = df.drop(columns=['%s'])
y = df['%s']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Check for complex object features that may cause issues
for col in categorical_features:
    sample = df[col].iloc[0]
    if isinstance(sample, dict) or isinstance(sample, list):
        print(f"Warning: Column {col} contains complex objects which may not work with standard transformations.")
        print(f"Sample value: {sample}")
        print(f"Consider extracting specific fields from this object or excluding it from your model.")
""" % (request.target_column, request.target_column)
    
    # Add feature transformations based on configurations with proper indentation
    transforms = []
    for feature_config in request.features:
        if not feature_config.use:
            continue
            
        feature_name = feature_config.name
        transform_type = feature_config.transform
        
        if transform_type == "log":
            transforms.append("# Apply log transformation to %s" % feature_name)
            transforms.append("if '%s' in numerical_features and (df['%s'] > 0).all():" % (feature_name, feature_name))
            transforms.append("    X['%s_log'] = np.log1p(X['%s'])" % (feature_name, feature_name))
            transforms.append("    print(f\"Created log transform: %s_log\")" % feature_name)
        elif transform_type == "scale":
            transforms.append("# Apply scaling to %s" % feature_name)
            transforms.append("if '%s' in numerical_features:" % feature_name)
            transforms.append("    scaler = StandardScaler()")
            transforms.append("    X['%s_scaled'] = scaler.fit_transform(X['%s'].values.reshape(-1, 1))" % (feature_name, feature_name))
            transforms.append("    print(f\"Created scaled feature: %s_scaled\")" % feature_name)
        elif transform_type == "onehot":
            transforms.append("# Apply one-hot encoding to %s" % feature_name)
            transforms.append("if '%s' in categorical_features:" % feature_name)
            transforms.append("    # Check if the feature contains complex values that need special handling")
            transforms.append("    if X['%s'].apply(lambda x: isinstance(x, (dict, list))).any():" % feature_name)
            transforms.append("        print(f\"Warning: Cannot apply one-hot encoding to complex objects in %s\")" % feature_name)
            transforms.append("        print(f\"Converting to string representation first\")")
            transforms.append("        X['%s'] = X['%s'].astype(str)" % (feature_name, feature_name))
            transforms.append("    try:")
            transforms.append("        onehot = pd.get_dummies(X['%s'], prefix='%s')" % (feature_name, feature_name))
            transforms.append("        X = pd.concat([X.drop('%s', axis=1), onehot], axis=1)" % feature_name)
            transforms.append("        print(f\"Created one-hot encoding for: %s\")" % feature_name)
            transforms.append("    except Exception as e:")
            transforms.append("        print(f\"Error one-hot encoding %s: {e}\")" % feature_name)
    
    if transforms:
        data_preprocessing_code += "\n" + "\n".join(transforms)
    
    # Add the train-test split after all transformations with proper indentation
    data_preprocessing_code += """

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=%s, random_state=%s
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
""" % (request.test_size, request.random_state)
    
    notebook.cells.append(new_code_cell(data_preprocessing_code))
    
    # Model training section
    notebook.cells.append(new_markdown_cell("## Model Training"))
    
    # Create model initialization code with proper indentation
    model_code = "# Initialize the model\n"
    if request.task_type == TaskType.REGRESSION:
        if request.model_type == ModelType.LINEAR_REGRESSION:
            model_code += "model = LinearRegression()\n"
        elif request.model_type == ModelType.DECISION_TREE:
            model_code += "model = DecisionTreeRegressor(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.RANDOM_FOREST:
            model_code += "model = RandomForestRegressor(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.GRADIENT_BOOSTING:
            model_code += "model = GradientBoostingRegressor(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.SVM:
            model_code += "model = SVR()\n"
        elif request.model_type == ModelType.KNN:
            model_code += "model = KNeighborsRegressor()\n"
    elif request.task_type == TaskType.CLASSIFICATION:
        if request.model_type == ModelType.LOGISTIC_REGRESSION:
            model_code += "model = LogisticRegression(random_state=%d, max_iter=1000)\n" % request.random_state
        elif request.model_type == ModelType.DECISION_TREE:
            model_code += "model = DecisionTreeClassifier(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.RANDOM_FOREST:
            model_code += "model = RandomForestClassifier(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.GRADIENT_BOOSTING:
            model_code += "model = GradientBoostingClassifier(random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.SVM:
            model_code += "model = SVC(probability=True, random_state=%d)\n" % request.random_state
        elif request.model_type == ModelType.KNN:
            model_code += "model = KNeighborsClassifier()\n"
    
    # Add custom model parameters if specified
    if request.model_parameters:
        model_code += "\n# Set custom model parameters\n"
        for param_name, param_value in request.model_parameters.items():
            model_code += "# model.set_params(%s=%s)\n" % (param_name, param_value)
    
    # Add pipeline creation and fitting
    model_code += """
# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

print("Model training complete!")
"""
    
    notebook.cells.append(new_code_cell(model_code))
    
    # Model evaluation
    notebook.cells.append(new_markdown_cell("## Model Evaluation"))
    
    eval_code = """# Make predictions on the test set
y_pred = pipeline.predict(X_test)

"""
    
    # Add metrics based on task type
    if request.task_type == TaskType.REGRESSION:
        eval_code += """# Evaluate regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()
"""
    elif request.task_type == TaskType.CLASSIFICATION:
        eval_code += """# Evaluate classification metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC curve and AUC (for binary classification)
if len(np.unique(y_test)) == 2:
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
        print(f"\\nAUC: {auc:.4f}")
    except:
        print("Could not calculate ROC curve and AUC.")
"""
    
    notebook.cells.append(new_code_cell(eval_code))
    
    # Feature importance visualization
    notebook.cells.append(new_markdown_cell("## Feature Importance"))
    
    feature_importance_code = """# Try to get feature importance if available
try:
    if hasattr(pipeline['model'], 'feature_importances_'):
        # Get feature names from the preprocessor
        feature_names = []
        for name, trans, cols in pipeline['preprocessor'].transformers_:
            if name == 'cat' and cols:
                # Get the one-hot encoded feature names
                cat_features = trans.named_steps['onehot'].get_feature_names_out(cols)
                feature_names.extend(cat_features)
            else:
                feature_names.extend(cols)
        
        # Get feature importances from the model
        importances = pipeline['model'].feature_importances_
        
        # Create a dataframe with feature importances
        if len(feature_names) == len(importances):
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
            print(feature_importance_df.head(15))
    else:
        print("Feature importances not available for this model.")
except Exception as e:
    print(f"Could not compute feature importances: {e}")
"""
    
    notebook.cells.append(new_code_cell(feature_importance_code))
    
    # Add final summary section
    notebook.cells.append(new_markdown_cell("## Summary and Next Steps"))
    
    summary_code = """# Summary of the model
print("Model Training Summary:")
print(f"Dataset: {len(X_train) + len(X_test)} samples ({len(X_train)} train, {len(X_test)} test)")
print(f"Task Type: %s")
print(f"Model Type: %s")
print(f"Features Used: {len(X.columns)}")

print("\\nNext Steps:")
print("1. Try different feature transformations to improve performance")
print("2. Experiment with hyperparameter tuning to find optimal model settings")
print("3. Consider feature selection to focus on the most important variables")
print("4. For deployment, save the model using joblib or pickle")
print("5. Monitor model performance over time and retrain as needed")
""" % (request.task_type, request.model_type)
    
    notebook.cells.append(new_code_cell(summary_code))
    
    # SHAP values
    notebook.cells.append(new_markdown_cell("## SHAP Values for Model Explainability"))
    
    shap_code = """# Try to compute SHAP values for model explainability
try:
    import shap
    
    # Sample a subset of the test data for SHAP analysis (for performance)
    n_samples = min(100, X_test.shape[0])
    X_sample = X_test.iloc[:n_samples]
    
    # Create a SHAP explainer
    try:
        # For sklearn models
        explainer = shap.Explainer(pipeline['model'], pipeline['preprocessor'].transform(X_sample))
        shap_values = explainer(pipeline['preprocessor'].transform(X_sample))
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, pipeline['preprocessor'].transform(X_sample))
        plt.tight_layout()
    except Exception as shap_error:
        print(f"First SHAP approach failed: {shap_error}")
        
        # Try alternative approach for tree-based models
        if hasattr(pipeline['model'], 'feature_importances_'):
            explainer = shap.TreeExplainer(pipeline['model'])
            # Transform the data first
            X_transformed = pipeline['preprocessor'].transform(X_sample)
            shap_values = explainer.shap_values(X_transformed)
            
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):  # For multi-class classification
                shap.summary_plot(shap_values[0], X_transformed)
            else:  # For regression or binary classification
                shap.summary_plot(shap_values, X_transformed)
            plt.tight_layout()
        else:
            print("Model type not supported for detailed SHAP analysis")
            
    plt.show()
except Exception as e:
    print(f"Could not compute SHAP values: {e}")
    print("Note: SHAP analysis requires additional setup in some environments.")
    print("If you want to use SHAP, try installing it separately with: pip install shap")
"""
    
    notebook.cells.append(new_code_cell(shap_code))
    
    # Model export
    notebook.cells.append(new_markdown_cell("## Model Export"))
    
    export_code = """# Save the trained model to a file
import joblib

try:
    # Save the pipeline (includes preprocessor and model)
    joblib.dump(pipeline, 'trained_model_pipeline.joblib')
    print("Model pipeline saved successfully!")
    
    # How to load the model
    print("\\nTo load and use this model in another script:")
    print("import joblib")
    print("loaded_pipeline = joblib.load('trained_model_pipeline.joblib')")
    print("predictions = loaded_pipeline.predict(new_data)")
except Exception as e:
    print(f"Error saving model: {e}")
"""
    
    notebook.cells.append(new_code_cell(export_code))
    
    # Add conclusion
    notebook.cells.append(new_markdown_cell("## Conclusion"))
    
    conclusion_md = f"""This notebook demonstrated how to:

1. Load and explore the dataset
2. Preprocess the data for machine learning
3. Train a {request.model_type} model for {request.task_type}
4. Evaluate the model's performance 
5. Understand feature importance and model explainability
6. Export the model for deployment

The model can be improved by:

* Feature engineering and selection
* Hyperparameter tuning
* Trying different algorithms
* Collecting more data
* Addressing class imbalance (if present)
"""
    
    notebook.cells.append(new_markdown_cell(conclusion_md))
    
    return notebook

def save_notebook(notebook_id: str, notebook: nbformat.NotebookNode) -> str:
    """
    Save a Jupyter notebook to the storage directory.
    """
    os.makedirs(NOTEBOOKS_PATH, exist_ok=True)
    
    notebook_path = os.path.join(NOTEBOOKS_PATH, f"{notebook_id}.ipynb")
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    return notebook_path 