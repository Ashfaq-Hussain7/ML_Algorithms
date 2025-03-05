# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()

# Create DataFrame for better visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display dataset overview
print(df.head())
print(df.info())

# Split into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']


# Check dataset distribution
sns.pairplot(df, hue='target', diag_kind='kde')
plt.show()

# Summary statistics
print(df.describe())

# Check for correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


from sklearn.feature_selection import SelectKBest, f_classif

# Select top 2 features based on ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Display selected features and scores
print("Feature Scores:", selector.scores_)
print("Selected Features Shape:", X_selected.shape)


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=rf.feature_importances_, y=X.columns)
plt.title('Feature Importance using Random Forest')
plt.show()

# Select top 2 important features
important_indices = np.argsort(rf.feature_importances_)[-2:]
X_rf_selected = X.iloc[:, important_indices]
print("Random Forest Selected Features:", X_rf_selected.columns.tolist())


from sklearn.svm import SVC
from sklearn.feature_selection import RFE

# Use SVM as the base model for RFE
svm = SVC(kernel="linear")
rfe = RFE(estimator=svm, n_features_to_select=2)
X_rfe_selected = rfe.fit_transform(X, y)

# Display selected features
print("Selected Features by RFE:", X.columns[rfe.support_])


from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# Function to evaluate model
def evaluate_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# Split original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("📊 Evaluation on Original Dataset (All Features):")
evaluate_model(X_train, X_test, y_train, y_test)


# Split dataset with SelectKBest features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

print("📊 Evaluation with SelectKBest Features:")
evaluate_model(X_train, X_test, y_train, y_test)


# Split dataset with Random Forest features
X_train, X_test, y_train, y_test = train_test_split(X_rf_selected, y, test_size=0.3, random_state=42)

print("📊 Evaluation with Random Forest Features:")
evaluate_model(X_train, X_test, y_train, y_test)


# Split dataset with RFE features
X_train, X_test, y_train, y_test = train_test_split(X_rfe_selected, y, test_size=0.3, random_state=42)

print("📊 Evaluation with RFE Features:")
evaluate_model(X_train, X_test, y_train, y_test)
