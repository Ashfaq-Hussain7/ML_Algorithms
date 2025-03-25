import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.utils import all_estimators
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Get all classifiers from sklearn
estimators = all_estimators(type_filter='classifier')
results = []

# Train and evaluate all classifiers
for name, Classifier in estimators:
    try:
        model = Classifier()
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        results.append((name, mean_score))
    except Exception as e:
        continue

# Sort classifiers based on accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Select top 4 classifiers
top_classifiers = results[:4]
print("Top 4 classifiers:")
for name, score in top_classifiers:
    print(f"{name}: {score:.4f}")

# Instantiate top 4 classifiers
best_models = {name: all_estimators(type_filter='classifier')[i][1]() for i, (name, _) in enumerate(top_classifiers)}

# Ensemble methods with corrected parameter names
bagging = BaggingClassifier(estimator=best_models[top_classifiers[0][0]], n_estimators=10, random_state=42)
boosting = AdaBoostClassifier(estimator=best_models[top_classifiers[1][0]], n_estimators=50, random_state=42)
stacking = StackingClassifier(estimators=[(name, model) for name, model in best_models.items()], final_estimator=best_models[top_classifiers[2][0]])

ensemble_models = {'Bagging': bagging, 'Boosting': boosting, 'Stacking': stacking}

# Evaluate ensemble models
for name, model in ensemble_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classifier Report:")
    print(classification_report(y_test, y_pred))
