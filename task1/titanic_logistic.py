"""
Titanic Survival Prediction using Logistic Regression

This script loads the Titanic dataset, preprocesses it, trains a logistic regression model,
and evaluates its performance with visualizations.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('titanic.csv')

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode Sex (direct mapping for binary)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked (nominal, no order)
df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb', drop_first=True)  # Drop_first avoids multicollinearity

# Feature engineering: FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Drop original SibSp/Parch to avoid redundancy
df = df.drop(['SibSp', 'Parch'], axis=1)

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify for balanced classes
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Enhanced model with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Cross-validation for robustness
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

# Optional: Hyperparameter tuning
# param_grid = {'C': [0.01, 0.1, 1, 10]}
# grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# grid.fit(X_train, y_train)
# model = grid.best_estimator_
# print(f'Best Params: {grid.best_params_}')

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Visualizations
# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()