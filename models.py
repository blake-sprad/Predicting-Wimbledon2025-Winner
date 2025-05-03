#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:33:48 2025

@author: blakespradlin
"""

#Import all models
from load_and_clean import load_and_clean_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load cleaned data
X_train, X_test, y_train, y_test = load_and_clean_data()

#%%
# Log regression

# Train data
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Evaluate
log_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)

#Print evals
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot ROC and AUC curve
y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Reg ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('logreg_roc_curve.png', dpi=300)
plt.show()

#%%
# K Nearest neighbors

#Test which k value is best
# Range of k values to test
k_values = list(range(1, 26))
accuracies = []

# Loop through k values and store accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='dodgerblue')
plt.xticks(k_values)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy on Test Set')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig('KNN_Accuracy_vs_Number_of_Neighbors.png', dpi = 300)

# Run after deciding whick k to use
knn = KNeighborsClassifier(n_neighbors=15)  # You can tweak 'k'
knn.fit(X_train, y_train)

# Predict on test
y_pred_knn = knn.predict(X_test)

# Evaluate
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

#%%
# RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on existing training data
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = rf.feature_importances_
features = X_train.columns

# Create a dataframe for plotting feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='RdGy')
plt.title('Feature Importance - Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=300)
plt.show()

estimator = rf.estimators_[0]

# Plot the tree
plt.figure(figsize=(20, 10))  # Wider figure for readability
plot_tree(estimator, 
          feature_names=X_train.columns, 
          class_names=['Loss', 'Win'], 
          filled=True, 
          max_depth=3,  # Limit depth to avoid complex plots
          fontsize=10)

plt.title("Random Forest - First Tree (Max Depth 3)")
plt.tight_layout()
plt.savefig('rf_first_tree.png', dpi=300)
plt.show()

#%%
# Naive Bayes

# train the model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict on test data
nb_preds = nb.predict(X_test)

# Evaluate accuracy
nb_accuracy = accuracy_score(y_test, nb_preds)
print("Naive Bayes Accuracy:", nb_accuracy)

y_probs = nb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("naive_bayes_roc_curve.png", dpi=300)
plt.show()

#%%
# Support Vector Machine

# train the model
svm = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Predict on test data
svm_preds = svm.predict(X_test)

# Evaluate
svm_accuracy = accuracy_score(y_test, svm_preds)
print("SVM Accuracy:", svm_accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_preds))
print("Classification Report:\n", classification_report(y_test, svm_preds))

svm_preds = svm.predict(X_test)
cm = confusion_matrix(y_test, svm_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Loss', 'Win'])

disp.plot(cmap='Purples')
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("svm_conf_matrix.png", dpi=300)
plt.show()

#%%
# Plot accuracies against each other
log_accuracy = accuracy_score(y_test, logreg.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
nb_accuracy = accuracy_score(y_test, nb.predict(X_test))
svm_accuracy = accuracy_score(y_test, svm.predict(X_test))

# Labels and values
model_names = ['Logistic Regression', 'K-Nearest Neighbors', 'Random Forest', 'Naive Bayes', 'Support Vector Machine']
accuracies = [log_accuracy, knn_accuracy, rf_accuracy, nb_accuracy, svm_accuracy]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=model_names, palette='Set2')
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1)

# Add value labels
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center')

plt.tight_layout()
plt.savefig('Model Accuracy Comparison All Models.png', dpi=300)
plt.show()