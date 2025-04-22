#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:05:46 2025

@author: blakespradlin
"""
# load data and packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df21 = pd.read_csv("atp_matches_2021.csv")
df22 = pd.read_csv("atp_matches_2022.csv")
df23 = pd.read_csv("atp_matches_2023 (1).csv")
df24 = pd.read_csv("atp_matches_2024.csv")

print(df24.columns)

#%%
# define year cleaning function

def process_year(df):
    df['rank_diff'] = df['winner_rank'] - df['loser_rank']
    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['ace_diff'] = df['w_ace'] - df['l_ace']
    df['df_diff'] = df['w_df'] - df['l_df']
    df['svpt_diff'] = df['w_svpt'] - df['l_svpt']
    df['bpSaved_diff'] = df['w_bpSaved'] - df['l_bpSaved']
    df['bpFaced_diff'] = df['w_bpFaced'] - df['l_bpFaced']
    df['year'] = df['tourney_id'].astype(str).str[:4].astype(int)

    df = pd.get_dummies(df, columns=['surface'], drop_first=True)

    features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
        'bpSaved_diff', 'bpFaced_diff'] + [col for col in df.columns if col.startswith('surface_')]

    df = df[features].dropna()
    df['target'] = 1

    # Create flipped (loser) version
    df_loser = df.copy()
    for col in ['rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff', 'bpSaved_diff', 'bpFaced_diff']:
        df_loser[col] = -df_loser[col]
    df_loser['target'] = 0

    # Combine winner and loser rows
    df_full = pd.concat([df, df_loser], ignore_index=True)
    return df_full

#%%
# Run process_year and combine
df21 = process_year(df21)
df22 = process_year(df22)
df23 = process_year(df23)
df24 = process_year(df24)

df_all = pd.concat([df21, df22, df23, df24], ignore_index=True)

#%%

df_train = df_all[df_all['year'] < 2024]
df_test = df_all[df_all['year'] == 2024]

X_train = df_train.drop(columns=['target'])
y_train = df_train['target']

X_test = df_test.drop(columns=['target'])
y_test = df_test['target']

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict on test set
y_pred = logreg.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create confusion matrix plot
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Loss', 'Predicted Win'], yticklabels=['Actual Loss', 'Actual Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

plt.savefig('Log_Confusion_Matrix.png', dpi = 300)

#%%
# K Nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

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

plt.savefig('KKN Accuracy vs. Number of Neighbors.png', dpi = 300)

# Run after deciding whick k to use
knn = KNeighborsClassifier(n_neighbors=15)  # You can tweak 'k'
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)

# Evaluate
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))


#%%
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on your existing training data
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))









