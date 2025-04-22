#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:20:00 2025

@author: blakespradlin
"""

import pandas as pd

df21 = pd.read_csv("atp_matches_2021.csv")
df22 = pd.read_csv("atp_matches_2022.csv")
df23 = pd.read_csv("atp_matches_2023 (1).csv")
df24 = pd.read_csv("atp_matches_2024.csv")

print(df24.columns)

#%%
# Creating variables for 2021
df21['rank_diff'] = df21['winner_rank'] - df21['loser_rank']
df21['age_diff'] = df21['winner_age'] - df21['loser_age']
df21['ace_diff'] = df21['w_ace'] - df21['l_ace']
df21['df_diff'] = df21['w_df'] - df21['l_df']
df21['svpt_diff'] = df21['w_svpt'] - df21['l_svpt']
df21['bpSaved_diff'] = df21['w_bpSaved'] - df21['l_bpSaved']
df21['bpFaced_diff'] = df21['w_bpFaced'] - df21['l_bpFaced']
df21['year'] = df21['tourney_id'].astype(str).str[:4].astype(int)

# creating numeric surface variable
df21 = pd.get_dummies(df21, columns=['surface'], drop_first=True)

#setting target winner prediction
df21['target'] = 1

# Keep only columns
features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
    'bpSaved_diff', 'bpFaced_diff'] + [col for col in df21.columns if col.startswith('surface_')]

df21 = df21[features + ['target']].dropna()
df21.head()

#%%
# Creating variables for 2022
df22['rank_diff'] = df22['winner_rank'] - df22['loser_rank']
df22['age_diff'] = df22['winner_age'] - df22['loser_age']
df22['ace_diff'] = df22['w_ace'] - df22['l_ace']
df22['df_diff'] = df22['w_df'] - df22['l_df']
df22['svpt_diff'] = df22['w_svpt'] - df22['l_svpt']
df22['bpSaved_diff'] = df22['w_bpSaved'] - df22['l_bpSaved']
df22['bpFaced_diff'] = df22['w_bpFaced'] - df22['l_bpFaced']
df22['year'] = df22['tourney_id'].astype(str).str[:4].astype(int)


# creating numeric surface variable
df22 = pd.get_dummies(df22, columns=['surface'], drop_first=True)

#setting target winner prediction
df22['target'] = 1

# Keep only columns
features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
    'bpSaved_diff', 'bpFaced_diff'] + [col for col in df22.columns if col.startswith('surface_')]

df22 = df22[features + ['target']].dropna()
df22.head()

#%%
# Creating variables for 2023
df23['rank_diff'] = df23['winner_rank'] - df23['loser_rank']
df23['age_diff'] = df23['winner_age'] - df23['loser_age']
df23['ace_diff'] = df23['w_ace'] - df23['l_ace']
df23['df_diff'] = df23['w_df'] - df23['l_df']
df23['svpt_diff'] = df23['w_svpt'] - df23['l_svpt']
df23['bpSaved_diff'] = df23['w_bpSaved'] - df23['l_bpSaved']
df23['bpFaced_diff'] = df23['w_bpFaced'] - df23['l_bpFaced']
df23['year'] = df23['tourney_id'].astype(str).str[:4].astype(int)

# creating numeric surface variable
df23 = pd.get_dummies(df23, columns=['surface'], drop_first=True)

#setting target winner prediction
df23['target'] = 1

# Keep only columns
features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
    'bpSaved_diff', 'bpFaced_diff'] + [col for col in df23.columns if col.startswith('surface_')]

df23 = df23[features + ['target']].dropna()
df23.head()

#%%
# Creating variables for 2024
df24['rank_diff'] = df24['winner_rank'] - df24['loser_rank']
df24['age_diff'] = df24['winner_age'] - df24['loser_age']
df24['ace_diff'] = df24['w_ace'] - df24['l_ace']
df24['df_diff'] = df24['w_df'] - df24['l_df']
df24['svpt_diff'] = df24['w_svpt'] - df24['l_svpt']
df24['bpSaved_diff'] = df24['w_bpSaved'] - df24['l_bpSaved']
df24['bpFaced_diff'] = df24['w_bpFaced'] - df24['l_bpFaced']
df24['year'] = df24['tourney_id'].astype(str).str[:4].astype(int)

# creating numeric surface variable
df24 = pd.get_dummies(df24, columns=['surface'], drop_first=True)

#setting target winner prediction
df24['target'] = 1

# Keep only columns
features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
    'bpSaved_diff', 'bpFaced_diff'] + [col for col in df24.columns if col.startswith('surface_')]

df24 = df24[features + ['target']].dropna()
df24.head()

#%%
# Combine all dataframes
df_all = pd.concat([df21, df22, df23, df24], ignore_index=True)

# Split into test and training datasets
df_train = df_all[df_all['year'] < 2024]
df_test = df_all[df_all['year'] == 2024]

#%%
X_train = df_train[features].dropna()
y_train = df_train['target']

X_test = df_test[features].dropna()
y_test = df_test['target']










