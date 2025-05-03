#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:29:57 2025

@author: blakespradlin
"""

import pandas as pd

# Define cleaning data function
def process_year(df):
    df['rank_diff'] = df['winner_rank'] - df['loser_rank']
    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['ace_diff'] = df['w_ace'] - df['l_ace']
    df['df_diff'] = df['w_df'] - df['l_df']
    df['svpt_diff'] = df['w_svpt'] - df['l_svpt']
    df['bpSaved_diff'] = df['w_bpSaved'] - df['l_bpSaved']
    df['bpFaced_diff'] = df['w_bpFaced'] - df['l_bpFaced']
    df['year'] = df['tourney_id'].astype(str).str[:4].astype(int)
    
    # Create a dummy variable for the surface type
    df = pd.get_dummies(df, columns=['surface'], drop_first=True)
    
    #Set features we want to keep
    features = ['year', 'rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff',
                'bpSaved_diff', 'bpFaced_diff'] + [col for col in df.columns if col.startswith('surface_')]
    
    #Drop data we don't need
    df = df[features].dropna()
    df['target'] = 1

    # Flipped (loser) version to balance data
    df_loser = df.copy()
    for col in ['rank_diff', 'age_diff', 'ace_diff', 'df_diff', 'svpt_diff', 'bpSaved_diff', 'bpFaced_diff']:
        df_loser[col] = -df_loser[col]
    df_loser['target'] = 0

    return pd.concat([df, df_loser], ignore_index=True)

#%%
# Define loading data to clean
def load_and_clean_data():
    df21 = process_year(pd.read_csv("atp_matches_2021.csv"))
    df22 = process_year(pd.read_csv("atp_matches_2022.csv"))
    df23 = process_year(pd.read_csv("atp_matches_2023.csv"))
    df24 = process_year(pd.read_csv("atp_matches_2024.csv"))

    #Combine all years
    df_all = pd.concat([df21, df22, df23, df24], ignore_index=True)

    # Set 2024 as test and the rest as training
    df_train = df_all[df_all['year'] < 2024]
    df_test = df_all[df_all['year'] == 2024]

    X_train = df_train.drop(columns=['target'])
    y_train = df_train['target']
    X_test = df_test.drop(columns=['target'])
    y_test = df_test['target']

    return X_train, X_test, y_train, y_test