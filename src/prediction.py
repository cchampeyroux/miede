# -*- coding: utf-8 -*-
#prediction

import sys
import os

print(sys.executable)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_src = "daily.csv"
path_parent = "../daily.csv"

import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.clustering import get_features_pdl


# On teste lequel des deux fonctionne pour ne plus avoir d'erreur
if os.path.exists(path_src):
    final_path = path_src
else :
    final_path = path_parent
#-------------------------------------------
# 1) Charger les labels (résultat du clustering)
# ---------------------------------------------------

df = pd.read_csv(final_path)
feature_cols = [
    "active_day_rate", "n_runs", "mean_run_len", "max_run_len",
    "mean_gap_len", "max_gap_len",
    "mean_daily_kwh", "p95_daily_kwh", "cv_daily_kwh",
    "active_rate_weekday", "active_rate_weekend",
    "mean_kwh_weekday", "mean_kwh_weekend",
    "winter_minus_summer", "seasonality_amp",
    "r_global", "r_mid", "r_summer", "r_winter",
]

# ---------------------------------------------------
# 3) Jointure features + labels
#    features_pdl contient pdl_id ; labels contient id
# ---------------------------------------------------
features_pdl = get_features_pdl()

df_model = (
    features_pdl.merge(
        labels[["id", "label"]],
        left_on="ID",
        right_on="id",
        how="inner"
    )
    .copy()
)

df['date'] = pd.to_datetime(df['date'])
# [1704875583, 6674572658, 9993623468, 10607320546, 11239534806, 14953875748, 16277393756]
# Trouver les ID des listes. 
liste_ids = df['ID'].unique().tolist()
print(f"\nListe des IDs uniques : {liste_ids}")

df_copy=df.copy()

df_copy=df_copy[df_copy["ID"]==6674572658]
#Moyenne glissante
window_day=3
df_copy['moyenne_glissante'] = df_copy['daily_kwh'].rolling(window=window_day).mean()
# 2. Création du graphique

plt.figure(figsize=(10, 6))
plt.scatter(df_copy['date'], df_copy['moyenne_glissante'], marker='o', linestyle='-', color='b', linewidth=2)

# 3. Personnalisation
plt.title('Évolution de la consommation quotidienne (kWh)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Consommation (kWh)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# Affichage
plt.show()

