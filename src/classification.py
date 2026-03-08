# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:57:08 2026

@author: cleli
"""

#classification

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.clustering import get_features_pdl

# helper for label I/O --------------------------------------------------
# we use joblib.Memory to persist a cache on disk so that loading
# sizeable CSVs/parquets is fast across separate Python sessions:
from joblib import Memory

# cache directory at project root (ignored by git via .gitignore)
_cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
memory = Memory(_cache_dir, verbose=0)

_LABEL_PATHS = (
    "data/RES2-6-9-labels.csv",
    "../data/RES2-6-9-labels.csv",
)


def _locate_labels_file() -> str:
    """Return the first existing path or raise FileNotFoundError."""
    for p in _LABEL_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("labels source file not found")


@memory.cache
def load_labels() -> pd.DataFrame:
    """Load the label file and return a DataFrame.

    The first time this function is called the CSV is read and the result
    is saved to the joblib cache; later calls (even in new Python
    processes) will reuse the cached DataFrame unless the source file has
    changed.  This gives us a *persistent* cache across runs.
    """
    source = _locate_labels_file()
    print("Reading labels from:", source)
    df = pd.read_csv(source)

    # to make subsequent loads fast we also dump a parquet copy nearby
    parquet_path = os.path.join(os.path.dirname(source), "RES2-6-9-labels.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:  # pragma: no cover - best effort
        pass
    return df


# load dataset -------------------------------------------------------------
labels = load_labels()

# ---------------------------------------------------
# 2) Définir les features (issues du notebook)
# ---------------------------------------------------
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


#c'est bon jusqu'ici

# Sécurisation (NaN / inf)
X = df_model[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df_model["label"].astype(int).copy()

print("Shape X:", X.shape)
print("Répartition y:\n", y.value_counts(normalize=False))

# ---------------------------------------------------
# 4) Split train/test (stratifié car classes déséquilibrées)
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("y_train:\n", y_train.value_counts())
print("y_test:\n", y_test.value_counts())




# ---------------------------------------------------
# 5) Fonction d'évaluation
# ---------------------------------------------------
def eval_classification(y_true, y_pred, model_name="modèle"):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== {model_name} ===")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {"model": model_name, "precision": p, "recall": r, "f1": f1}




# ---------------------------------------------------
# 6) Fonctions de métriques
# ---------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def print_eval(y_true, y_pred, model_name="modèle"):
    m = compute_metrics(y_true, y_pred)
    print(f"\n=== {model_name} ===")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print("\nMatrice de confusion")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return m

def apply_threshold(scores, threshold=0.5):
    return (scores >= threshold).astype(int)


# ---------------------------------------------------
# Modèle : Régression logistique
# ---------------------------------------------------
logreg_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight="balanced",   # important pour RS minoritaire
        solver="liblinear"         # robuste sur petits/moyens datasets
    ))
])

# Entraînement
logreg_model.fit(X_train, y_train)

# Probabilités de la classe 1 (RS)
y_score_logreg = logreg_model.predict_proba(X_test)[:, 1]

# Seuil par défaut (0.5)
threshold_logreg = 0.5
y_pred_logreg = apply_threshold(y_score_logreg, threshold=threshold_logreg)

# Évaluation
res_logreg = print_eval(y_test, y_pred_logreg, "Régression logistique")




#option : on peut ajuster le seuil selon l'objectif métier (si on veut rater moins de RS 0.35, si on veut plus de précision 0.65) )
for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
    y_pred_tmp = apply_threshold(y_score_logreg, threshold=th)
    m = compute_metrics(y_test, y_pred_tmp)
    print(f"Seuil={th:.2f} | Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1']:.3f}")




# ---------------------------------------------------
# Modèle : Réseau de neurones simple
# ---------------------------------------------------

# Préparation des données pour PyTorch
# Imputation et scaling des données
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Conversion en tensors PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Création des DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Définition du modèle
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialisation du modèle
input_size = X_train_scaled.shape[1]
model = SimpleNN(input_size)

# Fonction de perte et optimiseur
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Évaluation
model.eval()
with torch.no_grad():
    y_score_nn = model(X_test_tensor).squeeze().numpy()

# Seuil par défaut (0.5)
threshold_nn = 0.5
y_pred_nn = apply_threshold(y_score_nn, threshold=threshold_nn)

# Évaluation
res_nn = print_eval(y_test, y_pred_nn, "Réseau de neurones simple")

# Ajustement du seuil
for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
    y_pred_tmp = apply_threshold(y_score_nn, threshold=th)
    m = compute_metrics(y_test, y_pred_tmp)
    print(f"Seuil={th:.2f} | Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1']:.3f}")
