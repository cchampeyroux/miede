# -*- coding: utf-8 -*-
"""
Module de prédiction utilisant les modèles entraînés
"""

import sys
import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clustering import get_features_pdl

# ---------------------------------------------------
# Définition du modèle de réseau de neurones (même que dans classification.py)
# ---------------------------------------------------
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

# ---------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------
def load_models():
    """Charge les modèles entraînés et les préprocesseurs"""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    # Charger le modèle de régression logistique
    logreg_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    logreg_model = joblib.load(logreg_path)
    print(f"Modèle de régression logistique chargé : {logreg_path}")

    # Charger le modèle de réseau de neurones
    nn_model_path = os.path.join(models_dir, "neural_network_model.pth")
    model_config_path = os.path.join(models_dir, "model_config.pkl")
    imputer_path = os.path.join(models_dir, "imputer.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Charger la configuration du modèle
    config = joblib.load(model_config_path)
    input_size = config['input_size']

    # Recréer le modèle
    model = SimpleNN(input_size)
    model.load_state_dict(torch.load(nn_model_path, map_location=torch.device('cpu')))
    model.eval()

    # Charger les préprocesseurs
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    print(f"Modèle de réseau de neurones chargé : {nn_model_path}")
    print(f"Préprocesseurs chargés depuis : {imputer_path}, {scaler_path}")

    # Charger les colonnes de features
    feature_cols_path = os.path.join(models_dir, "feature_columns.pkl")
    feature_cols = joblib.load(feature_cols_path)
    print(f"Colonnes de features chargées : {feature_cols_path}")

    return logreg_model, model, imputer, scaler, feature_cols

# ---------------------------------------------------
# Fonction de prédiction
# ---------------------------------------------------
def predict_labels(features_df, model_type='both', threshold=0.5):
    """
    Prédit les labels (0 ou 1) pour de nouvelles données

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame contenant les features pour la prédiction
    model_type : str
        'logreg' pour régression logistique, 'nn' pour réseau de neurones, 'both' pour les deux
    threshold : float
        Seuil de décision (par défaut 0.5)

    Returns:
    --------
    dict : Résultats de prédiction
    """
    # Charger les modèles
    logreg_model, nn_model, imputer, scaler, feature_cols = load_models()

    # Vérifier que toutes les colonnes nécessaires sont présentes
    missing_cols = set(feature_cols) - set(features_df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les données : {missing_cols}")

    # Sélectionner seulement les colonnes de features
    X = features_df[feature_cols].copy()

    # Remplacer inf/-inf par NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    results = {}

    if model_type in ['logreg', 'both']:
        # Prédiction avec régression logistique
        print("\n=== Prédiction avec Régression Logistique ===")

        # Le pipeline gère l'imputation et le scaling automatiquement
        y_proba_logreg = logreg_model.predict_proba(X)[:, 1]
        y_pred_logreg = (y_proba_logreg >= threshold).astype(int)

        results['logreg'] = {
            'probabilities': y_proba_logreg,
            'predictions': y_pred_logreg,
            'threshold': threshold
        }

        print(f"Nombre de prédictions : {len(y_pred_logreg) if hasattr(y_pred_logreg, '__len__') else 1}")
        print(f"Classe 0 : {np.sum(y_pred_logreg == 0) if hasattr(y_pred_logreg, '__len__') else (1 if y_pred_logreg == 0 else 0)}")
        print(f"Classe 1 : {np.sum(y_pred_logreg == 1) if hasattr(y_pred_logreg, '__len__') else (1 if y_pred_logreg == 1 else 0)}")
        print(".3f")

    if model_type in ['nn', 'both']:
        # Prédiction avec réseau de neurones
        print("\n=== Prédiction avec Réseau de Neurones ===")

        # Imputation et scaling
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        # Conversion en tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Prédiction
        with torch.no_grad():
            y_proba_nn = nn_model(X_tensor).squeeze().numpy()

        # Seuil par défaut (0.5)
        threshold_nn = 0.5
        y_pred_nn = (y_proba_nn >= threshold).astype(int)

        results['nn'] = {
            'probabilities': y_proba_nn,
            'predictions': y_pred_nn,
            'threshold': threshold
        }

        print(f"Nombre de prédictions : {len(y_pred_nn) if hasattr(y_pred_nn, '__len__') else 1}")
        print(f"Classe 0 : {np.sum(y_pred_nn == 0) if hasattr(y_pred_nn, '__len__') else (1 if y_pred_nn == 0 else 0)}")
        print(f"Classe 1 : {np.sum(y_pred_nn == 1) if hasattr(y_pred_nn, '__len__') else (1 if y_pred_nn == 1 else 0)}")
        print(".3f")

    return results

# ---------------------------------------------------
# Exemple d'utilisation pour prédire sur de nouvelles données
# ---------------------------------------------------
def predict_single_example(features_dict, model_type='both', threshold=0.5):
    """
    Prédit le label pour un seul exemple

    Parameters:
    -----------
    features_dict : dict
        Dictionnaire contenant les valeurs des features
    model_type : str
        'logreg', 'nn', ou 'both'
    threshold : float
        Seuil de décision

    Returns:
    --------
    dict : Résultats de prédiction
    """
    # Convertir en DataFrame
    df = pd.DataFrame([features_dict])

    # Faire la prédiction
    results = predict_labels(df, model_type=model_type, threshold=threshold)

    return results

# ---------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------
if __name__ == "__main__":
    print("Chargement des features...")

    # Charger les features (comme dans classification.py)
    features_pdl = get_features_pdl()

    print(f"Features chargées : {features_pdl.shape}")
    print(f"Colonnes : {list(features_pdl.columns)}")

    # Faire des prédictions sur toutes les données
    try:
        results = predict_labels(features_pdl, model_type='both', threshold=0.5)

        # Afficher un résumé
        print("\n=== Résumé des prédictions ===")
        if 'logreg' in results:
            pred_logreg = results['logreg']['predictions']
            print(f"Régression logistique - Classe 0: {np.sum(pred_logreg == 0)}, Classe 1: {np.sum(pred_logreg == 1)}")

        if 'nn' in results:
            pred_nn = results['nn']['predictions']
            print(f"Réseau de neurones - Classe 0: {np.sum(pred_nn == 0)}, Classe 1: {np.sum(pred_nn == 1)}")

        # Optionnel : sauvegarder les résultats
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(output_dir, exist_ok=True)

        if 'logreg' in results:
            features_pdl['pred_logreg_proba'] = results['logreg']['probabilities']
            features_pdl['pred_logreg_class'] = results['logreg']['predictions']

        if 'nn' in results:
            features_pdl['pred_nn_proba'] = results['nn']['probabilities']
            features_pdl['pred_nn_class'] = results['nn']['predictions']

        output_path = os.path.join(output_dir, "predictions.csv")
        features_pdl.to_csv(output_path, index=False)
        print(f"\nRésultats sauvegardés dans : {output_path}")

        # Exemple de prédiction pour un seul exemple
        print("\n=== Exemple de prédiction pour un ID spécifique ===")
        example_id = features_pdl['ID'].iloc[0]  # Premier ID
        example_features = features_pdl.iloc[0].to_dict()

        single_results = predict_single_example(example_features, model_type='both', threshold=0.5)

        print(f"ID: {example_id}")
        if 'logreg' in single_results:
            proba = single_results['logreg']['probabilities']
            pred = single_results['logreg']['predictions']
            # Handle both array and scalar cases
            try:
                proba_val = proba[0]
                pred_val = pred[0]
            except (IndexError, TypeError):
                proba_val = proba
                pred_val = pred
            print(".3f")
        if 'nn' in single_results:
            proba = single_results['nn']['probabilities']
            pred = single_results['nn']['predictions']
            # Handle both array and scalar cases
            try:
                proba_val = proba[0]
                pred_val = pred[0]
            except (IndexError, TypeError):
                proba_val = proba
                pred_val = pred
            print(".3f")

    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        print("Assurez-vous d'avoir exécuté classification.py pour entraîner et sauvegarder les modèles.")

