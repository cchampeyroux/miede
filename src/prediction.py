# -*- coding: utf-8 -*-
"""
Module de prédiction de consommation avec plusieurs modèles
Entraîne et compare : Régression Linéaire, ARIMA, LSTM
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clustering import get_features_pdl
from src.forecast import (
    load_consumption_data,
    get_pdl_timeseries,
    train_and_predict_models,
)

# ---------------------------------------------------
# Chargement des informations de clustering
# ---------------------------------------------------
def load_cluster_assignments():
    """Charge les assignations de clusters"""
    features_pdl = get_features_pdl()
    return features_pdl[['ID', 'cluster']].drop_duplicates().set_index('ID')

# ---------------------------------------------------
# Affichage des résultats détaillés
# ---------------------------------------------------
def display_model_results(result_dict):
    """Affiche les résultats de tous les modèles pour un PDL"""
    
    if 'error' in result_dict:
        print(f"❌ Erreur : {result_dict['error']}")
        return
    
    pdl_id = result_dict['pdl_id']
    last_date = result_dict['last_date']
    lookback = result_dict['lookback']
    data_points = result_dict['data_points']
    
    print(f"\n{'='*70}")
    print(f"PDL {pdl_id}")
    print(f"{'='*70}")
    print(f"📊 Données historiques : {data_points} jours")
    print(f"📈 Profondeur d'historique utilisée (lookback) : {lookback} jours")
    print(f"📅 Dernière date : {last_date}")
    
    print(f"\n{'PRÉDICTIONS POUR LES 7 JOURS SUIVANTS':-^70}")
    print(f"{'Date':<15} {'Régr. Lin.':<15} {'ARIMA':<15} {'LSTM':<15} {'Ensemble':<15}")
    print(f"{'-'*70}")
    
    # Récupérer les prédictions de chaque modèle
    models = result_dict['models']
    ensemble = result_dict.get('ensemble', {})
    
    pred_lr = models.get('linear_regression', {}).get('predictions', [None] * 7)
    pred_arima = models.get('arima', {}).get('predictions', [None] * 7)
    pred_lstm = models.get('lstm', {}).get('predictions', [None] * 7)
    pred_ensemble = ensemble.get('predictions', [None] * 7)
    
    for i, date in enumerate(result_dict['dates']):
        date_str = date.strftime('%Y-%m-%d')
        
        # Formater les prédictions (ou afficher erreur si disponible)
        lr_str = f"{pred_lr[i]:.0f}" if pred_lr[i] is not None else "N/A"
        arima_str = f"{pred_arima[i]:.0f}" if pred_arima[i] is not None else "N/A"
        lstm_str = f"{pred_lstm[i]:.0f}" if pred_lstm[i] is not None else "N/A"
        ens_str = f"{pred_ensemble[i]:.0f}" if pred_ensemble[i] is not None else "N/A"
        
        print(f"{date_str:<15} {lr_str:>14} {arima_str:>14} {lstm_str:>14} {ens_str:>14}")
    
    # Résumé statistique
    print(f"\n{'RÉSUMÉ STATISTIQUE':-^70}")
    
    def print_stats(pred_list, model_name):
        pred_list_valid = [p for p in pred_list if p is not None]
        if pred_list_valid:
            print(f"{model_name:<20} Moyenne: {np.mean(pred_list_valid):>10.0f} | Total 7j: {np.sum(pred_list_valid):>10.0f} kWh")
        else:
            print(f"{model_name:<20} Erreur during training")
    
    print_stats(pred_lr, "Régression Linéaire")
    print_stats(pred_arima, "ARIMA")
    print_stats(pred_lstm, "LSTM")
    print_stats(pred_ensemble, "Ensemble (moyenne)")
    
    # Afficher les erreurs éventuelles pour chaque modèle
    if any('error' in models.get(m, {}) for m in models):
        print(f"\n{'DÉTAILS DES ERREURS':-^70}")
        for model_name, model_result in models.items():
            if 'error' in model_result:
                print(f"  • {model_result.get('model_name', model_name)}: {model_result['error']}")

# ---------------------------------------------------
# Sauvegarde des résultats
# ---------------------------------------------------
def save_detailed_results(all_results):
    """Sauvegarde les prédictions de tous les modèles"""
    
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Créer un DataFrame avec tous les résultats
    all_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
        
        pdl_id = result['pdl_id']
        models = result['models']
        ensemble = result.get('ensemble', {})
        
        pred_lr = models.get('linear_regression', {}).get('predictions', [])
        pred_arima = models.get('arima', {}).get('predictions', [])
        pred_lstm = models.get('lstm', {}).get('predictions', [])
        pred_ensemble = ensemble.get('predictions', [])
        
        for i, date in enumerate(result['dates']):
            all_data.append({
                'ID': pdl_id,
                'date': date,
                'day_number': i + 1,
                'linear_regression_kwh': pred_lr[i] if i < len(pred_lr) else None,
                'arima_kwh': pred_arima[i] if i < len(pred_arima) else None,
                'lstm_kwh': pred_lstm[i] if i < len(pred_lstm) else None,
                'ensemble_kwh': pred_ensemble[i] if i < len(pred_ensemble) else None,
            })
    
    df_results = pd.DataFrame(all_data)
    
    # Sauvegarder les résultats détaillés
    output_path = os.path.join(results_dir, "forecasts_models_comparison.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Résultats sauvegardés : {output_path}")
    
    # Créer un résumé par PDL
    summary = df_results.groupby('ID').agg({
        'linear_regression_kwh': ['sum', 'mean'],
        'arima_kwh': ['sum', 'mean'],
        'lstm_kwh': ['sum', 'mean'],
        'ensemble_kwh': ['sum', 'mean'],
    }).round(2)
    
    summary_path = os.path.join(results_dir, "forecasts_summary.csv")
    summary.to_csv(summary_path)
    print(f"✓ Résumé sauvegardé : {summary_path}")

# ---------------------------------------------------
# Utilisation principale
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SYSTÈME DE PRÉDICTION DE CONSOMMATION ÉNERGÉTIQUE")
    print("Entraînement de modèles ML/DL : Régression Linéaire, ARIMA, LSTM")
    print("="*70)
    
    print("\nChargement des données...")
    df_consumption = load_consumption_data()
    cluster_info = load_cluster_assignments()
    
    print(f"[OK] Donnees chargees : {df_consumption.shape[0]} observations")
    print(f"[OK] Periode : {df_consumption['date'].min().date()} a {df_consumption['date'].max().date()}")
    print(f"[OK] Nombre de PDLs : {df_consumption['ID'].nunique()}")
    
    # Faire des prédictions pour quelques PDLs
    print("\n" + "="*70)
    print("PRÉDICTIONS SUR LES 7 JOURS SUIVANTS")
    print("="*70)
    
    # Nombre de PDLs à traiter (réduire pour plus de rapidité)
    num_pdls_to_train = 10  # Modifiez ce nombre pour plus/moins de PDLs
    
    sample_pdls = df_consumption['ID'].unique()[:num_pdls_to_train]
    all_results = []
    
    for i, pdl_id in enumerate(sample_pdls, 1):
        print(f"\n[{i}/{len(sample_pdls)}] Entraînement des modèles pour PDL {pdl_id}...")
        result = train_and_predict_models(pdl_id, df_consumption, days_ahead=7)
        all_results.append(result)
        
        # Afficher les résultats détaillés seulement pour les 3 premiers
        if i <= 3:
            display_model_results(result)
    
    # Sauvegarder les résultats traités
    print("\n" + "="*70)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*70)
    print(f"\nSauvegarde des prédictions pour {len(sample_pdls)} PDLs...")
    
    all_results_full = all_results
    
    save_detailed_results(all_results_full)
    
    print("\n" + "="*70)
    print("✓ Prédictions terminées !")
    print("="*70)

