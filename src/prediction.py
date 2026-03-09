# -*- coding: utf-8 -*-
"""
Module de prédiction de consommation amélioré
Utilise le clustering et les données historiques pour faire du forecasting
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clustering import get_features_pdl
from src.forecast import (
    load_consumption_data,
    get_pdl_timeseries,
    forecast_consumption_lstm,
    forecast_consumption_trend,
)

# ---------------------------------------------------
# Chargement des informations de clustering
# ---------------------------------------------------
def load_cluster_assignments():
    """Charge les assignations de clusters"""
    features_pdl = get_features_pdl()
    return features_pdl[['ID', 'cluster']].drop_duplicates().set_index('ID')

# ---------------------------------------------------
# Prédiction avec clustering
# ---------------------------------------------------
def forecast_with_clustering(pdl_id, df_consumption, cluster_info=None, days_ahead=7):
    """
    Fait une prédiction en tenant compte du cluster du PDL
    
    Parameters:
    -----------
    pdl_id : int
        ID du PDL
    df_consumption : pd.DataFrame
        Données de consommation
    cluster_info : pd.DataFrame
        Info sur les clusters
    days_ahead : int
        Nombre de jours à prédire
    
    Returns:
    --------
    dict : Prédictions enrichies
    """
    # Récupérer la prédiction de base
    result = forecast_consumption_trend(pdl_id, df_consumption, days_ahead=days_ahead)
    
    if 'error' in result:
        return result
    
    # Ajouter l'info de cluster
    if cluster_info is not None and pdl_id in cluster_info.index:
        cluster = cluster_info.loc[pdl_id, 'cluster']
        result['cluster'] = cluster
        
        # Ajuster les prédictions selon le cluster
        cluster_adjustment = {
            0: 1.05, 1: 0.95, 2: 1.02, 3: 0.98,
            4: 1.00, 5: 1.03, 6: 0.97, 7: 1.01,
        }
        
        adjustment = cluster_adjustment.get(cluster, 1.0)
        result['predictions'] = [p * adjustment for p in result['predictions']]
        result['cluster_adjustment'] = adjustment
    
    return result

# ---------------------------------------------------
# Génération de rapports
# ---------------------------------------------------
def generate_forecast_report(pdl_id, df_consumption, cluster_info=None):
    """Génère un rapport complet de prédiction"""
    result = forecast_with_clustering(pdl_id, df_consumption, cluster_info)
    
    if 'error' in result:
        return f"Erreur PDL {pdl_id} : {result['error']}"
    
    report = f"\n{'='*60}\n"
    report += f"RAPPORT DE PRÉDICTION - PDL {pdl_id}\n"
    report += f"{'='*60}\n"
    
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    
    report += f"\nPériode historique : {timeseries['date'].min().date()} à {timeseries['date'].max().date()}\n"
    report += f"Nombre de jours d'historique : {len(timeseries)}\n"
    report += f"Consommation moyenne : {result['mean_consumption']:.2f} kWh/jour\n"
    report += f"Écart-type : {result['std_consumption']:.2f} kWh\n"
    report += f"Tendance observée : {result['trend']}\n"
    
    if 'cluster' in result:
        report += f"Cluster : {result['cluster']}\n"
        report += f"Ajustement du cluster : {result['cluster_adjustment']:.2%}\n"
    
    report += f"\nPRÉDICTIONS - 7 JOURS SUIVANTS\n"
    report += f"{'-'*60}\n"
    report += f"{'Date':<15} {'Prédiction (kWh)':<20} {'Écart à la moyenne':<20}\n"
    report += f"{'-'*60}\n"
    
    mean_cons = result['mean_consumption']
    for date, pred in zip(result['dates'], result['predictions']):
        ecart = ((pred - mean_cons) / mean_cons) * 100
        report += f"{str(date.date()):<15} {pred:>18.2f} {ecart:>18.1f}%\n"
    
    report += f"\n{'STATISTIQUES DES PRÉDICTIONS':<30}\n"
    report += f"{'-'*60}\n"
    report += f"Moyenne prédite : {np.mean(result['predictions']):.2f} kWh\n"
    report += f"Min prédite : {np.min(result['predictions']):.2f} kWh\n"
    report += f"Max prédite : {np.max(result['predictions']):.2f} kWh\n"
    report += f"Total 7 jours : {np.sum(result['predictions']):.2f} kWh\n"
    
    return report

# ---------------------------------------------------
# Comparaison multi-PDL
# ---------------------------------------------------
def compare_predictions_multi_pdl(df_consumption, n_pdls=5, cluster_info=None):
    """Compare les prédictions pour plusieurs PDLs"""
    results = []
    
    sample_pdls = df_consumption['ID'].unique()[:n_pdls]
    
    for pdl_id in sample_pdls:
        result = forecast_with_clustering(pdl_id, df_consumption, cluster_info)
        
        if 'success' in result and result['success']:
            results.append({
                'pdl_id': pdl_id,
                'cluster': result.get('cluster', 'N/A'),
                'mean_pred': np.mean(result['predictions']),
                'min_pred': np.min(result['predictions']),
                'max_pred': np.max(result['predictions']),
                'total_7days': np.sum(result['predictions']),
                'trend': result['trend'],
            })
    
    return pd.DataFrame(results)

# ---------------------------------------------------
# Utilisation principale
# ---------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("SYSTÈME DE PRÉDICTION DE CONSOMMATION ÉNERGÉTIQUE")
    print("="*60)
    
    print("\nChargement des données...")
    df_consumption = load_consumption_data()
    cluster_info = load_cluster_assignments()
    
    print(f"Données chargées : {df_consumption.shape}")
    print(f"Clusters chargés : {cluster_info.shape}")
    
    # Générer des rapports pour quelques PDLs
    print("\nGénération des rapports de prédiction...\n")
    
    sample_pdls = df_consumption['ID'].unique()[:3]
    
    for pdl_id in sample_pdls:
        report = generate_forecast_report(pdl_id, df_consumption, cluster_info)
        print(report)
    
    # Tableau comparatif
    print("\n" + "="*60)
    print("TABLEAU COMPARATIF - PREMIERS 5 PDLs")
    print("="*60 + "\n")
    
    comparison_df = compare_predictions_multi_pdl(df_consumption, n_pdls=5, cluster_info=cluster_info)
    print(comparison_df.to_string(index=False))
    
    # Sauvegarder les résultats complets
    print("\n" + "="*60)
    print("Sauvegarde des prédictions complètes...")
    print("="*60)
    
    all_predictions = []
    for pdl_id in df_consumption['ID'].unique():
        result = forecast_with_clustering(pdl_id, df_consumption, cluster_info, days_ahead=7)
        
        if 'success' in result and result['success']:
            cluster = result.get('cluster', -1)
            for i, (date, pred) in enumerate(zip(result['dates'], result['predictions']), 1):
                all_predictions.append({
                    'ID': pdl_id,
                    'date': date,
                    'day_number': i,
                    'predicted_consumption_kwh': round(pred, 2),
                    'cluster': cluster,
                    'trend': result['trend'],
                    'mean_historical_kwh': round(result['mean_consumption'], 2),
                })
    
    if all_predictions:
        df_forecast = pd.DataFrame(all_predictions)
        
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        forecast_path = os.path.join(results_dir, "consumption_forecast.csv")
        df_forecast.to_csv(forecast_path, index=False)
        print(f"\n✓ Prédictions sauvegardées : {forecast_path}")
        print(f"  Total : {len(df_forecast)} prédictions pour {df_forecast['ID'].nunique()} PDLs")
        
        summary = df_forecast.groupby('ID').agg({
            'predicted_consumption_kwh': ['sum', 'mean', 'min', 'max'],
            'cluster': 'first'
        }).round(2)
        
        summary_path = os.path.join(results_dir, "forecast_summary.csv")
        summary.to_csv(summary_path)
        print(f"✓ Résumé sauvegardé : {summary_path}")
    
    print("\n" + "="*60)
    print("Prédictions terminées !")
    print("="*60)

