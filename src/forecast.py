# -*- coding: utf-8 -*-
"""
Module de forecasting de consommation énergétique pour les jours suivants
Utilise les données historiques pour faire du forecasting
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------
# Chargement et préparation des données de consommation
# ---------------------------------------------------
def load_consumption_data():
    """Charge les données de consommation quotidienne"""
    path_options = [
        "data/daily.csv",
        "../data/daily.csv",
    ]
    
    for path in path_options:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values(['ID', 'date'])
    
    raise FileNotFoundError("Fichier daily.csv non trouvé")

def get_pdl_timeseries(pdl_id, df_consumption):
    """Récupère la série temporelle pour un PDL"""
    data = df_consumption[df_consumption['ID'] == pdl_id].copy()
    data = data.sort_values('date')
    return data

def prepare_sequences(data, lookback=30):
    """Prépare les séquences pour l'entraînement"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# ---------------------------------------------------
# Forecasting avec tendance et saisonnalité
# ---------------------------------------------------
def forecast_consumption_trend(pdl_id, df_consumption, days_ahead=7):
    """
    Prédit la consommation en utilisant la tendance historique et la saisonnalité
    
    Parameters:
    -----------
    pdl_id : int
        ID du PDL
    df_consumption : pd.DataFrame
        Données de consommation
    days_ahead : int
        Nombre de jours à prédire
    
    Returns:
    --------
    dict : Prédictions
    """
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    
    if len(timeseries) < 30:
        return {
            'error': 'Pas assez de données historiques (minimum 30 jours).'
        }
    
    # Calculs statistiques
    consumption = timeseries['daily_kwh'].values
    
    # Moyenne mobile (tendance)
    ma_7 = pd.Series(consumption).rolling(window=7).mean().iloc[-1]
    ma_30 = pd.Series(consumption).rolling(window=30).mean().iloc[-1]
    
    # Saisonnalité hebdomadaire
    timeseries['day_of_week'] = timeseries['date'].dt.dayofweek
    weekly_avg = timeseries.groupby('day_of_week')['daily_kwh'].mean().to_dict()
    
    # Saisonnalité mensuelle
    timeseries['month'] = timeseries['date'].dt.month
    monthly_avg = timeseries.groupby('month')['daily_kwh'].mean().to_dict()
    
    # Volatilité
    volatility = np.std(consumption[-30:])
    
    # Prédictions
    predictions = []
    last_date = timeseries['date'].iloc[-1]
    
    for i in range(days_ahead):
        future_date = last_date + timedelta(days=i+1)
        day_of_week = future_date.dayofweek
        month = future_date.month
        
        # Moyenne pondérée
        base_pred = 0.4 * ma_7 + 0.3 * ma_30 + 0.2 * weekly_avg.get(day_of_week, ma_7) + 0.1 * monthly_avg.get(month, ma_7)
        
        # Ajouter un peu de variation
        pred = base_pred + np.random.normal(0, volatility * 0.1)
        pred = max(0, pred)  # Pas de consommation négative
        
        predictions.append(pred)
    
    # Créer les dates futures
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    # Déterminer la tendance
    try:
        if abs(ma_7 - ma_30) / ma_30 < 0.1:
            trend = 'stable'
        elif ma_7 > ma_30:
            trend = 'hausse'
        else:
            trend = 'baisse'
    except:
        trend = 'stable'
    
    return {
        'success': True,
        'pdl_id': pdl_id,
        'predictions': predictions,
        'dates': future_dates,
        'last_date': last_date.date(),
        'mean_consumption': consumption.mean(),
        'std_consumption': consumption.std(),
        'trend': trend,
    }

# Alias LSTM pour compatibilité
def forecast_consumption_lstm(pdl_id, df_consumption, days_ahead=7, lookback=30, model_path=None):
    """Alias pour forecast_consumption_trend"""
    return forecast_consumption_trend(pdl_id, df_consumption, days_ahead)

# Fonction placeholder pour plot_forecast
def plot_forecast(pdl_id, df_consumption, predictions, dates, days_history=90):
    """Placeholder pour la visualisation"""
    pass
