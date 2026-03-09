# -*- coding: utf-8 -*-
"""
Module de forecasting de consommation énergétique
Entraîne plusieurs modèles (Régression Linéaire, ARIMA, LSTM, etc.)
pour prédire la consommation des jours suivants
"""

import sys
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Supprimer les avertissements
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Essayer d'importer ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except:
    HAS_ARIMA = False

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

def determine_lookback_window(timeseries_length):
    """
    Détermine la profondeur d'historique pertinente selon la longueur des données
    
    Parameters:
    -----------
    timeseries_length : int
        Nombre de points de données disponibles
    
    Returns:
    --------
    int : Nombre de jours à utiliser en historique
    """
    # Règle empirique :
    # - Si > 365 jours : utiliser 60-90 jours (environ 3 mois)
    # - Si > 180 jours : utiliser 30-60 jours (environ 1-2 mois)
    # - Si > 90 jours : utiliser 14-30 jours (environ 2-4 semaines)
    # - Sinon : utiliser 7-14 jours
    
    if timeseries_length >= 365:
        lookback = 60  # 2 mois pour capturer la saisonnalité
    elif timeseries_length >= 180:
        lookback = 30  # 1 mois
    elif timeseries_length >= 90:
        lookback = 14  # 2 semaines
    else:
        lookback = min(7, timeseries_length // 10)
    
    return max(7, lookback)

def prepare_sequences(data, lookback):
    """Prépare les séquences pour l'entraînement"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# ---------------------------------------------------
# Modèle LSTM pour forecasting
# ---------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# ---------------------------------------------------
# Entraînement des modèles de forecasting
# ---------------------------------------------------
def train_and_predict_models(pdl_id, df_consumption, days_ahead=7):
    """
    Entraîne plusieurs modèles et fait des prédictions
    
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
    dict : Résultats de prédiction pour chaque modèle
    """
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    
    if len(timeseries) < 30:
        return {'error': 'Pas assez de données historiques'}
    
    consumption = timeseries['daily_kwh'].values
    last_date = timeseries['date'].iloc[-1]
    
    # Déterminer la profondeur d'historique
    lookback = determine_lookback_window(len(timeseries))
    
    results = {
        'pdl_id': pdl_id,
        'last_date': last_date.date(),
        'lookback': lookback,
        'data_points': len(timeseries),
        'models': {}
    }
    
    # ===== 1. Régression Linéaire =====
    try:
        X_lr = np.arange(len(consumption) - lookback).reshape(-1, 1)
        y_lr = consumption[lookback:]
        
        model_lr = LinearRegression()
        model_lr.fit(X_lr, y_lr)
        
        # Prédictions
        X_future_lr = np.arange(len(consumption), len(consumption) + days_ahead).reshape(-1, 1)
        pred_lr = np.maximum(0, model_lr.predict(X_future_lr).flatten())
        
        results['models']['linear_regression'] = {
            'predictions': pred_lr.tolist(),
            'model_name': 'Régression Linéaire'
        }
    except Exception as e:
        results['models']['linear_regression'] = {'error': str(e)}
    
    # ===== 2. ARIMA =====
    if HAS_ARIMA:
        try:
            # Utiliser seulement une partie des données pour ARIMA (plus rapide)
            data_arima = consumption[-min(365, len(consumption)):]
            
            # Fit ARIMA(1,1,1) - ordre simple et rapide
            model_arima = ARIMA(data_arima, order=(1, 1, 1))
            fitted_model = model_arima.fit()
            
            # Prédictions
            forecast_arima = fitted_model.get_forecast(steps=days_ahead)
            pred_arima = np.maximum(0, forecast_arima.predicted_mean.values)
            
            results['models']['arima'] = {
                'predictions': pred_arima.tolist(),
                'model_name': 'ARIMA(1,1,1)'
            }
        except Exception as e:
            results['models']['arima'] = {'error': str(e)}
    
    # ===== 3. LSTM =====
    try:
        # Préparation des données
        X_lstm, y_lstm = prepare_sequences(consumption, lookback)
        
        if len(X_lstm) < 10:
            raise ValueError("Pas assez de séquences pour entraîner LSTM")
        
        # Normalisation
        scaler = MinMaxScaler()
        X_lstm_normalized = scaler.fit_transform(X_lstm.reshape(-1, 1)).reshape(X_lstm.shape)
        y_lstm_normalized = scaler.transform(y_lstm.reshape(-1, 1)).flatten()
        
        # Conversion en tensors
        X_lstm_tensor = torch.tensor(X_lstm_normalized.reshape(-1, lookback, 1), dtype=torch.float32)
        y_lstm_tensor = torch.tensor(y_lstm_normalized.reshape(-1, 1), dtype=torch.float32)
        
        # Entraînement
        device = torch.device('cpu')
        model_lstm = LSTMForecaster(input_size=1, hidden_size=32, num_layers=2)
        model_lstm.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.01)
        
        epochs = 20
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model_lstm(X_lstm_tensor)
            loss = criterion(outputs, y_lstm_tensor)
            loss.backward()
            optimizer.step()
        
        # Prédictions
        last_sequence = consumption[-lookback:]
        last_sequence_norm = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        predictions_lstm = []
        current_sequence = last_sequence_norm.copy()
        
        for _ in range(days_ahead):
            input_seq = torch.tensor(
                current_sequence.reshape(1, lookback, 1),
                dtype=torch.float32
            ).to(device)
            
            with torch.no_grad():
                next_pred_norm = model_lstm(input_seq).item()
            
            predictions_lstm.append(next_pred_norm)
            current_sequence = np.append(current_sequence[1:], next_pred_norm)
        
        # Dénormalisation
        predictions_lstm_array = np.array(predictions_lstm).reshape(-1, 1)
        pred_lstm = np.maximum(0, scaler.inverse_transform(predictions_lstm_array).flatten())
        
        results['models']['lstm'] = {
            'predictions': pred_lstm.tolist(),
            'model_name': 'LSTM (2 couches)'
        }
    except Exception as e:
        results['models']['lstm'] = {'error': str(e)}
    
    # Créer les dates futures
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    results['dates'] = future_dates
    
    # Ensemble voting - moyenne des prédictions
    all_predictions = []
    for model_name, model_result in results['models'].items():
        if 'predictions' in model_result:
            all_predictions.append(model_result['predictions'])
    
    if all_predictions:
        ensemble_pred = np.mean(all_predictions, axis=0)
        results['ensemble'] = {
            'predictions': ensemble_pred.tolist(),
            'model_name': f'Ensemble ({len(all_predictions)} modèles)'
        }
    
    return results

# ---------------------------------------------------
# Forecasting avec tendance et saisonnalité (pour compatibilité)
# ---------------------------------------------------
def forecast_consumption_trend(pdl_id, df_consumption, days_ahead=7):
    """
    Prédit la consommation en utilisant les modèles entraînés
    (Wrapper pour compatibilité avec les anciens scripts)
    """
    result = train_and_predict_models(pdl_id, df_consumption, days_ahead)
    
    if 'error' in result:
        return result
    
    # Retourner l'ensemble comme prédiction principale
    if 'ensemble' in result:
        ensemble_pred = result['ensemble']['predictions']
    elif 'models' in result:
        # Fallback sur le premier modèle disponible
        for model_name, model_result in result['models'].items():
            if 'predictions' in model_result:
                ensemble_pred = model_result['predictions']
                break
    else:
        return result
    
    return {
        'success': True,
        'pdl_id': result['pdl_id'],
        'predictions': ensemble_pred,
        'dates': result['dates'],
        'last_date': result['last_date'],
        'mean_consumption': np.mean(ensemble_pred),
        'std_consumption': np.std(ensemble_pred),
        'trend': 'stable',
    }

# Alias LSTM pour compatibilité
def forecast_consumption_lstm(pdl_id, df_consumption, days_ahead=7, lookback=30, model_path=None):
    """Alias pour train_and_predict_models"""
    return train_and_predict_models(pdl_id, df_consumption, days_ahead)

# Fonction placeholder pour plot_forecast
def plot_forecast(pdl_id, df_consumption, predictions, dates, days_history=90):
    """Placeholder pour la visualisation"""
    pass
