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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX # Plus robuste que ARIMA pour les saisons
from src.clustering import get_features_pdl
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
    print(HAS_ARIMA)
except:
    HAS_ARIMA = False

# ---------------------------------------------------
# Chargement et préparation des données de consommation
# ---------------------------------------------------
def load_consumption_data():
    """Charge les données de consommation quotidienne et extrait les jours"""
    path_options = [
        "data/daily.csv",
        "../data/daily.csv",
    ]
    
    for path in path_options:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Donne un chiffre de 0 (Lundi) à 6 (Dimanche)
            df['day_of_week'] = df['date'].dt.dayofweek 
            
            # Donne un chiffre de 1 (Janvier) à 12 (Décembre)
            df['month'] = df['date'].dt.month 
            
            # Bonus : Variable binaire très utile pour les modèles (1 = Week-end, 0 = Semaine)
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df = get_french_calendar_features(df)
            return df.sort_values(['ID', 'date'])
    
    raise FileNotFoundError("Fichier daily.csv non trouvé")

def load_cluster_assignments():
    """Charge les assignations de clusters"""
    features_pdl = get_features_pdl()
    return features_pdl[['ID', 'cluster']].drop_duplicates().set_index('ID')


import holidays

def get_french_calendar_features(df):
    """
    Ajoute les colonnes 'is_holiday' et 'is_school_holiday' au DataFrame
    """
    # Jours fériés français
    fr_holidays = holidays.France(years=df['date'].dt.year.unique())
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in fr_holidays else 0)
    
    # Vacances Scolaires (Logique simplifiée ou via API)
    # Pour une précision totale, il faudrait les dates officielles du gouv.
    # Ici, nous créons une colonne par défaut à 0. 
    # CONSEIL : Si vous avez un fichier CSV des vacances, joignez-le ici.
    df['is_school_holiday'] = 0 
    
    return df

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
import holidays

def train_and_predict_models(pdl_id, df_consumption, days_ahead=7,
                             lookback_lr=30, lookback_sarima=90, lookback_lstm=30):
    """
    Entraîne plusieurs modèles et fait des prédictions corrigées
    """
    # 1. Préparation de l'historique annuel
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    
    # Liste des features utilisée par la Régression (sans la cible)
    features_lr = ['day_of_week', 'month', 'trend', 'is_holiday', 'is_weekend']
    # Liste complète pour le LSTM (incluant la cible à l'index 0)
    features_lstm = ['daily_kwh', 'day_of_week', 'month', 'trend', 'is_holiday', 'is_weekend']
    print(f'Taille de la série {len(timeseries)}')
    if len(timeseries) < 365:
        print("Warning: Moins d'un an de données. La tendance annuelle sera estimée sur l'existant.")
    
    # Ajouter la colonne trend à TOUT le timeseries (ou à ce qu'on a)
    ts_for_decomp = timeseries.tail(365).copy() if len(timeseries) >= 365 else timeseries.copy()
    decomposition = seasonal_decompose(ts_for_decomp['daily_kwh'], model='additive', period=7)
    trend_values = decomposition.trend.fillna(method='bfill').fillna(method='ffill')
    
    # Créer une colonne trend pour tout le timeseries
    # Si on n'a pas 365 jours, utiliser la tendance calculée et remplir le reste
    if len(timeseries) > len(ts_for_decomp):
        last_trend_val = trend_values.iloc[-1]
        # Remplir les jours manquants au début avec la première valeur de la tendance
        first_trend_val = trend_values.iloc[0]
        trend_full = [first_trend_val] * (len(timeseries) - len(ts_for_decomp)) + trend_values.tolist()
        timeseries = timeseries.reset_index(drop=True)
        timeseries['trend'] = trend_full
    else:
        timeseries = timeseries.reset_index(drop=True)
        timeseries['trend'] = trend_values.values
    
    last_date = timeseries['date'].iloc[-1]
    last_trend = timeseries['trend'].iloc[-1]
    
    # Objet pour les jours fériés français
    fr_holidays = holidays.France()
    
    results = {
        'pdl_id': pdl_id,
        'last_date': last_date.date(),
        'lookback_lr': lookback_lr,
        'lookback_sarima': lookback_sarima,
        'lookback_lstm': lookback_lstm,
        'data_points': len(timeseries),
        'models': {}
    }
    
    # ===== 1. Régression Linéaire =====
    try:
        # Train linear model on the last `lookback_lr` days (or as many available)
        ts_lr = timeseries.tail(max(lookback_lr, 30)).copy()
        X_lr = ts_lr[features_lr]
        y_lr = ts_lr['daily_kwh']
        model_lr = LinearRegression().fit(X_lr, y_lr)
        
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        X_future_lr = pd.DataFrame({
            'day_of_week': [d.weekday() for d in future_dates],
            'month': [d.month for d in future_dates],
            'trend': [last_trend] * days_ahead,
            'is_holiday': [1 if d in fr_holidays else 0 for d in future_dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in future_dates]
        })
        results['models']['linear_regression'] = {
            'predictions': np.maximum(0, model_lr.predict(X_future_lr)).tolist(),
            'model_name': 'Régression'
        }
    except Exception as e:
        results['models']['linear_regression'] = {'error': str(e)}
        print(f'ERREUR Régression Linéaire: {e}')

    # ===== 2. SARIMA =====
    if HAS_ARIMA:
        try:
            # Fit SARIMA on the last `lookback_sarima` points (if available)
            sarima_series = timeseries['daily_kwh'].dropna().tail(max(lookback_sarima, 30)).values
            model_sarima = SARIMAX(
                sarima_series,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 0, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_sarima = model_sarima.fit(disp=False)
            forecast = fitted_sarima.get_forecast(steps=days_ahead)
            results['models']['arima'] = {
                'predictions': np.maximum(0, forecast.predicted_mean).tolist(),
                'model_name': 'SARIMA'
            }
        except Exception as e:
            results['models']['arima'] = {'error': str(e)}

    # ===== 3. LSTM (Séquences Longues) =====
    try:
        # For LSTM, use the recent window (default 365 days if available)
        ts_lstm = timeseries.tail(max(365, lookback_lstm + 10)).copy()
        features_df = ts_lstm[features_lstm].copy()

        # SÉCURITÉ : Vérifie si le dataset est plus grand que la taille de la séquence
        if len(features_df) <= lookback_lstm:
            raise ValueError(f"Historique de {len(features_df)}j trop court pour une séquence de {lookback_lstm}j.")
        
        scaler_X = MinMaxScaler()
        scaled_features = scaler_X.fit_transform(features_df.values)
        
        scaler_y = MinMaxScaler()
        scaler_y.fit(features_df[['daily_kwh']])
        
        # --- CRÉATION DES SÉQUENCES SUR TOUT LE DATASET ---
        X_lstm, y_lstm = [], []
        # La boucle parcourt TOUT le dataset moins la longueur de la fenêtre
        for i in range(len(scaled_features) - lookback_lstm):
            # X = Les N jours précédents (fenêtre complète)
            X_lstm.append(scaled_features[i:i+lookback_lstm, :])
            # Y = La consommation du jour SUIVANT la fenêtre
            y_lstm.append(scaled_features[i+lookback_lstm, 0])
            
        # Conversion en tenseurs 3D pour PyTorch (Batch, Sequence, Features)
        X_tensor = torch.tensor(np.array(X_lstm), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_lstm), dtype=torch.float32).unsqueeze(1)
        
        model_lstm = LSTMForecaster(input_size=len(features_lstm), hidden_size=64, num_layers=2)
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Entraînement sur l'ensemble des fenêtres générées
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model_lstm(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        # --- INFÉRENCE (Prédiction du futur) ---
        # On part de la TOUTE DERNIÈRE séquence connue du dataset
        curr_seq = scaled_features[-lookback_lstm:]
        preds_lstm = []
        
        for i in range(days_ahead):
            # On ajoute une dimension "Batch" fictive pour l'inférence
            input_t = torch.tensor(curr_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                p_norm = model_lstm(input_t).item()
            preds_lstm.append(p_norm)
            
            next_d = last_date + timedelta(days=i+1)
            
            # On construit la réalité de demain
            next_row_raw = np.array([[
                0, # Placeholder (remplacé juste après)
                next_d.weekday(), 
                next_d.month, 
                last_trend, 
                1 if next_d in fr_holidays else 0, 
                1 if next_d.weekday() >= 5 else 0
            ]])
            
            next_row_scaled = scaler_X.transform(next_row_raw)[0]
            next_row_scaled[0] = p_norm # Injection de la prédiction
            
            # Glissement de la fenêtre : on supprime le jour le plus vieux, on ajoute demain
            curr_seq = np.vstack((curr_seq[1:], next_row_scaled))
            
        # Dénormalisation propre
        preds_final = scaler_y.inverse_transform(np.array(preds_lstm).reshape(-1, 1)).flatten()
        
        results['models']['lstm'] = {
            'predictions': np.maximum(0, preds_final).tolist(),
            'model_name': 'LSTM'
        }
    except Exception as e:
        results['models']['lstm'] = {'error': f"LSTM: {str(e)}"}
        print(f'ERREUR LSTM: {e}')

    # Ensemble Voting
    results['dates'] = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    all_preds = [m['predictions'] for m in results['models'].values() if 'predictions' in m]
    if all_preds:
        results['ensemble'] = {
            'predictions': np.mean(all_preds, axis=0).tolist(),
            'model_name': f'Ensemble ({len(all_preds)} modèles)'
        }
    
    return results

# ---------------------------------------------------
# Forecasting avec tendance et saisonnalité (pour compatibilité)
# ---------------------------------------------------
def forecast_consumption_trend(pdl_id, df_consumption, days_ahead=7,
                               lookback_lr=30, lookback_sarima=90, lookback_lstm=30):
    """
    Prédit la consommation en utilisant les modèles entraînés
    (Wrapper pour compatibilité avec les anciens scripts)
    """
    result = train_and_predict_models(
        pdl_id, df_consumption, days_ahead,
        lookback_lr=lookback_lr, lookback_sarima=lookback_sarima, lookback_lstm=lookback_lstm
    )
    
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
        'all_models_results': result.get('models', {})
    }

def evaluate_and_plot_backtest(pdl_id, df_consumption, test_days=14,
                               lookback_lr=30, lookback_sarima=90, lookback_lstm=30):
    """
    Sépare les données en Train/Test pour évaluer la cohérence du modèle.
    Masque les 'test_days' derniers jours, prédit dessus, et affiche la comparaison
    avec TOUS les modèles individuels.
    """
    # 1. Récupérer la série complèt
    ts = get_pdl_timeseries(pdl_id, df_consumption)
    
    if len(ts) <= test_days + 30:
        print(f"⚠️ Pas assez de données pour faire un backtest sur {test_days} jours.")
        return
    
    # 2. Séparer Train (passé) et Test (les derniers jours qu'on veut vérifier)
    ts_train = ts.iloc[:-test_days]
    ts_test = ts.iloc[-test_days:]
    
    # 3. Créer un DataFrame tronqué pour simuler qu'on est dans le passé
    # Filtrer par PDL ET par dates de train
    train_dates = ts_train['date'].values
    df_train_temp = df_consumption[
        (df_consumption['ID'] == pdl_id) & 
        (df_consumption['date'].isin(train_dates))
    ].copy()
    
    print(f"\n⏳ Lancement du Backtest : Entraînement sur les données jusqu'au {ts_train['date'].iloc[-1].strftime('%Y-%m-%d')}...")
    
    # 4. Faire la prédiction (le modèle ne verra pas le Test)
    resultats_backtest = forecast_consumption_trend(
        pdl_id, df_train_temp, days_ahead=test_days,
        lookback_lr=lookback_lr, lookback_sarima=lookback_sarima, lookback_lstm=lookback_lstm
    )
    
    if 'error' in resultats_backtest:
        print("❌ Erreur pendant le backtest:", resultats_backtest['error'])
        return
        
    # 5. Calculer l'erreur absolue moyenne (MAE) pour l'Ensemble ET chaque modèle
    predictions_ensemble = resultats_backtest['predictions']
    realite = ts_test['daily_kwh'].values
    
    print("\n📊 RÉSULTATS DU BACKTEST (erreurs: MAE et RMSE — plus c'est bas, mieux c'est) :")
    mae_ensemble = mean_absolute_error(realite, predictions_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(realite, predictions_ensemble))
    print(f"   🎯 Ensemble (Moyenne)  : MAE={mae_ensemble:.2f} kWh/jour | RMSE={rmse_ensemble:.2f}")

    all_models = resultats_backtest.get('all_models_results', {})
    backtest_metrics = { 'Ensemble': {'MAE': float(mae_ensemble), 'RMSE': float(rmse_ensemble)} }
    for model_key, model_info in all_models.items():
        if 'predictions' in model_info:
            mae_model = mean_absolute_error(realite, model_info['predictions'])
            rmse_model = np.sqrt(mean_squared_error(realite, model_info['predictions']))
            nom_modele = model_info.get('model_name', model_key)
            print(f"   - {nom_modele:<20} : MAE={mae_model:.2f} | RMSE={rmse_model:.2f}")
            backtest_metrics[nom_modele] = {'MAE': float(mae_model), 'RMSE': float(rmse_model)}
        else:
            # Afficher si le modèle a une erreur
            if 'error' in model_info:
                print(f"   - {model_info.get('model_name', model_key):<20} : ERREUR = {model_info['error']}")
        
    # 6. Tracer le graphique de comparaison
    # Choisir une fenêtre d'historique raisonnable pour l'affichage
    history = ts_train.tail(max(lookback_lstm, lookback_lr, lookback_sarima) + test_days)
    
    # Création de la figure interactive Plotly
    fig = go.Figure()

    # 1. Tracer le Passé (Train)
    fig.add_trace(go.Scatter(
        x=history['date'], 
        y=history['daily_kwh'],
        mode='lines+markers',
        name='Historique (Entraînement)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=5)
    ))

    last_date = history['date'].iloc[-1]
    last_val = history['daily_kwh'].iloc[-1]
    dates_pred = list(resultats_backtest['dates']) # S'assurer que c'est une liste

    # Palette de couleurs pour les modèles
    colors = {'linear_regression': '#2ca02c', 'arima': '#9467bd', 'lstm': '#ff7f0e'}

    # 2. Tracer TOUS les modèles individuels
    for model_key, model_info in all_models.items():
        if 'predictions' in model_info:
            preds = list(model_info['predictions'])
            name = model_info.get('model_name', model_key)
            c = colors.get(model_key, '#8c564b')
            
            # On fusionne la dernière valeur d'historique avec les prédictions pour relier les courbes
            x_pred = [last_date] + dates_pred
            y_pred = [last_val] + preds
            
            fig.add_trace(go.Scatter(
                x=x_pred, 
                y=y_pred,
                mode='lines',
                name=f"Prédiction {name}",
                line=dict(color=c, width=1.5, dash='dashdot'),
                opacity=0.8
            ))

    # 3. Tracer la prédiction ENSEMBLE (plus épaisse, en rouge)
    x_ens = [last_date] + dates_pred
    y_ens = [last_val] + list(predictions_ensemble)

    fig.add_trace(go.Scatter(
        x=x_ens, 
        y=y_ens,
        mode='lines+markers',
        name='Prédictions (Ensemble)',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(symbol='x', size=8)
    ))

    # 4. Tracer la RÉALITÉ cachée au modèle (Test) - en NOIR
    x_test = [last_date] + list(ts_test['date'])
    y_test = [last_val] + list(ts_test['daily_kwh'])

    fig.add_trace(go.Scatter(
        x=x_test, 
        y=y_test,
        mode='lines+markers',
        name='Consommation historique (Réalité)',
        line=dict(color='black', width=3),
        marker=dict(symbol='circle', size=6)
    ))

    # 5. Mise en page du graphique
    fig.update_layout(
        title=f'Backtest des Prévisions - PDL : {pdl_id}',
        xaxis_title='Date',
        yaxis_title='Consommation (kWh)',
        hovermode='x unified', # MAGIQUE : Affiche une barre verticale avec toutes les valeurs au survol
        template='plotly_white',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # 6. Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # Retourner les métriques calculées pour affichage dans l'UI
    try:
        return {'metrics': backtest_metrics}
    except Exception:
        return {'metrics': {}}


def plot_forecast(pdl_id, df_consumption, dates, ensemble_preds=None, all_models_dict=None, days_history=60):
    """
    Affiche le graphique interactif (Plotly) de l'historique récent et toutes les prédictions
    (Ensemble + modèles individuels) pour le FUTUR.
    """
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    history = timeseries.tail(days_history)
    
    # Création de la figure interactive Plotly
    fig = go.Figure()
    
    # 1. Tracer l'historique
    fig.add_trace(go.Scatter(
        x=history['date'], 
        y=history['daily_kwh'],
        mode='lines+markers',
        name='Historique de consommation',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=5)
    ))
    
    last_date = history['date'].iloc[-1] if not history.empty else None
    last_val = history['daily_kwh'].iloc[-1] if not history.empty else None
    
    # S'assurer que les dates futures sont bien dans une liste
    dates_list = list(dates)

    # Palette de couleurs pour différencier les modèles
    colors = {'linear_regression': '#2ca02c', 'arima': '#9467bd', 'lstm': '#ff7f0e'}
    
    # 2. Tracer les modèles individuels
    if all_models_dict and last_date is not None:
        for model_key, model_info in all_models_dict.items():
            if 'predictions' in model_info:
                preds = list(model_info['predictions'])
                name = model_info.get('model_name', model_key)
                c = colors.get(model_key, '#8c564b') # Couleur marron par défaut si non trouvée
                
                # Relier le passé au futur
                x_pred = [last_date] + dates_list
                y_pred = [last_val] + preds
                
                fig.add_trace(go.Scatter(
                    x=x_pred, 
                    y=y_pred,
                    mode='lines',
                    name=f"Prédiction {name}",
                    line=dict(color=c, width=1.5, dash='dashdot'),
                    opacity=0.8
                ))

    # 3. Tracer la prédiction Ensemble (par dessus, plus épaisse)
    if ensemble_preds is not None and last_date is not None:
        x_ens = [last_date] + dates_list
        y_ens = [last_val] + list(ensemble_preds)
        
        fig.add_trace(go.Scatter(
            x=x_ens, 
            y=y_ens,
            mode='lines+markers',
            name='Ensemble (Moyenne)',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(symbol='x', size=8)
        ))
    
    # 4. Mise en page du graphique
    fig.update_layout(
        title=f'Prévisions de Consommation - Tous les Modèles - PDL ID : {pdl_id}',
        xaxis_title='Date',
        yaxis_title='Consommation (kWh)',
        hovermode='x unified', # Permet de voir toutes les valeurs au survol d'une date
        template='plotly_white',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # 5. Affichage direct dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# Bloc d'exécution principal pour tester le script
# ---------------------------------------------------
if __name__ == "__main__":
    print("Chargement des données en cours...")
    try:
        df_conso = load_consumption_data()
        print(f"✅ Données chargées avec succès ! ({len(df_conso)} lignes trouvées)")
        
        # ==========================================
        # 🎛️ TON PANNEAU DE CONTRÔLE CENTRALISÉ 🎛️
        # ==========================================
        pdl_test_id = df_conso["ID"].unique()[8]
        test_days = 14     # Nombre de jours cachés pour évaluer le modèle
        forecast_days = 7  # Nombre de jours à prédire dans le vrai futur
        lookback_window = 14 # <-- CHANGE CETTE VALEUR POUR TESTER L'IMPACT SUR LE LSTM !
        # ==========================================
        
        # --- 1. PHASE DE TEST (BACKTEST) ---
        print("\n" + "="*50)
        print(f"🔍 PHASE 1 : BACKTEST ENSEMBLE (Lookback: {lookback_window}j)")
        print("="*50)
        evaluate_and_plot_backtest(
            pdl_test_id, 
            df_conso, 
            test_days=test_days, 
            lookback=lookback_window
        )
        
        # # --- 2. PHASE DE VRAIE PRÉDICTION (ENSEMBLE) ---
        # print("\n" + "="*50)
        # print(f"🚀 PHASE 2 : VRAI FUTUR ({forecast_days} jours | Lookback: {lookback_window}j)")
        # print("="*50)
        # resultats_trend = forecast_consumption_trend(
        #     pdl_test_id, 
        #     df_conso, 
        #     days_ahead=forecast_days, 
        #     lookback=lookback_window
        # )

    except FileNotFoundError as e:
        print(f"\n❌ Erreur critique : {e}")