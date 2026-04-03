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
def train_and_predict_models(pdl_id, df_consumption, days_ahead=7,lookback=30):
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
    # 2. Vérification et extraction des N dernières valeurs
    # On vérifie si on a assez de données pour le lookback demandé
    if len(timeseries) < lookback:
        return {'error': f'Historique insuffisant ({len(timeseries)} points) pour un lookback de {lookback}'}
    
    # MISE À JOUR : On ne garde que les 'lookback' dernières lignes
    timeseries = timeseries.tail(lookback).copy()
    print(timeseries)
    consumption = timeseries['daily_kwh'].values
    last_date = timeseries['date'].iloc[-1]
    
    results = {
        'pdl_id': pdl_id,
        'last_date': last_date.date(),
        'lookback': lookback,
        'data_points': len(timeseries),
        'models': {}
    }
    
    # ===== 1. Régression Linéaire =====
    # ===== 1. Régression Linéaire (Améliorée avec features) =====
    try:
        # 1. Préparation des variables d'entraînement (X) et de la cible (y)
        X_lr = pd.DataFrame({
            'day_of_week': timeseries['day_of_week'].values,
            'is_weekend': timeseries['is_weekend'].values,
            'month': timeseries["month"].values
        })
        y_lr = timeseries['daily_kwh'].values
        
        # 2. Entraînement du modèle
        model_lr = LinearRegression(fit_intercept=False)
        model_lr.fit(X_lr, y_lr)
        
        # 3. Préparation des variables pour les jours futurs à prédire
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        future_day_of_week = [d.weekday() for d in future_dates] 
        future_is_weekend = [1 if d >= 5 else 0 for d in future_day_of_week]
        future_months = [d.month for d in future_dates] # NOUVEAU : On extrait le mois des dates futures
        
        X_future_lr = pd.DataFrame({
            'day_of_week': future_day_of_week,
            'is_weekend': future_is_weekend,
            'month': future_months # NOUVEAU : On utilise la liste générée juste au-dessus
        })
        # 4. Prédictions
        pred_lr = np.maximum(0, model_lr.predict(X_future_lr).flatten())
        
        results['models']['linear_regression'] = {
            'predictions': pred_lr.tolist(),
            'model_name': 'Régression Linéaire (Calendaire)'
        }
    except Exception as e:
        results['models']['linear_regression'] = {'error': str(e)}
    
    # ===== 2. ARIMA (Devenu SARIMA Saisonnier) =====
    if HAS_ARIMA:
        try:
            # On utilise une année de données
            data_arima = consumption[-min(365, len(consumption)):]
            
            # NOUVEAU : On ajoute seasonal_order=(1, 0, 0, 7)
            # Le "7" est magique : il lui dit de regarder ce qu'il s'est passé le même jour la semaine dernière !
            model_arima = ARIMA(
                data_arima, 
                order=(1, 1, 1), 
                seasonal_order=(1, 0, 0, 7)
            )
            
            # Le modèle va prendre 2 ou 3 secondes de plus à s'entraîner car il est plus intelligent
            fitted_model = model_arima.fit()
            
            # Prédictions
            forecast_arima = fitted_model.get_forecast(steps=days_ahead)
            pred_mean = np.array(forecast_arima.predicted_mean)
            pred_arima = np.maximum(0, pred_mean)
            
            results['models']['arima'] = {
                'predictions': pred_arima.tolist(),
                'model_name': 'SARIMA (Saisonnier)'
            }
        except Exception as e:
            results['models']['arima'] = {'error': str(e)}
    # ===== 3. LSTM (Multivarié / Calendaire) =====
    try:
        # 1. Sélectionner TOUTES les variables (Features)
        features_df = timeseries[['daily_kwh', 'day_of_week', 'is_weekend', 'month']].copy()
        feature_values = features_df.values
        num_features = feature_values.shape[1] # Vaut 4 désormais (au lieu de 1)
        
        # 2. Double Normalisation 
        # On utilise deux scalers : un pour normaliser toutes les entrées (X), 
        # et un spécifiquement pour la cible (y) afin de faciliter la dénormalisation à la fin
        scaler_X = MinMaxScaler()
        scaled_features = scaler_X.fit_transform(feature_values)
        
        scaler_y = MinMaxScaler()
        scaled_target = scaler_y.fit_transform(feature_values[:, 0].reshape(-1, 1)).flatten()
        
        # 3. Création des séquences multivariées (On remplace la fonction prepare_sequences)
        X_lstm = []
        y_lstm = []
        for i in range(len(scaled_features) - lookback):
            X_lstm.append(scaled_features[i:i+lookback, :])  # Historique avec les 4 colonnes
            y_lstm.append(scaled_features[i+lookback, 0])    # Cible = uniquement la conso (colonne 0)
            
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        if len(X_lstm) < 10:
            raise ValueError("Pas assez de séquences pour entraîner LSTM")
            
        # 4. Conversion en Tenseurs PyTorch
        X_lstm_tensor = torch.tensor(X_lstm, dtype=torch.float32)
        y_lstm_tensor = torch.tensor(y_lstm, dtype=torch.float32).unsqueeze(1)
        
        # 5. Paramétrage optimisé du LSTM
        device = torch.device('cpu')
        model_lstm = LSTMForecaster(input_size=num_features, hidden_size=64, num_layers=2)
        model_lstm.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001) # Learning rate plus doux
        
        # Entraînement plus long (60 au lieu de 10) pour bien apprendre la dynamique des 4 variables
        epochs = 60
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model_lstm(X_lstm_tensor)
            loss = criterion(outputs, y_lstm_tensor)
            loss.backward()
            optimizer.step()
            
        # 6. Prédiction Auto-régressive avec injection du calendrier futur
        last_sequence = scaled_features[-lookback:] # Les N derniers jours avec leurs 4 variables
        current_sequence = last_sequence.copy()
        
        predictions_lstm = []
        
        for i in range(days_ahead):
            # Préparer le tenseur d'entrée (1 batch, longueur du lookback, 4 variables)
            input_seq = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Prédire la conso de demain (normalisée)
            with torch.no_grad():
                next_pred_norm = model_lstm(input_seq).item()
            
            predictions_lstm.append(next_pred_norm)
            
            # --- LA MAGIE EST ICI ---
            # On construit la réalité calendaire de "demain" pour la glisser dans le modèle
            next_date = last_date + timedelta(days=i+1)
            next_dow = next_date.weekday()
            next_is_wend = 1 if next_dow >= 5 else 0
            next_month = next_date.month
            
            # On crée une ligne brute avec un faux 0 pour la conso, et les vraies infos calendaires
            new_row_raw = np.array([[0, next_dow, next_is_wend, next_month]]) 
            new_row_scaled = scaler_X.transform(new_row_raw)[0]
            
            # On remplace le faux 0 par la vraie prédiction du modèle !
            new_row_scaled[0] = next_pred_norm
            
            # On glisse la fenêtre : on enlève le jour le plus vieux (index 0), on ajoute "demain" à la fin
            current_sequence = np.vstack((current_sequence[1:], new_row_scaled))
            
        # 7. Dénormalisation finale (Uniquement avec scaler_y pour retrouver nos kWh)
        predictions_lstm_array = np.array(predictions_lstm).reshape(-1, 1)
        pred_lstm = np.maximum(0, scaler_y.inverse_transform(predictions_lstm_array).flatten())
        
        results['models']['lstm'] = {
            'predictions': pred_lstm.tolist(),
            'model_name': 'LSTM (2 couches)'
        }
    except Exception as e:
        results['models']['lstm'] = {'error': str(e)}
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
def forecast_consumption_trend(pdl_id, df_consumption, days_ahead=7,lookback=30):
    """
    Prédit la consommation en utilisant les modèles entraînés
    (Wrapper pour compatibilité avec les anciens scripts)
    """
    result = train_and_predict_models(pdl_id, df_consumption, days_ahead,lookback=lookback)
    
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
        'all_models_results': result.get('models', {}) # <-- NOUVEAU : On exporte le détail des modèles
    }


def evaluate_and_plot_backtest(pdl_id, df_consumption, test_days=14,lookback=30):
    """
    Sépare les données en Train/Test pour évaluer la cohérence du modèle.
    Masque les 'test_days' derniers jours, prédit dessus, et affiche la comparaison
    avec TOUS les modèles individuels.
    """
    # 1. Récupérer la série complète
    ts = get_pdl_timeseries(pdl_id, df_consumption)
    
    if len(ts) <= test_days + 30:
        print(f"⚠️ Pas assez de données pour faire un backtest sur {test_days} jours.")
        return
    
    # 2. Séparer Train (passé) et Test (les derniers jours qu'on veut vérifier)
    ts_train = ts.iloc[:-test_days]
    ts_test = ts.iloc[-test_days:]
    
    # 3. Créer un DataFrame tronqué pour simuler qu'on est dans le passé
    df_train_temp = df_consumption[df_consumption.index.isin(ts_train.index)]
    
    print(f"\n⏳ Lancement du Backtest : Entraînement sur les données jusqu'au {ts_train['date'].iloc[-1].strftime('%Y-%m-%d')}...")
    
    # 4. Faire la prédiction (le modèle ne verra pas le Test)
    resultats_backtest = forecast_consumption_trend(pdl_id, df_train_temp, days_ahead=test_days,lookback=lookback)
    
    if 'error' in resultats_backtest:
        print("❌ Erreur pendant le backtest:", resultats_backtest['error'])
        return
        
    # 5. Calculer l'erreur absolue moyenne (MAE) pour l'Ensemble ET chaque modèle
    predictions_ensemble = resultats_backtest['predictions']
    realite = ts_test['daily_kwh'].values
    
    print("\n📊 RÉSULTATS DU BACKTEST (Erreur Absolue Moyenne - plus c'est bas, mieux c'est) :")
    mae_ensemble = mean_absolute_error(realite, predictions_ensemble)
    print(f"   🎯 Ensemble (Moyenne)  : {mae_ensemble:.2f} kWh/jour")
    
    all_models = resultats_backtest.get('all_models_results', {})
    for model_key, model_info in all_models.items():
        if 'predictions' in model_info:
            mae_model = mean_absolute_error(realite, model_info['predictions'])
            nom_modele = model_info.get('model_name', model_key)
            print(f"   - {nom_modele:<20} : {mae_model:.2f} kWh/jour")
        
    # 6. Tracer le graphique de comparaison
    history = ts_train # Garder 45 jours d'historique pour la lisibilité
    
    plt.figure(figsize=(12, 6))
    
    # Tracer le Passé (Train)
    plt.plot(history['date'], history['daily_kwh'], 
             label='Historique (Entraînement)', color='#1f77b4', marker='.')
             
    last_date = history['date'].iloc[-1]
    last_val = history['daily_kwh'].iloc[-1]
    dates_pred = resultats_backtest['dates']
    
    # Palette de couleurs pour les modèles
    colors = {'linear_regression': '#2ca02c', 'arima': '#9467bd', 'lstm': '#ff7f0e'}
    
    # Tracer TOUS les modèles individuels
    for model_key, model_info in all_models.items():
        if 'predictions' in model_info:
            preds = model_info['predictions']
            name = model_info.get('model_name', model_key)
            c = colors.get(model_key, '#8c564b')
            plt.plot(dates_pred, preds, label=f"Prédiction ({name})", color=c, linewidth=1.5, linestyle='-.', alpha=0.8)
            plt.plot([last_date, dates_pred[0]], [last_val, preds[0]], color=c, linewidth=1.5, linestyle='-.', alpha=0.8)
    
    # Tracer la prédiction ENSEMBLE (plus épaisse, en rouge)
    plt.plot(dates_pred, predictions_ensemble, 
             label='Prédictions (Ensemble)', color='#d62728', linestyle='--', marker='X', linewidth=2.5)
    plt.plot([last_date, dates_pred[0]], [last_val, predictions_ensemble[0]], color='#d62728', linestyle='--', linewidth=2.5)

    # Tracer la RÉALITÉ cachée au modèle (Test) - en NOIR pour bien ressortir
    plt.plot(ts_test['date'], ts_test['daily_kwh'], 
             label='Vraie consommation (RÉALITÉ)', color='black', marker='o', linewidth=2.5)
    plt.plot([last_date, ts_test['date'].iloc[0]], [last_val, ts_test['daily_kwh'].iloc[0]], color='black', linewidth=2.5)
    
    plt.title(f'Backtest des Prévisions (Comparaison Réalité vs Tous Modèles) - PDL : {pdl_id}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Consommation (kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_forecast(pdl_id, df_consumption, dates, ensemble_preds=None, all_models_dict=None, days_history=60):
    """
    Affiche le graphique de l'historique récent et toutes les prédictions (Ensemble + modèles individuels).
    """
    timeseries = get_pdl_timeseries(pdl_id, df_consumption)
    history = timeseries.tail(days_history)
    
    plt.figure(figsize=(12, 6))
    
    # 1. Tracer l'historique
    plt.plot(history['date'], history['daily_kwh'], 
             label='Historique de consommation', 
             color='#1f77b4', linewidth=2, marker='.')
    
    last_date = history['date'].iloc[-1] if not history.empty else None
    last_val = history['daily_kwh'].iloc[-1] if not history.empty else None

    # Palette de couleurs pour différencier les modèles individuels
    colors = {'linear_regression': '#2ca02c', 'arima': '#9467bd', 'lstm': '#ff7f0e'}
    
    # 2. Tracer les modèles individuels
    if all_models_dict:
        for model_key, model_info in all_models_dict.items():
            if 'predictions' in model_info:
                preds = model_info['predictions']
                name = model_info.get('model_name', model_key)
                c = colors.get(model_key, '#8c564b') # Couleur par défaut si modèle inconnu
                
                plt.plot(dates, preds, label=name, color=c, linewidth=1.5, linestyle='-.', alpha=0.8)
                
                # Relier à l'historique
                if last_date is not None:
                    plt.plot([last_date, dates[0]], [last_val, preds[0]], color=c, linewidth=1.5, linestyle='-.', alpha=0.8)

    # 3. Tracer la prédiction Ensemble (par dessus, plus épaisse)
    if ensemble_preds is not None:
        plt.plot(dates, ensemble_preds, label='Ensemble (Moyenne)', color='#d62728', linewidth=3, linestyle='--', marker='o')
        if last_date is not None:
            plt.plot([last_date, dates[0]], [last_val, ensemble_preds[0]], color='#d62728', linewidth=3, linestyle='--')
    
    # Personnalisation
    plt.title(f'Prévisions de Consommation - Tous les Modèles - PDL ID : {pdl_id}', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Consommation (kWh)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
        pdl_test_id = df_conso["ID"].unique()[1]
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