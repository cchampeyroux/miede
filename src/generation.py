# -*- coding: utf-8 -*-
"""
Module de prédiction de consommation avec plusieurs modèles
Entraîne et compare : Régression Linéaire, ARIMA, LSTM
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ajout de la racine du projet au PATH pour les imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clustering import get_features_pdl
from src.forecast import load_consumption_data

# ---------------------------------------------------
# Chargement et préparation des données
# ---------------------------------------------------
def load_cluster_assignments():
    """Charge les assignations de clusters depuis les features."""
    features_pdl = get_features_pdl()
    return features_pdl[['ID', 'cluster']].drop_duplicates()

def prepare_cluster_data(df_consumption, cluster_info, cluster_id):
    """Fusionne les données et filtre les informations pour un cluster spécifique."""
    merged_df = pd.merge(df_consumption, cluster_info, on='ID', how='outer')
    
    # Filtrage sur le cluster
    df_cluster = merged_df[merged_df["cluster"] == cluster_id].copy()
    
    # Formatage de la date
    df_cluster['date'] = pd.to_datetime(df_cluster['date'])
    
    return df_cluster

import plotly.graph_objects as go

def plot_cluster_dispersion(df_cluster, cluster_id, ts_full, save_html=False):
    """
    Calcule la moyenne et l'écart type journaliers, 
    puis génère un graphique interactif Plotly incluant la courbe cible.
    """
    # Agréger les données par jour : Moyenne ET Écart type
    daily_series = df_cluster.groupby('date')['daily_kwh'].mean()
    daily_std = df_cluster.groupby('date')['daily_kwh'].std()
    
    # Création du graphique interactif
    fig = go.Figure()
    
    # --- ZONE OMBRÉE DE L'ÉCART TYPE ---
    fig.add_trace(go.Scatter(
        x=daily_series.index,
        y=daily_series.values + daily_std.values,
        mode='lines',
        line=dict(width=0), 
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_series.index,
        y=daily_series.values - daily_std.values,
        mode='lines',
        line=dict(width=0), 
        fill='tonexty',     
        fillcolor='rgba(173, 216, 230, 0.4)', 
        name='± 1 Écart type Cluster'
    ))
    
    # --- COURBE MOYENNE DU CLUSTER ---
    fig.add_trace(go.Scatter(
        x=daily_series.index,
        y=daily_series.values,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        name='Moyenne du Cluster'
    ))
    
    # --- COURBE CIBLE (LE PDL CHOISI) ---
    # Correction ici : on cible spécifiquement les colonnes date et daily_kwh
    fig.add_trace(go.Scatter(
        x=ts_full['date'],
        y=ts_full['daily_kwh'],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=4),
        name='Consommation Cible (PDL)'
    ))
    
    # Formatage du graphique
    fig.update_layout(
        title=f'Comparaison PDL vs Cluster {cluster_id}',
        xaxis_title='Date',
        yaxis_title='Consommation (kWh)',
        template='plotly_white', 
        hovermode='x unified',   
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) 
    )

    # Export HTML optionnel
    if save_html:
        filename = f"cluster_{cluster_id}_consommation.html"
        fig.write_html(filename)
        print(f"Graphique sauvegardé sous : {filename}")
        
    # IMPORTANT : On retourne la figure pour que Streamlit l'affiche proprement
    return fig

# ---------------------------------------------------
# Exécution principale
# ---------------------------------------------------
if __name__ == "__main__":
    TARGET_CLUSTER_ID = 1
    
    print("\nChargement des données...")
    df_conso = load_consumption_data()
    df_clusters = load_cluster_assignments()
    
    print(f"Préparation des données pour le cluster {TARGET_CLUSTER_ID}...")
    df_target_cluster = prepare_cluster_data(df_conso, df_clusters, TARGET_CLUSTER_ID)
    
    print("Génération du graphique interactif...")
    plot_cluster_dispersion(df_target_cluster, TARGET_CLUSTER_ID, save_html=False)