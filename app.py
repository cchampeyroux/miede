import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os
import time
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from src.forecast import (
    load_consumption_data,
    get_pdl_timeseries,
    evaluate_and_plot_backtest,
    forecast_consumption_trend,
    plot_forecast,
    load_cluster_assignments
)
from src.generation import generate_synthetic_curves_with_model, compute_r2_similarity
from src.clustering import get_features_pdl, cluster_to_label
from src.classification import (
    load_labels,
    feature_cols,
    SimpleNN
)
from src.models import ResidenceModel

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# --- CHARGEMENT DES DONNÉES (AVEC CACHE) ---
@st.cache_data
def cached_load_data():
    df_consumption = load_consumption_data()
    cluster_info = load_cluster_assignments()
    # On s'assure que cluster_info a bien une colonne 'ID' pour le merge
    if 'ID' not in cluster_info.columns and cluster_info.index.name == 'ID':
        cluster_info = cluster_info.reset_index()
    
    merged_df = pd.merge(df_consumption, cluster_info, on='ID', how='outer')
    return merged_df

# --- UTILITAIRES ---
def compute_optimal_threshold(y_true, y_scores):
    best = {
        'threshold': 0.50,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
    }
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best['f1']:
            best['threshold'] = float(threshold)
            best['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            best['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            best['f1'] = float(f1)
    return best


def get_reference_curve_for_evaluation(pdl_id, df_conso, n_days):
    ts = get_pdl_timeseries(pdl_id, df_conso)
    if ts.empty:
        return None
    if len(ts) < 1:
        return None
    return ts['daily_kwh'].tolist()

# --- INTERFACE STREAMLIT ---
def main():
    st.title("⚡ Dashboard de Forecasting Énergétique")

    # 1. Chargement des données
    try:
        df_conso = cached_load_data()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return

    # 2. Barre latérale (Contrôles Globaux)
    st.sidebar.header("Configuration")
    
    # Préparation des options pour la sidebar
    df_unique_pdl = df_conso[["ID", "cluster"]].dropna(subset=['cluster']).drop_duplicates()
    options = df_unique_pdl.apply(lambda x: f"{int(x['ID'])} | Cluster: {int(x['cluster'])}", axis=1).tolist()
    
    # Sélection unique dans la sidebar
    selection = st.sidebar.selectbox("Choisir le PDL (ID)", options, index=0)
    
    # Extraction globale de l'ID et du Cluster
    pdl_test_id = int(selection.split(" | ")[0])
    cluster_id = int(selection.split("Cluster: ")[1])
    
    # 3. Organisation en Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyse Historique", 
        "🎯 Backtesting & Prédiction",
        "🏷️ Analyse Classification",
        "🧠 Génération"
    ])

    # --- TAB 1 : HISTORIQUE ---
    with tab1:
        st.subheader(f"Historique du PDL : {pdl_test_id} (Cluster {cluster_id})")
        ts_full = get_pdl_timeseries(pdl_test_id, df_conso)
        st.line_chart(ts_full.set_index('date')['daily_kwh'])
        st.write(f"Nombre total de points : {len(ts_full)}")

    # --- TAB 4 : GÉNÉRATION ---
    with tab4:
        st.subheader("Génération de courbes synthétiques (GAN)")
        with st.expander("Générer des courbes synthétiques (GAN)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                synth_type = st.selectbox("Type de résidence", options=["principale", "secondaire"], index=0)
                synth_count = st.number_input("Nombre de courbes", min_value=1, max_value=50, value=3)
            with col2:
                # Option pour filtrer par cluster
                use_cluster = st.checkbox("Filtrer par cluster", value=True)
                if use_cluster:
                    # Filtrer les clusters selon le type de résidence
                    target_label = 0 if synth_type == "principale" else 1
                    available_clusters = sorted([c for c, l in cluster_to_label.items() if l == target_label])
                    
                    # Sélection par défaut : si le cluster du PDL sidebar est dans la liste, on le prend
                    default_idx = 0
                    if cluster_id in available_clusters:
                        default_idx = available_clusters.index(cluster_id)
                    
                    synth_cluster = st.selectbox("Numéro du cluster", options=available_clusters, index=default_idx)
                else:
                    synth_cluster = None
                
                synth_days = st.number_input("Jours par courbe", min_value=30, max_value=365, value=365)
            
            synth_seed = st.number_input("Seed (0 pour aléatoire)", min_value=0, value=0)
            
            if st.button("Générer et afficher"):
                seed_val = int(synth_seed) if synth_seed != 0 else None
                
                with st.spinner("Initialisation du modèle et calcul de la référence..."):
                    try:
                        # Initialiser le modèle une seule fois
                        model = ResidenceModel(
                            residence_type=synth_type,
                            seed=seed_val,
                            cluster_id=synth_cluster,
                            df_conso=df_conso
                        )
                        
                        # 1. Calculer la courbe de référence en utilisant les données du modèle
                        reference_curve = None
                        reference_dates = None
                        
                        if model.real_curves:
                            all_ref_vals = []
                            for curve in model.real_curves:
                                # Redimensionner chaque courbe réelle à la longueur demandée (interpolation)
                                if len(curve) != int(synth_days):
                                    indices = np.linspace(0, len(curve) - 1, int(synth_days))
                                    curve = np.interp(indices, np.arange(len(curve)), curve)
                                all_ref_vals.append(curve)
                            
                            if all_ref_vals:
                                reference_curve = np.mean(all_ref_vals, axis=0)
                                # Utiliser les mêmes dates de début que pour la génération
                                start_date = pd.to_datetime(df_conso['date'].min())
                                reference_dates = [start_date + timedelta(days=i) for i in range(int(synth_days))]
                    except Exception as e:
                        st.error(f"Erreur d'initialisation du modèle: {e}")
                        st.stop()
                
                with st.spinner("Génération des courbes synthétiques..."):
                    try:
                        # 2. Générer les courbes synthétiques
                        curves = []
                        start_date = pd.to_datetime(df_conso['date'].min())
                        for i in range(int(synth_count)):
                            synth_values = model.generate_synthetic_curve(n_days=int(synth_days))
                            
                            # Créer les timestamps et valeurs
                            timestamps = []
                            values = []
                            for day_idx, value in enumerate(synth_values):
                                date = start_date + timedelta(days=day_idx)
                                timestamps.append(date)
                                values.append(round(float(value), 4))
                            
                            curves.append({
                                "synthetic_id": i + 1,
                                "timestamps": timestamps,
                                "values": values,
                            })
                    except Exception as e:
                        st.error(f"Erreur lors de la génération: {e}")
                        st.stop()
                
                # 3. Créer le graphique
                fig = go.Figure()
                
                # Ajouter la courbe de référence en noir
                if reference_curve is not None and reference_dates is not None:
                    fig.add_trace(go.Scatter(
                        x=reference_dates, 
                        y=reference_curve, 
                        mode='lines', 
                        name='Moyenne de référence (Réel)',
                        line=dict(color='black', width=3),
                        opacity=0.9
                    ))
                
                # Ajouter les courbes synthétiques
                for c in curves:
                    fig.add_trace(go.Scatter(
                        x=c['timestamps'], 
                        y=c['values'], 
                        mode='lines', 
                        name=f"synth_{c['synthetic_id']}", 
                        opacity=0.8
                    ))
                
                title_suffix = f" (Cluster {synth_cluster})" if synth_cluster is not None else ""
                fig.update_layout(
                    title=f"Courbes synthétiques - {synth_type}{title_suffix}", 
                    xaxis_title='Date', 
                    yaxis_title='kWh',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Évaluation de la pertinence
                try:
                    if reference_curve is not None:
                        eval_scores = []
                        for c in curves:
                            r2 = compute_r2_similarity(reference_curve, c['values'])
                            eval_scores.append({'Courbe': f"synth_{c['synthetic_id']}", 'R2': r2})

                        df_eval = pd.DataFrame(eval_scores)
                        avg_r2 = df_eval['R2'].mean()

                        st.markdown("### 🔎 Pertinence des courbes générées")
                        st.markdown(
                            "Score R² calculé par rapport à la moyenne interpolée des données réelles du cluster/type sélectionné. "
                            "Plus le score est proche de 1, plus la forme et l'amplitude sont réalistes."
                        )
                        st.dataframe(df_eval.style.format({'R2': '{:.4f}'}), use_container_width=True)
                        st.metric("R² moyen", f"{avg_r2:.4f}")
                        
                        if avg_r2 >= 0.75:
                            st.success("Les courbes générées semblent globalement réalistes par rapport aux données réelles.")
                        elif avg_r2 >= 0.50:
                            st.info("Les courbes générées montrent une similarité raisonnable.")
                        else:
                            st.warning("Les courbes générées divergent significativement des données réelles.")
                    else:
                        st.markdown("### 🔎 Pertinence des courbes générées")
                        st.warning("Impossible de calculer la référence : aucune donnée réelle de longueur suffisante ou cluster vide.")
                except Exception as e:
                    st.error(f"Erreur lors du calcul du R² : {e}")

    # --- TAB 2 : BACKTESTING & PRÉDICTION ---
    with tab2:
        st.subheader("Évaluation de la performance")
        test_days = st.sidebar.slider("Jours de test (Backtest)", 1, 30, 14)

        # Lookback par modèle
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Lookback par modèle (jours)**")
        lookback_lr = st.sidebar.slider("Régression (linéaire)", 7, 365, 30)
        lookback_sarima = st.sidebar.slider("SARIMA", 30, 365, 90)
        lookback_lstm = st.sidebar.slider("LSTM", 7, 365, 14)

        # Explication dynamique des modèles selon la configuration choisie
        model_explanation = f"""
**Configuration sélectionnée :** Régression={lookback_lr}j, SARIMA={lookback_sarima}j, LSTM={lookback_lstm}j, Backtest={test_days}j.

- **Régression / Modèles linéaires :** rapide et stable, utilise les {lookback_lr} derniers jours pour estimer une tendance linéaire. Adapté aux comportements réguliers.
- **SARIMA (saisonnalité) :** capture saisonnalité et tendance; nécessite des séries suffisamment longues ({lookback_sarima}j). Instable si lookback < 30.
- **LSTM (Réseau de Neurones) :** apprend des patterns non-linéaires séquentiels; le temps d'entraînement augmente avec {lookback_lstm}j.
- **Ensemble :** moyenne/agrégation des prédictions ci‑dessus pour améliorer la robustesse.

Chaque modèle utilise son propre lookback configuré à gauche pour optimiser sa performance.
"""

        with st.expander("Description des modèles (basée sur la configuration)", expanded=False):
            st.markdown(model_explanation)

        if st.button("Lancer le Backtest"):
            with st.spinner("Entraînement des modèles en cours..."):
                res_backtest = evaluate_and_plot_backtest(
                    pdl_test_id,
                    df_conso,
                    test_days=test_days,
                    lookback_lr=lookback_lr,
                    lookback_sarima=lookback_sarima,
                    lookback_lstm=lookback_lstm
                )
                # Afficher les métriques retournées par le backtest (MAE et RMSE par modèle)
                if res_backtest and isinstance(res_backtest, dict) and 'metrics' in res_backtest:
                    metrics = res_backtest['metrics']
                    try:
                        rows = []
                        for name, vals in metrics.items():
                            if isinstance(vals, dict):
                                mae = vals.get('MAE')
                                rmse = vals.get('RMSE')
                            else:
                                mae = vals
                                rmse = None
                            rows.append({'Model': name, 'MAE': mae, 'RMSE': rmse})
                        df_metrics = pd.DataFrame(rows)
                        df_metrics['MAE'] = df_metrics['MAE'].astype(float)
                        if 'RMSE' in df_metrics.columns:
                            df_metrics['RMSE'] = df_metrics['RMSE'].astype(float)
                        st.markdown('### 📋 Métriques Backtest')
                        st.dataframe(df_metrics.style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}'}), use_container_width=True)
                    except Exception:
                        st.write(metrics)

        st.markdown("---")
        st.subheader("Prévisions du futur réel")
        forecast_days = st.number_input("Jours à prédire (Futur)", 1, 365, 7)

        if st.button("Générer les prévisions"):
            with st.spinner("Calcul en cours..."):
                resultats = forecast_consumption_trend(
                    pdl_test_id,
                    df_conso,
                    days_ahead=forecast_days,
                    lookback_lr=lookback_lr,
                    lookback_sarima=lookback_sarima,
                    lookback_lstm=lookback_lstm
                )
                
                if 'error' in resultats:
                    st.error(resultats['error'])
                else:
                    col1, col2 = st.columns(2)
                    col1.metric("Moyenne prédite", f"{resultats['mean_consumption']:.2f} kWh")
                    col2.metric("Date de départ", str(resultats['last_date']))

                    fig_forecast = plt.figure(figsize=(12, 6))
                    plot_forecast(
                        pdl_test_id, 
                        df_conso, 
                        resultats['dates'], 
                        ensemble_preds=resultats['predictions'],
                        all_models_dict=resultats['all_models_results']
                    )
                    st.pyplot(fig_forecast)
                    plt.close()

    # --- TAB 3 : ANALYSE CLASSIFICATION ---
    with tab3:
        st.subheader("🏷️ Analyse des Modèles de Classification")
        st.markdown("Comparaison des performances: Régression Logistique vs Réseau de Neurones")
        
        # Charger les données et modèles
        try:
            with st.spinner("Chargement des modèles et données..."):
                # Charger les features et labels
                features_pdl = get_features_pdl()
                labels = load_labels()
                
                # Fusion
                df_model = features_pdl.merge(
                    labels[["id", "label"]],
                    left_on="ID",
                    right_on="id",
                    how="inner"
                ).copy()
                
                X = df_model[feature_cols].replace([np.inf, -np.inf], np.nan)
                y = df_model["label"].astype(int).copy()
                
                # Split train/test (même config que classification.py)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )
                
                # Prétraitement
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                
                X_train_imputed = imputer.fit_transform(X_train)
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                
                X_test_imputed = imputer.transform(X_test)
                X_test_scaled = scaler.transform(X_test_imputed)
                
                # Paramètre visible pour la régression logistique : seul le seuil de décision
                st.markdown("#### Paramètre Régression Logistique")
                threshold = st.slider("Seuil de validation", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
                optimal_lr_placeholder = st.empty()
                optimal_nn_placeholder = st.empty()

                # (Cross-validation 5-fold removed — UI keeps only threshold)

                # Charger les modèles sauvegardés pour NN seulement
                models_dir = os.path.join(os.path.dirname(__file__), "models")
                nn_model_path = os.path.join(models_dir, "neural_network_model.pth")
                model_config_path = os.path.join(models_dir, "model_config.pkl")
                
                # Charger NN par défaut
                config = joblib.load(model_config_path)
                nn_model = SimpleNN(config['input_size'])
                nn_model.load_state_dict(torch.load(nn_model_path, map_location=torch.device('cpu')))
                nn_model.eval()
                y_score_nn = None
                y_pred_nn = None

                # Afficher la loss function du réseau de neurones
                if st.button("Afficher la loss du réseau de neurones"):
                    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
                    X_val_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                    model_nn = SimpleNN(config['input_size'])
                    criterion = torch.nn.BCELoss()
                    optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)

                    train_losses = []
                    val_losses = []
                    epochs = 30
                    # Early stopping parameters
                    patience = 3
                    best_val_loss = float('inf')
                    best_epoch = 0
                    counter = 0
                    stopped_epoch = None
                    best_model_state = None

                    for epoch in range(epochs):
                        model_nn.train()
                        epoch_loss = 0.0
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model_nn(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item() * X_batch.size(0)
                        train_losses.append(epoch_loss / len(train_dataset))

                        model_nn.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for X_batch, y_batch in val_loader:
                                outputs = model_nn(X_batch)
                                loss = criterion(outputs, y_batch)
                                val_loss += loss.item() * X_batch.size(0)
                        val_loss = val_loss / len(val_dataset)
                        val_losses.append(val_loss)

                        # Early stopping check
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = epoch + 1
                            best_model_state = model_nn.state_dict().copy()
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                stopped_epoch = epoch + 1
                                break
                    # Charger le meilleur modèle trouvé si disponible
                    if best_model_state is not None:
                        model_nn.load_state_dict(best_model_state)

                    n_epochs_done = len(train_losses)
                    with st.expander("Courbe de loss - Réseau de Neurones", expanded=True):
                        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                        ax_loss.plot(list(range(1, n_epochs_done + 1)), train_losses, label='Loss apprentissage')
                        ax_loss.plot(list(range(1, n_epochs_done + 1)), val_losses, label='Loss test')
                        ax_loss.set_xlabel('Epoch')
                        ax_loss.set_ylabel('Loss')
                        ax_loss.set_title('Courbe de loss - Réseau de Neurones')
                        ax_loss.legend()
                        ax_loss.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig_loss)
                        plt.close(fig_loss)
                    # Afficher info early stopping
                    if stopped_epoch is not None:
                        st.success(f"Early stopping déclenché à l'époque {stopped_epoch} (meilleur modèle à l'époque {best_epoch})")
                    else:
                        st.info(f"Entraînement terminé sur {n_epochs_done} époques (meilleur modèle à l'époque {best_epoch})")

                    y_score_nn = model_nn(X_val_tensor).detach().numpy().squeeze()
                    y_pred_nn = (y_score_nn >= 0.5).astype(int)

                # Prédictions par défaut du réseau de neurones pré-entraîné
                time_start_nn = time.time()
                
                if y_pred_nn is None:
                    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        y_score_nn = nn_model(X_test_tensor).squeeze().numpy()
                    y_pred_nn = (y_score_nn >= 0.5).astype(int)
                
                time_end_nn = time.time()
                time_nn = time_end_nn - time_start_nn

                # Entraîner la régression logistique sur les données prétraitées (hyperparamètres fixes)
                time_start_lr = time.time()
                
                lr_model = LogisticRegression(
                    random_state=42,
                    penalty='l2',
                    C=1.0,
                    solver='liblinear',
                    max_iter=2000,
                    class_weight='balanced'
                )
                lr_model.fit(X_train_scaled, y_train)

                # Prédictions - Régression Logistique
                y_score_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
                y_pred_lr = (y_score_lr >= float(threshold)).astype(int)
                
                time_end_lr = time.time()
                time_lr = time_end_lr - time_start_lr

                # Seuil optimal calculé sur les métriques d'évaluation (maximisation du F1)
                optimal_threshold_lr = compute_optimal_threshold(y_test, y_score_lr)
                optimal_lr_placeholder.success(
                    f"Seuil optimal LR : {optimal_threshold_lr['threshold']:.2f} "
                    f"(F1={optimal_threshold_lr['f1']:.4f}, Precision={optimal_threshold_lr['precision']:.4f}, Recall={optimal_threshold_lr['recall']:.4f})",
                    icon="✅"
                )

                optimal_threshold_nn = compute_optimal_threshold(y_test, y_score_nn)
                y_pred_nn = (y_score_nn >= optimal_threshold_nn['threshold']).astype(int)
                optimal_nn_placeholder.success(
                    f"Seuil optimal NN : {optimal_threshold_nn['threshold']:.2f} "
                    f"(F1={optimal_threshold_nn['f1']:.4f}, Precision={optimal_threshold_nn['precision']:.4f}, Recall={optimal_threshold_nn['recall']:.4f})",
                    icon="✅"
                )

        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")
            st.stop()
        
        # === SECTION 2 : MATRICES DE CONFUSION ===
        st.markdown("### Matrices de Confusion - Comparaison des Modèles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Régression Logistique")
            cm_lr = confusion_matrix(y_test, y_pred_lr)
            
            fig_cm_lr, ax_lr = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_lr,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=['Principale', 'Secondaire'],
                yticklabels=['Principale', 'Secondaire'],
                ax=ax_lr,
                annot_kws={'size': 14, 'weight': 'bold'}
            )
            ax_lr.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
            ax_lr.set_ylabel('Réalité', fontsize=12, fontweight='bold')
            ax_lr.set_title('Matrice de Confusion\nRégression Logistique', fontsize=12, fontweight='bold')
            st.pyplot(fig_cm_lr)
            plt.close()
        
        with col2:
            st.markdown("#### Réseau de Neurones")
            cm_nn = confusion_matrix(y_test, y_pred_nn)
            
            fig_cm_nn, ax_nn = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_nn,
                annot=True,
                fmt='d',
                cmap='Greens',
                cbar=True,
                square=True,
                xticklabels=['Principale', 'Secondaire'],
                yticklabels=['Principale', 'Secondaire'],
                ax=ax_nn,
                annot_kws={'size': 14, 'weight': 'bold'}
            )
            ax_nn.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
            ax_nn.set_ylabel('Réalité', fontsize=12, fontweight='bold')
            ax_nn.set_title('Matrice de Confusion\nRéseau de Neurones', fontsize=12, fontweight='bold')
            st.pyplot(fig_cm_nn)
            plt.close()
        
        # === SECTION 3 : MÉTRIQUES D'ÉVALUATION ===
        st.markdown("### Métriques d'Évaluation - Comparaison Détaillée")
        
        # Calculer les métriques
        metrics_lr = {
            'Precision': precision_score(y_test, y_pred_lr, zero_division=0),
            'Recall': recall_score(y_test, y_pred_lr, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_lr, zero_division=0),
        }
        
        metrics_nn = {
            'Precision': precision_score(y_test, y_pred_nn, zero_division=0),
            'Recall': recall_score(y_test, y_pred_nn, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred_nn, zero_division=0),
        }
        
        # Afficher les métriques sous forme de tableau
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Precision")
            st.metric(
                "Régression Logistique",
                f"{metrics_lr['Precision']:.4f}",
                delta=f"{(metrics_lr['Precision'] - metrics_nn['Precision']):.4f}",
                delta_color="inverse"
            )
            st.metric(
                "Réseau de Neurones",
                f"{metrics_nn['Precision']:.4f}"
            )
        
        with col2:
            st.markdown("#### 🎯 Recall")
            st.metric(
                "Régression Logistique",
                f"{metrics_lr['Recall']:.4f}",
                delta=f"{(metrics_lr['Recall'] - metrics_nn['Recall']):.4f}",
                delta_color="inverse"
            )
            st.metric(
                "Réseau de Neurones",
                f"{metrics_nn['Recall']:.4f}"
            )
        
        with col3:
            st.markdown("#### ⚖️ F1-Score")
            st.metric(
                "Régression Logistique",
                f"{metrics_lr['F1-Score']:.4f}",
                delta=f"{(metrics_lr['F1-Score'] - metrics_nn['F1-Score']):.4f}",
                delta_color="inverse"
            )
            st.metric(
                "Réseau de Neurones",
                f"{metrics_nn['F1-Score']:.4f}"
            )
        
        # Tableau comparatif détaillé
        st.markdown("---")
        st.markdown("#### 📈 Tableau Comparatif Complet")
        
        comparison_data = {
            'Métrique': ['Precision', 'Recall', 'F1-Score'],
            'Régression Logistique': [
                f"{metrics_lr['Precision']:.4f}",
                f"{metrics_lr['Recall']:.4f}",
                f"{metrics_lr['F1-Score']:.4f}",
            ],
            'Réseau de Neurones': [
                f"{metrics_nn['Precision']:.4f}",
                f"{metrics_nn['Recall']:.4f}",
                f"{metrics_nn['F1-Score']:.4f}",
            ],
            'Différence (NN - LR)': [
                f"{(metrics_nn['Precision'] - metrics_lr['Precision']):+.4f}",
                f"{(metrics_nn['Recall'] - metrics_lr['Recall']):+.4f}",
                f"{(metrics_nn['F1-Score'] - metrics_lr['F1-Score']):+.4f}",
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # === SECTION 4 : INSIGHTS ===
        st.markdown("---")
        st.markdown("### 💡 Insights & Recommandations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Régression Logistique")
            with st.container(border=True):
                st.markdown(f"""
                ✅ **Avantages:**
                - Modèle simple et interprétable
                - Entraînement rapide
                - Précision: {metrics_lr['Precision']:.1%}
                - Recall: {metrics_lr['Recall']:.1%}
                - F1-Score: {metrics_lr['F1-Score']:.4f}
                - ⏱️ Temps: {time_lr:.4f}s
                
                ⚠️ **Limitations:**
                - Peut être moins bonne pour patterns non-linéaires
                """)
        
        with col2:
            st.markdown("#### Réseau de Neurones")
            with st.container(border=True):
                st.markdown(f"""
                ✅ **Avantages:**
                - Capture patterns complexes
                - Plus flexible
                - Précision: {metrics_nn['Precision']:.1%}
                - Recall: {metrics_nn['Recall']:.1%}
                - F1-Score: {metrics_nn['F1-Score']:.4f}
                - ⏱️ Temps: {time_nn:.4f}s
                
                ⚠️ **Limitations:**
                - Moins interprétable ("boîte noire")
                - Entraînement plus lent
                """)

        
        # Recommandation finale
        st.markdown("---")
        meilleur_modele = "Réseau de Neurones" if metrics_nn['F1-Score'] > metrics_lr['F1-Score'] else "Régression Logistique"
        diff_f1 = abs(metrics_nn['F1-Score'] - metrics_lr['F1-Score'])
        
        if diff_f1 > 0.05:
            st.success(f"🏆 **{meilleur_modele}** est significativement meilleur (F1 difference: {diff_f1:.4f})", icon="✨")
        else:
            st.info(f"⚖️ Les deux modèles ont des performances comparables (F1 difference: {diff_f1:.4f})", icon="⚖️")
            st.markdown("**Recommandation:** Utiliser la **Régression Logistique** en production (plus rapide et interprétable)")


if __name__ == "__main__":
    main()
