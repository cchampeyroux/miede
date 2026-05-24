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
from src.generation import generate_synthetic_curves_with_model
from src.clustering import get_features_pdl
from src.classification import (
    load_labels,
    feature_cols,
    SimpleNN
)

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

# --- INTERFACE STREAMLIT ---
def main():
    st.title("⚡ Dashboard de Forecasting Énergétique")
    st.markdown("Analyse et prédiction multimodèle (LR, SARIMA, LSTM)")

    # 1. Chargement des données
    try:
        df_conso = cached_load_data()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return

    # 2. Barre latérale (Contrôles Globaux)
    st.sidebar.header("Configuration")
    
    # Préparation des options pour la sidebar
    df_unique_pdl = df_conso[["ID", "cluster"]].drop_duplicates()
    options = df_unique_pdl.apply(lambda x: f"{x['ID']} | Cluster: {x['cluster']}", axis=1).tolist()
    
    # Sélection unique dans la sidebar
    selection = st.sidebar.selectbox("Choisir le PDL (ID)", options, index=0)
    
    # Extraction globale de l'ID et du Cluster
    pdl_test_id = int(selection.split(" | ")[0])
    cluster_id = int(selection.split("Cluster: ")[1])
    
    # 3. Organisation en Onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analyse Historique", 
        "🎯 Backtesting", 
        "🚀 Prédiction Futur",
        "🏷️ Analyse Classification",
        "🧠 Génération"
    ])

    # --- TAB 1 : HISTORIQUE ---
    with tab1:
        st.subheader(f"Historique du PDL : {pdl_test_id} (Cluster {cluster_id})")
        ts_full = get_pdl_timeseries(pdl_test_id, df_conso)
        st.line_chart(ts_full.set_index('date')['daily_kwh'])
        st.write(f"Nombre total de points : {len(ts_full)}")

    # --- TAB 5 : GÉNÉRATION ---
    with tab5:
        st.subheader("Génération de courbes synthétiques (GAN)")
        with st.expander("Générer des courbes synthétiques (GAN)", expanded=True):
            synth_type = st.selectbox("Type de résidence", options=["principale", "secondaire"], index=0)
            synth_count = st.number_input("Nombre de courbes", min_value=1, max_value=50, value=3)
            synth_days = st.number_input("Jours par courbe", min_value=30, max_value=365, value=365)
            synth_seed = st.number_input("Seed (0 pour aléatoire)", min_value=0, value=0)
            if st.button("Générer et afficher"):
                seed_val = int(synth_seed) if synth_seed != 0 else None
                with st.spinner("Génération en cours..."):
                    curves = generate_synthetic_curves_with_model(
                        residence_type=synth_type,
                        n_curves=int(synth_count),
                        n_days=int(synth_days),
                        seed=seed_val,
                    )
                fig = go.Figure()
                for c in curves:
                    fig.add_trace(go.Scatter(x=c['timestamps'], y=c['values'], mode='lines', name=f"synth_{c['synthetic_id']}", opacity=0.8))
                fig.update_layout(title=f"Courbes synthétiques - {synth_type}", xaxis_title='Date', yaxis_title='kWh')
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 : BACKTESTING ---
    with tab2:
        st.subheader("Évaluation de la performance")
        lookback_window = st.sidebar.slider("Fenêtre d'historique (Lookback)", 3, 365, 14)
        test_days = st.sidebar.slider("Jours de test (Backtest)", 1, 30, 14)

        if st.button("Lancer le Backtest"):
            with st.spinner("Entraînement des modèles en cours..."):
                evaluate_and_plot_backtest(
                    pdl_test_id, 
                    df_conso, 
                    test_days=test_days, 
                    lookback=lookback_window
                )
                st.pyplot(plt.gcf())
                plt.clf()

    # --- TAB 3 : PRÉDICTION ---
    with tab3:
        st.subheader("Prévisions du futur réel")
        forecast_days = st.number_input("Jours à prédire (Futur)", 1, 30, 7)

        if st.button("Générer les prévisions"):
            with st.spinner("Calcul en cours..."):
                resultats = forecast_consumption_trend(
                    pdl_test_id, 
                    df_conso, 
                    days_ahead=forecast_days, 
                    lookback=lookback_window
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

    # --- TAB 4 : ANALYSE CLASSIFICATION ---
    with tab4:
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
                
                # Paramètres de la régression logistique contrôlables
                st.markdown("#### Paramètres Régression Logistique")
                col_param1, col_param2, col_param3 = st.columns(3)
                with col_param1:
                    penalty = st.selectbox("Pénalité", options=["l2", "l1", "none"], index=0)
                    C = st.number_input("Inverse de régularisation C", min_value=0.01, max_value=100.0, value=1.0, step=0.01, format="%.2f")
                with col_param2:
                    solver = st.selectbox("Solver", options=["liblinear", "lbfgs", "saga"], index=0)
                    max_iter = st.number_input("Max itérations", min_value=100, max_value=5000, value=2000, step=100)
                with col_param3:
                    threshold = st.slider("Seuil de décision", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
                    class_weight = st.selectbox("Pondération des classes", options=["balanced", "None"], index=0)

                # Forcer les combinaisons de solver compatibles
                if penalty == "l1" and solver == "lbfgs":
                    solver = "liblinear"
                if penalty == "none" and solver == "liblinear":
                    solver = "lbfgs"
                if penalty == "none" and solver == "saga":
                    solver = "lbfgs"

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
                        val_losses.append(val_loss / len(val_dataset))

                    with st.expander("Courbe de loss - Réseau de Neurones", expanded=True):
                        fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
                        ax_loss.plot(list(range(1, epochs + 1)), train_losses, label='Loss apprentissage')
                        ax_loss.plot(list(range(1, epochs + 1)), val_losses, label='Loss validation')
                        ax_loss.set_xlabel('Epoch')
                        ax_loss.set_ylabel('Loss')
                        ax_loss.set_title('Courbe de loss - Réseau de Neurones')
                        ax_loss.legend()
                        ax_loss.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig_loss)
                        plt.close(fig_loss)

                    y_score_nn = model_nn(X_val_tensor).detach().numpy().squeeze()
                    y_pred_nn = (y_score_nn >= 0.5).astype(int)

                # Prédictions par défaut du réseau de neurones pré-entraîné
                if y_pred_nn is None:
                    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        y_score_nn = nn_model(X_test_tensor).squeeze().numpy()
                    y_pred_nn = (y_score_nn >= 0.5).astype(int)

                # Entraîner la régression logistique sur les données prétraitées
                lr_model = LogisticRegression(
                    random_state=42,
                    penalty=penalty if penalty != "none" else None,
                    C=float(C),
                    solver=solver,
                    max_iter=int(max_iter),
                    class_weight=None if class_weight == "None" else class_weight,
                    l1_ratio=0.0 if penalty != "elasticnet" else 0.5,
                )
                lr_model.fit(X_train_scaled, y_train)

                # Prédictions - Régression Logistique
                y_score_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
                y_pred_lr = (y_score_lr >= float(threshold)).astype(int)
                
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles : {e}")
            st.stop()
        
        # === SECTION 2 : MATRICES DE CONFUSION ===
        st.markdown("### 2️⃣ Matrices de Confusion - Comparaison des Modèles")
        
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
        st.markdown("### 3️⃣ Métriques d'Évaluation - Comparaison Détaillée")
        
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
                
                ⚠️ **Limitations:**
                - Peut être suboptimale pour patterns non-linéaires
                - Performance: {metrics_lr['F1-Score']:.1%}
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
                
                ⚠️ **Limitations:**
                - Moins interprétable ("boîte noire")
                - Entraînement plus lent
                - Performance: {metrics_nn['F1-Score']:.1%}
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