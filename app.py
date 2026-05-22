import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from src.forecast import (
    load_consumption_data,
    get_pdl_timeseries,
    evaluate_and_plot_backtest,
    forecast_consumption_trend,
    plot_forecast,
    load_cluster_assignments
)
from src.generation import generate_synthetic_curves_with_model

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
    tab1, tab2, tab3= st.tabs([
        "📊 Analyse Historique", 
        "🎯 Backtesting", 
        "🚀 Prédiction Futur",
    ])

    # --- TAB 1 : HISTORIQUE ---
    with tab1:
        st.subheader(f"Historique du PDL : {pdl_test_id} (Cluster {cluster_id})")
        ts_full = get_pdl_timeseries(pdl_test_id, df_conso)
        st.line_chart(ts_full.set_index('date')['daily_kwh'])
        st.write(f"Nombre total de points : {len(ts_full)}")
        # Génération synthétique (GAN) - optionnel
        with st.expander("Générer des courbes synthétiques (GAN)"):
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

if __name__ == "__main__":
    main()