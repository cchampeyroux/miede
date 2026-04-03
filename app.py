import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.forecast import load_consumption_data,get_pdl_timeseries,evaluate_and_plot_backtest,forecast_consumption_trend,plot_forecast
# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Energy Forecast Dashboard", layout="wide")

# --- CHARGEMENT DES DONNÉES (AVEC CACHE) ---
@st.cache_data
def cached_load_data():
    return load_consumption_data()

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

    # 2. Barre latérale (Contrôles)
    st.sidebar.header("Configuration")
    
    pdl_list = df_conso["ID"].unique()
    pdl_test_id = st.sidebar.selectbox("Choisir le PDL (ID)", pdl_list, index=0)

    # 3. Organisation en Onglets
    tab1, tab2, tab3 = st.tabs(["📊 Analyse Historique", "🎯 Backtesting", "🚀 Prédiction Futur"])

    with tab1:
        st.subheader(f"Historique du PDL : {pdl_test_id}")
        ts_full = get_pdl_timeseries(pdl_test_id, df_conso)
        st.line_chart(ts_full.set_index('date')['daily_kwh'])
        st.write(f"Nombre total de points : {len(ts_full)}")

    with tab2:
        st.subheader("Évaluation de la performance")
        lookback_window = st.slider("Fenêtre d'historique (Lookback)", 7, 60, 14)
        test_days = st.slider("Jours de test (Backtest)", 7, 30, 14)

        if st.button("Lancer le Backtest"):
            with st.spinner("Entraînement des modèles en cours..."):
                # On utilise ta fonction de backtest existante
                # Petite astuce : Streamlit capture les plt.show(), 
                # mais il vaut mieux utiliser st.pyplot()
                evaluate_and_plot_backtest(
                    pdl_test_id, 
                    df_conso, 
                    test_days=test_days, 
                    lookback=lookback_window
                )
                st.pyplot(plt.gcf()) # Affiche le dernier graphique généré
                plt.clf()

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
                    # Affichage des métriques
                    col1, col2 = st.columns(2)
                    col1.metric("Moyenne prédite", f"{resultats['mean_consumption']:.2f} kWh")
                    col2.metric("Date de départ", str(resultats['last_date']))

                    # Affichage du graphique
                    fig_forecast = plt.figure(figsize=(12, 6))
                    plot_forecast(
                        pdl_test_id, 
                        df_conso, 
                        resultats['dates'], 
                        ensemble_preds=resultats['predictions'],
                        all_models_dict=resultats['all_models_results']
                    )
                    st.pyplot(plt.gcf())
                    plt.close()

if __name__ == "__main__":
    main()