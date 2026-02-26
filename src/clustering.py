# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:22:06 2026

@author: cleli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.io as pio
import os


#Préparation des données
#DATA_PATH = "C:/Users/cleli/OneDrive/Documents/ENPC/M1EDE/RES2-6-9/RES2-6-9.csv"
DATA_PATH = "../data/RES2-6-9.csv"

path_src = "data/RES2-6-9.csv"
path_parent = "../data/RES2-6-9.csv"

final_path = path_parent

# On teste lequel des deux fonctionne pour ne plus avoir d'erreur

if os.path.exists(path_src):
    final_path = path_src
else :
    final_path = path_parent


COL_PDL = "ID"
COL_DT = "horodate"
COL_PWR = "valeur"
#df = pd.read_csv(DATA_PATH, sep=";")
df = pd.read_csv(final_path, sep=";", parse_dates=[COL_DT])
df["horodate"] = pd.to_datetime(df["horodate"], utc=True)


print(df.dtypes)
df.head()


#Ajout des colonnes

df["date"] = df[COL_DT].dt.date
df["hour"] = df[COL_DT].dt.hour + df[COL_DT].dt.minute / 60
df["dow"] = df[COL_DT].dt.dayofweek # 0 = lundi
df["is_weekend"] = df["dow"] >= 5
df["hh_index"] = ((df[COL_DT].dt.hour * 60) + df[COL_DT].dt.minute) // 30

print(df.head())



print("n_rows:", len(df))
print("n_clients:", df[COL_PDL].nunique())
print(df.dtypes)



# énergie journalière par PDL (puissance * Pas de temps de 30min)
daily = (df
    .assign(energy_kwh_step=df[COL_PWR] * 0.5)
    .groupby([COL_PDL, "date"], as_index=False)
    .agg(
        daily_kwh=("energy_kwh_step", "sum"),
        daily_mean_kw=(COL_PWR, "mean"),
        daily_max_kw=(COL_PWR, "max"),
        n_steps=(COL_PWR, "size"),
    )
)


#on veut détecter les jours d'activité



# choix d'un seuil TH : par ex. quantile 20% des journées positives
#non_zero = daily["energy_kwh"][daily["energy_kwh"] > 0]
#TH = np.quantile(non_zero, 0.2)


#les jours actifs

#daily["is_active_day"] = daily["energy_kwh"] >= TH


# seuil par client sur les jours > 0
def q20_positive(s: pd.Series):
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    return s.quantile(0.2)

#Le seuil th_pdl pour chaque client est défini comme le quantile 20% (Q20)
daily["th_pdl"] = (daily.groupby(COL_PDL)["daily_kwh"]
                   .transform(q20_positive))

# fallback si un client n'a pas de jours >0 : on le marque jamais actif
daily["is_active_day"] = (daily["daily_kwh"] >= daily["th_pdl"]).fillna(False)






##Taux d'occupation global et saisonnier

daily["month"] = pd.to_datetime(daily["date"]).dt.month


def season(m):
    if m in [12, 1, 2]:
        return "winter"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "mid"


#trier les données par saison
daily["season"] = daily["month"].apply(season)




#enregistrer daily

daily.to_csv("daily.csv", index=False)

#Pour recuperer daily direct
daily = pd.read_csv("daily.csv", parse_dates=["date"])






### Paramètre au niveau client

daily2 = daily.copy()
daily2["date_ts"] = pd.to_datetime(daily2["date"])
daily2["month"] = daily2["date_ts"].dt.month


#Saisons
def season_from_month(m):
    # à adapter si vous avez une définition métier différente
    if m in (12, 1, 2):
        return "winter"
    if m in (6, 7, 8):
        return "summer"
    return "mid"

daily2["season"] = daily2["month"].map(season_from_month)



#season_stats pour avoir les comportement de saisonnailité
season_stats = (daily2
    .groupby([COL_PDL, "season"], as_index=False)
    .agg(mean_daily_kwh=("daily_kwh", "mean"))
    .pivot(index=COL_PDL, columns="season", values="mean_daily_kwh")
    .reset_index()
)

# colonnes manquantes -> 0 (si un client n'a pas de points dans une saison)
for c in ["winter", "summer", "mid"]:
    if c not in season_stats.columns:
        season_stats[c] = 0.0

global_mean = (daily2.groupby(COL_PDL, as_index=False)
               .agg(mean_daily_kwh_global=("daily_kwh", "mean")))

season_stats = season_stats.merge(global_mean, on=COL_PDL, how="left", validate="one_to_one")

eps = 1e-9
season_stats["r_global"] = 1.0  # par définition (référence)
season_stats["r_mid"]    = season_stats["mid"]    / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_summer"] = season_stats["summer"] / (season_stats["mean_daily_kwh_global"] + eps)
season_stats["r_winter"] = season_stats["winter"] / (season_stats["mean_daily_kwh_global"] + eps)

season_stats = season_stats[[COL_PDL, "r_global", "r_mid", "r_summer", "r_winter"]]  




#feature d'activité et de conso globale

activity = (daily
    .groupby(COL_PDL, as_index=False)
    .agg(
        n_days=(COL_PDL, "size"),
        n_active_days=("is_active_day", "sum"),
        active_day_rate=("is_active_day", "mean"),
        mean_daily_kwh=("daily_kwh", "mean"),
        p95_daily_kwh=("daily_kwh", lambda s: s.quantile(0.95)),
        cv_daily_kwh=("daily_kwh", lambda s: (s.std() / s.mean()) if s.mean() != 0 else np.nan),
    )
)



#features sur les séquences d'activité et d'inactivité
#Pour mieux caractériser l'occupation,
#nous analysons les séquences de jours actifs consécutifs (runs) et de jours inactifs consécutifs (gaps).
def runs_and_gaps(active_series: pd.Series):
    # active_series: bool list in chronological order
    runs = []
    gaps = []
    run = 0
    gap = 0
    
    for v in active_series.astype(bool):
        if v:
            run += 1
            if gap > 0:
                gaps.append(gap)
                gap = 0
        else:
            gap += 1
            if run > 0:
                runs.append(run)
                run = 0

    if run > 0:
        runs.append(run)
    if gap > 0:
        gaps.append(gap)

    return pd.Series({
        "n_runs": len(runs),
        "mean_run_len": float(np.mean(runs)) if runs else 0.0,
        "max_run_len": float(np.max(runs)) if runs else 0.0,
        "mean_gap_len": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_len": float(np.max(gaps)) if gaps else 0.0,
    })

runs_stats = (daily
    .sort_values([COL_PDL, "date"])
    .groupby(COL_PDL)["is_active_day"]
    .apply(runs_and_gaps)
    .unstack()
    .reset_index()
)




#features de profil hebdomadaire (semaine vs. week-end)

daily_dt = pd.to_datetime(daily["date"])
daily["dow"] = daily_dt.dt.dayofweek
daily["is_weekend"] = daily["dow"] >= 5

week_pattern = (daily
    .groupby([COL_PDL, "is_weekend"], as_index=False)
    .agg(active_rate=("is_active_day", "mean"),
         mean_kwh=("daily_kwh", "mean"))
    .pivot(index=COL_PDL, columns="is_weekend")
)

week_pattern.columns = [f"{a}_{'weekend' if b else 'weekday'}" for a, b in week_pattern.columns]
week_pattern = week_pattern.reset_index()




#Assemblage final des features
#features_pdl fusionnent tout en un unique df
features_pdl = (activity
    .merge(runs_stats, on=COL_PDL, how="left", validate="one_to_one")
    .merge(week_pattern, on=COL_PDL, how="left", validate="one_to_one")
    .merge(season_stats, on=COL_PDL, how="left", validate="one_to_one")
)

assert features_pdl[COL_PDL].is_unique, "ERREUR: plus d'une ligne par client => un merge a explosé"
print("OK: 1 ligne par client:", len(features_pdl), "clients:", features_pdl[COL_PDL].nunique())





#Création de features composites
# Amplitude de la saisonnalité (max des ratios saisonniers - min des ratios saisonniers).
features_pdl["seasonality_amp"] = features_pdl[["r_mid","r_summer","r_winter"]].max(axis=1) - features_pdl[["r_mid","r_summer","r_winter"]].min(axis=1)
#Différence entre le ratio d'hiver et d'été. Une valeur très positive indique un chauffage électrique important, typique d'une RP. Une valeur négative peut indiquer une RS d'été avec climatisation.
features_pdl["winter_minus_summer"] = features_pdl["r_winter"] - features_pdl["r_summer"]








###Clustering
#Sélection et préparation des données pour le clustering

feature_cols = [
    "active_day_rate", "n_runs", "mean_run_len", "max_run_len",
    "mean_gap_len", "max_gap_len",
    "mean_daily_kwh", "p95_daily_kwh", "cv_daily_kwh",
    "active_rate_weekday", "active_rate_weekend",
    "mean_kwh_weekday", "mean_kwh_weekend", "winter_minus_summer", 
    "seasonality_amp", "r_global",	"r_mid", "r_summer", "r_winter",
]

X = features_pdl[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Choix du nombre de clusters (K)

scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
print(best_k)

#on choisit 10 peut révéler des clutsers plus subtils
best_k = 10
kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
features_pdl["cluster"] = kmeans.fit_predict(X_scaled)
features_pdl[["ID","cluster"]].head()













###Analyse et Interprétation des Clusters
SHOW_PLOTS = False

cluster_profile = features_pdl.groupby("cluster")[feature_cols].mean().sort_index()

if SHOW_PLOTS:
    fig.show()


import plotly.express as px
import plotly.graph_objects as go

def plot_client_daily(pdl_id, daily_df, n_days=200):
    sub = daily_df[daily_df[COL_PDL] == pdl_id].sort_values("date").tail(n_days).copy()
    sub["date_ts"] = pd.to_datetime(sub["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["date_ts"], y=sub["daily_kwh"],
        mode="lines+markers", name="daily_kwh"
    ))

    active = sub[sub["is_active_day"]]
    fig.add_trace(go.Scatter(
        x=active["date_ts"], y=active["daily_kwh"],
        mode="markers", name="active_day"
    ))

    fig.update_layout(title=f"PDL {pdl_id} : daily_kwh + jours actifs (last {n_days} days)",
                      xaxis_title="date", yaxis_title="kWh/jour")
    if SHOW_PLOTS:
        fig.show()

for c in sorted(features_pdl["cluster"].unique()):
    sample = features_pdl[features_pdl["cluster"] == c].sample(min(1, (features_pdl["cluster"]==c).sum()), random_state=42)
    print("Cluster", c, "samples:", sample[COL_PDL].tolist())
    for pid in sample[COL_PDL].tolist():
        plot_client_daily(pid, daily, n_days=250)




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- prépare X ---
X = features_pdl[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_scaled = StandardScaler().fit_transform(X)

# --- PCA 2D ---
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X_scaled)

# X = features_pdl[feature_cols] (avant scaling) est déjà un DataFrame avec les colonnes feature_cols

pca_df = pd.concat(
    [
        pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1]}),
        features_pdl[[COL_PDL, "cluster"] + feature_cols].reset_index(drop=True)
    ],
    axis=1
)

pca_df["cluster"] = pca_df["cluster"].astype(str)
pca_df[COL_PDL] = pca_df[COL_PDL].astype(str)

fig = px.scatter(
    pca_df,
    x="PC1", y="PC2",
    color="cluster",
    hover_data=[COL_PDL] + feature_cols,
    title=(
        f"PCA (2D) — variance expliquée: "
        f"PC1={pca.explained_variance_ratio_[0]:.2%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.2%}"
    )
)
fig.update_traces(marker=dict(size=7, opacity=0.75))
if SHOW_PLOTS:
    fig.show()






# rattacher le cluster à daily
daily_cluster = daily.merge(
    features_pdl[[COL_PDL, "cluster"]],
    on=COL_PDL,
    how="left",
    validate="many_to_one"
)

assert daily_cluster["cluster"].notna().all()



mean_daily_cluster = (
    daily_cluster
    .groupby(["cluster", "date"], as_index=False)
    .agg(mean_daily_kwh=("daily_kwh", "mean"))
)



fig = px.line(
    mean_daily_cluster,
    x="date",
    y="mean_daily_kwh",
    color=mean_daily_cluster["cluster"].astype(str),
    title="Courbe moyenne journalière par cluster",
    labels={"mean_daily_kwh": "kWh / jour", "cluster": "cluster"}
)
if SHOW_PLOTS:
    fig.show()






df_cluster = df.merge(
    features_pdl[[COL_PDL, "cluster"]],
    on=COL_PDL,
    how="left",
    validate="many_to_one"
)

assert df_cluster["cluster"].notna().all()

mean_intraday_cluster = (
    df_cluster
    .groupby(["cluster", "hh_index"], as_index=False)
    .agg(mean_kw=(COL_PWR, "mean"))
)

mean_intraday_cluster["cluster"] = mean_intraday_cluster["cluster"].astype(str)



fig = px.line(
    mean_intraday_cluster,
    x="hh_index",
    y="mean_kw",
    color="cluster",
    title="Profil intrajournalier moyen par cluster (30 min)",
    labels={
        "hh_index": "Demi-heure (0 = 00:00)",
        "mean_kw": "Puissance moyenne (kW)",
        "cluster": "cluster"
    }
)

fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 48, 4)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
    )
)

if SHOW_PLOTS:
    fig.show()





df_cluster["is_weekend"] = df_cluster[COL_DT].dt.dayofweek >= 5

mean_intraday_wk = (
    df_cluster
    .groupby(["cluster", "is_weekend", "hh_index"], as_index=False)
    .agg(mean_kw=(COL_PWR, "mean"))
)

mean_intraday_wk["cluster"] = mean_intraday_wk["cluster"].astype(str)
mean_intraday_wk["day_type"] = mean_intraday_wk["is_weekend"].map({True: "weekend", False: "weekday"})

fig = px.line(
    mean_intraday_wk,
    x="hh_index",
    y="mean_kw",
    color="cluster",
    facet_col="day_type",
    title="Profil intrajournalier moyen par cluster — semaine vs week-end"
)

fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(0, 48, 4)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
    )
)

if SHOW_PLOTS:
    fig.show()



##Synthèse pour l'interprétation



summary = (features_pdl
    .groupby("cluster")
    .agg(
        n_clients=(COL_PDL, "size"),
        active_day_rate=("active_day_rate", "mean"),
        max_gap_len=("max_gap_len", "mean"),
        winter_minus_summer=("winter_minus_summer", "mean"),
        cv_daily_kwh=("cv_daily_kwh", "mean"),
        r_summer=("r_summer", "mean")
    )
)




#Interprétation des labels

# --- 1. Définition des étiquettes binaires (0 pour RP, 1 sinon) ---
# Basé sur notre interprétation des clusters
# RP : 0, 3, 5, 8, 9
# RS/Atypique : 1, 2, 4, 6, 7
cluster_to_label = {
    0: 0,
    3: 0,
    5: 0,
    8: 0,
    9: 0,
    1: 1,
    2: 1,
    4: 1,
    6: 1,
    7: 1
}





def get_features_pdl():
    return features_pdl



# --- 2. Création de la table de labels ---
# On part du dataframe `features_pdl` qui contient le `pdl_id` et le `cluster`
labels_df = features_pdl[[COL_PDL, "cluster"]].copy()

# On utilise la méthode .map() pour créer la colonne "label"
labels_df["label"] = labels_df["cluster"].map(cluster_to_label)

# On renomme la colonne 'pdl_id' en 'id' pour le fichier de sortie
output_df = labels_df[[COL_PDL, "label", "cluster"]].rename(columns={COL_PDL: "id"})


# --- 3. Sauvegarde dans un fichier CSV ---
output_path = "C:/Users/cleli/OneDrive/Documents/ENPC/M1EDE/RES2-6-9/RES2-6-9_labels_clelia.csv"

output_df.to_csv(output_path, index=False, sep=",")

print(f"Le fichier de labels a été sauvegardé avec succès ici : {output_path}")
print("\nAperçu des 5 premières lignes :")
output_df.head()





if __name__ == "__main__":
    SHOW_PLOTS = True
















