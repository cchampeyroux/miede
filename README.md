# Miede - Analyse et Prédiction de Consommation Énergétique

## Configuration Initiale

Mettre les CSV dans le dossier `data/`:
- `RES2-6-9.csv` - Données brutes de consommation
- `RES2-6-9-labels.csv` - Labels des types de résidences

### POUR GIT
```bash
# Cloner le repo depuis le dossier cible
git clone "https://url..."


# Ouvrir visual studio et ajouter le dossier au workspace
# File -> Add folder to workspace

# Créer environnement virtuel
python -m venv .venv 

# Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Mac/Linux

# Installer les paquets
pip install -r requirements.txt

# Avant de commit : vérifier qu'on est à jour
git pull

# Synchroniser
git push

# Si paquets additionnels ajoutés
pip freeze > requirements.txt && git add && git commit && git push
```

---


## Lancement du dashboard : 
```python
streamlit run .\app.py
```


## Architecture du Projet

### Structure `src/`

Le projet est organisé en modules spécialisés pour différentes tâches:

#### **📊 `clustering.py` - Extraction de Features et Clustering**

Module principalement de **traitement des données brutes** en features pour chaque PDL.

**Fonctions principales:**
- `load_labels()` - Charge les labels depuis le fichier CSV avec cache persistent (joblib)
- `get_features_pdl()` - Extrait les 16 features principales pour chaque PDL:
  - **Activité**: `active_day_rate`, `n_runs`, `mean_run_len`, `max_run_len`, `mean_gap_len`, `max_gap_len`
  - **Consommation quotidienne**: `mean_daily_kwh`, `p95_daily_kwh`, `cv_daily_kwh`
  - **Patterns hebdomadaires**: `active_rate_weekday`, `active_rate_weekend`, `mean_kwh_weekday`, `mean_kwh_weekend`
  - **Saisonnalité**: `winter_minus_summer`, `seasonality_amp`, `r_global`, `r_mid`, `r_summer`, `r_winter`
  - **Stats**: `cluster`, `n_days_complete`, `n_days_data`
- `_read_raw()` - Lit les données brutes avec parsing de dates et calcul de variables temporelles
- `_compute_daily()` - Agrège les données par jour avec détection des jours actifs
- `_compute_activity()` - Calcule les statistiques d'activité
- `_runs_and_gaps()` - Analyse les séquences de jours actifs/inactifs
- `cluster_and_save()` - Effectue le clustering K-means et sauvegarde les résultats

**Utilisation:**
```python
from src.clustering import get_features_pdl
features_df = get_features_pdl()  # retourne DataFrame avec features + cluster
```

---

#### **🏷️ `classification.py` - Classification des Types de Résidences**

Module de **classification supervisée** pour prédire si une résidence est principale ou secondaire.

**Données:**
- Features: Extraites de `clustering.py` (16 colonnes)
- Target: Label (0 = principale, 1 = secondaire)
- Split: 80% train / 20% test (stratifié)

**Fonctions principales:**
- `eval_classification()` - Évalue un modèle (précision, rappel, F1, matrice de confusion)
- Entraînement de modèles:
  - **Régression Logistique** avec pipeline (imputation + normalisation)
  - **Réseau de neurones PyTorch** (MLP multi-couches)
  - Validation croisée et tuning d'hyperparamètres

**Modèles disponibles:**
- Logistic Regression avec StandardScaler
- Neural Network (PyTorch): couches denses avec régularisation

---

#### **🔮 `forecast.py` - Prédiction de Consommation Énergétique**

Module de **forecasting multi-modèles** pour prédire la consommation des jours futurs.

**Données d'entrée:**
- `daily.csv` - Consommation quotidienne par PDL
- Features temporelles: jour de semaine, mois, jours fériés, vacances scolaires

**Modèles implémentés:**

1. **Régression Linéaire** (`LinearRegression`)
   - Simple baseline avec features temporelles

2. **ARIMA/SARIMAX** (`SARIMAX`)
   - Modèle classique de séries temporelles avec saisonnalité

3. **LSTM** (`LSTMForecaster`)
   - Réseau de neurones récurrent (PyTorch)
   - Architecture: Embedding → LSTM (32 hidden, 2 layers) → Dense
   - Input: fenêtre glissante de 30 jours → Output: 1 jour

4. **Ensemble** 
   - Moyenne des prédictions des 3 modèles

**Fonctions principales:**
- `load_consumption_data()` - Charge daily.csv avec features temporelles
- `get_pdl_timeseries()` - Extrait la série temporelle d'un PDL
- `determine_lookback_window()` - Calcule la profondeur d'historique adaptée
- `prepare_sequences()` - Prépare les séquences pour LSTM
- `train_and_predict_models()` - Entraîne les 3 modèles et retourne prédictions 7 jours
- `plot_forecast()` - Visualise les prédictions vs réalité

**Utilisation:**
```python
from src.forecast import train_and_predict_models
result = train_and_predict_models(pdl_id=12345)
# result['models']['lstm']['predictions'] -> array de 7 valeurs
# result['ensemble']['predictions'] -> moyenne des modèles
```

---

#### **🤖 `models.py` - Génération de Courbes Synthétiques**

Module de **génération de données synthétiques** utilisant des modèles basés sur les caractéristiques réelles.

**Concept:**
Crée des courbes synthétiques réalistes avec 3 niveaux de caractéristiques:
- **Niveau 1 (Distribution)**: Mean, std, skewness, kurtosis
- **Niveau 2 (Variationnelles)**: Patterns saisonniers et hebdomadaires
- **Niveau 3 (Individuelles)**: Variabilité quotidienne

**Classes principales:**
- `CurveCharacteristics` - Dataclass contenant tous les paramètres de caractérisation
- `ResidenceModel` - Modèle pour générer des courbes synthétiques par type de résidence (principale/secondaire)

**Fonctions principales:**
- `load_data_by_residence_type()` - Charge les données réelles filtrées par type
- `calculate_characteristics()` - Calcule les 3 niveaux de caractéristiques
- `generate_synthetic_curve()` - Génère une courbe synthétique basée sur les caractéristiques

---

#### **💻 `generation.py` - Interface de Génération**

Module de **haut niveau** pour générer des courbes synthétiques avec interface CLI.

**Fonction principale:**
- `generate_synthetic_curves_with_model(residence_type, n_curves, n_days, seed)`
  - `residence_type`: "principale" ou "secondaire"
  - `n_curves`: Nombre de courbes à générer
  - `n_days`: Longueur de chaque courbe (défaut: 365 jours)
  - `seed`: Seed pour reproductibilité
  - Retourne: Liste de dictionnaires avec timestamps et valeurs

**Utilisation CLI:**
```bash
python src/generation.py --type principale --count 5 --days 365 --seed 42
```

---

#### **📈 `prediction.py` - Utilitaires de Prédiction**

Module **d'affichage et d'analyse** des résultats de prédiction.

**Fonctions principales:**
- `load_cluster_assignments()` - Charge les assignations de clusters des PDLs
- `display_model_results()` - Affiche les prédictions de tous les modèles dans un format tabulaire avec statistiques

**Utilisation:**
```python
from src.prediction import display_model_results
display_model_results(result_dict)  # Affiche résultats détaillés
```

---

## Pipeline Typique

### 1. **Préparation des données**
```python
from src.clustering import get_features_pdl
features = get_features_pdl()  # Crée/charge les features et clusters
```

### 2. **Classification** (prédire principale/secondaire)
```python
from src.classification import eval_classification
# ... Entraîner modèle, puis évaluer
eval_classification(y_test, predictions, "Mon Modèle")
```

### 3. **Prédiction de consommation**
```python
from src.forecast import train_and_predict_models
result = train_and_predict_models(pdl_id=12345)
print(result['ensemble']['predictions'])  # Prédictions des 7 prochains jours
```

### 4. **Génération synthétique**
```python
from src.generation import generate_synthetic_curves_with_model
curves = generate_synthetic_curves_with_model(
    residence_type="principale",
    n_curves=10,
    n_days=365,
    seed=42
)
```

---

## Fichiers de Données

- `data/RES2-6-9.csv` - Données de consommation demi-horaire (30 min)
- `data/RES2-6-9-labels.csv` - Labels: ID et type de résidence (0/1)
- `data/daily.csv` - Agrégation quotidienne de la consommation
- `models/` - Poids sauvegardés des modèles LSTM et NN
- `cache/` - Cache joblib pour les calculs lourds

---

## Dépendances

Voir `requirements.txt` pour la liste complète. Principales:
- `pandas`, `numpy` - Manipulation de données
- `scikit-learn` - ML classique (classification, preprocessing)
- `torch` - Réseaux de neurones (LSTM, NN)
- `statsmodels` - Modèles de séries temporelles (ARIMA, SARIMAX)
- `plotly`, `matplotlib` - Visualisation
- `streamlit` - Application web interactive
- `joblib` - Cache persistant

