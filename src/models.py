"""
Modèles de génération de courbes synthétiques pour résidences principales et secondaires.
Basé sur les recommandations du document "DeepCourboGen - Génération de courbes de charge synthétiques"


Objectifs:
  - Générer des courbes de consommation réalistes
  - Capturer les 3 niveaux de caractéristiques (distribution, variationnelles, individuelles)
  - Assurer la diversité sans mode collapse
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
RES2_LABELS_FILE = DATA_DIR / "RES2-6-9-labels.csv"
DAILY_DATA_FILE = DATA_DIR / "daily.csv"


@dataclass
class CurveCharacteristics:
    """Caractéristiques d'une courbe (selon les 3 niveaux du document)"""
    # Niveau 1: Distribution
    mean: float
    std: float
    skewness: float
    kurtosis: float
    max_mean_ratio: float
    
    # Niveau 2: Variationnelles (longue échelle)
    seasonal_pattern: np.ndarray  # Moyennes par mois
    weekday_pattern: np.ndarray   # Moyennes par jour de semaine (0-6)
    
    # Niveau 3: Individuelles (courte échelle)
    daily_std: float              # Écart-type moyen des valeurs quotidiennes


def load_data_by_residence_type(residence_type: str) -> pd.DataFrame:
    """
    Charger les données réelles filtrées par type de résidence.
    
    Args:
        residence_type: "principale" (0) ou "secondaire" (1)
    
    Returns:
        DataFrame avec les données filtrées
    """
    res2_df = pd.read_csv(RES2_LABELS_FILE)
    daily_df = pd.read_csv(DAILY_DATA_FILE)
    
    if residence_type == "principale":
        label = 0
    elif residence_type == "secondaire":
        label = 1
    else:
        raise ValueError("residence_type doit être 'principale' ou 'secondaire'")
    
    # IDs correspondant au type
    ids = res2_df[res2_df['label'] == label]['id'].values
    
    # Filtrer les données daily
    filtered_df = daily_df[daily_df['ID'].isin(ids)].copy()
    return filtered_df


def calculate_characteristics(daily_values: np.ndarray, dates: np.ndarray = None) -> CurveCharacteristics:
    """
    Calculer les 3 niveaux de caractéristiques d'une courbe.
    
    Args:
        daily_values: Array de valeurs quotidiennes (365 points pour une année)
        dates: Array de dates (optionnel, pour déterminer les saisons/jours)
    
    Returns:
        CurveCharacteristics avec tous les paramètres calculés
    """
    # Niveau 1: Distribution
    mean = np.mean(daily_values)
    std = np.std(daily_values)
    skewness = float(stats.skew(daily_values))
    kurtosis = float(stats.kurtosis(daily_values))
    max_mean_ratio = np.max(daily_values) / mean if mean > 0 else 0
    
    # Niveau 2: Patterns variationnels
    if dates is None or len(dates) != len(daily_values):
        # Créer des dates par défaut
        dates = pd.date_range(start='2023-01-01', periods=len(daily_values), freq='D')
    else:
        dates = pd.to_datetime(dates)
    
    df_temp = pd.DataFrame({'date': dates, 'value': daily_values})
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['dayofweek'] = df_temp['date'].dt.dayofweek
    
    # Pattern saisonnier (mois)
    seasonal_pattern = df_temp.groupby('month')['value'].mean().values
    
    # Pattern jour de semaine
    weekday_pattern = df_temp.groupby('dayofweek')['value'].mean().values
    
    # Niveau 3: Caractéristique individuelle
    daily_std = float(np.std(df_temp.groupby(df_temp['date'].dt.date)['value'].std()))
    
    return CurveCharacteristics(
        mean=float(mean),
        std=float(std),
        skewness=skewness,
        kurtosis=kurtosis,
        max_mean_ratio=float(max_mean_ratio),
        seasonal_pattern=seasonal_pattern,
        weekday_pattern=weekday_pattern,
        daily_std=daily_std
    )


class ResidenceModel:
    """
    Modèle pour générer et évaluer des courbes synthétiques de consommation.
    
    Architecture: Modèle statistique conditionnel par échantillonnage
    - Utilise le filtrage par caractéristiques réelles (type, cluster) comme "condition"
    - Génère des courbes par application de perturbations aléatoires sur des bases réelles
    """
    
    def __init__(self, residence_type: str, seed: int = 42, cluster_id: int = None, df_conso: pd.DataFrame = None, labels_df: pd.DataFrame = None):
        """
        Initialiser le modèle.
        
        Args:
            residence_type: "principale" ou "secondaire"
            seed: Seed pour reproductibilité
            cluster_id: ID du cluster (optionnel)
            df_conso: DataFrame de consommation (optionnel)
            labels_df: DataFrame des labels (optionnel)
        """
        self.residence_type = residence_type
        self.seed = seed
        self.cluster_id = cluster_id
        np.random.seed(seed)
        
        # Charger les données si non fournies
        if df_conso is None:
            self.real_data = load_data_by_residence_type(residence_type)
        else:
            # Filtrer df_conso par type si les labels sont fournis
            if labels_df is not None:
                label = 0 if residence_type == "principale" else 1
                ids = labels_df[labels_df['label'] == label]['id'].values
                self.real_data = df_conso[df_conso['ID'].isin(ids)].copy()
            else:
                # Si df_conso est déjà filtré ou contient les infos nécessaires
                self.real_data = df_conso.copy()

        # Filtrer par cluster si demandé
        if cluster_id is not None and 'cluster' in self.real_data.columns:
            self.real_data = self.real_data[self.real_data['cluster'] == cluster_id].copy()
            print(f"[{residence_type}] Filtrage par cluster {cluster_id} : {len(self.real_data['ID'].unique())} PDLs")
        
        self._extract_real_characteristics()
    
    def _extract_real_characteristics(self):
        """Extraire les caractéristiques et profils des données réelles pour la génération"""        # Grouper par ID et calculer les caractéristiques
        self.real_characteristics = []
        self.real_curves = []
        
        for residence_id in self.real_data['ID'].unique()[:500]:  # Limiter pour la performance
            data_id = self.real_data[self.real_data['ID'] == residence_id]
            if len(data_id) < 300:
                continue
            
            values = data_id.sort_values('date')['daily_kwh'].values
            
            if len(values) > 0:
                chars = calculate_characteristics(values)
                self.real_characteristics.append(chars)
                self.real_curves.append(values)
        
        print(f"[{self.residence_type}] {len(self.real_characteristics)} courbes réelles chargées")
    
    def generate_synthetic_curve(self, n_days: int = 365) -> np.ndarray:
        """
        Générer une courbe synthétique en utilisant les patterns réels.
        
        Approche: Combiner les patterns réels de façon aléatoire
        - Sélectionner aléatoirement une courbe réelle comme "base"
        - Appliquer une perturbation basée sur les distributions observées
        - Ajouter du bruit réaliste
        
        Args:
            n_days: Nombre de jours (défaut: 365)
        
        Returns:
            Array de valeurs quotidiennes synthétiques
        """
        if not self.real_curves:
            raise ValueError("Aucune courbe réelle disponible")
        
        # Sélectionner une courbe réelle aléatoire comme base
        base_curve = self.real_curves[np.random.randint(0, len(self.real_curves))]
        
        # Redimensionner si nécessaire
        if len(base_curve) != n_days:
            indices = np.linspace(0, len(base_curve) - 1, n_days)
            base_curve = np.interp(indices, np.arange(len(base_curve)), base_curve)
        
        # Appliquer une transformation aléatoire
        # 1. Scaling aléatoire (±10% de la moyenne)
        scale_factor = np.random.normal(1.0, 0.05)
        
        # 2. Décalage aléatoire (±5% de la moyenne)
        shift = np.random.normal(0, base_curve.mean() * 0.05)
        
        # 3. Bruit Gaussien réaliste
        noise = np.random.normal(0, base_curve.std() * 0.08)
        
        synthetic_curve = base_curve * scale_factor + shift + noise
        return np.maximum(synthetic_curve, 0.01)  # Assurer des valeurs positives
    
    def generate_batch(self, n_curves: int = 10, n_days: int = 365) -> List[np.ndarray]:
        """
        Générer un batch de courbes synthétiques.
        
        Args:
            n_curves: Nombre de courbes à générer
            n_days: Nombre de jours par courbe
        
        Returns:
            Liste de courbes synthétiques
        """
        return [self.generate_synthetic_curve(n_days) for _ in range(n_curves)]
    
    def compute_similarity_cosine(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """
        Calculer la similarité cosinus entre deux courbes.
        
        Formula: cs(c1, c2) = (c1 · c2) / (||c1|| * ||c2||)
        
        Args:
            curve1, curve2: Arrays de valeurs
        
        Returns:
            Similarité cosinus (0-1, plus proche de 1 = plus similaire)
        """
        dot_product = np.dot(curve1, curve2)
        norm1 = np.linalg.norm(curve1)
        norm2 = np.linalg.norm(curve2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def compute_similarity_correlation(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """
        Calculer la similarité par coefficient de corrélation linéaire.
        
        Formula: r² = 1 - (cov(c1, c2) / (σ_c1 * σ_c2))
        
        Args:
            curve1, curve2: Arrays de valeurs
        
        Returns:
            Coefficient de corrélation (-1 à 1, 1 = similaire)
        """
        corr = np.corrcoef(curve1, curve2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    
    def validate_characteristics(self, synthetic_curve: np.ndarray, tolerance: float = 0.15) -> Dict:
        """
        Valider que la courbe synthétique respecte les caractéristiques réelles.
        
        Args:
            synthetic_curve: Courbe synthétique à valider
            tolerance: Tolérance relative pour comparer (15% par défaut)
        
        Returns:
            Dict avec validation de chaque caractéristique
        """
        # Calculer les caractéristiques de la courbe synthétique
        synth_chars = calculate_characteristics(synthetic_curve)
        
        # Moyennes des caractéristiques réelles
        real_mean = np.mean([c.mean for c in self.real_characteristics])
        real_std = np.mean([c.std for c in self.real_characteristics])
        real_skew = np.mean([c.skewness for c in self.real_characteristics])
        
        def check_tolerance(synth_val, real_val):
            if real_val == 0:
                return abs(synth_val) < tolerance
            return abs((synth_val - real_val) / real_val) < tolerance
        
        return {
            "mean_ok": check_tolerance(synth_chars.mean, real_mean),
            "std_ok": check_tolerance(synth_chars.std, real_std),
            "skewness_ok": check_tolerance(synth_chars.skewness, real_skew),
            "details": {
                "synth_mean": synth_chars.mean,
                "real_mean": real_mean,
                "synth_std": synth_chars.std,
                "real_std": real_std,
            }
        }
    
    def measure_repetition_rate(self, synthetic_curves: List[np.ndarray], top_n: int = 5) -> float:
        """
        Mesurer le taux de similitude entre les courbes générées (détection de redondance).
        
        Permet de vérifier si les perturbations appliquées offrent assez de diversité
        ou si les courbes générées se ressemblent trop.
        """
        if len(synthetic_curves) < 2:
            return 0.0
        
        similarities = []
        for i, curve in enumerate(synthetic_curves):
            # Calculer les similarités avec les autres courbes
            sim_vals = [
                self.compute_similarity_cosine(curve, other)
                for j, other in enumerate(synthetic_curves)
                if j != i
            ]
            # Prendre les top_n les plus proches
            sim_vals.sort(reverse=True)
            similarities.extend(sim_vals[:top_n])
        
        return float(np.mean(similarities))
    
    def get_statistics(self) -> Dict:
        """Obtenir les statistiques du modèle"""
        if not self.real_characteristics:
            return {}
        
        return {
            "residence_type": self.residence_type,
            "n_real_curves": len(self.real_characteristics),
            "mean_daily_consumption": float(np.mean([c.mean for c in self.real_characteristics])),
            "std_daily_consumption": float(np.mean([c.std for c in self.real_characteristics])),
            "mean_skewness": float(np.mean([c.skewness for c in self.real_characteristics])),
            "max_mean_ratio": float(np.mean([c.max_mean_ratio for c in self.real_characteristics])),
        }


def create_models() -> Tuple[ResidenceModel, ResidenceModel]:
    """
    Créer les deux modèles: principal et secondaire.
    
    Returns:
        Tuple contenant (model_principal, model_secondaire)
    """
    print("🔧 Création du modèle pour résidences PRINCIPALES...")
    model_principal = ResidenceModel("principale")
    
    print("🔧 Création du modèle pour résidences SECONDAIRES...")
    model_secondaire = ResidenceModel("secondaire")
    
    return model_principal, model_secondaire


if __name__ == "__main__":
    # Exemple d'utilisation
    print("=" * 60)
    print("CRÉATION ET VALIDATION DES MODÈLES")
    print("=" * 60)
    
    # Créer les modèles
    model_principal, model_secondaire = create_models()
    
    print("\n" + "=" * 60)
    print("STATISTIQUES DES MODÈLES")
    print("=" * 60)
    
    for model in [model_principal, model_secondaire]:
        stats_dict = model.get_statistics()
        print(f"\n📊 {stats_dict['residence_type'].upper()}")
        print(f"  Courbes réelles analysées: {stats_dict['n_real_curves']}")
        print(f"  Consommation moyenne: {stats_dict['mean_daily_consumption']:.2f} kWh/jour")
        print(f"  Écart-type: {stats_dict['std_daily_consumption']:.2f} kWh/jour")
        print(f"  Skewness moyen: {stats_dict['mean_skewness']:.3f}")
        print(f"  Ratio max/moyenne: {stats_dict['max_mean_ratio']:.2f}x")
    
    print("\n" + "=" * 60)
    print("GÉNÉRATION ET VALIDATION")
    print("=" * 60)
    
    # Générer des courbes synthétiques
    for residence_type, model in [("PRINCIPALE", model_principal), ("SECONDAIRE", model_secondaire)]:
        print(f"\n🎯 {residence_type}")
        
        synth_curves = model.generate_batch(n_curves=5)
        print(f"  ✓ {len(synth_curves)} courbes générées")
        
        # Valider la première
        validation = model.validate_characteristics(synth_curves[0])
        print(f"  ✓ Validation des caractéristiques:")
        print(f"    - Moyenne: {'✅' if validation['mean_ok'] else '❌'}")
        print(f"    - Écart-type: {'✅' if validation['std_ok'] else '❌'}")
        print(f"    - Skewness: {'✅' if validation['skewness_ok'] else '❌'}")
        
        # Mesurer le taux de répétition
        repetition_rate = model.measure_repetition_rate(synth_curves)
        print(f"  ✓ Taux de répétition: {repetition_rate:.3f} (< 0.5 idéal)")
    
    print("\n" + "=" * 60)
    print("✅ MODÈLES PRÊTS À L'EMPLOI")
    print("=" * 60)
