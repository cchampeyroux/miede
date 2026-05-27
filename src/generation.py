"""
Génération de courbes synthétiques de consommation électrique.

Utilise les modèles DC-WGAN Conditionnel pour générer des courbes réalistes
pour résidences principales et secondaires.
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict

VALID_RESIDENCE_TYPES = {"principale", "secondaire"}


def compute_r2_similarity(
    real_curve: List[float],
    synthetic_curve: List[float],
) -> float:
    """
    Calculer le coefficient de détermination R² entre deux courbes.

    Cette mesure évalue l'ajustement global en amplitude des valeurs.
    Elle ne prend pas en compte un décalage temporel entre les courbes.

    Args:
        real_curve: Courbe réelle (liste de valeurs)
        synthetic_curve: Courbe synthétique (liste de valeurs)

    Returns:
        R² : 1.0 = parfait, 0.0 = aussi bien que la moyenne, < 0 = pire que la moyenne
    """
    real = np.asarray(real_curve, dtype=float)
    synth = np.asarray(synthetic_curve, dtype=float)

    if real.shape != synth.shape:
        raise ValueError("Les deux courbes doivent avoir la même longueur pour calculer R².")

    ss_res = np.sum((real - synth) ** 2)
    ss_tot = np.sum((real - real.mean()) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def generate_synthetic_curves_with_model(
    residence_type: str,
    n_curves: int = 1,
    n_days: int = 365,
    seed=None,
    start_date: str = "2023-01-01",
) -> List[Dict]:
    """
    Générer des courbes synthétiques en utilisant les modèles DC-WGAN.
    
    Args:
        residence_type: "principale" ou "secondaire"
        n_curves: Nombre de courbes à générer
        n_days: Nombre de jours par courbe
        seed: Seed optionnelle pour reproductibilité
        start_date: Date de début des timestamps générés (YYYY-MM-DD)
    
    Returns:
        Liste de courbes synthétiques générées
    
    Raises:
        ValueError: Si residence_type est invalide
    """
    if residence_type not in VALID_RESIDENCE_TYPES:
        raise ValueError(f"Type de résidence invalide: {residence_type}. Utiliser: {sorted(VALID_RESIDENCE_TYPES)}")
    
    # Importer les modèles
    from src.models import ResidenceModel
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Créer le modèle pour le type de résidence
    model = ResidenceModel(residence_type, seed=seed)
    
    curves = []
    start_date = pd.to_datetime(start_date)
    for i in range(n_curves):
        synth_values = model.generate_synthetic_curve(n_days=n_days)
        
        # Créer les timestamps
        timestamps = []
        values = []
        
        for day_idx, value in enumerate(synth_values):
            date = start_date + pd.Timedelta(days=day_idx)
            timestamps.append(date.date().isoformat())
            values.append(round(float(value), 4))
        
        curves.append({
            "synthetic_id": i + 1,
            "type": residence_type,
            "freq": "D",
            "timestamps": timestamps,
            "values": values,
            "source": "synthetic_gan",
        })
    
    return curves


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Génération de courbes synthétiques de consommation électrique avec modèles GANs."
    )
    parser.add_argument(
        "--type",
        choices=sorted(VALID_RESIDENCE_TYPES),
        required=True,
        help="Type de résidence: 'principale' ou 'secondaire'.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Nombre de courbes à générer (défaut: 1).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Nombre de jours par courbe (défaut: 365).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed optionnelle pour rendre la génération reproductible.",
    )
    args = parser.parse_args()

    curves = generate_synthetic_curves_with_model(
        residence_type=args.type,
        n_curves=args.count,
        n_days=args.days,
        seed=args.seed,
    )
    
    for i, curve in enumerate(curves, start=1):
        print(
            f"courbe_{i} - type={curve['type']} - "
            f"points={len(curve['values'])} - source={curve['source']}"
        )