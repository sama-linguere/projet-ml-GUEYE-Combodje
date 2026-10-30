"""Projet 2 - Détection d'attaques réseau : prédiction.

Le correcteur automatique appelle predict(df).
Usage en ligne de commande : python src/predict.py chemin/vers/flux.csv
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

COLONNES = ["duration", "protocol", "dst_port", "src_bytes", "dst_bytes",
            "nb_packets", "nb_failed_logins", "nb_conn_same_host", "syn_rate"]

_modele = None  # chargé une seule fois


def _charger_modele():
    global _modele
    if _modele is None:
        _modele = joblib.load(MODEL_PATH)
    return _modele


def predict(df: pd.DataFrame) -> list:
    """Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne.

    df : mêmes colonnes que flux_train.csv, SANS 'attaque'.
    """
    modele = _charger_modele()
    X = df[COLONNES]  # ignore une éventuelle colonne en trop (ex. 'attaque')
    return [int(p) for p in modele.predict(X)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python src/predict.py chemin/vers/flux.csv")
    flux = pd.read_csv(sys.argv[1])
    print(predict(flux))
