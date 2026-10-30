"""Projet 2 - Détection d'attaques réseau : entraînement.

Usage (depuis la racine du dépôt) : python src/train.py

Étapes : exploration -> prétraitement (Pipeline) -> comparaison de plusieurs
modèles sur un jeu de validation -> choix du meilleur -> ré-entraînement sur
toutes les données -> sauvegarde dans models/model.joblib.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "flux_train.csv"
MODEL_PATH = BASE / "models" / "model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

CIBLE = "attaque"
COLONNES_NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
                "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
COLONNES_CAT = ["protocol"]
SEED = 42


def explorer(df):
    """Exploration rapide : taille, valeurs manquantes, répartition de la cible."""
    print("=== Exploration ===")
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[lambda s: s > 0])
    print("Répartition de 'attaque' (%) :\n",
          (df[CIBLE].value_counts(normalize=True) * 100).round(1))
    print()


def construire_pipeline(modele):
    """Prétraitement + modèle dans un seul Pipeline.

    - numériques : imputation par la médiane puis mise à l'échelle
    - protocol   : imputation par la valeur la plus fréquente puis one-hot
                   (handle_unknown='ignore' pour ne pas planter sur un
                   protocole jamais vu)
    """
    numerique = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categoriel = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pretraitement = ColumnTransformer([
        ("num", numerique, COLONNES_NUM),
        ("cat", categoriel, COLONNES_CAT),
    ])
    return Pipeline([("pretraitement", pretraitement), ("modele", modele)])


def main():
    df = pd.read_csv(DATA)
    explorer(df)

    X = df.drop(columns=[CIBLE])
    y = df[CIBLE]

    # Jeu de validation stratifié (garde ~28 % d'attaques dans les 2 jeux)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    modeles = {
        "Régression logistique": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=SEED),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED,
            n_jobs=-1),
    }

    resultats = {}
    for nom, modele in modeles.items():
        pipe = construire_pipeline(modele)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        resultats[nom] = {
            "f1_attaque": round(f1_score(y_val, pred), 4),
            "rappel_attaque": round(recall_score(y_val, pred), 4),
            "precision_attaque": round(precision_score(y_val, pred), 4),
            "matrice_confusion": {"vrais_negatifs": int(tn), "faux_positifs": int(fp),
                                  "faux_negatifs": int(fn), "vrais_positifs": int(tp)},
        }
        print(f"=== {nom} ===")
        print(classification_report(y_val, pred,
                                    target_names=["normal", "attaque"], digits=3))
        print("Matrice de confusion [[VN, FP], [FN, VP]] :\n",
              confusion_matrix(y_val, pred), "\n")

    # Choix du meilleur modèle : F1 de la classe « attaque »
    meilleur = max(resultats, key=lambda n: resultats[n]["f1_attaque"])
    print(f">>> Modèle retenu : {meilleur} "
          f"(F1 attaque = {resultats[meilleur]['f1_attaque']})")

    # Ré-entraînement sur TOUTES les données puis sauvegarde
    final = construire_pipeline(modeles[meilleur])
    final.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Modèle enregistré dans {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"modele_retenu": meilleur, "validation": resultats},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
