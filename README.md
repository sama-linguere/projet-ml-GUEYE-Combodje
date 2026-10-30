# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Nom / Prénom :** GUEYE Combodje (BTS RSIoT — ISEP Diamniadio)

Un modèle de classification décide, pour chaque flux réseau, s'il est **normal (0)** ou une **attaque (1)** (scan de ports, force brute, DoS).

## Lancer le projet

```bash
pip install -r requirements.txt
python src/train.py                       # entraîne, compare les modèles, enregistre models/model.joblib
python src/predict.py data/flux_train.csv # exemple de prédiction (affiche une liste de 0/1)
```

Dans le code : `from src.predict import predict` puis `predict(df)` (df sans la colonne `attaque`) renvoie une liste de 0 et de 1.

> Si `pip` signale une incompatibilité de version avec `models/model.joblib`, relancer `python src/train.py` pour régénérer le modèle avec vos versions.

## Démarche

1. **Exploration** : 3000 flux, 90 valeurs manquantes (uniquement `duration`), 27,8 % d'attaques → classes déséquilibrées, donc on suit le **F1 de la classe « attaque »** et pas seulement l'accuracy.
2. **Prétraitement (dans un `Pipeline`)** : imputation par la médiane + `StandardScaler` pour les colonnes numériques ; imputation + `OneHotEncoder` pour `protocol` (qui ignore un protocole inconnu au lieu de planter).
3. **Trois modèles** comparés sur un jeu de validation (20 %, stratifié, `random_state=42`) avec `class_weight="balanced"`.
4. **Choix du meilleur** (F1 attaque), ré-entraînement sur **toutes** les données, sauvegarde dans `models/model.joblib` (le Pipeline complet, prétraitement inclus).

## Résultats

**Nom / Prénom :** GUEYE Combodje

Jeu de validation : 600 flux (433 normaux, 167 attaques).

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|
| Régression logistique | 0,920 | 0,898 | 0,943 |
| Arbre de décision (profondeur 6) | 0,953 | 0,964 | 0,942 |
| **Forêt aléatoire (300 arbres)** | **0,982** | **0,976** | **0,988** |

Matrices de confusion (lignes : réalité, colonnes : prédiction) :

| Modèle | Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs |
|---|---|---|---|---|
| Régression logistique | 424 | 9 | 17 | 150 |
| Arbre de décision | 423 | 10 | 6 | 161 |
| Forêt aléatoire | 431 | 2 | 4 | 163 |

**Modèle retenu et pourquoi :** la **forêt aléatoire**. Elle a le meilleur F1 (0,982) et surtout le meilleur équilibre entre rappel et précision. Elle capture mieux que la régression logistique les relations non linéaires entre variables (par exemple `nb_conn_same_host` élevé combiné à `syn_rate` élevé), et elle est plus stable qu'un arbre unique. Une validation croisée à 5 plis donne un F1 moyen d'environ **0,96** : le 0,982 de la validation simple est donc un peu optimiste, et c'est plutôt 0,96 qu'il faut attendre sur des flux jamais vus. Les variables les plus utiles sont `dst_bytes`, `nb_conn_same_host`, `syn_rate` et `nb_failed_logins`.

**Analyse des erreurs (faux négatifs / faux positifs) :**

- **4 faux négatifs** (4 attaques sur 167 ratées, soit 2,4 %). Ces flux ont tous 0 échec de connexion, peu de connexions vers la même machine (3 à 7), un `syn_rate` faible (< 0,33) et des octets reçus normaux : ils **ressemblent à du trafic normal** (attaque discrète ou « lente »).
- **2 faux positifs** : des flux normaux qui ressemblent à une attaque (l'un a 46 connexions vers la même machine et un `syn_rate` de 0,90 ; l'autre a 9 échecs de connexion sur le port 22).
- **Pourquoi le faux négatif est le pire cas en sécurité :** une attaque non détectée passe sans alerte, l'attaquant peut continuer (accès, vol de données, indisponibilité) et personne ne réagit. Un faux positif coûte seulement du temps à l'analyste, qui vérifie l'alerte. On privilégie donc un **rappel élevé**, quitte à accepter quelques fausses alertes. Pour réduire encore les faux négatifs, on pourrait abaisser le seuil de décision (`predict_proba` > 0,4 au lieu de 0,5), au prix de plus de faux positifs.

## Structure

```
projet-ml-GUEYE-Combodje/
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py
└── models/
    ├── model.joblib
    └── metrics.json   (métriques de validation générées par train.py)
```
