# Bird & Wildlife Recognition - Dataset Documentation

Ce document détaille la structure des données pour le projet de classification audio d'espèces animales (oiseaux, mammifères, insectes, reptiles et amphibiens) du Pantanal (Brésil).

---

## Structure des Fichiers Audio

Tous les fichiers audio sont échantillonnés à **32 kHz** et au format **.ogg**.

### 1. `train_audio/`
Enregistrements courts d'individus isolés provenant de *xeno-canto.org* et *iNaturalist*.
- **Format du nom :** `[collection][file_id].ogg`.
- **Note :** Il est fortement déconseillé de télécharger des données supplémentaires sur ces plateformes pour ne pas surcharger leurs serveurs.

### 2. `train_soundscapes/`
Enregistrements d'une minute d'environnements complets (plusieurs espèces simultanées).
- Certains fichiers sont annotés par des experts dans `train_soundscapes_labels.csv`.
- Les labels sont fournis par segments de **5 secondes**.

### 3. `test_soundscapes/` (Données cachées)
Environ 600 enregistrements d'une minute utilisés pour le score final.
- **Format du nom :** `BC2026_Test_[ID]_[Site]_[Date]_[Heure_UTC].ogg`.
- **Important :** Certaines espèces présentes ici ne se trouvent **que** dans les `train_soundscapes` et non dans le `train_audio`.

---

## Métadonnées et Fichiers CSV

### `train.csv`
Contient les informations sur les enregistrements de `train_audio/`.
| Colonne | Description |
| :--- | :--- |
| `primary_label` | Code de l'espèce (eBird pour oiseaux, iNaturalist ID sinon). |
| `secondary_labels` | Autres espèces présentes en arrière-plan. |
| `rating` | Qualité du son (1 à 5). 0 = inconnu. |
| `latitude` / `longitude` | Coordonnées géographiques (utile pour les dialectes locaux). |
| `filename` | Nom du fichier correspondant dans `train_audio/`. |

### `taxonomy.csv`
Dictionnaire des **234 classes** à prédire.
- Regroupe les espèces par classe : *Aves* (oiseaux), *Amphibia*, *Mammalia*, *Insecta*, *Reptilia*.
- **Note sur les insectes :** Beaucoup sont classés par "sonotypes" (ex: `47158son16`) car l'espèce exacte n'est pas identifiée.

### `recording_location.txt`
Informations contextuelles sur la zone d'étude : **Le Pantanal, Brésil.**

---

## Format de Soumission

L'objectif est de prédire la présence de chaque espèce pour chaque segment de 5 secondes des fichiers de test.

- **Fichier :** `sample_submission.csv`
- **`row_id` :** Identifiant au format `[nom_du_fichier]_[temps_fin]`.
  - *Exemple :* `BC2026_Test_0001_S05_20250227_010002_20` (pour le segment 15s - 20s).
- **Prédictions :** Probabilité (0.0 à 1.0) pour chacune des 234 colonnes d'espèces.

---

## Points Clés à Retenir
1. **Déséquilibre :** Toutes les espèces du train ne sont pas forcément dans le test.
2. **Segmentation :** Le modèle doit être capable d'analyser des fenêtres de 5 secondes.
3. **Localisation :** La géographie est un facteur important pour affiner les probabilités de présence.