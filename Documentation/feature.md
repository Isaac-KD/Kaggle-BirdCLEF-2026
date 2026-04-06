# Feature Engineering Optimisé — BirdCLEF 2026 (Approche Arbres de Décision)

---

## 1. Features Cepstrales & Timbre

### 1.1. MFCCs (Mel-Frequency Cepstral Coefficients)
- **Quoi** : Représentation compacte de l'enveloppe spectrale (le timbre).
- **Paramètres** : 20 premiers coefficients (au-delà de 20, on capte le bruit de phase du micro).
- **Stats** : mean, std, max, min → **80 features**
- **Pourquoi** : Le cœur de la classification acoustique, très discriminant entre les espèces.

### 1.2. Delta MFCCs
- **Quoi** : Vitesse de variation du timbre dans le temps.
- **Paramètres** : Uniquement sur les 13 premiers MFCCs (les plus importants).
- **Stats** : std, max (la moyenne d'une dérivée sur 5s est proche de 0, elle est donc inutile) → **26 features**
- **Pourquoi** : Capture les attaques soudaines et la dynamique d'un cri.

---

## 2. Features Spectrales Globales

### 2.1. Spectral Centroid
- **Quoi** : Centre de gravité fréquentiel (où se concentre l'énergie).
- **Stats** : mean, std → **2 features**
- **Pourquoi** : Distingue instantanément les sons aigus (insectes, passereaux) des graves (amphibiens, gros oiseaux).

### 2.2. Spectral Contrast
- **Quoi** : Différence d'énergie entre pics et vallées spectrales.
- **Paramètres** : 7 bandes de fréquences.
- **Stats** : mean, std → **14 features**
- **Pourquoi** : Sépare les sons très harmoniques (vocalisations d'oiseaux) des sons plats (bruit de fond continu).

### 2.3. Spectral Flatness
- **Quoi** : Indice de "tonalité" (proche de 0 = sifflement pur, proche de 1 = bruit blanc/vent).
- **Stats** : mean, std → **2 features**
- **Pourquoi** : Excellent filtre anti-bruit ambiant (vent, pluie) spécifique au Pantanal.

### 2.4. Zero Crossing Rate (ZCR)
- **Quoi** : Taux de franchissement par zéro du signal brut.
- **Stats** : mean, std → **2 features**
- **Pourquoi** : Isole les bruits percussifs, les craquements (ou certains orthoptères/insectes) des sons tonals.

---

## 3. Features Fréquentielles & Harmoniques

### 3.1. Fréquence Fondamentale F0 (Pitch)
- **Quoi** : Hauteur tonale principale du signal détectée.
- **Stats** : mean, std, min, max → **4 features**
- **Pourquoi** : Chaque espèce possède une plage d'émission fondamentale très stricte.

### 3.2. Statistiques sur le Mel-Spectrogramme nettoyé
- **Quoi** : Distribution d'énergie après un `clean_spectrogram` (débruitage).
- **Paramètres** : 8 grandes bandes de fréquences.
- **Stats** : mean, std par bande → **16 features**
- **Pourquoi** : Donne une cartographie grossière mais très propre de l'occupation spectrale.

---

## 4. Features Bioacoustiques Avancées (Les "Pépites")

### 4.1. Proportion de frames actives
- **Quoi** : % de frames dépassant un seuil d'énergie (threshold dynamique).
- **Stats** : 1 valeur → **1 feature**
- **Pourquoi** : Différencie un fichier dense en cris d'un fichier avec un appel isolé.

### 4.2. Modulation de Fréquence (FM)
- **Quoi** : Variation temporelle de F0 (dérivée de la fréquence fondamentale).
- **Stats** : mean, std, max → **3 features**
- **Pourquoi** : Capture les trilles, glissandos, montées/descentes. Ultra discriminant pour les oiseaux.

### 4.3. Détection de Syllabes (Rythme)
- **Quoi** : Segmentation des moments de phonation (remplace l'algo de "Tempo" inadapté à la nature).
- **Stats** : nb de syllabes, durée de syllabe (mean/std), durée de silence entre syllabes (mean/std) → **5 features**
- **Pourquoi** : Code directement le rythme spécifique du chant de chaque espèce (trilles rapides vs cris espacés).

### 4.4. Plage Fréquentielle Active
- **Quoi** : Fréquences extrêmes effectivement utilisées dans l'audio.
- **Stats** : freq_min, freq_max, freq_range (max - min) → **3 features**
- **Pourquoi** : Filtre biologiquement absolu.

---

## Résumé des dimensions

| Feature | Nb Features |
|---|---|
| MFCCs (20 coeffs) | 80 |
| Delta MFCCs (13 coeffs) | 26 |
| Spectral Centroid | 2 |
| Spectral Contrast | 14 |
| Spectral Flatness | 2 |
| Zero Crossing Rate (ZCR) | 2 |
| Fréquence Fondamentale (F0) | 4 |
| Stats Mel-Spectrogramme | 16 |
| Frames actives | 1 |
| Modulation de Fréquence (FM) | 3 |
| Syllabes & Rythme | 5 |
| Plage Fréquentielle Act. | 3 |
| **TOTAL par extrait de 5s** | **158** |
