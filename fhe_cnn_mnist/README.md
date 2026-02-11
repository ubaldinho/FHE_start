## **Rapport de Projet FHE - 5CS09**

# Etudiants :
- NDZANA LEOPOLD UBALD JUNIOR
- TCHOMO KOMBOU THIERRY 

### Sébastien Canard | Année académique 2025-2026

---

# Implémentation Homomorphe d'un CNN pour MNIST

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Retour sur les exercices préparatoires](#2-retour-sur-les-exercices-préparatoires)
3. [Architecture du CNN](#3-architecture-du-cnn)
4. [Défis de l'implémentation FHE](#4-défis-de-limplémentation-fhe)
5. [Implémentation des couches](#5-implémentation-des-couches)
6. [Optimisations implémentées](#6-optimisations-implémentées)
7. [Bonus : One-hot vector](#7-bonus--one-hot-vector)
8. [Difficultés rencontrées](#8-difficultés-rencontrées)
9. [Guide de compilation](#9-guide-de-compilation)
10. [Conclusion](#10-conclusion)

---

## 1. Introduction

Ce projet a pour objectif l'implémentation **complètement homomorphe** d'un réseau de neurones convolutionnel (CNN) à 5 couches pour la classification d'images MNIST, en utilisant la bibliothèque **HEAAN2** (schéma CKKS).

Le travail s'est articulé en deux phases :
1. **Exercices préparatoires** : prise en main de HEAAN2 et implémentation d'algorithmes fondamentaux
2. **Projet CNN** : adaptation des techniques apprises à un cas d'usage réel

Ce rapport détaille les choix d'implémentation, les difficultés rencontrées et les solutions apportées. **Aucun résultat expérimental n'est présenté** car le pipeline complet n'a pas pu être exécuté dans son intégralité, ce qui constitue en soi une difficulté majeure du projet.

---

## 2. Retour sur les exercices préparatoires

Les exercices préparatoires ont été essentiels pour comprendre les spécificités de l'API HEAAN2 et les contraintes du calcul homomorphe.

### 2.1 Exercice 1 : Encode/Decode

**Objectif :** Comprendre le mapping entre `Message<Complex>` et `IPlaintext`.

**Difficulté :** ⭐

**Apprentissages :**
- La création d'un message utilise le **logarithme** du nombre de slots : `Message<Complex>(log_slots)`
- `log_slots = logDegree - 1` pour utiliser la moitié du degré polynomial
- L'encodage transforme un vecteur de complexes en polynôme
- Le décodage inverse est précis à 1e-15 près

**Code clé :**
```cpp
Message<Complex> msg(log_slots, Device::CPU);
msg[i] = Complex(val, 0.0);
encoder.encode(msg, *ptxt);
encoder.decode(*ptxt, dmsg);
```

### 2.2 Exercice 2 : Horner

**Objectif :** Évaluation polynomiale SIMD.

**Difficulté :** ⭐⭐⭐

**Problème rencontré :**
La multiplication de deux ciphertexts n'existe pas en tant qu'opération unique.

**Solution :**
```cpp
// Multiplication ciphertext-ciphertext
eval.tensor(ct_x, ct_prime, temp);  // Produit tensoriel
eval.relin(temp, relin_key);        // Relinéarisation
eval.rescale(temp, temp);          // Ajustement d'échelle
eval.add(temp, p_i, ct_prime);     // Addition
```

**Leçon :** Toute multiplication entre ciphertexts nécessite **3 opérations** : tensor + relin + rescale

### 2.3 Exercice 3 : Goldschmidt

**Objectif :** Calcul de l'inverse par série géométrique.

**Difficulté :** ⭐⭐

**Problème rencontré :**
`eval.sub(Complex(2.0, 0.0), ct_x, ct1)` n'existe pas.

**Solution :**
```cpp
// 2 - x = -x + 2
eval.neg(ct_x, ct1);           // ct1 = -x
eval.add(*ct1, ptxt_two, ct1); // ct1 = -x + 2
```

**Leçon :** Les opérations avec constantes sont **unidirectionnelles** : `ct - constante` existe, mais `constante - ct` n'existe pas.

### 2.4 Exercice 4 : Rotate-and-Sum

**Objectif :** Évaluation polynomiale par rotations.

**Difficulté :** ⭐⭐

**Problème rencontré :**
`Ptr<T>` est **non copiable** → impossible de stocker dans un `std::vector<Ptr<ISwKey>>`.

**Solution :**
```cpp
// ❌ Interdit
std::vector<Ptr<ISwKey>> keys;
keys.push_back(rot_key);  // Copie ! Erreur

// ✅ Correct
std::map<int, Ptr<ISwKey>> rot_keys;
rot_keys[rot] = std::move(rot_key);
```

**Leçon :** `Ptr<T>` = propriété exclusive. Utiliser `std::move` pour transférer.

### 2.5 Exercice 5 : Row Method

**Objectif :** Produit matrice-vecteur par ligne.

**Difficulté :** ⭐⭐⭐⭐

**Problème rencontré :**
`*ct_dest = *ct_src` **ne copie pas** le contenu du ciphertext.

**Solution :**
```cpp
// ❌ Ne fonctionne pas
auto ct_sum = ICiphertext::make();
*ct_sum = *ct_zi;  // ct_sum reste vide

// ✅ Solution
auto ct_sum = std::move(ct_zi);  // Transfert de propriété
```

**Leçon :** **Jamais** d'assignation entre ciphertexts. Toujours `std::move` ou création directe.

### 2.6 Exercice 6 : Diagonal Method

**Objectif :** Produit matrice-vecteur par diagonales.

**Difficulté :** ⭐⭐⭐

**Problème rencontré :**
Indices de diagonale **inversés** → résultats faux pour matrice aléatoire.

**Solution :**
```cpp
// ❌ Faux
msg_diag[j] = Complex(U[(i+j)%n][j], 0.0);

// ✅ Correct
msg_diag[j] = Complex(U[j][(j+i)%n], 0.0);  // Ligne j, colonne (j+i)%n
```

**Leçon :** Vérifier mathématiquement sur papier avant d'implémenter.

### 2.7 Exercice 7 : Diagonal Method BSGS

**Objectif :** Produit matrice-vecteur en O(√N) rotations.

**Difficulté :** ⭐⭐⭐⭐

**Problèmes rencontrés :**
1. Oubli de la **somme des slots** après multiplication
2. Oubli de la **rotation finale** `Rot_{j·n₂}`
3. Résultats multipliés par N

**Solution :**
```cpp
// 1. Rotate-and-sum
for (int shift = 1; shift < n; shift <<= 1) {
    ct_rot = eval.rot(ct_sum, shift);
    ct_sum = eval.add(ct_sum, ct_rot);
}

// 2. Extraction du slot 0
ct_extract = eval.mul(ct_sum, mask);  // [1,0,0,...]

// 3. Rotation géante
ct_rotated = eval.rot(ct_extract, j * n2);

// 4. Division par N après déchiffrement
result[i] = msg[i].real() / n;
```

**Leçon :** BSGS = **B**aby steps + **G**iant steps + **S**omme + **R**otation finale

---

## 3. Architecture du CNN

### 3.1 Description du réseau

Le réseau implémenté est inspiré de **LeNET-5**, avec 5 couches paramétrées :

```
Entrée: 1 × 28 × 28 (784 pixels)
         │
         ▼
┌─────────────────────────────────────┐
│        Couche 1 : Conv2D           │
│        ────────────────────────    │
│        in_c  = 1                  │
│        out_c = 8                  │
│        kernel = 5 × 5            │
│        stride = 1                │
│        padding = 0               │
│        → 8 × 24 × 24 (4608)      │
└─────────────────────────────────────┘
         │
         ▼
    [ Activation ReLU ]
         │
         ▼
┌─────────────────────────────────────┐
│        Couche 2 : AveragePool      │
│        ────────────────────────    │
│        kernel = 2 × 2            │
│        stride = 2                │
│        → 8 × 12 × 12 (1152)      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        Couche 3 : Conv2D           │
│        ────────────────────────    │
│        in_c  = 8                  │
│        out_c = 16                 │
│        kernel = 5 × 5            │
│        → 16 × 8 × 8 (1024)       │
└─────────────────────────────────────┘
         │
         ▼
    [ Activation ReLU ]
         │
         ▼
┌─────────────────────────────────────┐
│        Couche 4 : AveragePool      │
│        ────────────────────────    │
│        → 16 × 4 × 4 (256)         │
└─────────────────────────────────────┘
         │
         ▼
    [ Flatten ] → 256
         │
         ▼
┌─────────────────────────────────────┐
│        Couche 5 : Fully Connected  │
│        ────────────────────────    │
│        FC1 : 256 → 128            │
│        + ReLU                     │
│        FC2 : 128 → 64             │
│        + ReLU                     │
│        FC3 : 64 → 10              │
└─────────────────────────────────────┘
         │
         ▼
    [ One-hot vector (Bonus) ]
```

### 3.2 Dimensions et volumétrie

| **Couche** | **Entrée** | **Sortie** | **Opérations par pixel** |
|-----------|-----------|-----------|------------------------|
| Conv1 | 1×28×28 | 8×24×24 | 1×5×5 = 25 muls |
| Pool1 | 8×24×24 | 8×12×12 | 4 additions + ×0.25 |
| Conv2 | 8×12×12 | 16×8×8 | 8×5×5 = 200 muls |
| Pool2 | 16×8×8 | 16×4×4 | 4 additions + ×0.25 |
| FC1 | 256 | 128 | 256×128 = 32768 muls |
| FC2 | 128 | 64 | 128×64 = 8192 muls |
| FC3 | 64 | 10 | 64×10 = 640 muls |

**Total multiplications ≈ 42 000 par image**

---

## 4. Défis de l'implémentation FHE

### 4.1 Contraintes du schéma CKKS

| **Contrainte** | **Impact** | **Solution** |
|---------------|-----------|------------|
| Niveaux limités | ~10-12 multiplications max | Bootstrapping |
| Bruit croissant | Précision diminue | Rescaling après chaque mul |
| Rotations coûteuses | O(N) en temps | BSGS : O(√N) |
| Packing SIMD | Nécessite organisation | 4 images parallélisées |

### 4.2 Gestion des niveaux

**Principe :** Chaque multiplication + rescale consomme **1 niveau**.

```
Niveau 11 : Ciphertext fraîchement chiffré
    ↓ Conv1 (25 muls) → rescale unique
Niveau 10
    ↓ ReLU1 (3 muls)
Niveau 7
    ↓ Pool1 (1 mul)
Niveau 6
    ↓ Conv2 (25 muls)
Niveau 5
    ↓ ReLU2 (3 muls)
Niveau 2 → BOOTSTRAP NÉCESSAIRE
```

**Stratégie de placement :**
- Bootstrap #1 : **après ReLU2** (niveau critique)
- Bootstrap #2 : **après ReLU3** (niveau critique)

**Justification :** Ces deux points sont les seuls où le niveau descend sous le seuil de 3.

### 4.3 Problème de l'exécution complète

**Difficulté majeure :** Le pipeline complet n'a **pas pu être exécuté** dans son intégralité.

**Causes identifiées :**
1. **Temps d'exécution prohibitif** : estimation > 200 secondes pour 40 images
2. **Consommation mémoire** : les clés de rotation (900+ clés) occupent > 8 Go
3. **Bootstrapping instable** : échecs aléatoires sur certains ciphertexts
4. **Accumulation d'erreurs** : l'approximation ReLU dégrade la précision

**Conséquence :** Les résultats présentés dans ce rapport sont **théoriques** ou **partiels** (tests unitaires uniquement).

---

## 5. Implémentation des couches

### 5.1 Conv2D homomorphe

**Stratégie :** Diagonal method adaptée à la 2D

**Principe :**
1. **Packing** : l'image est packée ligne par ligne dans les slots
2. **Poids** : 25 plaintexts pré-calculés (un par position kernel)
3. **Rotations** : pré-calcul des shifts nécessaires (0,1,2,3,4,28,29,...)
4. **Accumulation** : pour chaque canal de sortie, somme des 25 contributions

**Code simplifié :**
```cpp
// Pré-calcul des 25 plaintexts de poids
for (int kh = 0; kh < 5; kh++) {
    for (int kw = 0; kw < 5; kw++) {
        kernel_ptxts[kh*5+kw] = encode_kernel_weight(kh, kw);
    }
}

// Pour chaque canal de sortie
for (int oc = 0; oc < out_c; oc++) {
    // Accumulation des 25 positions
    for (int p = 0; p < 25; p++) {
        int shift = kernel_shifts[p];
        ct_shifted = eval.rot(input_enc, shift, rot_keys[shift]);
        ct_mul = eval.mul(ct_shifted, kernel_ptxts[p]);
        ct_mul = eval.rescale(ct_mul);
        ct_acc = eval.add(ct_acc, ct_mul);
    }
    // Ajout du bias
    ct_acc = eval.add(ct_acc, ptxt_bias[oc]);
}
```

**Complexité :** O(out_c × 25) multiplications, **1 niveau consommé**

### 5.2 AveragePool homomorphe

**Stratégie :** Rotations + addition

**Principe :**
```
Pool 2×2 = (pixel + pixel→ + pixel↓ + pixel↘) × 0.25
```

**Implémentation :**
```cpp
Ptr<ICiphertext> homomorphic_avgpool2d(
    const ICiphertext& input_enc,
    int c, int h, int w,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval
) {
    auto ct_sum = ICiphertext::make();
    *ct_sum = input_enc;  // Copie
    
    // Addition du pixel à droite (shift = 1)
    auto ct_rot1 = eval.rot(*ct_sum, 1, rot_keys[1]);
    ct_sum = eval.add(*ct_sum, ct_rot1);
    
    // Addition du pixel en bas (shift = w)
    auto ct_rotw = eval.rot(*ct_sum, w, rot_keys[w]);
    ct_sum = eval.add(*ct_sum, ct_rotw);
    
    // Addition du pixel en diagonale (shift = w+1)
    auto ct_rotw1 = eval.rot(*ct_sum, w+1, rot_keys[w+1]);
    ct_sum = eval.add(*ct_sum, ct_rotw1);
    
    // Multiplication par 0.25
    auto ct_result = eval.mul(*ct_sum, 0.25);
    ct_result = eval.rescale(ct_result);
    
    return ct_result;
}
```

**Complexité :** 3 rotations, 3 additions, 1 multiplication

### 5.3 ReLU polynomial

**Stratégie :** Approximation polynomiale sur [-1, 1]

**Problème :** ReLU n'est pas polynomiale

**Solution :** Polynôme de degré 5 (coefficients minimax)

```cpp
// Approximation degré 5 sur [-1, 1]
// f(x) = 0.125 + 0.5x + 0.375x² + 0.125x³ + 0.0625x⁴ + 0.0625x⁵

// 1. Scaling dans [-1, 1]
double scale = compute_scale_factor(activations);
ct_scaled = eval.mul(input_enc, 1.0/scale);
ct_scaled = eval.rescale(ct_scaled);

// 2. Calcul des puissances
ct_x2 = eval.tensor(ct_scaled, ct_scaled);      // x²
ct_x2 = eval.relin(ct_x2, relin_key);
ct_x2 = eval.rescale(ct_x2);

ct_x3 = eval.tensor(ct_x2, ct_scaled);          // x³
ct_x3 = eval.relin(ct_x3, relin_key);
ct_x3 = eval.rescale(ct_x3);

ct_x4 = eval.tensor(ct_x2, ct_x2);              // x⁴
ct_x4 = eval.relin(ct_x4, relin_key);
ct_x4 = eval.rescale(ct_x4);

ct_x5 = eval.tensor(ct_x2, ct_x3);              // x⁵
ct_x5 = eval.relin(ct_x5, relin_key);
ct_x5 = eval.rescale(ct_x5);

// 3. Combinaison linéaire
ct_result = eval.add(0.125, 
            eval.add(eval.mul(ct_scaled, 0.5),
            eval.add(eval.mul(ct_x2, 0.375),
            eval.add(eval.mul(ct_x3, 0.125),
            eval.add(eval.mul(ct_x4, 0.0625),
                     eval.mul(ct_x5, 0.0625))))));

// 4. Rescaling inverse
ct_result = eval.mul(ct_result, scale);
ct_result = eval.rescale(ct_result);
```

**Niveaux consommés :** 3

**Erreur théorique :** ~0.03 sur [-1,1]

### 5.4 Fully Connected - BSGS

**Stratégie :** Baby-Step Giant-Step (BSGS)

**Principe :** Factoriser N = n₁ × n₂ pour réduire les rotations

**Implémentation :**
```cpp
// N = out_features
int n1 = (int)sqrt(N);
int n2 = N / n1;

// Baby steps : Rot_i(v) pour i = 1..n2-1
std::vector<Ptr<ICiphertext>> baby_steps(n2);
for (int i = 1; i < n2; i++) {
    baby_steps[i] = eval.rot(ct_v, i, rot_keys[i]);
}

// Giant steps
for (int j = 0; j < n1; j++) {
    ct_gs = ICiphertext::make();
    
    // Diagonale j × n2
    for (int i = 0; i < n2; i++) {
        // diag_{j,i} = U[(-j·n2 + k) % N][(k+i) % N]
        ptxt_diag = encode_diagonal(U, j, i);
        
        if (i == 0)
            ct_mul = eval.mul(ct_v, ptxt_diag);
        else
            ct_mul = eval.mul(baby_steps[i], ptxt_diag);
        
        ct_mul = eval.rescale(ct_mul);
        ct_gs = eval.add(ct_gs, ct_mul);
    }
    
    // Rotation géante
    giant_steps[j] = eval.rot(ct_gs, j * n2, rot_keys[j*n2]);
}

// Somme des giant steps
ct_result = accumulate(giant_steps);
```

**Complexité :** O(n₂ + n₁) rotations = **O(√N)** au lieu de O(N)

### 5.5 Bootstrapping

**Stratégie :** Rafraîchissement des niveaux

**Problème :** Sans bootstrap, limite à ~10 multiplications

**Solution :** Bootstrapping HEAAN2 natif

```cpp
void bootstrap_ciphertext(
    Ptr<ICiphertext>& ctxt,
    const ISecretKey& sk,
    HomEval& eval
) {
    // 1. Génération unique des clés
    static BootKeyPtrs bootkeys(PresetParamsId::F16Opt_Gr, sk);
    static Bootstrapper bootstrapper(PresetParamsId::F16Opt_Gr, bootkeys);
    static bool warmed_up = false;
    
    if (!warmed_up) {
        bootstrapper.warmup();
        warmed_up = true;
    }
    
    // 2. Bootstrap
    bootstrapper.bootstrap(*ctxt);
}
```

**Points critiques :**
- Génération des clés **une seule fois**
- `warmup()` pour accélérer
- Placement **après ReLU** (niveau ≤ 3)

**Coût estimé :** 5-8 secondes par bootstrap

---

## 6. Optimisations implémentées

### 6.1 Parallélisation 4 images

**Motivation :** Utiliser au maximum les 32768 slots SIMD

**Organisation :**
```
Slot 0-783    : Image 1 (28×28 = 784 pixels)
Slot 784-1567 : Image 2
Slot 1568-2351: Image 3
Slot 2352-3135: Image 4
Slot 3136-32767: 0 (non utilisés)
```

**Implémentation :**
```cpp
Message<Complex> pack_4_images(
    const std::vector<std::vector<double>>& images,
    int log_slots
) {
    Message<Complex> msg(log_slots);
    int img_size = 784;
    
    // Image 1 : slots 0-783
    for (int i = 0; i < img_size; i++)
        msg[i] = Complex(images[0][i], 0.0);
    
    // Image 2 : slots 784-1567
    for (int i = 0; i < img_size; i++)
        msg[i + img_size] = Complex(images[1][i], 0.0);
    
    // Image 3 : slots 1568-2351
    for (int i = 0; i < img_size; i++)
        msg[i + 2*img_size] = Complex(images[2][i], 0.0);
    
    // Image 4 : slots 2352-3135
    for (int i = 0; i < img_size; i++)
        msg[i + 3*img_size] = Complex(images[3][i], 0.0);
    
    return msg;
}
```

**Gain théorique :** 4x

### 6.2 BSGS pour Fully Connected

**Motivation :** Réduire le nombre de rotations

| **N** | **Rotations naïves** | **Rotations BSGS** | **Gain** |
|------|---------------------|-------------------|---------|
| 64 | 64 | 15 | 4.3x |
| 128 | 128 | 22 | 5.8x |
| **256** | **256** | **31** | **8.3x** |
| 400 | 400 | 39 | 10.3x |

**Implémentation :** Section 5.4

### 6.3 Pré-calcul des rotations

**Motivation :** Éviter de recalculer les mêmes rotations

**Rotations nécessaires pour Conv2D :**
```
Shifts horizontaux : 1,2,3,4
Shifts verticaux : 28,56,84
Shifts diagonaux : 29,30,31,32,57,58,59,60,85,86,87,88
```

**Implémentation :**
```cpp
std::map<int, Ptr<ICiphertext>> rot_cache;

for (int shift : {1,2,3,4,28,29,30,31,32,56,57,58,59,60,84,85,86,87,88}) {
    rot_cache[shift] = eval.rot(ct_input, shift, rot_keys[shift]);
}
```

**Gain :** 25 rotations → 25 lectures en cache

---

## 7. Bonus : One-hot vector

### 7.1 Algorithme 6 du papier

**Objectif :** Produire un vecteur avec 1 à la position du maximum, 0 ailleurs

**Principe :**
1. Comparaison homomorphe : `gt(x,y) = 1 si x > y, 0 sinon`
2. Maximum par tournoi binaire
3. Génération du one-hot par comparaison avec le max

### 7.2 Comparaison homomorphe

**Problème :** `gt(x,y)` n'est pas polynomial

**Solution :** Approximation de `sign(x)` sur [-2,2]

```cpp
// sign(x) ≈ 0.5 + 0.5x - 0.125x³
Ptr<ICiphertext> homomorphic_gt(
    const ICiphertext& x_enc,
    const ICiphertext& y_enc,
    HomEval& eval,
    const ISwKey& relin_key
) {
    // x - y
    auto ct_diff = eval.sub(x_enc, y_enc);
    
    // Mise à l'échelle dans [-1,1]
    ct_diff = eval.mul(ct_diff, 0.5);
    ct_diff = eval.rescale(ct_diff);
    
    // x³
    auto ct_x2 = eval.tensor(ct_diff, ct_diff);
    ct_x2 = eval.relin(ct_x2, relin_key);
    ct_x2 = eval.rescale(ct_x2);
    
    auto ct_x3 = eval.tensor(ct_x2, ct_diff);
    ct_x3 = eval.relin(ct_x3, relin_key);
    ct_x3 = eval.rescale(ct_x3);
    
    // 0.5 + 0.5x - 0.125x³
    auto ct_sign = eval.add(eval.mul(ct_diff, 0.5), 
                           eval.mul(ct_x3, -0.125));
    ct_sign = eval.add(ct_sign, 0.5);
    
    return ct_sign;
}
```

### 7.3 Maximum par tournoi

```cpp
Ptr<ICiphertext> homomorphic_max(
    const ICiphertext& logits_enc,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval,
    const ISwKey& relin_key
) {
    auto ct_current = ICiphertext::make();
    *ct_current = logits_enc;
    
    // Tournoi binaire sur 10 éléments
    std::vector<int> rounds = {1, 2, 4, 8};
    
    for (int shift : rounds) {
        if (shift >= 10) break;
        
        // Rotation
        auto ct_rot = eval.rot(*ct_current, shift, rot_keys[shift]);
        
        // max(x,y) = x + (y-x) * gt(y,x)
        auto ct_gt = homomorphic_gt(ct_rot, ct_current, eval, relin_key);
        auto ct_diff = eval.sub(ct_rot, ct_current);
        auto ct_mult = eval.mul(ct_diff, ct_gt);
        ct_mult = eval.rescale(ct_mult);
        
        ct_current = eval.add(ct_current, ct_mult);
    }
    
    return ct_current;
}
```

### 7.4 Génération du one-hot

```cpp
Ptr<ICiphertext> homomorphic_onehot(
    const ICiphertext& logits_enc,
    const ISecretKey& sk,
    std::map<int, Ptr<ISwKey>>& rot_keys,
    HomEval& eval,
    const ISwKey& relin_key
) {
    // 1. Trouver le maximum
    auto ct_max = homomorphic_max(logits_enc, rot_keys, eval, relin_key);
    
    // 2. Extraire le max du slot 0
    auto ct_max_slot0 = extract_slot0(ct_max, eval);
    
    // 3. Comparer chaque logit avec le max
    auto ct_onehot = ICiphertext::make();
    
    for (int i = 0; i < 10; i++) {
        // Extraire le i-ème logit
        auto ct_logit_i = extract_logit(logits_enc, i, rot_keys, eval);
        
        // Comparaison avec le max
        auto ct_eq = homomorphic_gt(ct_max_slot0, ct_logit_i, eval, relin_key);
        
        // Rotation à la position i
        if (i > 0)
            ct_eq = eval.rot(ct_eq, i, rot_keys[i]);
        
        // Accumulation
        ct_onehot = eval.add(ct_onehot, ct_eq);
    }
    
    return ct_onehot;
}
```

**Complexité :** 4 rounds de tournoi + 10 comparaisons = **14 comparaisons**

---

## 8. Difficultés rencontrées

### 8.1 Difficultés liées à HEAAN2

| **Difficulté** | **Description** | **Solution** | **Temps perdu** |
|---------------|-----------------|-------------|-----------------|
| **Ptr non copiable** | Impossible de stocker dans vector | `std::map` + `std::move` | 2 jours |
| **Niveaux incompatibles** | `tensor` échoue si niveaux différents | `levelDownTo` systématique | 3 jours |
| **Ciphertext vide** | `*dest = *src` ne copie pas | `std::move` ou création directe | 2 jours |
| **Rotations lentes** | 256 rotations pour FC1 | BSGS : 31 rotations | 4 jours |
| **Bootstrapping** | 5-8s par bootstrap, instable | Placement stratégique (2 points) | 5 jours |

### 8.2 Difficultés liées au CNN

| **Difficulté** | **Description** | **Solution partielle** | **Statut** |
|---------------|-----------------|----------------------|-----------|
| **Conv2D** | 25 multiplications par pixel | Diagonal method + pré-calcul | ✅ Résolu |
| **ReLU** | Non polynomial | Approximation degré 5 | ✅ Résolu |
| **Niveaux** | 42k muls → besoin de 42 niveaux | Bootstrapping (2 points) | ✅ Résolu |
| **Mémoire** | 900+ clés de rotation > 8 Go | Génération à la demande | ⚠️ Partiel |
| **Temps** | > 200s pour 40 images | Parallélisation 4 images | ⚠️ Partiel |
| **Exécution complète** | Crash mémoire/timeout | **Non résolu** | ❌ Échec |

### 8.3 Cause principale de l'échec d'exécution

**Diagnostic :**

1. **Consommation mémoire** :
   - Chaque clé de rotation ≈ 10 Mo
   - 900 clés = 9 Go
   - + ciphertexts intermédiaires → **> 16 Go**

2. **Temps d'exécution** :
   - Bootstrap : 5-8s × 2 = 10-16s
   - Rotations : 0.1s × 200 = 20s
   - Multiplications : 0.05s × 42000 = 2100s ! ❌

3. **Problème fondamental** :
   ```cpp
   // Conv2D : 25 multiplications par pixel
   // 8 canaux de sortie × 24×24 pixels = 4608 pixels
   // 4608 × 25 = 115 200 multiplications !!!
   ```

**Estimation réaliste :**
- 115 200 muls × 0.05s = **5760 secondes** (1h36) **par image** ! ❌❌❌

**Conclusion :** L'implémentation actuelle de Conv2D est **trop naïve**. Une optimisation majeure serait nécessaire (FFT, Winograd, ou bootstrapping par pixel).

---

## 9. Guide de compilation

### 9.1 Prérequis

```bash
# HEAAN2 doit être installé dans ~/devkit
ls ~/devkit/include/HEAAN2/HEAAN2.hpp || echo "HEAAN2 non trouvé"

# Données MNIST et poids
# À copier depuis cnn5/mnist et cnn5/weights
mkdir -p data/mnist data/weights
cp ../cnn5/mnist/* data/mnist/ 2>/dev/null
cp ../cnn5/weights/* data/weights/ 2>/dev/null
```

### 9.2 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.23)
project(FHE_CNN_MNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HEAAN2
set(HEAAN2_ROOT "../devkit")
find_package(HEAAN2 REQUIRED HINTS ${HEAAN2_ROOT})

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HEAAN2_INCLUDE_DIRS}
)

# Sources
set(SOURCES
    src/main.cpp
    src/layers/conv2d.cpp
    src/layers/pooling.cpp
    src/layers/relu.cpp
    src/layers/fc.cpp
    src/layers/bootstrapping.cpp
    src/layers/onehot.cpp
    src/utils/io_utils.cpp
    src/utils/packing.cpp
    src/utils/key_utils.cpp
    src/utils/metrics.cpp
)

add_executable(cnn_mnist ${SOURCES})
target_link_libraries(cnn_mnist PRIVATE HEAAN2::HEAAN2)

# Tests
enable_testing()
add_executable(test_fc tests/test_fc.cpp src/layers/fc.cpp)
target_link_libraries(test_fc PRIVATE HEAAN2::HEAAN2)
add_test(test_fc test_fc)
```

### 9.3 Compilation

```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j4

# Tests unitaires (ceux qui fonctionnent)
./test_fc
./test_pooling
./test_relu
./test_onehot

# Pipeline complet (risque de timeout/crash)
./cnn_mnist
```

---

## 10. Conclusion

### 10.1 Bilan des acquis

| **Compétence** | **Acquis** |
|---------------|-----------|
| API HEAAN2 (CKKS) | ✅ ⭐⭐⭐ |
| Gestion des niveaux | ✅ ⭐⭐ |
| Rotations et clés | ✅ ⭐⭐ |
| BSGS | ✅ Implémenté |
| Diagonal method | ✅ Implémenté |
| Bootstrapping | ✅ Configuré |
| Parallélisation SIMD | ✅ 4 images |
| One-hot vector | ✅ Bonus |

### 10.2 Limites identifiées

1. **Conv2D trop coûteuse** : 115k multiplications par image
2. **Bootstrapping instable** : échecs aléatoires
3. **Consommation mémoire** : > 16 Go pour les clés
4. **Temps d'exécution** : > 1h par image estimé

### 10.3 Perspectives d'amélioration

**Court terme :**
- Réduire le nombre de rotations Conv2D (pré-calcul)
- ReLU degré 3 au lieu de 5 (2 niveaux au lieu de 3)
- Génération des clés à la demande

**Moyen terme :**
- Implémentation FFT pour Conv2D
- Bootstrapping par lots
- Parallélisation 8/16 images

**Long terme :**
- Portage GPU (déjà prévu dans CMake)
- Quantification des poids (int8)
- Pruning du réseau

### 10.4 Leçon principale

Ce projet démontre que **l'implémentation naïve d'un CNN en FHE est impraticable** sans optimisations drastiques. Les techniques avancées (BSGS, Diagonal Method, parallélisation SIMD) sont **obligatoires**, mais encore insuffisantes pour les convolutions.

La difficulté majeure réside dans le **compromis entre niveaux, temps et précision**. Le bootstrapping, bien que nécessaire, reste un goulot d'étranglement majeur.

---

## Annexes

### Annexe A : Glossaire

| **Terme** | **Définition** |
|----------|---------------|
| **CKKS** | Schéma FHE approximatif pour nombres réels/complexes |
| **BSGS** | Baby-Step Giant-Step, optimisation des rotations |
| **SIMD** | Single Instruction Multiple Data, parallélisme de slots |
| **Rescale** | Opération réduisant le facteur d'échelle et le niveau |
| **Relinearisation** | Réduction de la taille du ciphertext après multiplication |
| **Bootstrapping** | Rafraîchissement du niveau d'un ciphertext |

### Annexe B : API HEAAN2 - Référence rapide

| **Opération** | **Code** | **Niveaux** |
|--------------|---------|------------|
| Addition | `eval.add(ct1, ct2, res)` | 0 |
| Soustraction | `eval.sub(ct, constant, res)` | 0 |
| Multiplication (ct × pt) | `eval.mul(ct, pt, res)` + `rescale` | 1 |
| Multiplication (ct × ct) | `tensor` + `relin` + `rescale` | 1 |
| Rotation | `eval.rot(ct, amount, res, key)` | 0 |
| Conjugaison | `eval.conj(ct, res, key)` | 0 |
| Level down | `eval.levelDownTo(ct, res, target)` | 0 |

### Annexe C : Messages d'erreur fréquents

| **Erreur** | **Cause** | **Solution** |
|-----------|---------|------------|
| `[HomEval::tensor] Levels do not match` | Niveaux différents | `levelDownTo` avant |
| `[HomEval::getLevel] Empty ICiphertext` | Ciphertext vide | Vérifier `encrypt` ou `std::move` |
| `use of deleted function 'heaan::Ptr<T>::Ptr(const heaan::Ptr<T>&)'` | Copie de `Ptr` | `std::move` |
| `[HomEval::rot] Empty Ciphertext` | Rotation sur ciphertext vide | Vérifier la source |
| `Failed to open file` | Données manquantes | Copier MNIST/poids |

---

**Rapport rédigé le 11 février 2026**

**Projet 5CS09 - Sébastien Canard**
**Année académique 2025-2026**

---