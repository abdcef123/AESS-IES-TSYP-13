# AESS-IES-TSYP-13: Système FDIR Autonome par XGBoost Accéléré sur FPGA

Ce projet présente un système de **Détection, Isolation et Récupération des Fautes (FDIR)** embarqué pour un CubeSat 3U. Notre solution garantit une **vitesse électronique** et une **résilience spatiale** en accélérant un modèle d'Intelligence Artificielle (XGBoost) directement sur la logique reconfigurable d'un FPGA de grade spatial (Rad-Tolerant). Ceci élimine le besoin d'intervention immédiate de l'ordinateur de bord (OBC) ou du sol, assurant une autonomie critique.

### Objectifs Clés
* **Vitesse :** Latence de diagnostic totale de **5–10 µs** par traitement parallèle RTL.
* **Fiabilité :** Implémentation de la **Triple Modular Redundancy (TMR)** pour la résilience aux radiations.
* **Autonomie :** Rétablissement déterministe et immédiat des fautes critiques (ADCS/EPS).
* ---

## 🛠️ Architecture et Technologies

L'architecture est basée sur une boucle de contrôle fermée et ultra-rapide (voir `5_DOCUMENTATION/Synoptic_Diagram.png`).

| Composant | Technologie | Rôle dans le FDIR |
| :--- | :--- | :--- |
| **Flux** | Capteurs $\to$ ADC $\to$ FPGA $\to$ Actuateurs | **Vitesse Électronique :** Confirme la connexion directe pour une faible latence. |
| **Algorithme** | **XGBoost** (eXtreme Gradient Boosting) | **Détection et Isolation :** Classifie les fautes avec une précision élevée. |
| **Matériel** | **FPGA** (Grade Spatial / RTL) | Garantit la **Vitesse** et implémente la **TMR** pour la résilience. |
| **Simulation** | **MATLAB/Simulink** | Modélisation de la dynamique orbitale et génération des jeux de données d'entraînement réalistes. |
---

## 📂 Structure du Dépôt

Cette structure reflète le flux de travail de l'ingénierie du système (Data → Logiciel → Matériel → Vérification).

| Dossier | Contenu | Objectif |
| :--- | :--- | :--- |
| `1_DATA_SIMULATION/` | Scripts MATLAB/Simulink et données d'entraînement brutes. | Fournir la base factuelle (données simulées) pour l'entraînement IA. |
| `2_MODEL_DEVELOPMENT/` | Code Python initial pour XGBoost, scripts de quantification (4-8 bits). | Démontrer le choix de l'algorithme et la préparation pour l'exportation RTL. |
| `3_HARDWARE_RTL/` | Code C/C++ HLS et les fichiers Verilog/VHDL finaux (avec TMR). | Contenir l'implémentation physique et résiliente sur FPGA. |
| `4_VERIFICATION/` | Fichiers Testbench pour la vérification RTL et logs de simulation. | Prouver que la logique matérielle fonctionne correctement et atteint les cibles de latence. |
| `5_DOCUMENTATION/` | Rapport technique final, schémas synoptiques et de fonctionnement. | Fournir le contexte et les résultats de l'ingénierie du système. |

---
## 📈 Résultats Clés (KPIs)

| Métrique | Résultat Obtenu | Preuve du Succès |
| :--- | :--- | :--- |
| **Latence d'Inférence** | **5−10 µs** | Respect strict du critère de **Vitesse Électronique** pour l'ADCS. |
| **Précision d'Isolation** | **> 96.6%** | Haute fiabilité du diagnostic permettant un rétablissement autonome. |
| **Résilience** | **TMR Actif** | Validation de la conception pour l'environnement spatial (Mitigation des SEU). |

---

## ✅ Prochaines Étapes

Pour exécuter et vérifier le projet :

1.  Clonez le dépôt.
2.  Commencez par `1_DATA_SIMULATION/` pour comprendre l'injection des fautes.
3.  Vérifiez les résultats du Testbench dans `4_VERIFICATION/` pour confirmer la latence de $\mathbf{5-10\ \mu s}$.
