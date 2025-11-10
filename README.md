# AESS-IES-TSYP-13: Système FDIR Autonome par XGBoost Accéléré sur FPGA

Ce projet propose un système de **Détection, Isolation et Récupération des Fautes (FDIR)** embarqué pour un CubeSat 3U. Notre solution garantit une **vitesse électronique** et une **résilience spatiale** en accélérant un modèle d'Intelligence Artificielle (XGBoost) directement sur la logique reconfigurable d'un FPGA de grade spatial (Rad-Tolerant), éliminant ainsi le besoin d'intervention immédiate de l'ordinateur de bord (OBC) ou du sol.

### Objectifs Clés
* **Vitesse :** Latence de diagnostic totale de **5–10 µs** par traitement parallèle RTL.
* **Fiabilité :** Implémentation de la **Triple Modular Redundancy (TMR)** pour la résilience aux radiations.
* **Autonomie :** Rétablissement déterministe et immédiat des fautes critiques (ADCS/EPS).

---

## Architecture, Technologies et Flux de Travail

L'architecture est basée sur une boucle de contrôle fermée et ultra-rapide (voir les diagrammes dans `5_DOCUMENTATION/`). Le flux de données est direct : **Capteurs ADCS/EPS** $\to$ **ADC** $\to$ **FPGA** $\to$ **Actuateurs/Drivers**.

Le cœur du diagnostic est l'algorithme **XGBoost (eXtreme Gradient Boosting)**, choisi pour sa précision supérieure en **Détection et Isolation** des fautes sur les données tabulaires des capteurs. Ce modèle est exécuté sur le **FPGA** qui, au-delà de sa **vitesse électronique**, implémente la **TMR** au niveau RTL pour assurer la fiabilité des diagnostics. La preuve du concept repose sur la **Simulation MATLAB/Simulink**, utilisée pour modéliser la dynamique orbitale et générer des jeux de données d'entraînement réalistes, assurant ainsi la pertinence de l'algorithme.

---

## Structure du Dépôt

La structure reflète le flux de travail d'ingénierie du système :

* **`1_DATA_SIMULATION/`** contient les scripts MATLAB/Simulink et les données d'entraînement brutes.
* **`2_MODEL_DEVELOPMENT/`** est dédié au code Python et aux scripts de quantification du modèle (4-8 bits).
* **`3_HARDWARE_RTL/`** héberge le code C/C++ HLS et les fichiers Verilog/VHDL finaux (avec TMR).
* **`4_VERIFICATION/`** contient les testbenches pour la vérification RTL et les logs de simulation.
* **`5_DOCUMENTATION/`** regroupe le rapport final, le schéma synoptique et le diagramme de fonctionnement.

---

## Résultats Clés et Preuve de Succès

Le succès de la conception est confirmé par l'atteinte des indicateurs de performance critiques :

| Métrique | Résultat Obtenu | Preuve du Succès |
| :--- | :--- | :--- |
| **Latence d'Inférence** | **5−10 µs** | Respect strict du critère de **Vitesse Électronique** pour l'ADCS. |
| **Précision d'Isolation** | **> 96.6%** | Haute fiabilité du diagnostic permettant un rétablissement autonome. |
| **Résilience** | **TMR Actif** | Validation de la conception pour l'environnement spatial (Mitigation des SEU). |

Le système FDIR par XGBoost sur FPGA offre une solution **précise, rapide et résiliente aux radiations** pour les opérations critiques d'un CubeSat 3U.
