# SPEECH TO TEXT : l'art de retranscrire la parole

## Introduction

Ce projet a pour but de faire une preuve de concept pour le déploiement d'une technologie Speech to text sur un Cortex M7 embarquant peu de mémoire RAM.
Ce projet a été réalisé par Elisa BONIFAS, Alex DUFOUR ,Bechir MNAKRI et Felipe NEGRELLI WOLTER dans le cadre des Projets Industriels de TELECOM NANCY. Il est encadrer par Yoan et Mme MAIMOUR


## Organisation 
Ce projet git implemente la partie intelligence artificiel du projet.
Il comporte différente branche qui implémente chacune une phase, partie ou fonctionnalité diférentes.
Il y a ainsi les branches:

- "phase_1" qui implemente un modele de reconnaissance vocale sur 10 mots.
- "phase_1_extra" qui augmente le modele de la phase 1 sur 30 mots.
- "deepspeech_custom" qui implemente un modèle de reconnaissance vocale basé sur l'architecture deepspeech.
- "wav2vec" qui implemente un modèle de reconnaissance vocale basé sur l'architecture wav2vec.

Chaque branche a un README qui detaille les prérequis, la procédure d'execution, l'architecture du modele et les choix qui ont été fait.
On trouve aussi un dockerfile pour pouvoir exécuter le projet dans un containeur.



