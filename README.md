# Readme IA :

Description de l'utilisation du docker, de l'organisation du git, 

## Organisation du git

Le git est séparer en fonction des phase du projet.
Tous les fichiers hors ressource et paramétrage sont placé dans le dossier src.

### Phase 1

#### dossiers
./donnees_traitees :
contient toutes les données qui on été préparer pour l'entrainement du modèle.

./modeles : 
Contient les dossier avec les différents modele. Chacun a un dossier.

./conversion_tflite contient les exemples de conversion de modèle avec "quantization" (réduction de la précision des nombre pour un modèle plus léger et plus rapide mais moins précis)

#### fichiers


##### Préparation des données

pretraitement.py : télécharge, redimmensionne et transforme en spectogramme de mel les données du dataset : "hub://activeloop/speech-commands-train"

reduction_de_bruit.py : affiche un filtre de réduction de bruit à un fichier donné ( EN COURS)

##### Construction du modèle

modele.py : Construit et entraine le modèle avec les données prétraitées.

version_model.txt : contient les versions et la justesse du modele
##### Test manuel du modèle

main.py : 
applique le modèle a tous les audios du repertoire \audio

affichage.py : 
fait l'affichage de la table récapitulative de l'analyse des audio

test_vocal.py : fait un enregistrement de 3s et analyse cet audio avec le modèle et renvoi le mot compris. Le signal est enregistré au format wav sous le nom output.

test_fichier.py: fait l'analyse d'un fichier .wav  et renvoi le mot compris.

model_fct.py : fonction pour appliqué un model a un tableau fournit en entrée.

##### Conversion en .tflite

post_quantization.py : Conversion du modèle phase1 en appliquant une "quantization" après l'entrainement. Les chiffres passent de int32 à int8

q_aware_training.py : Le modèle est ré-entrainé (fine-tuned) pour être préparé à la "quantization". 


### Phase 2 

Pour la phase 2, nous pourrons créer des dossiers différents en fonction des modèles.

#### wav2vec 

pretrained_test.py: Importation d'un modèle français et possibilité de tester en s'enregistrant.





