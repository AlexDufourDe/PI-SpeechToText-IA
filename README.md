# SPEECH TO TEXT : l'art de retranscrire la parole

Ce README concerne la branche "deepspeech_custom". Elle implemente un modèle de reconnaissance de phrase basée sur le modèle Deepspeech. Ce dernier est un projet opensource de mozilla : "https://github.com/mozilla/DeepSpeech"
On peut télécharger et prétraiter les données ainsi que entrainer le modèle.

## Prerequis


### Modules Python
Concernant les modules nécessaires à l'exécution du modèle, ils sont spécifiés dans le fichier requirements.txt.
Pour tous les installés, il suffit de lancer la commande:
```
pip3 install -r requirements.txt
```
Ces installations sont déjà présentes dans le docker fournit avec le projet.


### données d'entrainements

Les données d'entrainement proviennent de Mozilla Common voice pour l'entrainement en français et de LJ Speech de Keith Ito pour l'entrainement en anglais.

#### Mozilla common voice

Ce dataset est un ensemble données en français
Pour telechargé les données de mozilla common voice, il suffit de lancer le script pretraitement_json. Il est necessaire que le fichier mozilla_commonvoice.json soit dans le même dossier que le script sinon il faut indiqué le chemin vers ce fichier.
On peut le lancer avec la commande:
```
python3 src/phase2/download_mozillacm.py (CHEMIN_JSON CHEMIN_SAUVEGARDE PRETRAITEMENT)
```
Deux dossiers par défaut vont alors etre créer : mozilla_commonvoice qui contient les audios originaux et mozilla_common_voice_pretraitee qui contient les données prétraité (si on a signifié avec pretraitement que l'on voulait enregistrer les fichier pretraité). Si un chemain de sauvegarder a étét préciser alors les données originales seront sauvergardées dans un dossier du nom indiqué et les donnée prétraitée dans le dossier CHEMIN_SAUVEGARDE+'_pretraitee'

Cette étape est facultative. En effet, si lors de l'entrainement du modèle, les données n'ont pas été téléchargé ce derniers va lancer le script de prétraitement. Il faudrat ainsi prendre en compte la durée de téléchargement en plus du temps d'entrainement du modèle.

#### LJSpeech
Ce dataset est un dataset d'entrainement en anglais.
Les données de LJ Speech n'ont pas besoin d'etre téléchargé avant.


### Docker 

Un docker est fournit pour simplifier l'utilastion de ce projet.
Il y a deux manières de le lancer, soit à partir du dockerfile soit à partir de la sauvegarde.

Si on veut le lancer à partir du dockerfile, il suffit de construire le conteuneur avec ( en ayant au préalable lancer de demon docker):
```
docker build -t deepspeech_custom .
```
On peut ensuite lancer le docker en mode interactif avec :
```
docker run -it deepspeech_custom
```
Ctrl+D permet de fermer le docker.

## Exécution

### Entrainement du modèle

Pour entrainer le modèle, on doit lancer le script train_deepseepch_custom.py.  On peut preciser un certain nombre de paramètres. Ceux obligatoires sont le chemin vers les données, le chemin vers les meta données et le type de modele a entrainer. On peut choisir entre '2CNN+5RNN', 'CNN+RNN' et '3CNN'.   Pour cela on utilise la commmande ( soit directement dans le terminal soit dans le terminal du docker):
```
python3 src/phase2/train_deepspeech_custom.py -d CHEMIN_DATA -m CHEMIN_METADATA -mo TYPE_MODELE -o NOM_FICHER_SORTIE -e NOMBRE_EPOCHS -s VALIDATION_SPLIT 
```

Il est possible d'utiliser --help pour avoir le detail de chaque option.


## Modèle et Usage

### prétraitement

Avant d'être envoyé en entrée au modèle, les données sont prétraitées. Elles sont tout d'abord redimensionner à la même longueur puis transformé en spectogramme de mel avec un echelle en décibel.

### Modèle

#### Deepspeech 
Le modèle utilisé est un modèle composé de 3 couches dense une couche récurente et deux couches denses. Il est ensuite compilé à l'aide de la loss de l'algoritme CTC.

### Valeur par défaut

Les données prétraitée ont une durrée d'enregistrement de 1s et la fréquence d'chantillonnage est de 16 000 HZ.

Par défaut, les données prétraitées sont enregistrées dans le dossier './donnees_traitees_extra' et les modèles dans './modeles_extra'.

De même par défaut,le modèle est entrainé sur 8 epochs.



## Details des fichiers

##### Téléchargement des données
download_mozillacm.py : télécharge les données voulues et effectue le prétraitement si préciser.

##### Construction du modèle
build_model.py : construit les différents type de modeles  en ajoutant les différentes couches

train_deepseepch_custom.py : entraine le modele selon les paramètres spécifié par l'utilisateur

version_en.txt : contient les versions du modele entrainé en anglais
version_fr.txt :contient les versions du modele entrainé en français



