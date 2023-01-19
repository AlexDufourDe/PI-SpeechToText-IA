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
python3 src/phase2/download_mozillacmt.py (CHEMIN_JSON CHEMIN_SAUVEGARDE PRETRAITEMENT)
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

Pour entrainer le modèle, on doit lancer le script train.py.  On doit préciser la langue d'entrainement. On peut également  préciser le nombre d'epoch,le repertoire dans lequel enregistrer le modeleainsi que son nom. Pour cela on utilise la commmande ( soit directement dans le terminal soit dans le terminal du docker):
```
python3 src/phase2/train.py  (LANGUE NB_EPOCH CHEMIN_MODELE NOM_MODELE )
```



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

##### Péparation des données
pretraitrement.py : télécharge les données voulues et effectue le prétraitement.

##### Construction du modèle
build_model.py : construit le modele en ajoutant les différentes couches

callback.py: affiche un exemple de traduction d'une phrase à la fin de chaque epoch

ctc_loss: renvoi la loss de l'algoritme ctc

prediction.py : effectue la prediction du phrase par le modèle

train.py : entraine le modele selon les paramètres spécifié par l'utilisateur

version_en.txt : contient les versions du modele entrainé en anglais
version_fr.txt :contient les versions du modele entrainé en français


##### Test manuel du modèle
test_vocal_extra.py : fait un enregistrement de 3s, analyse cet audio avec le modèle et renvoi le mot compris. Le signal est enregistré au format wav sous le nom "output.wav".

test_fichier_extra.py: fait l'analyse d'un fichier .wav  et renvoi le mot compris.

