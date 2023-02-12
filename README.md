# SPEECH TO TEXT : l'art de retranscrire la parole

Ce README concerne la branche "phase_1". Elle implemente un modèle de commande vocale.
On peut télécharger et prétraiter les données ainsi que entrainer le modèle et réaliser des prédictions.

## Prerequis


### Modules Python
Concernant les modules nécessaires à l'exécution du module, ils sont spécifiés dans le fichier requirements.txt.
Pour tous les installés, il suffit de lancer la commande:
```
pip3 install -r requirements.txt
```
Ces installations sont déjà présentes dans le docker fournit avec le projet.


### données d'entrainements

Les données d'entrainement proviennent de "ActiveLoop" dans le dataset "speech-commands-train".
Elles peuvent etre téléchargée et enregistrée après prétraitement en lançant le scipt pretraitenement.py et en donnant en argument le nom du dossier où stocker les données. La commande est la suivante
```
python3 src/phase1/pretraitement.py NOM_DU_DOSSIER
```
Cette étape est facultative. En effet, si lors de l'entrainement du modèle, les données n'ont pas été téléchargé ce derniers va lancer le script de prétraitement. Il faudrat ainsi prendre en compte la durée de téléchargement en plus du temps d'entrainement du modèle.

### Docker 

Un docker est fournit pour simplifier l'utilastion de ce projet.
Il y a deux manières de le lancer, soit à partir du dockerfile soit à partir de la sauvegarde.

Si on veut le lancer à partir du dockerfile, il suffit de construire le conteuneur avec ( en ayant au préalable lancer de demon docker):
```
docker build -t model_run .
```
On peut ensuite lancer le docker en mode interactif avec :
```
docker run -it model_run
```
Ctrl+D permet de fermer le docker.

## Exécution

### Entrainement du modèle

Pour entrainer le modèle, on doit lancer le script modele.py. On peut preciser le nom du modele que l'on souhaite créer, le nombre d'epochs  ainsi que le repertoire des données d'entrainement. Pour cela on utilise la commmande ( soit directement dans le terminal soit dans le terminal du docker):
```
python3 src/phase1/modele.py  -n NOM_MODELE -e NOMBRE_EPOCH -p REPERTOIRE_DONNEES
```

### Test 

#### Spectrogramme
On peut tester les spectogrammes produit par la carte afin de s'assurer de la ressemblance avec le spectrogramme utilisé pendant l'entrainement. Pour celon on dispose de plusieurs outils.

Tout d'abord un fichier qui va afficher le spectrogramme fait par la carte enregitrer en tant que fichier binaire. On doit donc lui passer en paramètre le chemin du fichier. On peut egalement préciser le mot représenté, si on souhaite l'enregistrer et le nom du fichier à sauvegarder
```
python3 test/lecture_spectrogramme.py -p CHEMIN_FICHIER -w MOT -s SAVE -n NAME
```

On a ensuite un fichier qui fait l'opération inverse c'est a dire qu'il enregistre un spectrogramme sous forme de fichier binaire. Il faut juste lui passer en paramètre le chemin du fichier
```
python3 test/ecriture_spectrogramme.py -p CHEMIN_FICHIER -w MOT -s SAVE -n NAME
```

On a également un fichier qui compare deux spectogrammes, un fait par la carte d'evaluation , l'autre à l'aide d'un module de tensorflow. Il fait de plus la prediction mots représenté par les spectogrammes. 
Il prend différents paramètres en entrée tel que les chemins des deux fichiers, le modèle a utilisé. On peut aussi precisé le mot qui doit etre predit, si on souhaite afficher le graphique, si on veut l'enregistrer et le nom du fichier sauvegarder le mot 

```
python3 .\test\comparaison_spectogramme.py  -c CHEMIN_SPECTOGRAMME_CARTE -a CHEMAIN_WAV -w MOT -d DISPLAY -s SAVE -n NOM
```

#### Modèle
Pour tester le modèle, on peut lancer le script test_vocal qui va effectuer un enregistrement pendant 3 seconde et renvoyer le mots compris par le modèle.
```
python3 src/phase1/test_vocal.py
```

On peut également faire un test à partir d'un fichier avec le script test_fichier. Il faut alors lui fournir le chemin vers le fichier. On peut également lui fournit le chemin du modele a utilisé, par défaut il s'agit de "modeles/mel-cnn"
Si il n'y a qu'un seul mot prononcé lors de l'enregistrement, on utilise:
```
python3 src/phase1/test_fichier.py -p CHEMIN_FICHIER -m CHEMIN_MODELE
```
Si il y en a plusieurs:
```
python3 src/phase1/test_fichier_long.py  -p CHEMIN_FICHIER -m CHEMIN_MODELE
```

## Modèle et Usage

### prétraitement

Avant d'être envoyé en entrée au modèle, les données sont prétraitées. Elles sont tout d'abord redimensionner à la même longueur puis transformé en pectogramme de mel avec un echelle en décibel.

### Modèle

Le modèle utilisé est un modèle CNN composé de 3 couches convolutives et de deux couches denses. Il est ensuite compilé à l'aide de la loss "sparse_categorical_crossentropy" et de l'optimiseur "RMSprop". La métrique utilisé est l'accuracy.


### Valeur par défaut
Nous avons définit certaines valeurs par défaut.
En effet, dans cette partie nous utilisons que 10 mots de commande : 'yes', 'no', 'up', 'down', 'right', 'left', 'stop', 'go', 'on' et 'off'.

Les données prétraitée ont une durrée d'enregistrement de 1s et la fréquence d'chantillonnage est de 16 000 HZ.

Par défaut, les données prétraitées sont enregistrées dans le dossier './donnees_traitees' et les modèles dans './modeles'.

De même par défaut,le modèle est entrainé sur 8 epochs.



## Details des fichiers

##### Péparation des données
pretraitement.py : télécharge, redimmensionne et transforme en spectogramme de mel les données du dataset : "hub://activeloop/speech-commands-train"

##### Construction du modèle
modele.py : Construit et entraine le modèle avec les données prétraitées.

version_model.txt : contient les versions et la justesse du modele

##### Test manuel du modèle
main.py :
applique le modèle a tous les audios du repertoire \audio

affichage.py :
fait l'affichage de la table récapitulative de l'analyse des audios

test_vocal.py : fait un enregistrement de 3s, analyse cet audio avec le modèle et renvoi le mot compris. Le signal est enregistré au format wav sous le nom "output.wav".

test_fichier.py: fait l'analyse d'un fichier .wav  et renvoi le mot compris.

model_fct.py : fonction pour appliquer un model à un tableau fournit en entrée.
