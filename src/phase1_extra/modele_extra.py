""""
Dans ce fichier, le modèle est défini et entrainé avec les données pré-traitées sur 30 mots.
Pour enregistrer le modèle entrainé, il suffit de décommenter les 3 dernières lignes du fichier

This file defined and  train the model using the pre-processed data on 30 words.
To save the model, you have to uncomment the last three ligns.
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import datetime
import matplotlib.pyplot as plt


from pretraitement_extra import download_data

DATA="hub://activeloop/speech-commands-train"
NB_MOTS = 30 # Changer en fonction du nombre de mots du corpus
CHEMIN_SAUVEGARDE_MODELE = './src/phase1_extra/modeles_extra' # Dossier de sauvegarde des modèles entrainés


# Command line parser

parser = argparse.ArgumentParser(description="Test the transcription of a wav file with a CNN model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",  help="Path to the folder containing the data, the default value is './src/phase1_extra/donnees_traitees_extra'")
parser.add_argument("-n","--name",help="name of the model to train, by default it is 'mel-cnn-enhance' " )
parser.add_argument("-e","--epochs",default=1,help="Number of epochs for training, by default it is 1")

args = vars(parser.parse_args())

if (args['path']):
    CHEMIN_DONNEES = args['path'] 
else:   
    CHEMIN_DONNEES= './src/phase1_extra/donnees_traitees_extra' # Dossier contenant les données pré-traitées

if (args['name']):
    NOM_MODEL = args['name']
else: 
    NOM_MODEL='mel-cnn-enhance'

if (args['epochs']):
    NB_EPOCH = int(args['epochs'])
else: 
    NB_EPOCH=1

print(f"nb_epoch : {type(NB_EPOCH)} {NB_EPOCH}\nNom model : {NOM_MODEL}\nChemin donnes : {CHEMIN_DONNEES}")

if not os.path.exists(CHEMIN_DONNEES): 
    download_data(DATA,CHEMIN_DONNEES)

# Importation des données
x_train, y_train = np.load(CHEMIN_DONNEES+'/train_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/train_labels.npy')
x_test, y_test = np.load(CHEMIN_DONNEES+'/test_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/test_labels.npy')


# Architecture du modèle
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NB_MOTS, activation='softmax'))

# Compilation du modèle
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

# Entrainement du modèle
h = model.fit(x_train, y_train, validation_split= 0.15, epochs=NB_EPOCH)

# Evaluation du modèle
acc=model.evaluate(x_test, y_test)

max=np.max(h.history['val_accuracy'])
ind=h.history['val_accuracy'].index(max)


version = open("src/phase1_extra/version_model_extra.txt", "a")
version.write("\n\n")
version.write(str(datetime.datetime.today()))
version.write("  "+NOM_MODEL+ " entraine sur "+CHEMIN_DONNEES+"\n")
version.write("loss : "+str(acc[0])+", accuracy : "+str(acc[1]))
version.write(" epoch :"+ str(NB_EPOCH))
version.write("\n meilleure val_accuracy "+ str(max)+" atteinte lors de l'epoch "+str(ind))
version.close()


plt.figure()
plt.plot(h.history['accuracy'], label="loss")
plt.plot(h.history['val_accuracy'], label="val_loss")
plt.plot([acc[1] for i in range(len(h.history['accuracy']))],label='final accuracy')
plt.legend()
plt.show()

# Sauvegarde du modèle entraîné (decommenter la ligne si le modèle est satisfaisant)
if not os.path.exists(CHEMIN_SAUVEGARDE_MODELE):
    os.makedirs(CHEMIN_SAUVEGARDE_MODELE)
model.save(CHEMIN_SAUVEGARDE_MODELE+"/"+NOM_MODEL)
