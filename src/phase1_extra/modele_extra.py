""""
Dans ce fichier, le modèle est défini et entrainé avec les données pré-traitées sur 30 mots.
Pour enregistrer le modèle entrainé, il suffit de décommenter les 3 dernières lignes du fichier

This file defined and  train the model using the pre-processed data on 30 words.
To save the model, you have to uncomment the last three ligns.
"""

import numpy as np
import tensorflow as tf
import os
import datetime

from pretraitement2 import download_data2

DATA="hub://activeloop/speech-commands-train"
NB_MOTS = 30 # Changer en fonction du nombre de mots du corpus
CHEMIN_DONNEES= './donnees_traitees_extra' # Dossier contenant les données pré-traitées
CHEMIN_SAUVEGARDE_MODELE = './modeles_extra' # Dossier de sauvegarde des modèles entrainés
NOM_MODEL='mel-cnn-enhance'
NB_EPOCH=8

if not os.path.exists(CHEMIN_DONNEES):
    download_data2(DATA,CHEMIN_DONNEES)


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
history = model.fit(x_train, y_train, validation_split= 0.15, epochs=NB_EPOCH)

# Evaluation du modèle
acc=model.evaluate(x_test, y_test)

version = open("version_model_extra.txt", "a")
version.write("\n")
version.write(str(datetime.datetime.today()))
version.write("  "+NOM_MODEL+ " entrainé sur "+CHEMIN_DONNEES+"\n")
version.write("loss : "+str(acc[0])+", accuracy : "+str(acc[1]))
version.write("epoch :"+ str(NB_EPOCH))
version.close()

# Sauvegarde du modèle entraîné (decommenter la ligne si le modèle est satisfaisant)
if not os.path.exists(CHEMIN_SAUVEGARDE_MODELE):
    os.makedirs(CHEMIN_SAUVEGARDE_MODELE)
model.save(CHEMIN_SAUVEGARDE_MODELE+"/"+NOM_MODEL)