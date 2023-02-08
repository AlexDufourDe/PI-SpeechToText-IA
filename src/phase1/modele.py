""""
Dans ce fichier, le modèle est défini et entrainé avec les données pré-traitées sur 10 mots.
Le model est ensuite sauvegarder et la version du modele avec la loss et le nombre d'epoch est ensuite sauvegarder dans le fichier version_model.txt

This file defined and  train the model using the pre-processed data on 10 words.
then the model is save. The version, the loss and the number of epoch is save in the file version_model.txt
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import datetime
import sys

from pretraitement import download_data 

DATA="hub://activeloop/speech-commands-train"
NB_MOTS = 10 # Changer en fonction du nombre de mots du corpus
CHEMIN_SAUVEGARDE_MODELE = './src/phase1/modeles' # Dossier de sauvegarde des modèles entrainés

parser = argparse.ArgumentParser(description="Train a custom model of speech recognition on 10 words",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",default= './src/phase1/donnees_traitees', help="Path to the folder containing the data, by default it is '.src/phase1/donnees_traitees'")
parser.add_argument("-n","--name",default='mel-cnn',help="name of the model to train, by default it is 'mel-cnn' " )
parser.add_argument("-e","--epochs",default=8,help="Number of epochs for training, by default it is 8")
args = vars(parser.parse_args())

CHEMIN_DONNEES = args['path'] 
NOM_MODELE = args['name']
NB_EPOCH = args['epochs']



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

model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
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

version = open("src/phase1/version_model.txt", "a")
version.write("\n")
version.write(str(datetime.datetime.today()))
version.write("  "+NOM_MODEL+ " entrainé sur "+CHEMIN_DONNEES+"\n")
version.write("loss : "+str(acc[0])+", accuracy : "+str(acc[1]))
version.write("epoch :"+ str(NB_EPOCH))
version.close()

if not os.path.exists(CHEMIN_SAUVEGARDE_MODELE):
    os.makedirs(CHEMIN_SAUVEGARDE_MODELE)
model.save(CHEMIN_SAUVEGARDE_MODELE+"/"+NOM_MODEL)
