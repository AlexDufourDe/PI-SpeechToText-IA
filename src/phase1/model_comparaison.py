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
import matplotlib.pyplot as plt

from pretraitement import download_data 

DATA="hub://activeloop/speech-commands-train"
NB_MOTS = 10 # Changer en fonction du nombre de mots du corpus
CHEMIN_SAUVEGARDE_MODELE = './src/phase1/modeles' # Dossier de sauvegarde des modèles entrainés

parser = argparse.ArgumentParser(description="Train a custom model of speech recognition on 10 words",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e","--epochs",default=8,help="Number of epochs for training, by default it is 8")
parser.add_argument("-i","--i",default=3,help="Number of epochs for training, by default it is 3")
args = vars(parser.parse_args())


NB_COUCHES=int(args['i'])
CHEMIN_DONNEES= './src/phase1/donnees_traitees' # Dossier contenant les données pré-traitées
NB_EPOCH = int(args['epochs'])


if not os.path.exists(CHEMIN_DONNEES):
    download_data(DATA,CHEMIN_DONNEES)


# Importation des données
x_train, y_train = np.load(CHEMIN_DONNEES+'/train_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/train_labels.npy')
x_test, y_test = np.load(CHEMIN_DONNEES+'/test_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/test_labels.npy')

accuracy_comp=[]
loss_comp=[]
for i in range(NB_COUCHES):
    print(f"\n###################  {i} couches ###############")
    NOM_MODEL='mel-'+str(i)+'cnn'
    # Architecture du modèle
    model = tf.keras.models.Sequential()
    for k in range(i):
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
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

    accuracy_comp.append(acc[1])
    loss_comp.append(acc[0])
    version = open("src/phase1/version_model.txt", "a")
    version.write("\n")
    version.write(str(datetime.datetime.today()))
    version.write("  "+NOM_MODEL+ " entrainé sur "+CHEMIN_DONNEES+"\n")
    version.write("loss : "+str(acc[0])+", accuracy : "+str(acc[1]))
    version.write("epoch :"+ str(NB_EPOCH))
    version.close()

    if not os.path.exists('./graphique_modeles'):
        os.makedirs('./graphique_modeles')
    plt.figure()
    plt.plot(history.history['accuracy'], label="accuracy")
    plt.plot(history.history['val_accuracy'], label="val_accuracy")
    plt.title('Accuracy pour reseau avec '+str(i+1)+' couches convolutives')
    plt.legend()
    plt.savefig('./graphique_modeles/'+NOM_MODEL+'_accuracy.png')
  
    

    if not os.path.exists(CHEMIN_SAUVEGARDE_MODELE):
        os.makedirs(CHEMIN_SAUVEGARDE_MODELE)
    model.save(CHEMIN_SAUVEGARDE_MODELE+"/"+NOM_MODEL)

plt.figure()
plt.plot(accuracy_comp,label="accuracy des differents model")
plt.xlabel('epoch')
plt.savefig('./graphique_modeles/comparaison_accuracy.png')
plt.figure()
plt.plot(loss_comp,label="loss des differents model")
plt.xlabel('epoch')
plt.savefig('./graphique_modeles/comparaison_loss.png')
