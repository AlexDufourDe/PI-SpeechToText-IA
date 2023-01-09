"""
Ce fichier permet l'importation et le pré-traitement de données audio de mots simples.
Le pré-traitement par défaut est la transformation en spectogramme mel.
Les données pré-traitées sont sauvegardées dans deux fichiers, un pour l'entrainement et l'autre pour le test.


This file download and pre-process the data. It is audio of simple word.
The default preprocessing is mel spectogram.
The pre- precessed data are saved in two file, one foe the training of the model and the other for the testing. 
"""

import deeplake
import numpy as np
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
from time import time
import os
import sys

#url="hub://activeloop/speech-commands-train"

def download_data(url,chemin):

    # Ajouter les mots souhaités, et l'index correspondant. Les mots disponibles sont donnés en bas de page
    MOTS = ['Sheila','Zero','Go','Bed','Bird','Stop','Marvin','Yes','Four','House','Off','Tree','Wow','Happy','Nine','Up','Three','Right','Five','Two','One','Left','Eight','Six','Down','Dog','No','Cat','On','Seven']
    index_mots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30]

    CHEMIN_SAUVEGARDE =chemin    #Choisir le dossier de sauvegarde des données
    
    
    data = deeplake.load(url)
    X = []
    Y = []

    print("Importation des données. Cela peut prendre plus d'une heure")
    debut = time()
    for i in range(64726):
        if (i %100 ==0):
            print(i)
        index = data.labels[i].numpy()[0]
        if index in index_mots :
            wav = data.audios[i].numpy()
            if wav.shape != (16000, 1):
                continue
            wav = wav.reshape(16000)
            #Début pré-traitement : changer les lignes suivantes pour essayer d'autres prétraitements
            fade = tfio.audio.fade(wav, fade_in=1000, fade_out=2000, mode="logarithmic")
            spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
            mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
            dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
            #Fin pré-traitement

            X.append(dbscale_mel_spectrogram)
            Y.append(index_mots.index(index))
    print(str(len(Y)) + ' elements on étés importés en ' + str(time()-debut))


    print("Découpage en données d'entraitement et test, et sauvegarde")
    # Les données sont mélangées et découpées en une partie pour l'entrainement et une seconde pour le test.
    train_data, test_data, train_labels, test_labels = train_test_split(np.array(X), np.array(Y), test_size=0.3, random_state=83)

    # Les tableaux numpy sont sauvegardés pour une utilisation rapide.
    
    if not os.path.exists(CHEMIN_SAUVEGARDE):
        os.makedirs(CHEMIN_SAUVEGARDE)
    np.save(CHEMIN_SAUVEGARDE+'/train_data.npy', train_data)
    np.save(CHEMIN_SAUVEGARDE+'/train_labels.npy', train_labels)
    np.save(CHEMIN_SAUVEGARDE+'/test_data.npy', test_data)
    np.save(CHEMIN_SAUVEGARDE+'/test_labels.npy', test_labels)
    print("Données enregistrées dans le dossier " + CHEMIN_SAUVEGARDE)

"""
Dictionnaire des mots
index 0: Sheila ---Debut 0 fin 1733
index 1: Zero ---Debut 1734 fin 4109
index 2: Go ---Debut 4110 fin 6481
index 3: Bed ---Debut 6482 fin 8194
index 4: Bird ---Debut 8195 fin 9925
index 5: Stop ---Debut 9926 fin 12305
index 6: Marvin ---Debut 12306 fin 14051
index 7: Yes ---Debut 14052 fin 16428
index 8: Four ---Debut 16429 fin 18800
index 9: House ---Debut 18801 fin 20550
index 10: Off ---Debut 20551 fin 22907
index 11: Tree Debut 22908 fin 24640
index 12: Wow ---Debut 24641 fin 26385
index 13: Happy ---Debut 26386 fin 28127
index 14: Nine ---Debut 28128 fin 30491
index 15: Up ---Debut 30492 fin 32866
index 16: Three ---Debut 32867 fin 35222
index 17: !!!Bug ---Debut 35223 fin 35228
index 18: Right ---Debut 35229 fin 37595
index 19: Five ---Debut 37596 fin 39952
index 20: Two ---Debut 39953 fin 42325
index 21: One ---Debut 42326 fin 44695
index 22: Left ---Debut 44696 fin 47048
index 23: Eight ---Debut 47049 fin 49400
index 24: Six ---Debut 49401 fin 51769
index 25: Down ---Debut 51770 fin 54128
index 26: Dog ---Debut 54129 fin 55874
index 27: No ---Debut 55875 fin 58249
index 28: Cat ---Debut 58250 fin 59982
index 29: On ---Debut 59983 fin 62349
index 30: Seven ---Debut 62350 fin 64726
"""

if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)
    if (n>2):
        print(f"Too many argument, expected :1 , got {n}")
    elif n==2:
        download_data("hub://activeloop/speech-commands-train",sys.argv[1])
    else:   
        download_data("hub://activeloop/speech-commands-train",'./donnees_traitees_extra')







