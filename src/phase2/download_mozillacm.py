import numpy as np 
import pandas as pd
from time import time
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
import os
import json
import sys
import wget
from scipy.io import wavfile
import csv

""" Ce fichier fait le prétraitement des données pour la phase 2 c'est à dire des phrases completes"""
# data_chemin="mozilla-foundation/common_voice_9_0
# description des données :https://huggingface.co/datasets/mozilla-foundation/common_voice_7_0

def mozillacm_download(data_chemin,save_chemin,pretraitement=False):

    """Cette fonction  pretraite les donnée de la bassée de donnée mozilla common voice pour les mettre à la bonne longueur"""

    # Parametres
    chemin_sauvegarde =  save_chemin
    if not os.path.exists(chemin_sauvegarde):
        os.makedirs(chemin_sauvegarde)

    
    
    

    # Lecture json
    fileObject = open(data_chemin, "r")
    jsonContent = fileObject.read()
    data_json = json.loads(jsonContent)

    #transcript
    n=len(data_json['rows'])
    X = []
    Y = []

    debut = time()
    sampling_rate=data_json['features'][2]['type']['sampling_rate']

    meta_data=open(chemin_sauvegarde+'/metadata.csv','w')
    writer=csv.writer(meta_data,delimiter='|')
    for i in range(n):
    
        url=data_json['rows'][i]['row']['audio'][1]['src']
        response = wget.download(url,chemin_sauvegarde+"/audio_"+str(i)+".wav")
        samplerate, audio= wavfile.read(chemin_sauvegarde+"/audio_"+str(i)+".wav")

        if pretraitement:
            Y.append(data_json['rows'][i]['row']['sentence'])
            #pré-traitement
            fade = tfio.audio.fade(audio, fade_in=1000, fade_out=2000, mode="logarithmic")
            spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
            mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sampling_rate, mels=128, fmin=0, fmax=8000)
            dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
            X.append(dbscale_mel_spectrogram)

        row=["audio_"+str(i),data_json['rows'][i]['row']['sentence'],data_json['rows'][i]['row']['sentence']]
        writer.writerow(row)

    print(str(n) + ' elements on étés importés en ' + str(time()-debut))

   
    # Les données sont mélangées et découpées en une partie pour l'entrainement et une seconde pour le test.

    if pretraitement:
        print("Découpage en données d'entraitement et test, et sauvegarde")
        train_data, test_data, train_labels, test_labels = train_test_split(np.array(X), np.array(Y), test_size=0.3, random_state=83)

        if not os.path.exists(chemin_sauvegarde+'_pretraitee'):
            os.makedirs(chemin_sauvegarde+'_pretraitee')

        ###### Save as .npy #####

        np.save(chemin_sauvegarde+'_pretraitee/train_data.npy', train_data)
        np.save(chemin_sauvegarde+'_pretraitee/train_labels.npy', train_labels)
        np.save(chemin_sauvegarde+'_pretraitee/test_data.npy', test_data)
        np.save(chemin_sauvegarde+'_pretraitee/test_labels.npy', test_labels)
        print("Données enregistrées dans le dossier " + chemin_sauvegarde)


        ###### Save as csv #####
        
        train_df=pd.DataFrame()
        train_df['spectogramme']=train_data
        train_df['label']=train_labels
        train_df.to_csv(chemin_sauvegarde+'_pretraitee/train.csv')
        test_df=pd.DataFrame(test_data,test_labels)
        test_df['spectogramme']=test_data
        test_df['label']=test_labels
        test_df.to_csv(chemin_sauvegarde+'_pretraitee/test.csv')



if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)
    if (n>4):
        print(f"Too many argument, expected :3 , got {n-1}")
    elif n==4:
        mozillacm_download(sys.argv[1],sys.argv[2],sys.argv[3])
    elif n==3:
        mozillacm_download(sys.argv[1],sys.argv[2])
    elif n==2:
        mozillacm_download(sys.argv[1],"mozilla_common_voice")
    else:   
        mozillacm_download("src/phase2/mozilla_commonvoice.json","src/phase2/mozilla_common_voice")



