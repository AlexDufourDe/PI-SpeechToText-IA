import numpy as np 
from time import time
import deeplake
from datasets import load_dataset
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import os

login("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")
""" Ce fichier fait le prétraitement des données pour la phase 2 c'est à dire des phrases completes"""
# data_chemin="mozilla-foundation/common_voice_9_0
# description des données :https://huggingface.co/datasets/mozilla-foundation/common_voice_7_0

def data_download(data_chemin,save_chemin):

    """Cette fonction  pretraite les donnée de la bassée de donnée mozilla common voice pour les mettre à la bonne longueur"""

    # Parametres
    chemin_sauvegarde =  save_chemin
    # durree_enregistrement=
    ds = load_dataset(data_chemin, "fr", use_auth_token=True)
    n=len(ds)
    print(f"longueur du dataset: {n}")
    transcription = ds["sentence"]
    audio=ds[0]["audio"]

    X = []
    Y = []


    debut = time()
    max_size=0
    for i in range(n):
        wav=audio[i].numpy()
        if wav.shape[0]>max_size:
            max_size=wav.shape[0]

    for i in range(n):
        wav=audio[i].numpy()
        wav = wav.reshape(max_size)

        #pré-traitement
        fade = tfio.audio.fade(wav, fade_in=1000, fade_out=2000, mode="logarithmic")
        spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
        mel_spectrogram = tfio.audio.melscale(spectrogram, rate=ds['audio']['sampling_rate'], mels=128, fmin=0, fmax=8000)
        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
        X.append(dbscale_mel_spectrogram)
        Y.append(transcription[i])

        
    print(str(len(Y)) + ' elements on étés importés en ' + str(time()-debut))


    print("Découpage en données d'entraitement et test, et sauvegarde")
    # Les données sont mélangées et découpées en une partie pour l'entrainement et une seconde pour le test.
    train_data, test_data, train_labels, test_labels = train_test_split(np.array(X), np.array(Y), test_size=0.3, random_state=83)

    # Les tableaux numpy sont sauvegardés pour une utilisation rapide.
    
    if not os.path.exists(chemin_sauvegarde):
        os.makedirs(chemin_sauvegarde)

    np.save(chemin_sauvegarde+'/train_data.npy', train_data)
    np.save(chemin_sauvegarde+'/train_labels.npy', train_labels)
    np.save(chemin_sauvegarde+'/test_data.npy', test_data)
    np.save(chemin_sauvegarde+'/test_labels.npy', test_labels)
    print("Données enregistrées dans le dossier " + chemin_sauvegarde)


data_download("mozilla-foundation/common_voice_7_0","mozilla_common_voice_pretraitee")