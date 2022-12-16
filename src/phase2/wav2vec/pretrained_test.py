"""
Dans ce fichier, nous faisons appel à la bibliothèque transformers (de HuggingFace) afin d'essayer un modèle wav2vec2
francais avec un enregistrement vocal.
"""
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from time import sleep

# On importe le modèle
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")

# On prépare l'enregistrement, comme pour la phase1. Nous choisissons le temps que nous voulons pour parler (ici 10s)
fs = 16000  # Fréquence d'echantillonage (en Hz)
secondes = 10  # Durée de l'enregistrement


print("Enregistrement dans 3")
sleep(1)
print("2")
sleep(1)
print("1")
sleep(1)
print('go!')
enregistrement = sd.rec(int(secondes * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print('fin')

audio = enregistrement
print("l'enregistrement est sauvegardé dans un fichier 'output.wav' pour vérification")
write('output.wav', fs, enregistrement)

# Préparation de l'enregistrement à passer à l'entrée du modèle. Nous remarquons qu'il n'y a pas de calcul e spectrogramme
input_values = processor(audio.reshape(160000), return_tensors="pt", padding="longest").input_values

# Appel du modèle
logits = model(input_values).logits

# Decodage du resultat abec le processor (tokenizer ?)
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

# Affichage de la transcription
print(transcription)


