import wave
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np



#Path to file
f_name = "../../audio/mal/yes_e_down.wav"
samplerate, data = wavfile.read(f_name)


fps = 16000  # Fréquence d'echantillonage (en Hz)
duration = 3  # Durée de l'enregistrement


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


fc=1000
filtered_sg = butter_highpass_filter(data, fc, fps)
plt.figure(figsize = (10, 5))
plt.subplot(211)
plt.plot(range(len(data)),data)
plt.title("Signal")
plt.subplot(212)
plt.plot(range(len(filtered_sg)), filtered_sg)
plt.title("Filtered Signal")
#plt.show()



nom=f_name.split('/')[-1]
nom_f=nom.split('.')[0]
write("../../audio/filtre/"+nom+"_filtre.wav", fps, filtered_sg.astype(np.int16))
