import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
import pickle
import os

# odczytanie nazw z folderu o nazwie path_folder
path_folder = 'eval'
filenames = os.listdir('./' + path_folder + '/')
names = [f[:3] for f in filenames]


# wyświetla nazwy plików by sprawdzić czy odpowiednie pliki wybrane do analizy
print(names)

# stworzenie słownika mfcc_dict
mfcc_dict = {}
for i in range(0, len(names)):
    fs, data = wavfile.read('./' + path_folder + '/' + names[i] + '.wav')
    mfcc_from_file = mfcc(data, samplerate=16000,
                          winlen=0.025,
                          winstep=0.01,
                          numcep=13, nfilt=26,
                          nfft=512, lowfreq=0,
                          highfreq=None,
                          preemph=0,
                          ceplifter=22,
                          appendEnergy=True,
                          winfunc=np.hamming)

    mfcc_dict[i] = mfcc_from_file

# zapisanie pliku pickle
with open('mfcc_dict_eval.pickle', 'wb') as handle:
    pickle.dump(mfcc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

