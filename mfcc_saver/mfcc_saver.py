import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
import pickle
import os

# odczytanie nazw z folderu o nazwie path_folder
path_folder = 'train'
filenames = os.listdir('./' + path_folder + '/')
names_ids = [f[:5] for f in filenames]
names = list(set(names_ids))

# wyświetla nazwy plików by sprawdzić czy odpowiednie pliki wybrane do analizy
print(names)

# stworzenie słownika mfcc_dict
mfcc_dict = {}
for i in range(0, len(names)):
    tmp_dict = {}
    for j in range(0, 10):
        fs, data = wavfile.read('./train/' + names[i] + '_' + str(j) + '_.wav')
        mfcc_from_file = mfcc(data, samplerate=16000,
                winlen=0.025,
                winstep=0.01,
                numcep=13,nfilt=26,
                nfft=512,lowfreq=0,
                highfreq=None,
                preemph=0.97,
                ceplifter=22,
                appendEnergy=True,
                winfunc=np.hamming)
        tmp_dict[j] = mfcc_from_file
    mfcc_dict[i] = tmp_dict

# zapisanie pliku pickle
with open('mfcc_dict.pickle', 'wb') as handle:
    pickle.dump(mfcc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

