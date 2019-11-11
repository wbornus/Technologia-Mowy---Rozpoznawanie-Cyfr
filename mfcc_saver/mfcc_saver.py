import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile
import pickle


names = ['AO1M1', 'BC1M1', 'DG1M1', 'JC1M1', 'JK1M1', 'JO1M1', 'JP1M1', 'JP2M1', 'JS1M1', 'KD1M1', 'KD2M1', 'MR1M1', 'MS1M1',
         'PL1M1', 'PS1M1', 'PW1M1', 'RG1M1', 'SG1M1', 'SG2M1', 'SG3M2', 'SP1M1', 'SW1M1']

tmp = []
filenames = [[tmp for i in range(0, len(names))] for j in range(0, 10)]

print(len(names))
print(len(filenames[0]))

tmp_dict = {}
mfcc_dict = {}

for i in range(0, len(names)):
    tmp_dict.clear()
    for j in range(0, 10):
        filenames[j][i] = names[i] + '_' + str(j) + '_.wav'
        #print(filenames[j][i])
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
        mfcc_from_file = mfcc_from_file.tolist()
        tmp_dict[j] = mfcc_from_file
    mfcc_dict[i] = tmp_dict

#  tu sobie sprawdzałem czy działa
# print(mfcc_dict[2][3])
# plt.pcolormesh(mfcc_dict[2][3])
# plt.show()
#

with open('mfcc_dict.pickle', 'wb') as handle:
    pickle.dump(mfcc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
