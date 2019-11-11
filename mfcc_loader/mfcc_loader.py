import pickle
import matplotlib.pyplot as plt

# wczytanie mfcc_dict z pliku pickle

with open('mfcc_dict.pickle', 'rb') as handle:
    mfcc_dict = pickle.load(handle)



# indexy dla 'id_mówcy'
print(mfcc_dict.keys())

# indexy dla 'wypowiedzianej_liczby'
print(mfcc_dict[0].keys())

# przyklad wczytania mfcc dla id_mówcy = 2 i wypowiedzianej_liczby = 3
print(mfcc_dict[2][3])

