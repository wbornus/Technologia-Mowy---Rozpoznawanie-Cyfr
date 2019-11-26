from scipy.io import wavfile

from utils import predict_digit

directory = './own_test_data/PG/'

predictions = []
correct = 0
for it in range(10):
    digit = str(it)
    fs, data = wavfile.read(directory+digit+'.wav')
    prediction = predict_digit(data, fs, model_type='conv')
    if prediction == it:
        correct += 1
    predictions.append(prediction)

acc = correct / 10
print(predictions)
print('accuracy: ', acc)