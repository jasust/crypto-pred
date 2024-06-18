import math
import numpy as np
import matplotlib.pyplot as plt
from data_processing import DataProcessor
from lstm_model import LSTMModel

### Constants
filePath = 'data/Ethereum/ETH-USD.csv'
save_dir = 'results'
offset = 800
trainValidSplit = 0.85
testLength = 86
window_size = 50
batch_size = 32
epochs = 12
prediction_length = 7
input_col = ['Close', 'Volume']

### Helper functions
def plot_results(predicted_data, true_data):
    plt.figure()
    plt.title('Predviđanje LSTM modela 1 dan unapred')
    plt.plot(true_data, label='Prava cena')
    plt.plot(predicted_data, label='Predviđena cena')
    plt.legend()
    plt.show()

    print('RMSE = ', rmse(predicted_data, true_data))

def plot_results_multiple(predicted_data, true_data, prediction_len):
    plt.figure()
    plt.title('Predviđanje LSTM modela %s dana unapred' % (prediction_len))
    plt.plot(true_data, label='Prava cena')
    for i in range(predicted_data.shape[0]):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + np.ndarray.tolist(predicted_data[i]))
    plt.legend()
    plt.show()

    predicted_array = predicted_data.flatten()
    print('RMSE = ', rmse(predicted_array, true_data))

def rmse(predicted, true_data):
    rms = 0
    num_data = min(len(predicted), len(true_data))
    for i in range(num_data):
        rms += (predicted[i]-true_data[i])**2
    return math.sqrt(rms/num_data)


if __name__ == '__main__':
    ### Load and plot data
    data = DataProcessor(filePath, trainValidSplit, testLength, offset, input_col)
    data.plot_train_valid()

    ### Buil LSTM model
    model = LSTMModel()
    model.build_model(len(input_col))

    ### Train LSTM model
    steps_per_epoch = math.ceil((data.len_train - window_size) / batch_size)
    model.train(
        data_gen=data.get_train_data_LSTM(
            window_size=window_size,
            batch_size=batch_size
        ),
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        save_dir=save_dir
    )

    ### Validation
    x_valid, y_valid = data.get_valid_data(window_size)
    yd_valid = data.data_valid[window_size-window_size//10:-window_size//10, 0]

    if (prediction_length == 1):
        predictions = model.predict_point_by_point(x_valid)
        predictions_denorm = data.denormalise_data(data.data_valid[:-window_size-1, 0], predictions)
        plot_results(predictions_denorm, yd_valid)
    else:
        predictions = model.predict_sequences_multiple(x_valid, window_size, prediction_length)
        predictions_denorm = data.denormalise_data(data.data_valid[:-window_size-prediction_length:prediction_length, 0], predictions)
        plot_results_multiple(predictions_denorm, yd_valid, prediction_length)

    ### Testing
    if (prediction_length == 7):
        x_test, y_test = data.get_test_data(window_size)
        yd_test = data.data_test[window_size-window_size//10:-window_size//10, 0]
        predictions = model.predict_sequences_multiple(x_test, window_size, prediction_length)
        predictions_denorm = data.denormalise_data(data.data_test[:-window_size-prediction_length:prediction_length, 0], predictions)
        plot_results_multiple(predictions_denorm, yd_test, prediction_length)

    mika = np.random.rand(10)
    print(mika < 0.6)