import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor():
    ### Split data into train/valid/test arrays
    def __init__(self, filePath, split, testLength, offset, cols):
        dataframe = pd.read_csv(filePath)
        self.name = filePath.split('/')[1]
        i_split = int((len(dataframe)-offset-testLength) * split) + offset
        self.time_frame = dataframe.get('Date').values[offset:]
        self.data_train = dataframe.get(cols).values[offset:i_split]
        self.data_valid = dataframe.get(cols).values[i_split:-testLength]
        self.data_test = dataframe.get(cols).values[-testLength:]
        self.len_train = len(self.data_train)
        self.len_valid = len(self.data_valid)
        self.len_test = testLength
        
    ### Plot data for visual representation
    def plot_train_valid(self):
        plt.figure()
        plt.plot(self.time_frame[:self.len_train], self.data_train[:,0], color="blue")
        plt.plot(self.time_frame[self.len_train:-self.len_test], self.data_valid[:,0], color="green")
        plt.title(self.name + ' Close Price')
        plt.xticks(self.time_frame[:-self.len_test:100], self.time_frame[:-self.len_test:100], rotation=45)
        plt.show()

        plt.figure()
        plt.plot(self.time_frame[:self.len_train], self.data_train[:,1], color="blue")
        plt.plot(self.time_frame[self.len_train:-self.len_test], self.data_valid[:,1], color="green")
        plt.title(self.name + ' Trading Volume')
        plt.xticks(self.time_frame[:-self.len_test:100], self.time_frame[:-self.len_test:100], rotation=45)
        plt.show()

    ### Get windowed test data
    def get_test_data(self, window_size):
        data_windows = []
        for i in range(self.len_test - window_size):
            data_windows.append(self.data_test[i:i+window_size])
        data_windows = self.normalise_windows(np.array(data_windows))

        return data_windows[:, :-1], data_windows[:, -1, [0]]

    ### Get windowed validation data
    def get_valid_data(self, window_size):
        data_windows = []
        for i in range(self.len_valid - window_size):
            data_windows.append(self.data_valid[i:i+window_size])
        data_windows = self.normalise_windows(np.array(data_windows))

        return data_windows[:, :-1], data_windows[:, -1, [0]]

    ### Get batched and windowed training data for keras LSTM model
    def get_train_data_LSTM(self, window_size, batch_size):
        i = 0
        while i < (self.len_train - window_size):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - window_size):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                window = self.normalise_windows([self.data_train[i:i+window_size]])[0]
                x = window[:-1]
                y = window[-1, [0]]
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    ### Normalize data as an_i = a_i/a_1 - 1
    def normalise_windows(self, window_data):
        normalised_data = []
        for window in window_data:
            normalised_window = []
            for col in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col])) - 1) for p in window[:, col]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    ### Inverse operation of normalization
    def denormalise_data(self, x_true, y_data):
        denormalised_data = []
        for i in range(len(x_true)):
            denormalised_window = np.array([(float(y+1.) * float(x_true[i])) for y in y_data[i]]).T
            denormalised_window = np.array(denormalised_window).T
            denormalised_data.append(denormalised_window)
        return np.array(denormalised_data)