import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

### Constants
filePath = 'data/Bitcoin/BTC-USD.csv'
save_dir = 'results'
offset = 700
trainTestSplit = 0.8
window_size = 10
n_estimators = 1000
prediction_length = 10
input_col = ['Close']

### Read data
dataframe = pd.read_csv(filePath)
data = dataframe.get(input_col).values

### Normalizing data
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

### Split into train and test
i_split = int((len(dataframe)-offset) * trainTestSplit) + offset
data_train = data[offset:i_split]
data_test = data[i_split:]

### Convert into windowed data
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-window_size-1):
        a = dataset[i:(i+window_size), 0]
        dataX.append(a)
        dataY.append(dataset[i + window_size, 0])
    return np.array(dataX), np.array(dataY)

X_train, y_train = create_dataset(data_train)
X_test, y_test = create_dataset(data_test)

### Train XGB model
xgb_model = XGBRegressor(n_estimators=n_estimators)
xgb_model.fit(X_train, y_train)

### Plot results
predictions = xgb_model.predict(X_test)
print("RMSE = " + str(math.sqrt(mean_squared_error(y_test, predictions))))
test_predict = scaler.inverse_transform(predictions.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure()
plt.title('Predviđanje XGBoost modela 1 dan unapred')
plt.plot(original_ytest, label='Prava cena')
plt.plot(test_predict, label='Predviđena cena')
plt.legend()
plt.show()

print("RMSE = " + str(math.sqrt(mean_squared_error(original_ytest, test_predict))))

### Predict next 10 days
prediction_seqs = []
for i in range(int((len(data)-window_size-1)/prediction_length)):
    curr_frame = data_test[i*prediction_length:i*prediction_length+window_size]
    if (curr_frame.shape[0]<window_size): 
        break
    predicted = []
    for j in range(prediction_length):
        predicted.append(xgb_model.predict(curr_frame.reshape(1,-1))[0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    prediction_seqs.append(predicted)

plt.figure()
plt.title('Predviđanje XGBoost modela %s dana unapred' % (prediction_length))
plt.plot(original_ytest, label='Prava cena')
scaled_predict = np.zeros((len(prediction_seqs)*prediction_length,1))
predictions = np.zeros((len(prediction_seqs)*prediction_length,1))
for i, seq in enumerate(prediction_seqs):
    scaled_predict[i*prediction_length:(i+1)*prediction_length] = scaler.inverse_transform(np.array(seq).reshape(-1,1))
    predictions[i*prediction_length:(i+1)*prediction_length] = np.array(seq).reshape(-1,1)
    plt.plot(np.arange(i*prediction_length,(i+1)*prediction_length), scaled_predict[i*prediction_length:(i+1)*prediction_length])
plt.legend()
plt.show()

print("RMSE = " + str(math.sqrt(mean_squared_error(y_test, predictions[:len(original_ytest)]))))
print("RMSE = " + str(math.sqrt(mean_squared_error(original_ytest, scaled_predict[:len(original_ytest)]))))
