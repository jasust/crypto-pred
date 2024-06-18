import os
import math
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

class LSTMModel():
	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('Loading LSTM model from a file')
		self.model = load_model(filepath)

	def build_model(self, in_dim):
		self.model.add(LSTM(100, input_shape=(49, in_dim), return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(100, return_sequences=True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(100, return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1, activation='linear'))

		self.model.compile(loss='mse', optimizer='adam')

		print('Model Compiled')

	def train(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		print('Training Started')
		print('Specs: %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, 'e%s.h5' % (str(epochs)))
		callbacks = [ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)]

		self.model.fit(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('Training Completed.')

    ### Predict one day ahead
	def predict_point_by_point(self, data):
		print('Predicting 1 day ahead...')
		predicted = self.model.predict(data, verbose=0)
		# predicted = np.reshape(predicted, (predicted.size,))
		return predicted

    ### Predict prediction_len days ahead
	def predict_sequences_multiple(self, data, window_size, prediction_len):
		print('Predicting %s days ahead...' % (prediction_len))
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[np.newaxis,:,:], verbose=0)[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs