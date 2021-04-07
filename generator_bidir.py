import d_m as dm
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, LSTM, Input, Bidirectional, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
import numpy as numpy
import random
import sys
import os
import collections

# Data paths
data_file = os.path.join('data', 'data_labeled.txt')
data_stripped = os.path.join('data', 'data_stripped.txt')
file_unlabeled = os.path.join('data', 'data_unlabeled.txt')
file_unlabel = os.path.join('data', 'unlabeled_reviews.txt')
settings = dm.read_settings('settings.txt')

# Strip data files to text
dm.strip_data(file_unlabeled, file_unlabel)
dm.strip_data(data_file, data_stripped)

# Get training files
seq_length = settings['seq_length']
split = 0.8
max_len, tokenizer, X_train, X_test, Y_train, Y_test = dm.train_gen_data(data_file, seq_length, split)

path = settings['path']
model_name = settings['model_name']
model_name = os.path.join(path, model_name)
new_model = settings['new_model']
model = None

if new_model:
    # Create model
    lstm_out = settings['lstm_out']
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_out, activation = 'relu', dropout = 0.6, return_sequences = True), input_shape = X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(300, activation = 'sigmoid'))
else:
    print('Loading bidir model...', end = '')
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    print('\rLoaded bidir model from disk')

optimizer = Adam(lr = 0.005)
callbacks = [EarlyStopping(patience = 2, monitor = 'val_loss')]
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = [categorical_accuracy])
print(model.summary())

# Train model
cycles = settings['cycles']
save_epochs = settings['save_epochs']
batch_size = settings['batch_size']
reset_train = settings['reset_train']

cy = cycles // reset_train

for i in range(cy):
    max_len, tokenizer, X_train, X_test, Y_train, Y_test = dm.train_gen_data(data_file, seq_length, split)
    for j in range(reset_train):
        print('Cycle', i + 1, ' starting... Saving in ', save_epochs, ' epochs.')
        model.fit(X_train, Y_train, nb_epoch = save_epochs, batch_size = batch_size, verbose = 2, validation_split = 0.1)
        # Save
        print('Saving bidir model...', end = '')
        model_json = model.to_json()
        with open(model_name + '.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(model_name + '.h5')
        print('\rSaved bidir model to disk')