from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, TimeDistributed, BatchNormalization, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import os
import random
import d_m as dm
import numpy as np

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
split = 0.8
max_len, tokenizer, X_train, X_test, Y_train, Y_test, _rev_train, _rev_test, _val_train, _val_test = dm.train_data(data_file, split)

unlabeled, _unlabeled = dm.open_doc(file_unlabel, True)

path = settings['path']
model_name = settings['model_name']
model_name = os.path.join(path, model_name)
new_model = settings['new_model']
model = None
if new_model:
    # Create Model
    input('Creating new model may override old model. Press enter to continue')
    model = Sequential()
    model.add(Conv1D(filters = 32, input_shape = X_train.shape[1:], kernel_size = 8, activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
else:
    print('Loading cnn model...', end = '')
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    print('\rLoaded cnn model from disk')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

# Train model
cycles = settings['cycles']
save_epochs = settings['save_epochs']
batch_size = settings['batch_size']
reset_train = settings['reset_train']

cy = cycles // reset_train

for i in range(cy):
    max_len, tokenizer, X_train, X_test, Y_train, Y_test, _rev_train, _rev_test, _val_train, _val_test = dm.train_data(data_file, split)
    for j in range(reset_train):
        print('Cycle', i + 1, ' starting... Saving in ', save_epochs, ' epochs.')
        model.fit(X_train, Y_train, nb_epoch = save_epochs, batch_size = batch_size, verbose = 2, validation_split = 0.1)
        # Save
        print('Saving cnn model...', end = '')
        model_json = model.to_json()
        with open(model_name + '.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(model_name + '.h5')
        print('\rSaved cnn model to disk')

# Print results
print("----------------------------------------   TRAINING SET   ----------------------------------------")

loss, acc = model.evaluate(X_train, Y_train, verbose = 0)


prediction = model.predict(X_train)

char_to_display = 140
allowed_difference = 0.1

total = 0
right = 0
for i in range(len(X_train)):
    predict = prediction[i][0]

    difference = abs(predict - Y_train[i])

    predict_stars = dm.to_stars(predict)
    actual_stars = dm.to_stars(Y_train[i])

    text = str(i + 1) + ') ' + _rev_train[i]

    if difference <= allowed_difference:
        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars)
        right += 1
    else:
        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars + ' *****')
    total += 1

print('Loss:', loss, 'Accuracy:', str(right / total * 100) + '% |', str(right) + '/' + str(total))

print("----------------------------------------     TEST SET     ----------------------------------------")

loss, acc = model.evaluate(X_test, Y_test, verbose = 0)

prediction = model.predict(X_test)

total = 0
right = 0
for i in range(len(X_test)):
    predict = prediction[i][0]

    difference = abs(predict - Y_test[i])

    predict_stars = dm.to_stars(predict)
    actual_stars = dm.to_stars(Y_test[i])

    text = str(i + 1) + ') ' + _rev_test[i]

    if difference <= allowed_difference:
        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars)
        right += 1
    else:
        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars + ' *****')
    total += 1

print('Loss:', loss, 'Accuracy:', str(right / total * 100) + '% |', str(right) + '/' + str(total))

print("----------------------------------------  SAMPLE REVIEWS  ----------------------------------------")

tests = dm.lines_to_input(file_unlabel)

prediction = model.predict(tests)

for i in range(len(unlabeled)):
    predict = prediction[i][0]

    predict_stars = dm.to_stars(predict)

    text = str(i + 1) + ') ' + _unlabeled[i]

    print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars)

print("----------------------------------------  CUSTOM REVIEWS  ----------------------------------------")
while True:
    inp = input('Sample review:\n')
    
    z = dm.data_to_input(inp)

    prediction = model.predict(z)
    predict = prediction[0][0]
    
    predict_stars = dm.to_stars(predict)

    print(dm.truncate_text('', char_to_display) + ' ----- predicted ' + predict_stars)