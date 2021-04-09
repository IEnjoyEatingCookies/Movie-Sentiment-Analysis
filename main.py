import d_m as dm
from keras.models import load_model, model_from_json
import os
import numpy as np
import generator as gen
import naive as nai

# File paths
data_file = os.path.join('data', 'data_labeled.txt')
data_stripped = os.path.join('data', 'data_stripped.txt')
file_unlabeled = os.path.join('data', 'data_unlabeled.txt')
file_unlabel = os.path.join('data', 'unlabeled_reviews.txt')
settings = dm.read_settings('settings.txt')

# Strip data files to text
dm.strip_data(file_unlabeled, file_unlabel)
dm.strip_data(data_file, data_stripped)

# Get Files
max_len, tokenizer, X_data, _, Y_data, _, _rev_data, _, _val_data, _ = dm.train_data(data_file, 1)
unlabeled, _unlabeled = dm.open_doc(file_unlabel, True)

path = settings['path']
model_name = settings['model_name']
model_name = os.path.join(path, model_name)
model = None
def model_load():
    global model
    print('Loading model...', end = '')
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name + '.h5')
    print('\rLoaded model from disk')

char_to_display = settings['char_to_display']
allowed_difference = 0.1
def generate_set(X_data, _rev_data):
    print("----------------------------------------     Data Set     ----------------------------------------")
    print('Model: ' + model_name)
    z = []
    for data in X_data:
        z.append(dm.data_to_input(data)[0])
    X_data = np.array(z)

    prediction = model.predict(X_data)

    for i in range(len(_rev_data)):
        predict = prediction[i][0]

        predict_stars = dm.to_stars(predict)

        text = str(i + 1) + ') ' + _rev_data[i]

        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars)

def generate_set_vals(X_data, Y_data, _rev_data, pr = True):
    print("----------------------------------------     Data Set     ----------------------------------------")
    print('Model: ' + model_name)
    prediction = model.predict(X_data)

    total = 0
    right = 0
    for i in range(len(X_data)):
        predict = prediction[i][0]

        difference = abs(predict - Y_data[i])

        predict_stars = dm.to_stars(predict)
        actual_stars = dm.to_stars(Y_data[i])

        text = str(i + 1) + ') ' + _rev_data[i]

        if difference <= allowed_difference:
            if pr:
                print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars)
            right += 1
        else:
            if pr:
                print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars + ' *****')
        total += 1

    print('Accuracy:', str(right / total * 100) + '% |', str(right) + '/' + str(total))

stop_words = ['--e', '--n', '--q']
def generate(forever = False):
    inp = ''
    print('Model: ' + model_name)
    while forever or not any(inp == w for w in stop_words):
        inp = input('Sample review:\n')
    
        z = dm.data_to_input(inp)

        prediction = model.predict(z)
        predict = prediction[0][0]
        
        predict_stars = dm.to_stars(predict)

        print(dm.truncate_text('', char_to_display) + ' ----- predicted ' + predict_stars)

def generate_set_naive(X_data):
    print("----------------------------------------     Data Set     ----------------------------------------")
    print('Model: ' + model_name)
    prediction = naive_predict_data(X_data)

    for i in range(len(X_data)):
        predict = prediction[i]

        predict_stars = dm.to_stars(predict)

        text = str(i + 1) + ') ' + X_data[i]

        print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars)

def generate_set_vals_naive(X_data, Y_data, pr = True):
    print("----------------------------------------     Data Set     ----------------------------------------")
    print('Model: ' + model_name)
    prediction = naive_predict_data(X_data)

    total = 0
    right = 0
    for i in range(len(X_data)):
        predict = prediction[i]

        difference = abs(predict - Y_data[i])

        predict_stars = dm.to_stars(predict)
        actual_stars = dm.to_stars(Y_data[i])

        text = str(i + 1) + ') ' + X_data[i]

        if difference <= allowed_difference:
            if pr:
                print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars)
            right += 1
        else:
            if pr:
                print(dm.truncate_text(text, char_to_display) + ' ----- predicted ' + predict_stars + '----- actually ' + actual_stars + ' *****')
        total += 1

    print('Accuracy:', str(right / total * 100) + '% |', str(right) + '/' + str(total))

stop_words = ['--e', '--n', '--q']
def generate_naive(forever = False):
    inp = ''
    print('Model: ' + model_name)
    while forever or not any(inp == w for w in stop_words):
        inp = input('Sample review:\n')

        prediction = naive_predict(inp)
        print(dm.truncate_text('', char_to_display) + ' ----- predicted ' + prediction)

def predict_data(data):
    z = []
    for d in data:
        z.append(dm.data_to_input(d)[0])
    z = np.array(z)

    prediction = zip(z, model.predict(data))

    return prediction

def predict(txt):
    inp = dm.data_to_input(txt)

    prediction = model.predict(inp)
    predict = prediction[0][0]
    predict_stars = dm.to_stars(predict)

    return predict_stars

def naive_predict(txt):
    return nai.to_stars(nai.get_val(txt))

def naive_predict_data(data):
    prediction = []

    for val in data:
        prediction.append(nai.get_val(val))
    return prediction

# NN models
models_to_test = ['model_cnn', 'model_lstm', 'model_bidir']
for model_nam in models_to_test:
    model_name = os.path.join(path, model_nam)
    model_load()

    #Generate Vals
    #generate_set_vals(X_data, Y_data, _rev_data, False)
    # generate_set(unlabeled, _unlabeled)
    generate(False)

# Naive model
model_name = 'Naive'
#generate_set_vals_naive(_rev_data, Y_data, False)
# generate_set_naive(_unlabeled)
generate_naive(False)

generate = True
# Generate Text
if generate:
    while True:
        input('\nPress Enter to Generate Review:\n')
        txt = gen.generate_text(4, 20)
        stars = predict(txt)
        print(stars)
        print(txt)