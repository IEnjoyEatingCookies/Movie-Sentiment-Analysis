import d_m as dm
from keras.models import load_model, model_from_json
import generator as gen
import os
import pickle
import random

settings = dm.read_settings('settings.txt')

seq_length = settings['seq_length']
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
model_load()

def init(start_save = 'start_tokens.txt'):
    path = 'Tokens'

    start_save = os.path.join(path, start_save)
    start_tokens = []
    if os.path.exists(start_save):
        print('Loading files...', end = '')
        with open(start_save, mode = 'rb') as fp:
                start_tokens = pickle.load(fp)
        print('\rFiles loaded.')
    else:
        print('Failed to initialize...')
        assert 'Answer to the Ultimate Question of Life, the Universe, and Everything' == 42
    
    return start_tokens

s = init()

def generate(pmin, pmax, temp = 1):
    sentence = random.choice(s).split(' ')
    
    s_len = random.randint(pmin, pmax)

    print(s_len)
    for i in range(s_len):
        seed = ' '.join(sentence[i: i + seq_length])
        print('Seed: ' + seed)
        val = dm.data_to_input(seed, 2)
        print(val)
        vec = model.predict(val)[0]
        print(vec)
        w = dm.vec2word(vec, temp)
        print('word: ' +  w)
        sentence.append(w)
    
    return ' '.join(sentence)

input('Press enter to generate!\n')
s = generate(4, 10, 2)
print(s)

while True:
    input('Press enter to generate!\n')
s = generate(4, 10, 2)
print(s)