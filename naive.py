
from nltk.stem.porter import PorterStemmer
import re
import os

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Data paths
data_file = os.path.join('data', 'data_labeled.txt')
data_stripped = os.path.join('data', 'data_stripped.txt')
file_unlabeled = os.path.join('data', 'data_unlabeled.txt')
file_unlabel = os.path.join('data', 'unlabeled_reviews.txt')

# Stemming
ps = PorterStemmer()


word_vals = {
    'like': 1,
    'amaz': 3,
    'good': 1,
    'great': 2.5,
    'enjoy': 3,
    'spectacular': 4,
    'spectacl': 4,
    'excelsior': 4,
    'better': 2,
    'satisfact': 3.5,
    'satisfi': 3.5,
    'satisfactori': 3.5,
    'entertain': 3,
    'love': 3,
    'fit': 1,
    'happi': 3,
    'cri': 1.5,
    'emot': 2.5,
    'masterpiec': 4,
    'master': 4,
    'celebr': 3.5,
    'reward': 1,
    'inspir': 2,
    'right': 5,
    'best': 5,
    'blow': 2,
    'blew': 2,
    'perfect': 5,
    'winner': 5,
    'win': 5,
    'won': 5,
    'joy': 3,
    'succeed': 4.5,
    'miracl': 4.5,
    'miracul': 4.5,
    'real': 1,
    'beauti': 4,
    'epic': 4,
    'work': 1,
    'power': 1.5,
    'worth': 3,
    'interest': 3,
    'captiv': 2.5,
    'motiv': 2.5,
    'funni': 1,
    'hilari': 1,
    'top': 2,
    'tremend': 1,
    'must': 3,
    'laughter': 3.5,
    'laugh': 3.5,

    'sad': -2,
    'bad': -2,
    'terribl': -4,
    'gross': -4,
    'trash': -4,
    'garbag': -4,
    'hate': -3,
    'worst': -5,
    'wors': -5,
    'forget': -1.5,
    'forgot': -1.5,
    'fail': -5,
    'ridicul': -3.5,
    'long': -1,
    'dismiss': -3,
    'nowhere': -4,
    'risk': -2,
    'sacrific': -4,
    'slow': -1,
    'bore': -4,
    'lame': -2,
    'wast': -4,
    'piss': -2,
    'angri': -4,
    'left': -1
}

word_muls = {
    'veri': 1.3,
    'really': 1.6,
    'super': 2,
    'much': 1.5,
    'such': 1.5,
    'too': 2,
    'absolut': 4,
    'especially': 3,
    'well': 1.2,
    'all': 1.2,

    'never': -0.8,
    'not': -0.2,
    'didn\'t': -0.2,
    'couldn\'t': -0.2
}

def to_stars(percent):
    val = round(percent * 10.0) / 2.0
    if val <= 0:
        val = 0.5
    if val > 5:
        val = 5
    return str(val) + '/5.0 stars'

def clean_txt(txt):
    # Lowercase
    txt = txt.lower()

    # Remove Punctuation
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    txt = re.sub('[' + punc + ']', '', txt)

    tokens = txt.split()

    # Remove non alpha words
    tokens = [w for w in tokens if w.isalpha()]

    # Remove one letter words
    tokens = [w for w in tokens if len(w) > 1]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return tokens

def get_val(txt):
    tokens = clean_txt(txt)

    value = 0
    tokens.reverse()
    for word in tokens:
        if word in word_vals:
            value += word_vals[word]
        elif word in word_muls:
            value *= word_muls[word]

    mapped = sigmoid(value)
    return mapped


# f = open(data_stripped, mode = 'r', encoding = 'utf8')

# lines = f.read().splitlines()

# for line in lines:
#     print(line)
#     print(to_stars(get_val(line)))

# while True:
#     inp = input("text:\n")
#     print(ps.stem(inp))