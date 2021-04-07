import nltk
import re
import os
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import spacy
import gensim
from gensim.models import KeyedVectors

nlp = spacy.load("en_core_web_sm")
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
print('Loaded Pre-processing Tools')


def read_file(file_name):
    f = open(file_name, mode = 'r', encoding = 'utf8')

    txt = f.read()
    f.close()
    return txt

def clean(txt):
    doc = nlp(txt)
    for ent in doc.ents:
        txt = txt[:ent.start_char] + ent.label_ + txt[ent.end_char:]

    
    # Lowercase
    txt = txt.lower()

    # Remove Punctuation
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    txt = re.sub('[' + punc + ']', '', txt)
    
    tokens = nltk.word_tokenize(txt)
    
    # Remove non alpha words
    tokens = [w for w in tokens if w.isalpha()]

    # Remove one letter words
    tokens = [w for w in tokens if len(w) > 1]

    # Stemming
    # ps = PorterStemmer()
    # tokens = [ps.stem(word) for word in tokens]

    txt = ' '.join(tokens)

    return txt

def clean_tokens(txt):
    doc = nlp(txt)
    for ent in doc.ents:
        txt[ent.start_char:ent.end_char] = ent.label_

    # Lowercase
    txt = txt.lower()

    # Remove Punctuation
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    txt = re.sub('[' + punc + ']', '', txt)
    
    tokens = nltk.word_tokenize(txt)
    
    # Remove non alpha words
    tokens = [w for w in tokens if w.isalpha()]

    # Remove one letter words
    tokens = [w for w in tokens if len(w) > 1]

    # Stemming
    # ps = PorterStemmer()
    # tokens = [ps.stem(word) for word in tokens]

    return tokens

def get_reviews(file_name):
    f = open(file_name, mode = 'r', encoding = 'utf8')

    txts = []
    lines = f.read().splitlines()
    for line in lines:
        line = clean(line)
        txts.append(line)
    
    f.close()
    return txts

def TF_IDF(txts):
    tf_idf = TfidfVectorizer()
    tf_idf.fit(txts)

    x = tf_idf.transform(txts)

    return tf_idf, x

def convert_tfidf(txts):
    tf_idf, x = TF_IDF(txts)

    encoded_txts = []
    for txt in txts:
        encoded = []
        for word in txt.split(' '):
            encoding = float(x[1, tf_idf.vocabulary_[word]])
            encoded.append(encoding)
        encoded_txts.append(encoded)
    
    return encoded_txts

def open_doc(file_name, shuffle = False):
    f = open(file_name, mode = 'r', encoding = 'utf8')

    doc = []
    _doc = []

    lines = f.read().splitlines()
    for line in lines:
        if shuffle:
            index = random.randint(0, len(doc))
            doc.insert(index, clean(line))
            _doc.insert(index, line)
        else:
            doc.append(clean(line))
            _doc.append(line)
    
    return doc, _doc

def process_doc(file_name, shuffle = False):
    f = open(file_name, mode = 'r', encoding = 'utf8')

    doc = []
    _doc = []

    lines = f.read().splitlines()
    for i in range(0, len(lines), 3):
        if float(lines[i + 1]) == 0:
            continue
        review = round(float(lines[i]) / float(lines[i + 1]), 5)

        line = lines[i + 2]
        if shuffle:
            index = random.randint(0, len(doc))
            doc.insert(index, (clean(line), review))
            _doc.insert(index, (line, review))
        else:
            doc.append((clean(line), review))
            _doc.append((line, review))
    
    return doc, _doc

def data_to_input(txt, max_words = 20):
    txt = clean(txt)

    max_len = max_words

    z = []

    s = txt.split()
    s = s[:max_words]

    z.append([])
    padding = max_len - len(s)

    l = 0
    for i, word in enumerate(s):
        if word in model:
            z[0].append([])
            w = model[word].tolist()
            z[0][i - l] = [v for v in w]
        else:
            l += 1
    for f in range(padding + l):
        z[0].append([0] * 300)
    txt = np.asarray(z)

    return txt

def lines_to_input(file_name, max_words = 20):
    data, _data = open_doc(file_name)

    data = [x[0] for x in data]

    max_len = max_words
    ''' USING W2V '''
    z = []
    for i, s in enumerate(data):
        s = s.split()
        s = s[:max_words]

        z.append([])
        padding = max_len - len(s)
        
        l = 0
        for j, word in enumerate(s):
            if word in model:
                z[i].append([])
                w = model[word].tolist()
                z[i][j - l] = [v for v in w]
            else:
                l += 1
        for f in range(padding + l):
            z[i].append([0] * 300)
    data = np.asarray(z)

    return data


def train_data(file_name, split):
    data, reviews = process_doc(file_name, shuffle = True)

    split = int(len(reviews) * split)
    X_train = [x[0] for x in data[:split]]
    X_test = [x[0] for x in data[split:]]
    combined = X_train + X_test

    max_len = max(len(s) for s in combined)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined)
    

    ''' USING TOKENIZER '''
    # vs = len(tokenizer.word_index) + 1
    # X_train = tokenizer.texts_to_sequences(X_train)
    # X_test = tokenizer.texts_to_sequences(X_test)
    # 
    # X_train = pad_sequences(X_train, maxlen = max_len, padding = 'post')
    # X_test = pad_sequences(X_test, maxlen = max_len, padding = 'post')

    max_words = 20
    max_len = max_words
    ''' USING W2V '''
    z = []
    for i, s in enumerate(X_train):
        s = s.split()
        s = s[:max_words]

        z.append([])
        padding = max_len - len(s)

        
        l = 0
        for j, word in enumerate(s):
            if word in model:
                z[i].append([])
                w = model[word].tolist()
                z[i][j - l] = [v for v in w]
            else:
                l += 1
        for f in range(padding + l):
            z[i].append([0] * 300)
    X_train = np.asarray(z)
    z = []
    for i, s in enumerate(X_test):
        s = s.split()
        s = s[:max_words]

        z.append([])
        padding = max_len - len(s)

        
        l = 0
        for j, word in enumerate(s):
            if word in model:
                z[i].append([])
                w = model[word].tolist()
                z[i][j - l] = [v for v in w]
            else:
                l += 1
        for f in range(padding + l):
            z[i].append([0] * 300)
    X_test = np.asarray(z)


    
    

    Y_train = [x[1] for x in data[:split]]
    Y_test = [x[1] for x in data[split:]]

    _rev_train = [x[0] for x in reviews[:split]]
    _rev_test = [x[0] for x in reviews[split:]]

    _val_train = [x[1] for x in reviews[:split]]
    _val_test = [x[1] for x in reviews[split:]]

    return max_len, tokenizer, X_train, X_test, Y_train, Y_test, _rev_train, _rev_test, _val_train, _val_test

def train_gen_data(file_name, seq_length, split):
    data, reviews = process_doc(file_name, shuffle = True)

    split = int(len(reviews) * split)
    X_train = [x[0] for x in data[:split]]
    X_test = [x[0] for x in data[split:]]
    combined = X_train + X_test

    max_len = max(len(s) for s in combined)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(combined)

    max_words = 20
    max_len = max_words

    ''' USING W2V '''
    z = []
    for i, s in enumerate(X_train):
        s = s.split()
        s = s[:max_words]

        z.append([])
        padding = max_len - len(s)

        
        l = 0
        for j, word in enumerate(s):
            if word in model:
                z[i].append([])
                w = model[word].tolist()
                z[i][j - l] = [v for v in w]
            else:
                l += 1
        for f in range(padding + l):
            z[i].append([0] * 300)
    X_train = np.asarray(z)

    z = []
    for i, s in enumerate(X_test):
        s = s.split()
        s = s[:max_words]

        z.append([])
        padding = max_len - len(s)

        
        l = 0
        for j, word in enumerate(s):
            if word in model:
                z[i].append([])
                w = model[word].tolist()
                z[i][j - l] = [v for v in w]
            else:
                l += 1
        for f in range(padding + l):
            z[i].append([0] * 300)
    X_test = np.asarray(z)

    z = []
    y = []
    for data in X_train:
        for i in range(len(data) - seq_length):
            z.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
    X_train = np.asarray(z)
    Y_train = np.asarray(y)

    z = []
    y = []
    for data in X_test:
        for i in range(len(data) - seq_length):
            z.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
    X_test = np.asarray(z)
    Y_test = np.asarray(y)

    return max_len, tokenizer, X_train, X_test, Y_train, Y_test


def split_data(file_name, pos_to, neg_to, unlabel_to):
    f = open(file_name, mode = 'r', encoding = 'utf8')
    lines = f.read().splitlines()

    pos = open(pos_to, mode = 'w', encoding = 'utf8')
    neg = open(neg_to, mode = 'w', encoding = 'utf8')
    unlabel = open(unlabel_to, mode = 'w', encoding = 'utf8')

    for i in range(0, len(lines), 3):
        if float(lines[i + 1]) == 0:
            unlabel.write(lines[i + 2] + '\n')
            continue
        review = float(lines[i]) / float(lines[i + 1])
        if review >= 0.7:
            pos.write(lines[i + 2] + '\n')
        else:
            neg.write(lines[i + 2] + '\n')
    
    f.close()
    pos.close()
    neg.close()
    unlabel.close()
        
def strip_data(file_name, file_to):
    f = open(file_name, mode = 'r', encoding = 'utf8')
    lines = f.read().splitlines()

    f_to = open(file_to, mode = 'w', encoding = 'utf8')

    for i in range(0, len(lines), 3):
        f_to.write(lines[i + 2] + '\n')
    
    f.close()
    f_to.close()

def to_stars(percent):
    val = round(percent * 10.0) / 2.0
    if val <= 0:
        val = 0.5
    if val > 5:
        val = 5
    return str(val) + '/5.0 stars'

def read_settings(file_name):
    f = open(file_name, mode = 'r', encoding = 'utf8')
    
    settings = {}
    lines = f.read().splitlines()
    for line in lines:
        setting = line.split(': ')

        s = setting[1]
        try:
            if int(s) == float(s):
                s = int(s)
            else:
                s = float(s)
        except Exception:
            s = setting[1]
        
        if s == 'True':
            s = True
        if s == 'False':
            s = False
        settings[setting[0]] = s
    
    return settings

def truncate_text(txt, length):
    if len(txt) > length:
        return txt[:length - 3] + '...'
    return txt + ' ' * (length - len(txt))


def save_arr(arr, file_name):
    np.save(file_name, arr)
    print('Saved to ' + file_name)

def load_arr(file_name):
    np.load(file_name)

def vec2word(vector, temp = 1):
    print(vector)
    sim = model.most_similar([vector], [], topn = temp)

    
    w = random.choice(sim)
    return w[0]