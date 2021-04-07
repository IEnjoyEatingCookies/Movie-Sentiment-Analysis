import os
import nltk
import random
import re
import sys
import json
import os
import pickle
import re
import gensim
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True, limit = 10000)

stop_words = ['a', 'an', 'and', 'the', 'no', 'of']

data_stripped = os.path.join('data', 'data_stripped.txt')

f = open(data_stripped, mode = 'r', encoding = 'utf8')
z = list(f.read().splitlines())

lead_w = 2
end_w = 1

s = []
for sent in z:
        if not sent[-1] in '.?!':
                sent += '.'
        s.extend(re.split('(?<=[.!?]) ', sent))

def tokenizer(txt):
        tokens = txt.split()

        # Parentheses ()
        i = 0
        paren_tag = False
        ntokens = []
        paren = []
        while i < len(tokens):
                if '(' in tokens[i]:
                        paren_tag = True
                        paren.append(tokens[i])
                elif ')' in tokens[i]:
                        paren_tag = False
                        paren.append(tokens[i])
                        ntokens.append(' '.join(paren))
                        paren = []
                elif paren_tag:
                        paren.append(tokens[i])
                else:
                        ntokens.append(tokens[i])
                i += 1
        tokens = ntokens

        # Parentheses []
        i = 0
        paren_tag = False
        ntokens = []
        paren = []
        while i < len(tokens):
                if '[' in tokens[i]:
                        paren_tag = True
                        paren.append(tokens[i])
                elif ']' in tokens[i]:
                        paren_tag = False
                        paren.append(tokens[i])
                        ntokens.append(' '.join(paren))
                        paren = []
                elif paren_tag:
                        paren.append(tokens[i])
                else:
                        ntokens.append(tokens[i])
                i += 1
        tokens = ntokens

        # Parentheses {}
        i = 0
        paren_tag = False
        ntokens = []
        paren = []
        while i < len(tokens):
                if '{' in tokens[i]:
                        paren_tag = True
                        paren.append(tokens[i])
                elif '}' in tokens[i]:
                        paren_tag = False
                        paren.append(tokens[i])
                        ntokens.append(' '.join(paren))
                        paren = []
                elif paren_tag:
                        paren.append(tokens[i])
                else:
                        ntokens.append(tokens[i])
                i += 1
        tokens = ntokens

        # Quotes ""
        i = 0
        quote_tag = False
        ntokens = []
        quote = []
        while i < len(tokens):
                if '"' in tokens[i]:
                        if quote_tag:
                                quote.append(tokens[i])
                                q = ' '.join(quote)
                                ntokens.append(q)
                                quote = []
                                quote_tag = False
                        else:
                                quote_tag = True
                                quote.append(tokens[i])
                elif quote_tag:
                        quote.append(tokens[i])
                else:
                        ntokens.append(tokens[i])
                i += 1
        tokens = ntokens
        
        return tokens

tags = ['NN', 'JJ', 'VBG']

def init(txts, start_save = 'start_tokens.txt', end_save = 'end_tokens.txt', chain_save = 'chain_tokens.txt'):
        path = 'Tokens'

        start_save = os.path.join(path, start_save)
        end_save = os.path.join(path, end_save)
        chain_save = os.path.join(path, chain_save)

        start_tokens = []
        end_tokens = []
        chain = {}

        if os.path.exists(start_save) and os.path.exists(end_save) and os.path.exists(chain_save):
                print('Loading files...', end = '')
                with open(start_save, mode = 'rb') as fp:
                        start_tokens = pickle.load(fp)
                with open(end_save, mode = 'rb') as fp:
                        end_tokens = pickle.load(fp)
                with open(chain_save, mode = 'r') as fp:
                        chain = json.load(fp)
                print('\rFiles loaded.')
        else:
                print('Files don\'t exist, creating files...')
                l = 50
                count = len(txts)
                counter = 0
                print('Reading...')
                print('Reading docs [' + ' ' * l  + '] 0/' + str(count), end ='')
                for txt in txts:
                        tokens = tokenizer(txt)
                        tokens_alt = [model.most_similar(positive = [x])[0][0] if x in model and nltk.pos_tag([x])[0][1] in tags else x for x in tokens]


                        if len(tokens) < lead_w + end_w:
                                continue
                        
                        s_t = ' '.join(tokens[:lead_w])
                        if not s_t in start_tokens:
                                start_tokens.append(s_t)
                        e_t = ' '.join(tokens[-end_w:])
                        if not e_t in end_tokens:
                                if not e_t[-1] in '.?!':
                                        e_t += '.'
                                end_tokens.append(e_t)

                        for i in range(len(tokens) - lead_w - end_w + 1):
                                lead = ' '.join(tokens[i: i + lead_w]).lower()
                                follow = ' '.join(tokens[i + lead_w: i + lead_w + end_w])
                                if lead in chain:
                                        if not follow in chain[lead]:
                                                chain[lead].append(follow)
                                else:
                                        chain[lead] = [follow]
                        
                        s_t = ' '.join(tokens_alt[:lead_w])
                        if not s_t in start_tokens:
                                start_tokens.append(s_t)
                        e_t = ' '.join(tokens_alt[-end_w:])
                        if not e_t in end_tokens:
                                if not e_t[-1] in '.?!':
                                        e_t += '.'
                                end_tokens.append(e_t)

                        for i in range(len(tokens_alt) - lead_w - end_w + 1):
                                lead = ' '.join(tokens_alt[i: i + lead_w]).lower()
                                follow = ' '.join(tokens_alt[i + lead_w: i + lead_w + end_w])
                                if lead in chain:
                                        if not follow in chain[lead]:
                                                chain[lead].append(follow)
                                else:
                                        chain[lead] = [follow]
                                
                        counter += 1

                        percent = int(counter / count * l)
                        print('\rReading docs [' + '█' * percent + ' ' * (l - percent) +  '] ' + str(counter) + '/' + str(count), end = '')
                
                print('Saving files...', end = '')
                try:
                        os.mkdir(path)
                except OSError:
                        print ("Creation of the directory %s failed" % path)
                else:
                        print ("Successfully created the directory %s " % path)
                with open(start_save, mode = 'wb') as fp:
                        pickle.dump(start_tokens, fp)
                with open(end_save, mode = 'wb') as fp:
                        pickle.dump(end_tokens, fp)
                with open(chain_save, mode = 'w') as fp:
                        json.dump(chain, fp)
                print('\rSaved files.')
        return start_tokens, end_tokens, chain

print('Reading docs')
s, e, c = init(s)
print('Finished.')

# print(s)
# print()
# print(e)
# print()
# print(c)

def generate_text(pmin, pmax):
        sentence = random.choice(s).split(' ')
        while True:
                l = len(sentence)

                seed = ' '.join(sentence[-lead_w:]).lower()
                if not seed in c:
                        if not any(x in sentence[-1] for x in '.?!'):
                                sentence[-1] = sentence[-1] + '.'
                        break
                poss = c[seed]

                p = 'ERROR'
                if l >= pmax:
                        ends = [x for x in poss if x in e]
                        if len(ends) > 0:
                                p = random.choice(ends)
                        else:
                                p = random.choice(poss)
                elif l < pmin:
                        not_ends = [x for x in poss if not x in e]
                        if len(not_ends) > 0:
                                p = random.choice(not_ends)
                        else:
                                p = random.choice(poss)
                else:
                        p = random.choice(poss)
                
                p = p.split()
                sentence.extend(p)
                if p in e:
                        break 
        return ' '.join(sentence)