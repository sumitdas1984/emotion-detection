'''
description : this module trains an LSTM based emotion classifier
author      : sumit das
date        : 16/09/2018
'''

import os
import sys
import time
import numpy as np
import pickle
import re

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,SpatialDropout1D,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.regularizers import L1L2
from keras.layers.embeddings import Embedding


def read_text_file(file_name):
    data_list  = []
    with open(file_name,'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data_list.append([label, text])

    return data_list


def extract_labels(text_list):
    label_list = []
    text_list = [text_list[i][0].replace('[','') for i in range(len(text_list))]
    label_list = [list(np.fromstring(text_list[i], dtype=float, sep=' ')) for i in range(len(text_list))]
    return label_list


def extract_text_msgs(text_list):
    msg_list = []
    msg_list = [text_list[i][1] for i in range(len(text_list))]
    return msg_list


def read_glove_vector(glove_file):
    with open(glove_file,'r',encoding='UTF-8') as file:
        words = set()
        word_to_vec = {}
        for line in file:
            line = line.strip().split()
            line[0] = re.sub('[^a-zA-Z]', '', line[0])
            if len(line[0]) > 0:
                words.add(line[0])
                try:
                    word_to_vec[line[0]] = np.array(line[1:],dtype=np.float64)
                except:
                    print('Error has occured')
                    print('-'*50)
                    print(line[1:])

        i = 1
        word_to_index = {}
        index_to_word = {}
        for word in sorted(words):
            word_to_index[word] = i
            index_to_word[i] = word
            i = i+1
    return word_to_index,index_to_word,word_to_vec


def create_embedding_layer(word_to_index,word_to_vec):
    corpus_len = len(word_to_index) + 1
    embed_dim = word_to_vec['word'].shape[0]

    embed_matrix = np.zeros((corpus_len,embed_dim))

    for word, index in word_to_index.items():
        embed_matrix[index,:] = word_to_vec[word]

    embedding_layer = Embedding(corpus_len, embed_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embed_matrix])

    return embedding_layer


def create_lstm_model(input_shape,embedding_layer):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer =  embedding_layer
    embeddings = embedding_layer(sentence_indices)
    reg = L1L2(0.01, 0.01)

    X = Bidirectional(LSTM(128, return_sequences=True,bias_regularizer=reg,kernel_initializer='he_uniform'))(embeddings)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = LSTM(64)(X)
    X = Dropout(0.5)(X)
    X = Dense(7, activation='softmax')(X)
    X =  Activation('softmax')(X)
    model = Model(sentence_indices, X)

    return model


if __name__ == '__main__':
    textlist = read_text_file("data/data.txt")
    label_list = extract_labels(textlist)
    msg_list = extract_text_msgs(textlist)
    word_to_index,index_to_word,word_to_vec = read_glove_vector('resources/glove.6B.50d.txt')
    x_train, x_test, y_train, y_test = train_test_split(msg_list, label_list,stratify = label_list,        test_size = 0.2, random_state = 123)
    tk = Tokenizer(lower = True, filters='')
    tk.fit_on_texts(msg_list)
    train_tokenized = tk.texts_to_sequences(x_train)
    test_tokenized = tk.texts_to_sequences(x_test)
    maxlen = 50
    X_train = pad_sequences(train_tokenized, maxlen = maxlen)
    X_test = pad_sequences(test_tokenized, maxlen = maxlen)
    if os.path.exists('models/tokenizer.pickle'):
        os.remove('models/tokenizer.pickle')
        with open('models/tokenizer.pickle', 'wb') as tokenizer:
            pickle.dump(tk, tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

    embedding_layer = create_embedding_layer(word_to_index,word_to_vec)
    model = create_lstm_model((maxlen,),embedding_layer)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train), epochs = 1, batch_size = 32, shuffle=True)
    model.save('models/emoji_model.h5')
#     model = load_model('models/emoji_model.h5')
    loss, accuracy = model.evaluate(X_train, np.array(y_train), verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, np.array(y_test), verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

