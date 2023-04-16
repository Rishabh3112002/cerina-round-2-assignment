import pandas as pd
import re
import string
import numpy as np
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Packages for model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional
from keras.models import Model
import keras

print('Setting up Server')
print('[=', end='')
data = pd.read_csv("Suicide_Detection.csv")
print('=', end='')

x_train, x_test, y_train, y_test = train_test_split(
    data.text, data['class'], test_size=0.2)


def to_lower(word):
    result = word.lower()
    return result


def remove_number(word):
    result = re.sub(r'\d+', '', word)
    return result


def remove_punctuation(word):
    result = word.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return result


def remove_whitespace(word):
    result = word.strip()
    return result


def replace_newline(word):
    return word.replace('\n', '')


def clean_up_pipeline(sentence):
    cleaning_utils = [replace_newline,
                      to_lower,
                      remove_number,
                      remove_punctuation, remove_whitespace]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


print('=', end='')

x_train = [clean_up_pipeline(o) for o in x_train]
print('=', end='')
x_test = [clean_up_pipeline(o) for o in x_test]
print('=', end='')

le = LabelEncoder()
train_y = le.fit_transform(y_train.values)
test_y = le.transform(y_test.values)
print('=', end='')

embed_size = 100
max_feature = 50000
max_len = 2000
tokenizer = Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(x_train)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('=]')


def train_model(x_train=x_train, x_test=x_test, train_y=train_y, test_y=test_y):
    x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
    x_test_features = np.array(tokenizer.texts_to_sequences(x_test))

    x_train_features = pad_sequences(x_train_features, maxlen=max_len)
    x_test_features = pad_sequences(x_test_features, maxlen=max_len)

    embedding_vecor_length = 32

    model = tf.keras.Sequential()
    model.add(Embedding(max_feature, embedding_vecor_length, input_length=max_len))
    model.add(Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train_features, train_y, batch_size=512,
                        epochs=2, validation_data=(x_test_features, test_y))

    model.save("models")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.savefig('accuracy.jpg')
    plt.show()


if __name__ == '__main__':
    train_model()
