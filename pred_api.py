from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import re
import string
import numpy as np
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import pickle
import keras

from keras.utils import pad_sequences

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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


def get_pred(text):
    net = keras.models.load_model("/Users/rishabh/Downloads/content 3/models")
    text = clean_up_pipeline(text)
    text = [text]
    my_input_features = tokenizer.texts_to_sequences(text)
    my_input_features = pad_sequences(my_input_features, maxlen=2000)
    pred = net.predict(my_input_features)
    pred = pred[0][0]
    print(pred)
    res = int(pred > 0.6)
    print(res)
    return res


app = FastAPI()

origins = ['*']
headers = ['*']
methods = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=methods,
    allow_headers=headers
)


@app.post('/get_prediction')
async def test(text: str = Form(...)):
    pred = get_pred(text)
    if pred == 1:
        res = "The text contains references to self-harm"
    else:
        res = "The text does not contain references to self-harm"
    return {
        "code": "success",
        "error": False,
        "data": {
            "text": text,
            "pred": res
        }
    }
