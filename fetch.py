from bs4 import BeautifulSoup
import os
import sys
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import numpy as np
from gensim.models import Word2Vec

path = "./reuters-21578/"

x_train = []
x_test = []
y_train = []
y_test = []


def stem(words):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words


def unique(arr):
    return list(set(arr))


for file in os.listdir(path):
    if file.endswith(".sgm"):
        print("reading ", path + file)
        f = open(path + file, 'r', encoding="ISO-8859-1")
        data = f.read()

        soup = BeautifulSoup(data, "html5lib")
        posts = soup.findAll("reuters")

        for post in posts:
            tokenizer = RegexpTokenizer("[A-Z]\w+")
            words = tokenizer.tokenize(post.text)
            stemmed = stem(words)

            topics = post.topics.findAll("d")
            ts = []
            for topic in topics:
                ts.append(topic.text)

            cross_validation_type = post["lewissplit"]
            if (cross_validation_type == "TRAIN"):
                x_train.append(stemmed)
                y_train.append(unique(ts))
            else:
                x_test.append(stemmed)
                y_test.append(unique(ts))

x_train_np = np.array(x_train).reshape(len(x_train), -1)
y_train_np = np.array(y_train).reshape(len(y_train), -1)

x_test_np = np.array(x_test).reshape(len(x_test), -1)
y_test_np = np.array(y_test).reshape(len(y_test), -1)


# train model
model = Word2Vec(x_train, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
