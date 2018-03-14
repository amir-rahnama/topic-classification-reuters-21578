"""Fetch and do some basic cleaning."""
import os
import numpy as np
import nltk
import sys
import re
import xml.sax.saxutils as saxutils

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from gensim.models import Word2Vec

from multiprocessing import cpu_count

from keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('stopwords')

this = sys.modules[__name__]
this.vocabulary = []
this.categories = []


this.stop_words = set(stopwords.words('english'))

# Initialize tokenizer
# It's also possible to try with a stemmer or to mix a stemmer and a lemmatizer
this.tokenizer = RegexpTokenizer('[\'a-zA-Z]+')

# Initialize lemmatizer
this.lemmatizer = WordNetLemmatizer()

# Tokenized document collection


# def stem(words):
#     stemmer = SnowballStemmer("english", ignore_stopwords=True)
#     stemmed_words = [stemmer.stem(word) for word in words]

#     return stemmed_words


def unescape(text):
    return saxutils.unescape(text)


def unique(arr):
    return list(set(arr))


def add_to_vocab(elements):
    for element in elements:
        if element not in this.vocabulary:
            this.vocabulary.append(element)


def add_to_categories(elements):
    for element in elements:
        if element not in this.categories:
            this.categories.append(element)


def transform_to_indices(elements):
    res = []
    for element in elements:
        res.append(this.vocabulary.index(element))
    return res


def transform_to_category_indices(element):
    return this.categories.index(element)


def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()


def generate_categories():
    topics = 'all-topics-strings.lc.txt'

    with open('./reuters-21578/' + topics, 'r') as file:
        for category in file.readlines():
            this.categories.append(category.strip().lower())


def to_category_onehot(categories):
    target_categories = this.categories
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0

    return vector


def save_data(x_train, y_train, x_test, y_test_np):
    np.save('./data/x_train_np.npy', x_train)
    np.save('./data/y_train_np.npy', y_train)
    np.save('./data/x_test_np.npy', x_test)
    np.save('./data/y_test_np.npy', y_test)

    np.save('./data/vocabulary.npy', this.vocabulary)
    np.save('./data/categories.npy', this.categories)


def tokenize(document):
    words = []

    for sentence in sent_tokenize(document):
        tokens = [this.lemmatizer.lemmatize(t.lower()) for t in this.tokenizer.tokenize(sentence)
                  if t.lower() not in this.stop_words]
        words += tokens

    return words


def read_retuters_files(path="./reuters-21578/"):
    # x_train = []
    # x_test = []
    # y_train = []
    # y_test = []

    this.x_train = {}
    this.x_test = {}
    this.y_train = {}
    this.y_test = {}

    for file in os.listdir(path):
        if file.endswith(".sgm"):
            print("reading ", path + file)
            f = open(path + file, 'r', encoding="ISO-8859-1")
            data = f.read()

            soup = BeautifulSoup(data, "html5lib")
            posts = soup.findAll("reuters")

            for post in posts:
                post_id = post['newid']
                body = unescape(strip_tags(str(post('text')[0].content))
                                .replace('reuter\n&#3;', ''))
                post_categories = []

                topics = post.topics.contents

                for topic in topics:
                    post_categories.append(strip_tags(str(topic)))

                # tokenizer = RegexpTokenizer("[A-Z]\w+")

                # words = tokenizer.tokenize(post.text)
                # stemmed = stem(words)

                # add_to_vocab(stemmed)
                # tokenz = transform_to_indices(stemmed)[0]

                # topics = post.topics.contents
                # doc_id = posts["newid"]

                # ts = []
                # if len(topics) == 0:
                #     ts.append('NA')

                # for topic in topics:
                #     if topic not in ts:
                #         ts.append(topic.text)

                # add_to_categories(ts)
                # catz = transform_to_category_indices(ts[0])

                category_onehot = to_category_onehot(post_categories)

                cross_validation_type = post["lewissplit"]
                if (cross_validation_type == "TRAIN"):
                    x_train[post_id] = body
                    y_train[post_id] = category_onehot
                    # x_train.append(tokenz)
                    # y_train.append(catz)
                else:
                    x_test[post_id] = body
                    y_test[post_id] = category_onehot
                    # x_test.append(body)
                    # y_test.append(catz)

    save_data(x_train, y_train, x_test, y_test)

    return (x_train, y_train), (x_test, y_test)

    # x_train_np = np.array(x_train).reshape(len(x_train), -1)
    # y_train_np = np.array(y_train).reshape(len(y_train), -1)

    # x_test_np = np.array(x_test).reshape(len(x_test), -1)
    # y_test_np = np.array(y_test).reshape(len(y_test), -1)


def tokenize_docs(document):
    tokenized_docs = []
    this.number_of_documents = len(document)

    for key in document.keys():
        tokenized_docs.append(tokenize(document[key]))

    return tokenized_docs


def vectorize_docs(documents, w2v_model):
    document_max_num_words = 100
    num_features = 500

    x = np.zeros(shape=(this.number_of_documents, document_max_num_words,
                        num_features)).astype(np.float32)

    empty_word = np.zeros(num_features).astype(np.float32)

    for idx, document in enumerate(documents):
        for jdx, word in enumerate(document):
            if jdx == document_max_num_words:
                break

            else:
                if word in w2v_model:
                    x[idx, jdx, :] = w2v_model[word]
                else:
                    x[idx, jdx, :] = empty_word

    return x


def vectorize_categories(categories):
    num_categories = len(this.categories)

    y = np.zeros(shape=(this.number_of_documents, num_categories)).astype(np.float32)

    for idx, key in enumerate(categories.keys()):
        y[idx, :] = categories[key]

    return y


def model_lstm(X_train, Y_train, X_test, Y_test):
    document_max_num_words = 100
    num_features = 500
    num_categories = len(this.categories)

    model = Sequential()

    model.add(LSTM(int(document_max_num_words * 1.5), input_shape=(document_max_num_words, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(num_categories))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))

    # Evaluate model
    score, acc = model.evaluate(X_test, Y_test, batch_size=128)

    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)


def read_numpy_files():
    x_train = np.load('./data/x_train_np.npy')
    y_train = np.load('./data/y_train_np.npy')

    x_test = np.load('./data/x_test_np.npy')
    y_test = np.load('./data/y_test_np.npy')

    return (x_train, y_train), (x_test, y_test)


def model_mlp(X_train, Y_train, X_test, Y_test):
    document_max_num_words = 100
    num_classes = 135
    batch_size = 128
    nb_epoch = 5

    print(num_classes, 'classes')

    print('Vectorizing sequence data...')
    # tokenizer = Tokenizer(num_words=document_max_num_words)
    # x_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    # x_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')
    # y_train = keras.utils.to_categorical(Y_train, num_classes)
    # y_test = keras.utils.to_categorical(Y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print('Building model...')
    model = Sequential()
    model.add(Dense(512, input_shape=(document_max_num_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=1,
              validation_split=0.1)

    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    generate_categories()
    cache = True

    if (cache):
        (x_train, y_train), (x_test, y_test) = read_numpy_files()
    else:
        (x_train, y_train), (x_test, y_test) = read_retuters_files()

    num_features = 500
    x_train_token = tokenize_docs(x_train)

    w2v_model = Word2Vec(x_train_token, size=num_features, min_count=1, window=10, workers=cpu_count())
    w2v_model.init_sims(replace=True)
    w2v_model.save('./model/reuters_train.word2vec')

    x_train = vectorize_docs(x_train_token, w2v_model)
    y_train = vectorize_categories(y_train)

    x_test_token = tokenize_docs(x_test)

    w2v_model = Word2Vec(x_test_token, size=num_features, min_count=1, window=10, workers=cpu_count())
    w2v_model.init_sims(replace=True)
    w2v_model.save('./model/reuters_test.word2vec')

    x_test = vectorize_docs(x_test_token, w2v_model)
    y_test = vectorize_categories(y_test)

    model_mlp(x_train, y_train, x_test, y_test)
