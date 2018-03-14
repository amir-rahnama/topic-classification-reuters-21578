"""Fetch and do some basic cleaning."""
import numpy as np

import sys

from gensim.models import Word2Vec

from multiprocessing import cpu_count

from model import lstm
from data import reader


this = sys.modules[__name__]


def read_numpy_files():
    """Instead of running the entire pipeline at all times."""
    x_train = np.load('./data/x_train_np.npy')
    y_train = np.load('./data/y_train_np.npy')

    x_test = np.load('./data/x_test_np.npy')
    y_test = np.load('./data/y_test_np.npy')

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    cache = False
    reader.generate_categories()

    # TODO: Fix the big the cache mechanism
    # if (cache):
    #     (x_train, y_train), (x_test, y_test) = read_numpy_files()
    # else:
    #     (x_train, y_train), (x_test, y_test) = read_retuters_files()

    (x_train, y_train), (x_test, y_test) = reader.read_retuters_files()

    num_features = 500
    x_train_token = reader.tokenize_docs(x_train)

    w2v_model = Word2Vec(x_train_token,
                         size=num_features,
                         min_count=1,
                         window=10,
                         workers=cpu_count())
    w2v_model.init_sims(replace=True)
    w2v_model.save('./model/reuters_train.word2vec')

    x_train = reader.vectorize_docs(x_train_token, w2v_model)
    y_train = reader.vectorize_categories(y_train)

    x_test_token = reader.tokenize_docs(x_test)

    w2v_model = Word2Vec(x_test_token,
                         size=num_features,
                         min_count=1,
                         window=10,
                         workers=cpu_count())

    w2v_model.init_sims(replace=True)
    w2v_model.save('./model/reuters_test.word2vec')

    x_test = reader.vectorize_docs(x_test_token, w2v_model)
    y_test = reader.vectorize_categories(y_test)

    lstm.lstm(x_train, y_train, x_test, y_test)
