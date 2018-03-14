
import sys
import re
import xml.sax.saxutils as saxutils
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')


this = sys.modules[__name__]


this.tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
this.lemmatizer = WordNetLemmatizer()
this.vocabulary = []
this.categories = []
this.stop_words = set(stopwords.words('english'))


def generate_categories():
    """Generate the list of categories."""
    topics = 'all-topics-strings.lc.txt'

    with open('./reuters-21578/' + topics, 'r') as file:
        for category in file.readlines():
            this.categories.append(category.strip().lower())


def vectorize_docs(documents, w2v_model):
    """A weird oneshot representation for word2vec."""
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


def unescape(text):
    """Unescape charactes."""
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
    """String tags for a better vocabulary."""
    return re.sub('<[^<]+?>', '', text).strip()


def to_category_onehot(categories):
    """Create onehot vectors for categories."""
    target_categories = this.categories
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0

    return vector


def save_data(x_train, y_train, x_test, y_test):
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


def tokenize_docs(document):
    tokenized_docs = []
    this.number_of_documents = len(document)

    for key in document.keys():
        tokenized_docs.append(tokenize(document[key]))

    return tokenized_docs


def read_retuters_files(path="./reuters-21578/"):
    x_train = {}
    x_test = {}
    y_train = {}
    y_test = {}

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

                category_onehot = to_category_onehot(post_categories)

                cross_validation_type = post["lewissplit"]
                if (cross_validation_type == "TRAIN"):
                    x_train[post_id] = body
                    y_train[post_id] = category_onehot
                else:
                    x_test[post_id] = body
                    y_test[post_id] = category_onehot

    save_data(x_train, y_train, x_test, y_test)

    return (x_train, y_train), (x_test, y_test)
