#Python script for unigrams and tf-idf representations

"""
Implements the perceptron algorithm on a data set containing reviews and rating of the reviews.
Using unigram and unigram tf-idf representations to predict the rating. Use of nltk prohibited.
"""


import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter

# The data csv contains only two columns: one for text (review) and one for labels (rating of the review)
# The csv has been preprocessed to remove stopwords
train_preprocessed = pd.read_csv('train_preprocessed2.csv')

X_train_pp = train_preprocessed["text"]
y_train_pp = train_preprocessed['label']

# convert series into array, and reshape to fit into sparse matrix later
y_train_pp = y_train_pp.values.reshape([y_train_pp.shape[0], 1])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass


def get_corpus_word_doc_count(data):
    """
    :param data: Pandas dataframe
        contains training data corpus
    :return: dict
        keys are words in the corpus, values are the number of documents in the corpus that contains the word
    """
    #word_doc_count = {}
    word_doc_count = Counter()
    for index, text in data.iteritems():
        words = text.split(' ')

        # remove words that contain '_' or '.' as they're probably invalid words that won't appear in other documents
        words = [word for word in words if '_' or '.' not in word]

        unique_words = set(words)
        word_doc_count.update([w for w in unique_words if not float(w)])
        '''
        for i in unique_words:
            if is_number(i):
                continue 
            #elif i not in word_doc_count:
            #    word_doc_count[i] = 1
            #else:
            #    word_doc_count[i] += 1
        '''

    return word_doc_count


word_doc_count = get_corpus_word_doc_count(X_train_pp)


# remove words that only appear in one document out of the entire corpus
word_doc_count = {k:v for k,v in word_doc_count.items() if v!=1}
# sort dictionary alphabetically by key
word_doc_count = dict(sorted(word_doc_count.items(), key=lambda x: x[0]))


def create_word_index_map(word_doc_count_dict):
    """
    :param word_doc_count_dict:
        dictionary containing words in the corpus
    :return: dict
        contains words as key and an assigned number as value
    """
    word_map = {}
    for x in enumerate(word_doc_count_dict):
        val, key = x
        word_map[key] = val
    return word_map

word_index_map = create_word_index_map(word_doc_count)


def get_key(index):
    """
    Purpose: Since the sparse matrix only allows numbers as column name, this function retrieves
             the word associated with each column.
    :param index:
        column number in the
    :return:
        word mapped to the column
    """
    return list(word_index_map.keys())[list(word_index_map.values()).index(index)]


def get_doc_word_freq(document, word_index_map):
    """
    :param document: string
        document in the corpus
    :param word_map: dict
        dictionary containing all the words in the corpus that we're interested in
    :return: dict
        key: unique word in document, value: number of times that word appears in document
    """
    word_freq = {}
    words = document.split(' ')
    words = [word for word in words if word in word_index_map]

    for word in words:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    return word_freq


def build_sparse_unigram_dtm(X_data, word_index_map):
    """
    :param X_data: Pandas dataframe
        Contains the corpus of review text
    :param word_map: dict
        key is word that appears in text and value is the assigned index
    :return: sparse dok document-term-matrix
        Each column is mapped to a word in the corpus, each row represents one review (document),
        and [row, column] represents the number of times the word appears in the document
    """
    # initialize sparse matrix with a row for each document, a column for each unique word in the corpus,
    # plus one column for the weights to be calculated and one column for labels
    dok = sp.dok_matrix((X_data.shape[0], len(word_index_map)+2), dtype=np.float64)

    for index, text in X_data.iteritems():
        word_freq = get_doc_word_freq(text, word_index_map)

        for k, v in word_freq.items():
            dok[index, word_index_map[k]] = v

    return dok

unigram_dok = build_sparse_unigram_dtm(X_train_pp, word_index_map)


# Add w_0 and label vectors to sparse unigram matrix
# initialize all weights as 1
w_0 = np.ones([unigram_dok.shape[0], 1])
unigram_dok[:, len(word_index_map)] = w_0
unigram_dok[:, len(word_index_map)+1] = y_train_pp


# convert unigram dok_matrix to csr_matrix for faster row-wise operations
unigram_csr = unigram_dok.tocsr()



def build_sparse_tfidf_dtm(X_data, word_index_map, word_doc_count):
    """
    :param X_data: Pandas dataframe
        Contains the corpus of review text
    :param word_map: dict
        key is word that appears in text and value is the assigned index
    :param word_doc_count: dict
        key is a word that appears in the corpus and value is the number of documents within the corpus that contains the word
    :return: sparse dok document-term-matrix
        Each column is mapped to a word in the corpus, each row represents one review (document),
        and [row, column] represents the tf-idf of the word that appears in the document
    """
    # initialize sparse matrix
    dok = sp.dok_matrix((X_data.shape[0], len(unigram_map) + 2), dtype=np.float64)
    D = X_data.shape[0]

    def idf(t):
        return D/word_doc_count.get(t)

    def tf_idf(t, tf):
        return tf * np.log10(idf(t))

    for index, text in X_data.iteritems():
        word_freq = get_doc_word_freq(text, word_index_map)

        for k, v in word_freq.items():
            dok[index, word_index_map[k]] = tf_idf(k, v)

    return dok


tfidf_dok = build_sparse_tfidf_dtm(X_train_pp, word_index_map, word_doc_count)
w_0 = np.ones([tfidf_dok.shape[0], 1])
tfidf_dok[:, len(word_index_map)] = w_0
tfidf_dok[:, len(word_index_map)+1] = y_train_pp

# Convert to csr_matrix for faster row-wise operations
tfidf_csr = tfidf_dok.tocsr()


def train(train_csr, word_map, n_epoch=2, sample_frac=0.8):
    '''
    Return: a tuple of number of errors made in each epoch, and the final averaged weight vector.
    '''
    # Initialize w vector of zeros with dimensions (1,
    w = np.zeros([1, train_csr.shape[1] - 1])
    w_sum = 0

    # Compute dot product of w and x vectors
    def net_input(x):
        return (np.dot(w, x.T))

    # Compute predicted y
    def predict(x):
        return (np.where(net_input(x) >= 0.0, 1, 0))

    train_accuracy = []
    for epoch in range(n_epoch):
        n = 0
        error = 0

        # Get training samples and shuffle the data
        train_size = int(sample_frac * train_csr.shape[0])
        index = np.arange(train_csr.shape[0])
        np.random.shuffle(index)
        if sample_frac != 1.0:
            train_index = index[:train_size]
            train_set = train_csr[train_index, :]
            test_set = train_csr[-train_index, :]
        else:
            train_set = train_csr

        for i in train_set:
            d = train_set[i].toarray()
            y = d[:, -1:]  # label
            x = d[:, :-1]  # x array
            update = y - predict(x)
            w += update * x
            error += int(update != 0.0)
            n += 1
            if epoch == 1:
                w_sum += w
        train_accuracy.append((train_set.shape[0] - error) / train_set.shape[0])

    w_final = w_sum / n + 1

    ## Test set
    # Compute dot product of w_final and x vectors
    if sample_frac != 1.0:
        def net_input_test(x):
            return (np.dot(w_final, x.T))

        # Compute predicted y
        def predict_test(x):
            return (np.where(net_input_test(x) >= 0.0, 1, 0))

        test_error = 0
        for i in test_set:
            d = i.toarray()
            y = d[:, -1:]  # label
            x = d[:, :-1]  # x array
            if predict_test(x) != y: test_error += 1
        test_accuracy = (test_set.shape[0] - test_error) / test_set.shape[0]

    else:
        test_accuracy = 0

    return (train_accuracy, test_accuracy, w_final)




unigram_train_results = train(unigram_csr, unigram_map, sample_frac = .9)
tfidf_train_results = train(tfidf_csr, unigram_map, sample_frac = .9)


# Import test set
test_data = pd.read_csv('reviews_te.csv', sep = ',')
X_test = test_data["text"]
y_test = test_data['label']
y_test.as_matrix()
y_test = y_test.values.reshape([y_test.shape[0], 1])

# construct unigram test matrix
test_unigram_dok = build_sparse_unigram(X_test, unigram_map)
w_0 = np.ones([X_test.shape[0], 1])
test_unigram_dok[:, len(unigram_map)] = w_0
test_unigram_dok[:, len(unigram_map)+1] = y_test
test_unigram_csr = test_unigram_dok.tocsr()



# construct tf-idf test matrix
test_tfidf_dok = build_sparse_tfidf(X_test, unigram_map, unigrams_docs_count)
w_0 = np.ones([X_test.shape[0], 1])
test_tfidf_dok[:, len(unigram_map)] = w_0
test_tfidf_dok[:, len(unigram_map)+1] = y_test
test_tfidf_csr = test_tfidf_dok.tocsr()


# Test function
def test_prediction(test_csr, training_results):
    # Get final weights from training set
    w = training_results[2]

    # Compute dot product of w and x vectors
    def net_input(x):
        return (np.dot(w, x.T))

    # Compute predicted y
    def predict(x):
        return (np.where(net_input(x) >= 0.0, 1, 0))

    test_error = 0
    for i in test_csr:
        d = i.toarray()
        y = d[:, -1:]  # label
        x = d[:, :-1]  # x array
        if predict(x) != y: test_error += 1

    return (test_csr.shape[0] - test_error) / test_csr.shape[0]


unigram_test_result = test_prediction(test_tfidf_csr, unigram_train_results)
tfidf_test_result = test_prediction(test_tfidf_csr, tfidf_train_results)

