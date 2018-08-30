import numpy as np
import scipy.sparse as sp
from collections import Counter


def get_corpus_word_doc_count(data):
    """
    :param data: Pandas dataframe
        contains training data corpus
    :return: dict
        {word in the corpus: number of documents in the corpus that contains the word}
    """
    word_doc_count = Counter()
    for index, text in data.iteritems():
        words = text.split(' ')

        # remove words that contain '_' or '.' as they're probably invalid words that won't appear in other documents
        words = [word for word in words if '_' or '.' not in word]

        unique_words = set(words)
        word_doc_count.update([w for w in unique_words if not w.isnumeric()])

    return word_doc_count


def create_word_index_map(word_doc_count_dict):
    """
    :param word_doc_count_dict:
        dictionary containing words in the corpus
    :return: dict
        {word in corpus: unique index number}
    """
    word_map = {}
    for x in enumerate(word_doc_count_dict):
        val, key = x
        word_map[key] = val
    return word_map


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
    :param word_index_map: dict
        dictionary containing all the words in the corpus that we're interested in
    :return: dict
        {unique word in document: number of times that word appears in document}
    """

    words = document.split(' ')
    word_freq = Counter([word for word in words if word in word_index_map])

    return word_freq


def build_sparse_unigram_dtm(X, y, word_index_map):
    """
    :param X_data: Pandas dataframe
        Contains the corpus of review text
    :param word_index_map: dict
        key is word that appears in text and value is the assigned index
    :return: sparse dok document-term-matrix
        Each column is mapped to a word in the corpus, each row represents one review (document),
        and [row, column] represents the number of times the word appears in the document
    """
    # initialize sparse matrix with a row for each document, a column for each unique word in the corpus,
    # plus one column for the weights to be calculated and one column for labels
    dok = sp.dok_matrix((X.shape[0], len(word_index_map)+2), dtype=np.float64)

    # convert series into array, and reshape to fit into sparse matrix
    y = y.values.reshape([y.shape[0], 1])

    for index, text in X.iteritems():
        word_freq = get_doc_word_freq(text, word_index_map)

        for k, v in word_freq.items():
            dok[index, word_index_map[k]] = v

    # append initial w vector of ones and the labels
    w_0 = np.ones([X.shape[0], 1])
    dok[:, len(word_index_map)] = w_0
    dok[:, len(word_index_map) + 1] = y

    # convert to csr for efficient arithmetic operations
    dtm_csr = dok.tocsr()
    return dtm_csr


def build_sparse_tfidf_dtm(X, y, word_index_map, word_doc_count):
    """
    :param X_data: Pandas dataframe
        Contains the corpus of review text
    :param word_index_map: dict
        key is word that appears in text and value is the assigned index
    :param word_doc_count: dict
        key is a word that appears in the corpus and value is the number of documents within the corpus that contains the word
    :return: sparse dok document-term-matrix
        Each column is mapped to a word in the corpus, each row represents one review (document),
        and [row, column] represents the tf-idf of the word that appears in the document
    """
    # initialize sparse matrix
    dok = sp.dok_matrix((X.shape[0], len(word_index_map)+2), dtype=np.float64)
    D = X.shape[0]

    # convert series into array, and reshape to fit into sparse matrix
    y = y.values.reshape([y.shape[0], 1])

    def idf(t):
        return D/word_doc_count.get(t)

    def tf_idf(t, tf):
        return tf * np.log10(idf(t))

    for index, text in X.iteritems():
        word_freq = get_doc_word_freq(text, word_index_map)

        for k, v in word_freq.items():
            dok[index, word_index_map[k]] = tf_idf(k, v)

    # append initial w vector of ones and the labels
    w_0 = np.ones([X.shape[0], 1])
    dok[:, len(word_index_map)] = w_0
    dok[:, len(word_index_map) + 1] = y

    dtm_csr = dok.tocsr()
    return dtm_csr