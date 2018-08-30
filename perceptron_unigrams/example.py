
import numpy as np
from perceptron_unigrams.utils import get_corpus_word_doc_count, create_word_index_map, get_key, get_doc_word_freq, build_sparse_unigram_dtm, build_sparse_tfidf_dtm
from sklearn.model_selection import train_test_split
from perceptron_unigrams.perceptron import Perceptron


def main():

    data = pd.read_csv('review_data.csv')

    X = data['text']
    y = data['label']

    word_doc_count = get_corpus_word_doc_count(X)

    # remove words that only appear in one document out of the entire corpus
    word_doc_count = {k: v for k, v in word_doc_count.items() if v != 1}

    # sort dictionary alphabetically by key
    word_doc_count = dict(sorted(word_doc_count.items(), key=lambda x: x[0]))

    word_index_map = create_word_index_map(word_doc_count)

    unigram_dtm = build_sparse_unigram_dtm(X, y, word_index_map)

    train, test = train_test_split(unigram_dtm, test_size=0.25, random_state=11)

    clf = Perceptron(n_epochs=2)
    clf.fit(train)

    y_pred = clf.predict(test)
    return y_pred


if __name__ == "__main__":
    main()