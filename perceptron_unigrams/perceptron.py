
"""
Implements the perceptron algorithm on a data set containing reviews and rating of the reviews.
Using unigram and unigram tf-idf representations to predict the rating. Use of nltk prohibited.
"""

import numpy as np


class Perceptron():

    def __init__(self, n_epochs=2):
        self.n_epochs = n_epochs

    def fit(self, train_data):
        n_samples = train_data.shape[0]
        n_features = train_data.shape[1] - 1

        self.w = np.zeros([1, n_features])
        self.epoch_accuracy = []

        def net_input(x):
            return np.dot(self.w, x.T)

        def predict_on_train(x):
            return np.where(net_input(x) >= 0, 1, 0)

        for epoch in range(self.n_epochs):
            n_errors = 0

            for i, row in train_data.iterrows():
                y = row.iloc[:, -1]
                x = row.iloc[:, :-1]

                # update = 1 if mistake made on positive label, -1 if on negative label
                update = y - predict_on_train(x)
                self.w += update * x
                n_errors += int(update != 0)

            self.epoch_accuracy.append((n_samples - n_errors)/n_samples)

    def predict(self, test_data):
        X = test_data.iloc[:, :-1]
        y_pred = np.where(np.dot(self.w, X.T) >= 0, 1, 0)
        return y_pred


