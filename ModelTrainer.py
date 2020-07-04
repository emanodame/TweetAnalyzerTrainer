import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from ModelEvaluator import evaluate


def split_data(processed_text, sentiment):
    return train_test_split(processed_text, sentiment, test_size=0.05, random_state=0)


def tf_idf_vectoriser(x_train, x_test):
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    vectoriser.fit(x_train)
    transformed_X_train = vectoriser.transform(x_train)
    transformed_X_test = vectoriser.transform(x_test)
    return transformed_X_train


def bernoulli(X_train, y_train):
    bnb_model = BernoulliNB(alpha=2)
    bnb_model.fit(X_train, y_train)
    file = open('Sentiment-BNB.pickle', 'wb')
    pickle.dump(bnb_model, file)
    evaluate(bnb_model)


def linear_scv(X_train, y_train):
    svc_model = LinearSVC()
    svc_model.fit(X_train, y_train)
    file = open('linear-scv.pickle', 'wb')
    pickle.dump(svc_model, file)
    evaluate(svc_model)


def logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    file = open('Sentiment-LR.pickle', 'wb')
    pickle.dump(lr_model, file)
    file.close()
    evaluate(lr_model)
