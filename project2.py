import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np
import nltk


def predictor(args):
    print(args)
    y, corpus = readJson()
    tfidf = TfidfVectorizer()
    X_train, X_test, Y_train, Y_test = feature_Extraction_Train(tfidf,  y, corpus)
    predictionModel(X_train, X_test, Y_train, Y_test)
    predictUserInput(args, tfidf)


def readJson():
    dataFrame = []
    dataFrame = pd.read_json('yummly.json')

    y = []
    for values in dataFrame['cuisine']:
        y.append(values)

    corpus = []
    for ingredients in dataFrame.ingredients:
        corpus.append(" ".join(ingredients))

    return y, corpus


def feature_Extraction_Train(tfidf, y, corpus):
    frequency = tfidf.fit_transform(corpus)
    X_train, X_test, Y_train, Y_test = train_test_split(frequency, y, test_size=0.2, shuffle=False)
    return X_train, X_test, Y_train, Y_test


def predictionModel(X_train, X_test, Y_train, Y_test):
    svc = LinearSVC()
    clf = CalibratedClassifierCV(svc)
    clf.fit(X_train, Y_train)
    y_prob = clf.predict(X_test)
    print("score:", clf.score(X_test, Y_test), "test values:", y_prob)
    joblib.dump(clf, "linearSVCModel.pkl")


def predictUserInput(args, tfidf):
    clf_model = joblib.load("linearSVCModel.pkl")
    userPref = nltk.flatten(args.ingredient)
    userPref = np.asarray(userPref, dtype=object)
    userInput = tfidf.transform(userPref)
    y_pred = clf_model.predict(userInput)
    y_proba = clf_model.predict_proba(userInput)[0]
    print(y_pred[0], y_pred, y_proba[0] * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help=" Specify the number of suggestions",  nargs="?", required=True)
    parser.add_argument("--ingredient", help="Specify the type of ingredients", action="append", nargs="+", required=True)
    args = parser.parse_args()
    predictor(args)
