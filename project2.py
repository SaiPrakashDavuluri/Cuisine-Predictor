import pandas as pd
import numpy as np
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import sys
import json


def predictor(args):
    resultJson = {}
    fileName = 'yummly.json'
    userIngredients = nltk.flatten(args.ingredient)
    cuisine, corpus, dataFrame = readJson(fileName)
    _sentences = lemmatization(corpus)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train, X_test, y_train, y_test = feature_Extraction_Train(tfidf, cuisine, _sentences)
    le, Y_train, Y_test = labelEncoding(cuisine, y_train, y_test)
    resultJson = predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson)
    resultJson = nearestNeighbours(dataFrame, corpus, cuisine, tfidf, userIngredients, args.N, resultJson)
    exportResultJson(resultJson)


def readJson(fileName):
    try:
        dataFrame = pd.read_json(fileName)
    except Exception as ex:
        sys.stderr(f"Error whilst reading the yummly.json file. Error:{ex}")
    cuisine = []
    for values in dataFrame['cuisine']:
        cuisine.append(values)

    corpus = []
    for ingredients in dataFrame['ingredients']:
        corpus.append(" ".join(ingredients))
    # I tried to add ingredients passed through command line to features but received errors when converting to features.
    # temp = ''
    # for values in userIngredients:
    #     temp = temp+" "+values
    # print("text:", temp.strip())
    # corpus.append(temp.strip())
    return cuisine, corpus, dataFrame


def lemmatization(corpus):
    lemma = WordNetLemmatizer()
    _sentences = []
    for sentences in corpus:
        _words = word_tokenize(sentences)
        temp = []
        for word in _words:
            words_lemma = lemma.lemmatize(word)
            temp.append(words_lemma)
        _sentences.append(' '.join(temp))

    return _sentences


def feature_Extraction_Train(tfidf, cuisine, _sentences):
    frequency = tfidf.fit_transform(_sentences)
    X_train, X_test, y_train, y_test = train_test_split(frequency, cuisine, test_size=0.2, shuffle=False, random_state=42)
    return X_train, X_test, y_train, y_test


def labelEncoding(cuisine, y_train, y_test):
    le = LabelEncoder()
    le.fit(cuisine)
    Y_train = le.transform(y_train)
    Y_test = le.transform(y_test)
    return le, Y_train, Y_test


def predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson):
    svc = LinearSVC(penalty='l2', multi_class='ovr')
    clf_model = CalibratedClassifierCV(svc)
    clf_model.fit(X_train, Y_train)
    clf_model.predict(X_test)
    userPref = np.asarray(lemmatization(userIngredients), dtype=object)
    userInput = tfidf.transform(userPref)
    y_prediction = clf_model.predict(userInput)
    y_probability = clf_model.predict_proba(userInput)[0]
    cuisine_prediction = le.inverse_transform(y_prediction)[0]
    resultJson["cuisine"] = cuisine_prediction
    result = [i for i in zip(le.classes_, y_probability)]
    for index in range(0, len(result), 1):
        if result[index][0] == cuisine_prediction:
            resultJson["score"] = result[index][1]
    return resultJson


def nearestNeighbours(dataFrame, corpus, cuisine, tfidf, userIngredients, N, resultJson):
    knn = KNeighborsClassifier(n_neighbors=int(N))
    X_train, X_test, y_train, y_test = feature_Extraction_Train(tfidf, cuisine, lemmatization(corpus))
    knn.fit(X_train, y_train)
    user_ingredient_transform = tfidf.transform(userIngredients)
    dish_id_probability, dish_ids = knn.kneighbors(user_ingredient_transform, int(N))
    closest = []
    for i in range(len(dish_ids[0])):
        tempDict = {}
        tempDict["id"] = str(dataFrame.id[dish_ids[0][i]])
        tempDict["score"] = dish_id_probability[0][i]
        closest.append(tempDict)
    resultJson["closest"] = closest
    return resultJson


def exportResultJson(resultJson):
    sys.stdout.write(json.dumps(resultJson, indent=4))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help=" Specify the number of suggestions",  nargs="?", required=True)
    parser.add_argument("--ingredient", help="Specify the type of ingredients", action="append", nargs="+", required=True)
    args = parser.parse_args()
    predictor(args)
