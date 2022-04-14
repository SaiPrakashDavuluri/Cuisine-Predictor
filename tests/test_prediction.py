import pandas
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from project2 import readJson, lemmatization, feature_Extraction_Train, labelEncoding, predictionModel, \
    nearestNeighbours

corpus = [
        'plain flour sugar butter eggs fresh ginger root salt ground cinnamon milk vanilla extract ground ginger powdered sugar baking powder',
        'roma tomatoes kosher salt purple onion jalapeno chilies lime chopped cilantro',
        'olive oil bread slices great northern beans garlic cloves pepper shrimp sage leaves salt',
        'melted butter matcha green tea powder white sugar milk all-purpose flour eggs salt baking powder chopped walnuts']


def test_readJson():
    fileName = 'yummly_test.json'
    cuisine, corpus, dataFrame = readJson(fileName)
    assert type(cuisine) == list
    assert type(corpus) == list
    assert type(dataFrame) == pandas.core.frame.DataFrame
    assert cuisine is not None
    assert corpus is not None
    assert dataFrame.ingredients is not None
    assert dataFrame.cuisine is not None


def test_lemmatization():
    lemmaSentences = lemmatization(corpus)
    assert type(lemmaSentences) == list
    assert lemmaSentences is not None
    for sentence in lemmaSentences:
        assert type(sentence) == str
        assert sentence is not None


def test_feature_Extraction_Train():
    fileName = 'yummly_test.json'
    cuisine, ingredients, dataFrame = readJson(fileName)
    lemmaSentences = lemmatization(ingredients)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train, X_test, y_train, y_test = feature_Extraction_Train(tfidf, cuisine, lemmaSentences)
    assert type(X_train) == scipy.sparse._csr.csr_matrix
    assert type(X_test) == scipy.sparse._csr.csr_matrix
    assert type(y_train) == list
    assert type(y_test) == list
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None


def test_labelEncoding():
    fileName = 'yummly_test.json'
    cuisine, ingredients, dataFrame = readJson(fileName)
    lemmaSentences = lemmatization(ingredients)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train, X_test, y_train, y_test = feature_Extraction_Train(tfidf, cuisine, lemmaSentences)
    le, Y_train, Y_test = labelEncoding(cuisine, y_train, y_test)
    assert type(y_train) == list
    assert type(y_test) == list
    assert y_train is not None
    assert y_test is not None
    assert type(le) == sklearn.preprocessing._label.LabelEncoder
    assert le is not None


def test_Prediction_Model():
    userIngredients = [
      "brown sugar",
      "asian fish sauce",
      "thai chile",
      "green papaya",
      "fresh lime juice"
    ]
    resultJson = {}
    fileName = 'yummly_test.json'
    cuisine, ingredients, dataFrame = readJson(fileName)
    lemmaSentences = lemmatization(ingredients)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train, X_test, y_train, y_test = feature_Extraction_Train(tfidf, cuisine, lemmaSentences)
    le, Y_train, Y_test = labelEncoding(cuisine, y_train, y_test)
    resultJson = predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson)
    assert type(resultJson) == dict
    assert resultJson is not None
    assert resultJson["cuisine"] is not None
    assert resultJson["score"] is not None


def test_nearestNeighbours():
    userIngredients = [
        "brown sugar",
        "asian fish sauce",
        "thai chile",
        "green papaya",
        "fresh lime juice"
    ]
    resultJson = {}
    N = 5
    fileName = 'yummly_test.json'
    cuisine, ingredients, dataFrame = readJson(fileName)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    resultJson = nearestNeighbours(dataFrame, ingredients, cuisine, tfidf, userIngredients, N, resultJson)
    assert type(resultJson) == dict
    assert resultJson is not None
    assert resultJson["closest"] is not None
    assert type(resultJson["closest"]) == list
