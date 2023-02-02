### Goal
The aim of the project is to collect ingredients from user and create two models to predict the cuisine that can be prepared by the passed ingredients and closest cuisines.

### How to run the application

#### steps:
1) First clone the project using : git clone URL
2) check to see if there is a directory named as 'cs5293sp22-project2' using ls
3) open the directory using cd cs5293sp22-project2
4) Run the below command to start the project:

pipenv run python project2.py --N 5 --ingredient "chile pepper" --ingredient "salt" --ingredient "ground cumin" --ingredient "diced tomatoes" --ingredient "ground black pepper" --ingredient "onions" 

#### WEB OR EXTERNAL LIBRARIES:

1) pandas
2) numpy
3) sklearn
4) joblib
5) nltk

#### INSTALLATION OF ABOVE LIBRARIES:
1) pipenv install pandas
2) pipenv install numpy
3) pipenv install scikit-learn
4) pipenv install joblib
5) pipenv install nltk

### Assumptions

* There is a possibility that defined models might return few discrepancies for the predicted cuisine when user provides ingredients which are not present in the yummly.json file.
* I haven't saved the model because I tried adding the user ingredients which are not available in the json file to the model for training purpose. But haven't succeeded in doing so.

### metrics for the machine learning models:

* I have used below code to display the metrics:


                from sklearn.metrics import precision_score, recall_score, f1_score
                print("precision:", precision_score(Y_test, y_prob, average='macro'))
                print("recall score:", recall_score(Y_test, y_prob, average='macro'))
                print("f1_score:", f1_score(Y_test, y_prob, average='macro'))

                output:
                precision: 0.7756477286741336
                recall score: 0.6923521672201868
                f1_score: 0.7267156133460182


### Reason for choosing these models for my project:

* I have tested below models and came to the conclusion that linearSVC and KNeighborsClassifier classifier has better accuracy and results.
     
                   
                   Models Tested:
                   LogisticRegression
                   Multinomial Naive Bayes
                   RandomForestClassifier
                   DecisionTreeClassifier


### Functions and approach

predictor(args) function:

* This function acts as base layer for remaining functions. It takes user provided arguments called args and first flattens list of 'args.ingredient' to store the result in 'userIngredients'.
* I have initialized a dictionary called resultJson to store the predicted cuisine, similarity score, the closest id's and their similarity scores.
* I have statically defined the file name as yummly.json and storing the name in filename variable.
* Sending the filename to readJson() function to convert the data in the json file to a dataframe which contains three columns called id, cuisine, Ingredients. At the same time, I am storing the values of cuisine in the dataframe to cuisine list and the values of ingredients to corpus list.
* The corpus is then passed to lemmatization(corpus) function to lemmatize the ingredients.
* I have initialized the TfidfVectorizer() in this function. So, that I can pass it as a parameter to other functions when required.
* Next, I am calling feature_Extraction_Train(tfidf, cuisine, _sentences) method to split the data into train and test.
* To do label encoding for the y_train and y_test, I have created a function called labelEncoding(cuisine, y_train, y_test).
* For the first model to predict the cuisine and score, I have defined a function called predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson) which returns the predicted cuisine and score.
* To find the nearest neighbours, I have defined a function called nearestNeighbours(dataFrame, corpus, cuisine, tfidf, userIngredients, args.N, resultJson) which returns the closest cuisine's id and similarity score.
* To print the final output as json format, I have defined a function called exportResultJson(resultJson) which prints the output to command line.

readJson(fileName):

* It takes the filename as parameter, using pandas library converts the json file to dataframe. This newly created dataframe has 3 columns called id, cuisine, ingredients.
* I have initialized two lists called cuisine and corpus to hold the dataframe column values of cuisine and ingredients.
* I have used for loops to iterate the dataframe columns and stored them in the list.
* Finally, returned the cuisine, corpus, and dataframe as output.


lemmatization(corpus):

* This function takes corpus as input parameter and iterates through corpus to lemmatize the words.
* I have used word_tokenize() method from nltk library to break down the sentences into words and pass them to lemma.lemmatize() method to lemmatize every word.
* Again, joining the words in a list together using ' '.join() and finally storing every value as string in the _sentences list.


feature_Extraction_Train(tfidf, cuisine, _sentences):

* I have used TfidfVectorizer to convert documents into matrix of TF-IDF features.
* This function takes the cuisine and _sentences and splits the data into specified test size. For example, if test_size was given as 0.2 then train set size will be 80% and test size set will be 20%.
* First, to train data and transform features, I am using fit_transform() method. So, the model can learn the mean and variance of the features of the training set.
* This method returns train and test data separately for cuisine and ingredients.

labelEncoding(cuisine, y_train, y_test):

* This function takes cuisine, y_train, and y_test parameters and does label encoding for train and test data of y.
* We do label encoding to convert categorical data into numerical format. I have used fit() and transform() methods to apply the functionality of label encoder to the y train and test data.
* Finally, this function returns le which is label encoder along with y_train and y_test.

predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson):

* In this method, I am using pre-defined machine learning model to predict the cuisine and the similarity score.
* I have defined LinearSVC() model then to predict the cuisine and probability scores, I am passing the linearSVC model to CalibratedClassifierCV().
* I have used fit() method to fit the model with X_train and Y_train data to train the model. I am passing X_test to predict() method to test the trained data. By doing this, I am training the testing the model.
* To actually predict the cuisine from the user provided ingredients, I am first lemmatizing them and converting them into nparray.
* Using TfidfVectorizer() and transform() method, I am transforming the user provided ingredients into features. 
* When it comes to predicting the cuisine and similarity score using 'clf_model.predict(userInput)' and 'clf_model.predict_proba(userInput)[0]'.
* Finally, I am storing the predicted cuisine and score to a dictionary.

nearestNeighbours(dataFrame, corpus, cuisine, tfidf, userIngredients, N, resultJson):

* To predict the nearest neighbours, I am using KNeighborsClassifier. I am using the feature_Extraction_Train() function to split ingredients and cuisine into  train and test data.
* Using fit() method, to fit the model with X_train and Y_train data to train the model. I am passing X_test to predict() method to test the trained data. By doing this, I am training the testing the model.
* By utilizing knn.kneighbors(user_ingredient_transform, int(N)) method, I am retrieving row numbers.
* Using for loop, I am iterating the row numbers and storing the ids in the dataFrame to closestIDs list and applying the same logic to store the ingredients to closestIngredients list.
* To convert user provided ingredients into features, I am first iterating them and append the values to a string and storing them as document in a list.
* To find the similarity score between the user provided ingredients and nearest neighbours, I am using cosine similarity.
* Before passing them to cosine_similarity(frequency, y), I am applying the fit_transform() method on nearest neighbours and converting them to features to send as input parameter.
* After converting the user provided ingredients to list, I am applying transform() method to convert into features.
* Finally, finding out the similarity score using cosine_similarity() method. Using for loop, I am storing the results that we secured to the resultJson dictionary.

exportResultJson(resultJson):

* Using sys.stdout.write() method, I am printing the final output to command line.

### Test cases

* Test cases to test above functionalities are available in tests folder.
* Command to run test cases: pipenv run python -m pytest
* I have taken sample json file called yummly_test.json which is a smaller version of yummly.json.

test_readJson():

* I have predefined the filename and passing it to readJson(fileName) method to receive outputs for cuisine, corpus, and dataFrame objects.
* I am using assert statements to check the type and length of the above variables and also to see if they are not none.

test_lemmatization():

* I have predefined the corpus as class level variable and passing it to lemmatization(corpus) method to receive output as lemmaSentences which is list of sentences.
* I am using assert statements to check the type and length of the lemmaSentences list and also to see if it is not none.
* Using for loop, I have used assert statements to check the type of values in the list.


test_feature_Extraction_Train():

* For this test function, to prepare variables to pass as input parameters for feature_Extraction_Train(tfidf, cuisine, lemmaSentences) function I am making function calls to readJson(fileName), lemmatization(ingredients).
* I also initalizing TfidfVectorizer(ngram_range=(1, 2), stop_words="english") to pass as input parameter.
* Finally, using assert statements I am checking the type and length of the above variables and also to see if they are not none.


test_labelEncoding():

* To test this function, to prepare variables to pass as input parameters for labelEncoding(cuisine, y_train, y_test) function I am making function calls to readJson(fileName), lemmatization(ingredients), and feature_Extraction_Train(tfidf, cuisine, lemmaSentences).
* I also initalizing TfidfVectorizer(ngram_range=(1, 2), stop_words="english") to pass as input parameter.
* Finally, using assert statements I am checking the type and length of the above variables and also to see if they are not none.

test_Prediction_Model():

* In this function, I am pre-defining the user provided ingredients to pass them as input to this function predictionModel(X_train, X_test, Y_train, Y_test, userIngredients, tfidf, le, resultJson).
* To gather the input parameters to above function, I am making function calls to readJson(fileName), lemmatization(ingredients), feature_Extraction_Train(tfidf, cuisine, lemmaSentences), and labelEncoding(cuisine, y_train, y_test).
* Finally, after getting output in variables, I am using assert statements, I am checking the type and length of the above variables and also to see if they are not none.


test_nearestNeighbours():

* In this function, I am pre-defining the file name as fileName, user provided ingredients and 'N' variable as 5 to pass them as input to this function nearestNeighbours(dataFrame, ingredients, cuisine, tfidf, userIngredients, N, resultJson).
* I have made function call to readJson(fileName) To gather the input parameters for above function.
* Finally, after getting output in variables, I am using assert statements, I am checking the type and length of the above variables and also to see if they are not none.


### GitHub:
The above-mentioned files need to be added, committed, and pushed to GitHub repository by using the following commands.

git add file-name;

git commit -m "commit message"

git push origin main
