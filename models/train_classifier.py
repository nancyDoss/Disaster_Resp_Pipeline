# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import pickle

from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])



def load_data(database_filepath):
    """
    loads the data to the dataframe from the database
    param: database file path
    
    return: X, Y and label

    """
 # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table("DisasterResponse", conn)
    X = df['message'] #only the message text, and not others.
    Y= df.drop(['id','genre','message','original'],axis=1)
    category_names = list(Y.columns) #Y should only contain category colunm
    return X, Y, category_names 


def tokenize(text):
    """
    This function performs NLP. tokenizing data
    param: text
    
    return:cleaned token

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    builds the pipeline, with the appropriate classifier
    param:None
    
    return: pipeline

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf',MultiOutputClassifier( RandomForestClassifier(bootstrap = bootstrap_p)))
        #('clf',MultiOutputClassifier( KNN_modelAdaBoostClassifier)) # accr = 0.73
       ('clf',MultiOutputClassifier( AdaBoostClassifier())) # accu = 0.75
    ])

    n_estimators = [int(x) for x in np.linspace(start = 5 , stop = 25, num = 10)] # returns 10 numbers 

    parameters = {'vect__min_df': [1, 5],
                  'clf__estimator__n_estimators':[10]}
    
    cv = GridSearchCV(pipeline, param_grid =parameters )
        
    return cv



def display_results(y_test, y_pred):
    """
    displays the label and accuarcy score
    
    param:y_test and y_pred
    
    return:none

    """
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)

    
def evaluate_model(model, X_test, Y_test, category_names):
    """
     This function evaluates the model, displays the scores.
     The f1 score, precision and recall for the test set is
     outputted for each category
     
    param: X, Y test values, tranined Model, label
    
    return: none

    """
    y_pred = model.predict(X_test)
    display_results(Y_test, y_pred)
    print("model score: %.3f" % model.score(X_test, Y_test))
    #Report the f1 score, precision and recall for each output category of the dataset
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    pickle fitted model
    param:model object
    model filepath
    
    return: None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
