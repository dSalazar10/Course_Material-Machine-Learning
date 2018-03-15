#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:45:36 2018

@author: Daniel Salazar
"""

"""
This uses GridSearchCV to determine the most efficient combination of parameters for 
CountVectorizer, TfidfTransformer, and LogisticRegression
31104 fits done in 189207.729s

Best score: 0.419
Best parameters set:
	clf__C: 10
	clf__max_iter: 10
	clf__multi_class: 'multinomial'
	clf__penalty: 'l2'
	clf__solver: 'sag'
	tfidf__norm: 'l2'
	tfidf__use_idf: True
	vect__analyzer: 'word'
	vect__max_df: 0.75
	vect__max_features: None
	vect__ngram_range: (1, 2)
	vect__stop_words: None
"""

import pandas as pd
import numpy as np
from pprint import pprint
from time import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def merge_df(df_):
    df[['year','rating']] = df[['year','rating']].astype(str)
    # extract the class column
    target_ = df_["genre"].copy()
    # delete the column from the dataset
    df_ = df_.drop(columns = ["genre"])
    # merge all columns
    columnTitles=['title','director','actor1','actor2','year','rating']
    df_ = df_[columnTitles].apply(lambda x: ' '.join(x), axis=1)
    df_ = pd.concat([df_, target_], axis=1)
    df_.columns=['Movie', 'Genre']
    return df_

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())])

parameters = {
    # CountVectorizer
    'vect__analyzer': ['word'],
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__stop_words' : ('english', None),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    
    # TfidfTransformer
    'tfidf__norm': ('l1','l2'),
    'tfidf__use_idf': (True, False),
    
    # LogisticRegression
    'clf__penalty': ['l2'],
    'clf__C': (0.001,0.01,0.1,1,10,100),
    'clf__solver': ('newton-cg', 'lbfgs', 'sag'),
    'clf__max_iter': (10, 50, 80),
    'clf__multi_class': ['multinomial']
}

def train_model(df_):
    df_ = merge_df(df_)
    X_train, X_test, y_train, y_test = train_test_split(df_['Movie'].values, df_['Genre'].values, test_size = 0.25)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=6, verbose=1)
    pipeline.fit(X_train, y_train)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

df = pd.read_csv('clean_data.csv')
train_model(df)

"""
done in 189207.729s

Best score: 0.419
Best parameters set:
	clf__C: 10
	clf__max_iter: 10
	clf__multi_class: 'multinomial'
	clf__penalty: 'l2'
	clf__solver: 'sag'
	tfidf__norm: 'l2'
	tfidf__use_idf: True
	vect__analyzer: 'word'
	vect__max_df: 0.75
	vect__max_features: None
	vect__ngram_range: (1, 2)
	vect__stop_words: None
"""

