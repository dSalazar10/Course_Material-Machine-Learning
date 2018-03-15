#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:45:36 2018

@author: Daniel Salazar
"""

"""
Now that I have my dataset ready and my arguments figured out, it is time for 
data cleaning and training This program will model the count of words in the 
dataset. Unfortunately, the dataset is way too small to be remotely useful, but
I managed to improve my Logistic Regression model from 36% accuracy to upwards 
of 48% depending on the RNG. I heard from other students that they were able 
to get predictions over 70%. In comparison, I feel that this model choice was 
not as efficient as I had hoped. 
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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

pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),max_df=0.75,analyzer='word')),
                     ('tfidf', TfidfTransformer(use_idf=True,norm='l2')),
                     ('clf', LogisticRegression(solver='sag',penalty='l2',multi_class='multinomial',max_iter=1000,C=10))])

def train_model(df_):
    df_ = merge_df(df_)
    X_train, X_test, y_train, y_test = train_test_split(df_['Movie'].values, df_['Genre'].values, test_size = 0.25)
    text_clf = pipeline.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    return np.mean(predicted == y_test)

# remove outliers
df = pd.read_csv('clean_data.csv',index_col=0)

# directors
director_counts = df.director.value_counts()
threshold = 2
remove_directors = director_counts[director_counts < threshold].index
df = df[~df['director'].isin(remove_directors)]
df['director'].replace('None', np.nan, inplace=True)
df['director'].loc[df['director'].isnull()] = df['director'].loc[df['director'].isnull()].apply(lambda x: random.choice(remove_directors))

# years
year_counts = df.year.value_counts()
threshold = 200
remove_years = year_counts[year_counts < threshold].index
df = df[~df['year'].isin(remove_years)]

# ratings
rating_counts = df.rating.value_counts()
threshold = 10
remove_ratings = rating_counts[rating_counts < threshold].index
df = df[~df['rating'].isin(remove_directors)]

# genres
genres = ['Action','Adventure','Sci-Fi','Fantasy','Drama','Comedy','Horror','Thriller','Documentary',
          'Romance','Animation','Biography','Family','Crime','Western','Adult','Mystery','Musical']
df = df[df['genre'].isin(genres)]

train_model(df)

"""
0.46241276171485546
"""
