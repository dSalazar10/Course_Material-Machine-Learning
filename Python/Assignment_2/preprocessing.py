#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:45:36 2018

@author: Daniel Salazar
"""
"""
This is the preprocessing portion:
- read the dataset 'usa-training.json'
- removes the following columns: rating, year, imdb, country, director
- reorganizes the columns logically
- removes any rows that have a genre which doesn't match the allowable types
- removes any rows that have a country other than USA
- separate the genre column
- isolate only year in releasedate
- removes any symbols from the dataset
- join all the columns into one
- concatenate labels back into dataframe
- saves the test and train sets to csv for next stage
"""

import pandas as pd
import re
import unicodedata

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 60)

#read json into dataframe
df = pd.read_csv('usa-training.csv')

# filter out letters from string
def clean_feature(raw_feature):
    if isinstance(raw_feature, unicode):
        raw_feature = unicodedata.normalize('NFKD', raw_feature).encode('ascii','ignore')
    letters_only = re.sub(u"[^a-zA-Z0-9 ]", "", raw_feature)
    words = letters_only.lower().split()
    clean_feature = [w for w in words]
    return(" ".join(clean_feature))

# save dataframe to csv
def save_df(df_,filename):
    # write to csv
    df_.to_csv(filename, encoding='utf-8')
    
# remove country and releasedate columns
df = df.drop(columns = ['country', 'releasedate'])
# convert numbers to string
df[['year','rating']] = df[['year','rating']].astype(str)
# remove all symbols
for col in df.columns:
    df[col].apply(lambda x: clean_feature(x))
# reorganize dataframe
columnTitles=['title','director','actor1','actor2','year','rating','genre']
df = df.reindex(columns=columnTitles)
# save dataframe
save_df(df,'clean_data.csv')
