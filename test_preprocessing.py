# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:31:41 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
import joblib
import data_preprocessing as dp
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import hstack
import joblib

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]



if __name__ == '__main__':
    print('test preprocessing started')
    test = pd.read_csv('./test.csv')
    submission = pd.read_csv('./sample_submission_UVKGLZE.csv')
    test_data = test.drop('ID',axis = 1)



    test_feature_clean = dp.data_cleaning(test_data)
    test_feature_clean.to_csv('./test_feature_clean.csv')

    title = test_feature_clean['TITLE'].values
    abstract = test_feature_clean['ABSTRACT'].values
    print("="*100)
    title_tfidf = dp.Tf_idf(title)
    abstract_tfidf = dp.Tf_idf(abstract)


    final_test_dataset = hstack((title_tfidf, abstract_tfidf)).tocsr()
    print(f"final_test_dataset:{final_test_dataset.shape}")
    joblib.dump(final_test_dataset,'./final_test_dataset.pkl')
