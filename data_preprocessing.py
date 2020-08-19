# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:44:34 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
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

def decontracted(phrase):
    phrase = [word for word in phrase.split() if len(word) > 3]
    phrase = ' '.join(phrase)
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = phrase.replace('\\r', ' ')
    phrase = phrase.replace('\\n', ' ')
    phrase = phrase.replace('\\"', ' ')
    phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase)
    phrase= ' '.join(e for e in phrase.split() if e.lower() not in stopwords)
    return phrase

def data_cleaning(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x : decontracted(str(x)))
    return df


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    stems = ' '.join(stems)
    return stems

def tokenized_data(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: tokenize_and_stem(x))
    return df

def Tf_idf(text):
    vectorizer = TfidfVectorizer(min_df = 20,max_features = 5000)
    text_tfidf = vectorizer.fit_transform(text)
    print("Shape of matrix after one hot encodig ",text_tfidf.shape)
    return text_tfidf



if __name__ == '__main__':
    print('working on jantahack')
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    cols = train.columns
    target = train.loc[:,['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']]
    feature = train.drop(['ID','Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance'],axis = 1)
    feature = feature.append(test)
    feature_clean = data_cleaning(feature)

    title = feature_clean['TITLE'].values
    abstract = feature_clean['ABSTRACT'].values
    print("="*100)
    title_tfidf = Tf_idf(title)
    abstract_tfidf = Tf_idf(abstract)


    final_dataset = hstack((title_tfidf, abstract_tfidf)).tocsr()
    joblib.dump(final_dataset,'./final_dataset.pkl')

