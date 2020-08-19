# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:13:52 2020

@author: Alok Garg
"""
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def training(X,target):
    x_train,x_test,y_train,y_test = train_test_split(X, target, random_state=42, test_size=0.33,stratify= target)
    # clf = LogisticRegression(n_jobs=-1,verbose=4)
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    score = accuracy_score(y_test, pred)
    # estimator = clf.best_estimator_
    # print(estimator)
    return pred,score,clf
def training_fullData(X,target):
    clf = GaussianNB()
    clf.fit(X,target)
    return clf




if __name__ == "__main__":
    print('training the model')
    train = pd.read_csv('./train.csv')
    target = train.loc[:,['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']]
    final_dataset = joblib.load('./final_dataset.pkl')
    final_dataset = final_dataset.toarray()
    # computer_science = np.array(target['Computer Science']).reshape(-1,1)
    # Physics = np.array(target['Physics']).reshape(-1,1)
    Mathematics = np.array(target['Mathematics']).reshape(-1,1)
    # Statistics = np.array(target['Statistics']).reshape(-1,1)
    # Quantitative_Biology = np.array(target['Quantitative Biology']).reshape(-1,1)
    # Quantitative_Finance = np.array(target['Quantitative Finance']).reshape(-1,1)



    # pred_Mathematics,score_Mathematics,clf = training(final_dataset,Mathematics)
    # joblib.dump(clf,'./jantahack.pkl')
    # print(score)
    # model_computer_science = training_fullData(final_dataset,computer_science)
    # joblib.dump(model_computer_science,'./model_computer_science.pkl')

    # model_Physics = training_fullData(final_dataset,Physics)
    # joblib.dump(model_Physics,'./model_Physics.pkl')

    model_Mathematics = training_fullData(final_dataset,Mathematics)
    joblib.dump(model_Mathematics,'./model_Mathematics.pkl')

    # model_Statistics = training_fullData(final_dataset,Statistics)
    # joblib.dump(model_Statistics,'./model_Statistics.pkl')

    # model_Quantitative_Biology = training_fullData(final_dataset,Quantitative_Biology)
    # joblib.dump(model_Quantitative_Biology,'./model_Quantitative_Biology.pkl')

    # model_Quantitative_Finance = training_fullData(final_dataset,Quantitative_Finance)
    # joblib.dump(model_Quantitative_Finance,'./model_Quantitative_Finance.pkl')