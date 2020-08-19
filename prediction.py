# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:44:24 2020

@author: Alok Garg
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


def predict(data,model):
    predicted = model.predict(data)
    return predicted





if __name__ == '__main__':
    print('prediction started')
    submission = pd.read_csv('./sample_submission_UVKGLZE.csv')
    test_feature_clean = pd.read_csv('./test_feature_clean.csv')
    cols = submission.columns
    test_data = joblib.load('./test_vector.pkl')

    model_computer_science = joblib.load('./model_computer_science.pkl')
    model_Mathematics = joblib.load('./model_Mathematics.pkl')
    model_Physics = joblib.load('./model_Physics.pkl')
    model_Quantitative_Biology = joblib.load('./model_Quantitative_Biology.pkl')
    model_Quantitative_Finance = joblib.load('./model_Quantitative_Finance.pkl')
    model_Statistics = joblib.load('./model_Statistics.pkl')

    pred_peds = predict(test_data,model_computer_science)

    submission['Computer Science'] = np.array(predict(test_data,model_computer_science))
    submission['Physics'] = np.array(predict(test_data,model_Physics))
    submission['Mathematics'] = np.array(predict(test_data,model_Mathematics))
    submission['Statistics'] = np.array(predict(test_data,model_Statistics))
    submission['Quantitative Biology'] = np.array(predict(test_data,model_Quantitative_Biology))
    submission['Quantitative Finance'] = np.array(predict(test_data,model_Quantitative_Finance))

    submission.to_csv('./submission.csv',index= False)