#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:32:26 2019

@author: hdeva
"""

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#import xgboost, numpy, textblob, string
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
import pandas as pd

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, valid_y)


def main():

    
    lyrics_df = pd.read_csv("Rap_Lyrics_From_Different_Eras.csv")
    
    lyrics_df_2010 = lyrics_df.loc[lyrics_df['Era'] == '2010-2020']
    lyrics_df_2000 = lyrics_df.loc[lyrics_df['Era'] == '2000-2010']
    lyrics_df_1980 = lyrics_df.loc[lyrics_df['Era'] == '1980-2000']
    
    print(lyrics_df_2010.head(10))
    print(lyrics_df_2000.head(10))
    print(lyrics_df_1980.head(10))
    
    
    wordcloud = WordCloud(
            width = 1000,
            height = 800,
            background_color = 'black',
            stopwords = STOPWORDS).generate(str(lyrics_df_2010.values))
    
    fig = plt.figure(
            figsize = (40, 30),
            facecolor = 'k',
            edgecolor = 'k')
    
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(lyrics_df['Lyrics'], lyrics_df['Era'])
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
    count_vect.fit(lyrics_df['Lyrics'])
    
    # Transforming our x_train data using our fit cvec.
    # And converting the result to a DataFrame.
    X_train = pd.DataFrame(count_vect.transform(lyrics_df['Lyrics']).todense(),
                       columns=count_vect.get_feature_names())
    
    # Which words appear the most?
    word_counts = X_train.sum(axis=0)
    print(word_counts.sort_values(ascending = False).head(20))

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    
    print(xtrain_count)
    print(xvalid_count)
    
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(lyrics_df['Lyrics'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    
    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(lyrics_df['Lyrics'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, valid_y)
    print("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
    print("NB, WordLevel TF-IDF: ", accuracy)

    # SVM on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
    print("SVM, N-Gram Vectors: ", accuracy)
    
    
    
    # Fitting Naive Bayes to the Training set
    classifier = naive_bayes.MultinomialNB()
    classifier.fit(xtrain_count, train_y)
    predictions = classifier.predict(xvalid_count)
    print(predictions)

    
if __name__ == '__main__':
    main()