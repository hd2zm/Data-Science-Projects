#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:32:26 2019

@author: hdeva

SEE FOR REFERENCE: https://github.com/emmagrimaldi/Content_based_movie_recommender/blob/master/EG_project5.ipynb

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake

import joblib
import pandas as pd

def convert_lyrics_to_keywords(lyrics_df):
     # assigning the key words to the new column
    lyrics_df['Key_Words'] = ""
      
    for index, row in lyrics_df.iterrows():
        lyric = row['Lyrics']
        
        # instantiating Rake, by default is uses english stopwords from NLTK
        # and discard all puntuation characters
        r = Rake()

        # extracting the words by passing the text
        r.extract_keywords_from_text(lyric)

        # getting the dictionary whith key words and their scores
        key_words_dict_scores = r.get_word_degrees()
    
        # assigning the key words to the new column
        row['Key_Words'] = ' '.join(list(key_words_dict_scores.keys()))
        lyrics_df.at[index, 'Key_Words'] = row['Key_Words']


    lyrics_df.drop(columns = ['Lyrics'], inplace = True)
    
    # merging together hip hop artist names to treat as unique values
    lyrics_df['Artists_Lower'] = lyrics_df['Artists'].map(lambda x: x.split(' '))
    for index, row in lyrics_df.iterrows():
        row['Artists_Lower'] = ''.join(row['Artists_Lower']).lower()
        lyrics_df.at[index, 'Artists_Lower'] = row['Artists_Lower']

    lyrics_df['Key_Words'] = lyrics_df['Artists_Lower'] + ' ' + lyrics_df['Key_Words']
    lyrics_df.drop(columns = ['Artists_Lower'], inplace = True)
    
    return lyrics_df


def clean_data_frame(lyrics_df):
    #Remove duplicate values and where songs = artists (errors from parsing)
    lyrics_df = lyrics_df.drop_duplicates(subset=['Songs', 'Artists'], keep='first')
    lyrics_df = lyrics_df.drop(lyrics_df[lyrics_df['Songs'] == lyrics_df['Artists']].index)
    
    #Reset index to songs
    lyrics_df.set_index('Songs', inplace = True)
    lyrics_df.drop(columns = ['Unnamed: 0'], inplace = True) 
    
    return lyrics_df


def get_cosine_sim_matrix(lyrics_df):
    # instantiating and generating the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(lyrics_df['Key_Words'])
    
    # generating the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix) 
    
    return cosine_sim


def get_recommended_songs(lyrics_df, cosine_sim, song):  
    recommended_songs = []
    
    # creating a Series for the song titles so they are associated to an ordered numerical
    indices = pd.DataFrame(lyrics_df.index, lyrics_df['Artists'])
    indices = indices.reset_index()
    
    # gettin the index of the song that matches the title
    idx = indices[indices['Songs'] == song].index[0]
    
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar songs
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching songs
    for i in top_10_indexes:
        recommended_songs.append(list(lyrics_df['Artists'])[i] + ': ' + list(lyrics_df.index)[i] + " " + list(lyrics_df['Era'])[i])
    
    return recommended_songs

def main():

    lyrics_df = pd.read_csv("Rap_Lyrics_From_Different_Eras.csv")
    lyrics_df = convert_lyrics_to_keywords(lyrics_df)
    lyrics_df = clean_data_frame(lyrics_df)
    cosine_sim_matrix = get_cosine_sim_matrix(lyrics_df)
    
    joblib.dump(lyrics_df, 'lyrics.pkl')
    print('Exported Lyrics to pkl format')
    joblib.dump(cosine_sim_matrix, 'csm.pkl')
    print('Exported Cosine Similarity Matrix to pkl format')
    
if __name__ == '__main__':
    main()