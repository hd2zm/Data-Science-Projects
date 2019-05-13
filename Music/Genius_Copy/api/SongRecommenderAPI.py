#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:06:39 2019

@author: hdeva
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to the hip hop recommender system!"

@app.route("/recommend/<string:song>")
def get_recommended_songs(song):
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    lyrics_df = joblib.load('lyrics.pkl')
    cosine_sim = joblib.load('csm.pkl')
    try:
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
    
        return str(recommended_songs)
    except:
        return "Song is not a #1 hip hop single. Please try again."

'''
https://www.datacamp.com/community/tutorials/machine-learning-models-api-python

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        json_ = request.json
        print(json_)

        prediction = list(lr.predict(query))

        return jsonify({'prediction': str(prediction)})

    except:

        return jsonify({'trace': traceback.format_exc()})
'''

if __name__ == '__main__':
    app.run(host="0.0.0.0")
    #app.run(debug=True)