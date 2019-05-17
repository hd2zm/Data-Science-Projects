#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:06:39 2019

@author: hdeva
"""

from flask import Flask, request, jsonify
from sklearn.externals import joblib
import SongRecommender

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
    lyrics = joblib.load('lyrics.pkl')
    csm = joblib.load('csm.pkl')
    try:
        return str(SongRecommender.get_recommended_songs(lyrics, csm, song))
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