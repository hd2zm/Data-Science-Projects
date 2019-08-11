from flask import Flask, request, render_template, json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import boto
from boto.s3.key import Key
from boto.s3.connection import S3Connection
import os.path


# Constant variables
MODEL_FILE_NAME = 'finalized_model.p'
MODEL_LOCAL_PATH = MODEL_FILE_NAME

# create a flask object
sp = None
model = None
app = Flask(__name__)

# initiate spotify credentials
def initiate_spotify_credentials():
    global sp

    cid ="403359461f0b4827adf4b8c3b6dd87d9" 
    secret = "f4837ee0941b4a769c345cc4b7b98156" 
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
    sp.trace=False 

# load model
def load_model():
    global model

    # download model from aws bucket if it doesn't exist in local path
    if not os.path.isfile(MODEL_LOCAL_PATH):

        AWS_ACCESS_KEY_ID = 'AKIAJNCGPD6VW6G6Q47A'
        AWS_SECRET_ACCESS_KEY = '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'
        REGION_HOST = 's3.us-east-2.amazonaws.com'
        BUCKET_NAME = 'music-genre-recommender-model'

        conn = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, host=REGION_HOST)
        bucket = conn.get_bucket(BUCKET_NAME)

        key_obj = Key(bucket)
        key_obj.key = MODEL_FILE_NAME

        key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
    
    model = joblib.load(MODEL_LOCAL_PATH)

# select genre
def select_genre(track_data):

    input_df = [track_data]

    # make a prediction
    prediction = model.predict(input_df)[0]

    #predition message
    message_array = ["Your song is Electronic",
                     "Your song is Hip-Hop",
                     "Your song is Rock"]
    
    return message_array[prediction]

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/predict_genre_song/', methods=['GET', 'POST'])
def render_message_song():

    artist = request.form['artist']
    song = request.form['song']

    song_parsed = song.replace("'", "")
    q = 'artist:' + artist + ' track:' + song_parsed
    q = q.encode(encoding='UTF-8',errors='strict')
    track_search = sp.search(q=q, type="track", limit=1)
    track_items = track_search['tracks']['items']
    if len(track_items) > 0:
        audio_features = sp.audio_features(track_items[0]['id'])
        music_data_categories = ['danceability', 'energy', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness', 'valence', 'tempo']  
        key_audio_features = { key: audio_features[0][key] for key in music_data_categories }
        final_message = select_genre(list(key_audio_features.values()))
    else:
        final_message = 'Cannot find track listing for ' + artist + '-' + song

    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    initiate_spotify_credentials()
    load_model()
    app.run(host='0.0.0.0', port=80)