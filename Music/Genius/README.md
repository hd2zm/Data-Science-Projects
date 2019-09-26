# Genius

Genius uses data from billboard top 100 hip hop from 3 different eras: 1980-2000, 2000-2010, and 2010-present. You input a top 100 hip hop song into the Flask app and it outputs 10 songs that are similar to that song (using cosine similarity). 

## Files
* GeniusParser.py - Web scrapes song lyrics from the 3 different eras using Spotify and Genius.
* LyricsAnalysis.py - Using NLTK and Rake to extract key words to feature in word clouds. Also creating classification models to predict which era a song is in based on its lyrics.
* SongRecommender.py - Uses cosine similarity to recommend top 10 songs based on input.
* SongRecommenderApi.py - Flask app that calls methods in GeniusParser and SongRecommender to make recommendations. 
