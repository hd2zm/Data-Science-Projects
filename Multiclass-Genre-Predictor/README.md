# Rock, Hip-Hop, or Electronic?

This project uses a variety of classification algorithms to predict whether a song is rock, hip-hop, or electronic. 
The final model (Grid Searched Random Forest) was uploaded to an AWS S3 bucket, which a web application downloaded. The web application checked if the song exists in Spotify, gathers its music data from Echonest API, inputs the dta into that model, and outputs the prediction of whether the song should be classified as Rock, Hip-Hop, or Electronic. 

## Files 

* FMA_Echonest_Data.ipynb - Download echonest data from Free Music Archive (FMA), a collection of free music. 
* Billboard_Echonest_Data.ipynb - Web scraped Billboard Year End Charts from 2003-2019 for Rock, Hip-Hop, and Electronic. Input data into Echonest API to get echonest data for respective songs.
* Logistic Regression Modeling.ipynb - Run different classification models to predict genres, using echonest data from both FMA and Billboard Year End Charts. 
* webapp/ - folder that contains a dockerized Flask application. User inputs song name and artist, and the app outputs which genre it things the song belongs to.

