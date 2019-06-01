#start with 3.7.2 version of python
FROM python:3.7.2

#ENV PYTHONUNBUFFERED 1

# Create directory for volume
RUN mkdir -p "genius_copy_data/"

COPY SongRecommender.py SongRecommender.py
COPY Rap_Lyrics_From_Different_Eras.csv Rap_Lyrics_From_Different_Eras.csv

# Copy over the serialized objects into the genius_copy_data directory that will be mounted as a volume
COPY lyrics.pkl genius_copy_data/lyrics.pkl
COPY csm.pkl genius_copy_data/csm.pkl

# Mount the directory as a volume
VOLUME [ "genius_copy_data/" ]

#update packages instlaled in the os
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran

#Copy the list of python packages to install
COPY requirements.txt requirements.txt

#Now install the python packages
RUN pip install -r requirements.txt

#Separately install nltk and dependencies 
RUN pip install -U nltk
RUN pip install -U numpy
RUN python -m nltk.downloader all
RUN pip install rake-nltk

#Give permissions to container to run main.py
RUN chmod a+x *.py

#Allow access on port 8000
EXPOSE 8000

#Run script
CMD ["sh","-c","./SongRecommender.py && tail -f /dev/null"]

