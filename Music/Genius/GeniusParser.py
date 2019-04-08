#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:48:42 2019

@author: hdeva
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

defaults = {
    'request': {
        'token': 'G6O3jp-KNiWJd5iKakg9AaT18JFEeCjQTxFIa0U6b-GRu-laAh3Sl2ZeAFqkiiWb',
        'base_url': 'https://api.genius.com'
    },
    'message': {
        'search_fail': 'The lyrics for this song were not found!',
        'wrong_input': 'Wrong number of arguments.\n' \
                       'Use two parameters to perform a custom search ' \
                       'or none to get the song currently playing on Spotify.'
    },
    'client': {
         'client_id':'Pbmyj-VntXPsngM83j1ztSpN261bofh8jYlU6rHQtIMMc_3Y32i2xDTep7L4-LSc',
         'client_secret':'kBPmMm_91_gkUclM9TFXXq0lIBf7XC6zJjhdlZPAYJidFcuEMVVcId2ltnggBOZqiIHHUiicqhMkrLOdKnvT2Q'       
    }
}

def request_song_info(song_title, artist_name):
    base_url = defaults['request']['base_url']
    headers = {'Authorization': 'Bearer ' + defaults['request']['token']}
    search_url = base_url + '/search'
    data = {'q': song_title + ' ' + artist_name}
    #data = {'q': artist_name}
    #data = {'q': song_title}
    response = requests.get(search_url, data=data, headers=headers)

    return response

def scrap_song_url(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    [h.extract() for h in html('script')]
    lyrics = html.find('div', class_='lyrics').get_text()
    lyrics = re.sub('\[.*?\]', '', str(lyrics))

    return lyrics

def get_lyrics(songs_dict, decade_range):
    
    lyrics = []
    remote_song_info = None
    
    print('Parsing lyrics in range ' + decade_range)
    for artist, songs in songs_dict.items():
        for song in songs:
            response = request_song_info(song, artist)
            json = response.json()
            
            for hit in json['response']['hits']:
                if artist.lower() in hit['result']['primary_artist']['name'].lower():
                    remote_song_info = hit
                    break           
            
            if remote_song_info:
                song_url = remote_song_info['result']['url']
                lyrics.append(scrap_song_url(song_url))
            else:
                print('The lyrics for song ' + song + ' with artist ' + artist + ' were not found')
    print('Completed parsing in range ' + decade_range)
    return lyrics

#List of NBA All Stars
def get_song_data(url):

    song_data = {}
    
    song_soup = BeautifulSoup(requests.get(url).content)
    
    song_table = song_soup.find('table',{'class': 'plainrowheaders sortable wikitable'})
    
    rows = song_table.findAll('tr')
    
    for tr in rows:
        cols = tr.findAll('td')
        if len(cols) > 0:
            if cols[0].find('a'):
                key = cols[0].find('a').find(text=True)
                value = tr.find('a').find(text=True)
                if key not in song_data.keys():
                    song_data[key] = [value]
                else:
                    if value not in song_data[key]:
                        song_data[key].append(value)
    
    #song_df = pd.DataFrame({'2010': song_data})
    
    return song_data

def main():


    
    song_data_2010 = get_song_data('https://en.wikipedia.org/wiki/List_of_Billboard_Hot_Rap_Songs_number-one_songs_of_the_2010s')
    song_data_2000 = get_song_data('https://en.wikipedia.org/wiki/List_of_Billboard_number-one_rap_singles_of_the_2000s')
    song_data_1980 = get_song_data('https://en.wikipedia.org/wiki/List_of_Billboard_number-one_rap_singles_of_the_1980s_and_1990s')
    
    lyrics = get_lyrics(song_data_2010, '2010-2020')
    lyrics_df = pd.DataFrame(lyrics, columns=['Lyrics'])
    lyrics_df['Era'] = '2010-2020'
    
    temp_lyrics = get_lyrics(song_data_2000, '2000-2010')
    temp_lyrics_df = pd.DataFrame(temp_lyrics, columns=['Lyrics'])
    temp_lyrics_df['Era'] = '2000-2010'
    
    lyrics_df = lyrics_df.append(temp_lyrics_df.copy())
    
    temp_lyrics = get_lyrics(song_data_1980, '1980-2000')
    temp_lyrics_df = pd.DataFrame(temp_lyrics, columns=['Lyrics'])
    temp_lyrics_df['Era'] = '1980-2000'    
    
    lyrics_df = lyrics_df.append(temp_lyrics_df.copy())
    
    print(lyrics_df)
    
    lyrics_df.to_csv('Rap_Lyrics_From_Different_Eras.csv')
    
    '''
    
    song_data_2000 = get_song_data('https://en.wikipedia.org/wiki/List_of_Billboard_number-one_rap_singles_of_the_2000s')
   
    print(song_data_2000)
    
    songs_dict = {'Jay-Z': ['Empire State of Mind']}
    

    
    songs_dict = {'Jay-Z': ['Empire State of Mind'],
                  'Eminem': ['Stan', 'Without Me']}
    
    songs_dict_two = {'50 Cent': ['21 Questions', 'P.I.M.P.', 'Candy Shop'],
                      'Kanye West': ['Gold Digger', 'Good Life', 'Heartless']
                    }

    
    lyrics = get_lyrics(songs_dict, '2010-2020')

    lyrics_two = get_lyrics(songs_dict_two, '2000-2010')
    
    lyrics_df = pd.DataFrame(lyrics, columns=['Lyrics'])
    lyrics_df['Era'] = '2010-2020'
    
    lyrics_df_two = pd.DataFrame(lyrics_two, columns=['Lyrics'])
    lyrics_df_two['Era'] = '2000-2010'
    
    lyrics_df = lyrics_df.append(lyrics_df_two)
    '''
    
        

if __name__ == '__main__':
    main()