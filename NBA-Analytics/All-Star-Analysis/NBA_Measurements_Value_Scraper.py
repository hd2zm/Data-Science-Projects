#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:19:21 2019

@author: hdeva
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
from selenium import webdriver
import threading
from multiprocessing import Pool  # This is a thread-based Pool
from multiprocessing import cpu_count


def convert_to_inches(stat):
    if stat:
        stat_ = stat.split("' ")
        ft = float(stat_[0])
        inch = float(stat_[1].replace("'",""))    
        return (12*ft) + inch
    else:
        return 0


'''

Draft measurements Years
'''

def get_measurements_content(url, measurements_data):
    #Set as Beautiful Soup Object
    driver = webdriver.Chrome("/usr/local/bin/chromedriver")
    
    driver.get(url)

    measurements_soup = BeautifulSoup(driver.page_source)
            
    driver.quit()
    # Go to the section of interest
    measurements_summary = measurements_soup.find('div', attrs={'class':'nba-stat-table'})
    
    #print(measurements_summary)
    
    # Find the tables in the HTML
    measurements_tables = measurements_summary.find_all('table')
        
    # Set rows as first indexed object in tables with a row
    rows = measurements_tables[0].findAll('tr')
    
    # now grab every HTML cell in every row
    for tr in rows:
        cols = tr.findAll('td')
        # Check to see if text is in the row
        measurements_data.append([])
        for td in cols:
            text = td.find(text=True) 
            measurements_data[-1].append(text)

def get_measurements(start_year, end_year):

    measurements_base_url = 'https://stats.nba.com/draft/combine-anthro/'

    measurements_data = []
    
    urls=[]
    while start_year < end_year:
        urls.append(measurements_base_url + '#!?SeasonYear=' + str(start_year) + '-' + str(start_year+1)[-2:])
        start_year = start_year + 1

    threads = []
    
    for url in urls:
        process = threading.Thread(target=get_measurements_content, args=(url, measurements_data))
        process.start()
        threads.append(process) 
     
    for process in threads:
        process.join()

    
    print(measurements_data)

    measurements_data.pop(0)  
    measurements_df = pd.DataFrame(measurements_data)
    measurements_df.columns = ['PLAYER', 'POS', 'BODY FAT', 'HAND LENGTH', 'HAND WIDTH', 'HEIGHT', 'HEIGHT WITH SHOES', 'STANDING REACH', 'WEIGHT', 'WINGSPAN']
    measurements_df = measurements_df.loc[:, measurements_df.columns.intersection(['PLAYER', 'BODY FAT', 'HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN' ])]

    measurements_df['HEIGHT'] = measurements_df['HEIGHT'].apply(lambda x: convert_to_inches(x))
    measurements_df['STANDING REACH'] = measurements_df['STANDING REACH'].apply(lambda x: convert_to_inches(x))
    measurements_df['WINGSPAN'] = measurements_df['WINGSPAN'].apply(lambda x: convert_to_inches(x))
    
    return measurements_df

'''

def get_measurements (start_year, end_year):

    measurements_base_url = 'https://stats.nba.com/draft/combine-anthro/'

    measurements_data = []
    
    urls=[]
    while start_year < end_year:
        urls.append(measurements_base_url + '#!?SeasonYear=' + str(start_year) + '-' + str(start_year+1)[-2:])
        start_year = start_year + 1

    for url in urls:   
    
        #Set as Beautiful Soup Object
        driver = webdriver.Chrome("/usr/local/bin/chromedriver")
        driver.get(url)
        measurements_soup = BeautifulSoup(driver.page_source)
        driver.quit()
        
        # Go to the section of interest
        measurements_summary = measurements_soup.find('div', attrs={'class':'nba-stat-table'})
        
        #print(measurements_summary)
    
        # Find the tables in the HTML
        measurements_tables = measurements_summary.find_all('table')
        
        # Set rows as first indexed object in tables with a row
        rows = measurements_tables[0].findAll('tr')
    
        # now grab every HTML cell in every row
        for tr in rows:
            cols = tr.findAll('td')
            # Check to see if text is in the row
            measurements_data.append([])
            for td in cols:
                text = td.find(text=True) 
                measurements_data[-1].append(text)
    
    measurements_data.pop(0)
    measurements_df = pd.DataFrame(measurements_data)
    measurements_df.columns = ['PLAYER', 'POS', 'BODY FAT', 'HAND LENGTH', 'HAND WIDTH', 'HEIGHT', 'HEIGHT WITH SHOES', 'STANDING REACH', 'WEIGHT', 'WINGSPAN']
    measurements_df = measurements_df.loc[:, measurements_df.columns.intersection(['PLAYER', 'BODY FAT', 'HEIGHT', 'STANDING REACH', 'WEIGHT', 'WINGSPAN' ])]

    measurements_df['HEIGHT'] = measurements_df['HEIGHT'].apply(lambda x: convert_to_inches(x))
    measurements_df['STANDING REACH'] = measurements_df['STANDING REACH'].apply(lambda x: convert_to_inches(x))
    measurements_df['WINGSPAN'] = measurements_df['WINGSPAN'].apply(lambda x: convert_to_inches(x))

    print(measurements_df.tail(5))
    
    return measurements_df

'''

def get_value_link_content(url, va_data):
#def get_value_link_content(url):
    #va_data = []
    #Set as Beautiful Soup Object    
    va_soup = BeautifulSoup(requests.get(url).content)
    
    # Go to the section of interest
    va_summary = va_soup.find("div",{'class':'col-main', 'id':'my-players-table'})
    
    # Find the tables in the HTML
    va_tables = va_summary.find_all('table')
        
    # Set rows as first indexed object in tables with a row
    rows = va_tables[0].findAll('tr')
    
    # now grab every HTML cell in every row
    for tr in rows:
        cols = tr.findAll('td')
        # Check to see if text is in the row
        va_data.append([])
        for td in cols:
            text = td.find(text=True) 
            va_data[-1].append(text)
            
    return va_data

def get_value_added():
    
    va_base_url = 'http://insider.espn.com/nba/hollinger/statistics' 
    max_va_page = 8
    i = 2
    va_data = []
    
    urls = [va_base_url]
    while i <= max_va_page:
        urls.append(va_base_url + '/_/page/' + str(i))
        i = i + 1
    
    threads = []
    
     
    for url in urls:

        process = threading.Thread(target=get_value_link_content, args=(url, va_data))
        process.start()
        threads.append(process)
    
    
    
    for process in threads:
        process.join()
    
    
    '''
    with Pool(10) as p:
        va_data = p.map(get_value_link_content, urls)
    '''
    
    threads = [threading.Thread(target=get_value_link_content, args=(url, va_data))
               for url in urls]
    
    for t in threads:
        t.start()
    
    print(len(va_data))
    
    va_data.pop(0)
    
    print(va_data)
    
    va_df = pd.DataFrame(va_data[1:], columns=va_data[0])
    #va_df = va_df[va_df.PLAYER != 'PLAYER']
    #va_df = va_df.sort_values(by=['VA'])
    #va_df = va_df.loc[:, va_df.columns.intersection(['PLAYER','VA'])]


    return va_df
    
    '''
    
    #Value Added Data Frame
    va_base_url = 'http://insider.espn.com/nba/hollinger/statistics'
    va_next_url = va_base_url
    max_va_page = 8
    
    va_data = []
    
    i = 2
    
    urls = [va_base_url]
    while i <= max_va_page:
        urls.append(va_base_url + '/_/page/' + str(i))
        i = i + 1
    
    for url in urls:   
        #Set as Beautiful Soup Object    
        va_soup = BeautifulSoup(requests.get(url).content)
    
        # Go to the section of interest
        va_summary = va_soup.find("div",{'class':'col-main', 'id':'my-players-table'})
    
        # Find the tables in the HTML
        va_tables = va_summary.find_all('table')
        
        # Set rows as first indexed object in tables with a row
        rows = va_tables[0].findAll('tr')
    
        # now grab every HTML cell in every row
        for tr in rows:
            cols = tr.findAll('td')
            # Check to see if text is in the row
            va_data.append([])
            for td in cols:
                text = td.find(text=True) 
                va_data[-1].append(text)
        
    
    va_data.pop(0)
    
    va_df = pd.DataFrame(va_data[1:], columns=va_data[0])
    va_df = va_df.loc[:, va_df.columns.intersection(['PLAYER','VA'])]

    return va_df

    '''

#List of NBA All Stars
def get_all_stars():

    all_star_data = []
    
    all_star_soup = BeautifulSoup(requests.get('https://en.wikipedia.org/wiki/List_of_NBA_All-Stars').content)
    
    all_star_table = all_star_soup.find('table',{'class': 'wikitable sortable'})
    
    rows = all_star_table.findAll('tr')
    
    for tr in rows:
        cols = tr.findAll('td')
        if len(cols) > 0:
            all_star_data.append(cols[0].find(text=True) )
    
    all_star_df = pd.DataFrame({'PLAYER': all_star_data})
    all_star_df['ALL-STAR'] = 1
    
    return all_star_df



if __name__ == "__main__":

    # Merge everything
      
    va_df = get_value_added()
    va_df = va_df[va_df.PLAYER != 'PLAYER']
    va_df = va_df.sort_values(by=['RK'])
    print(va_df.head(50))
    va_df = va_df.loc[:, ['PLAYER','VA']]
    

    all_star_df = get_all_stars()  
    
    va_df = va_df.merge(all_star_df, on="PLAYER", how='left')
    va_df = va_df[va_df.PLAYER != 'PLAYER']
    va_df = va_df.fillna(0)
    
    va_df.to_csv('NBA_Value_Added.csv')
    
    measurements_df = get_measurements(2003, 2019)
    
    va_df = va_df.merge(measurements_df, on="PLAYER", how='left')
    
    va_df.to_csv('NBA_Measurements_Value_Added.csv')
    
    print(va_df.head(15))
    
    
    measurements_df.to_csv('NBA_Measurements.csv')

