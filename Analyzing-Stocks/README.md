# Analyzing Tech Stocks of the Big 5 Tech Companies

This project takes the big 5 tech companies (Apple, Google, Microsoft, Facebook, and Amazon) and uses time series decomposition to forecast two things.

1. Market capitalization trend of tech stock 1 year from now
2. Stock price of tech stock in a 45 day period. 

## Files

* airflow/ - folder to schedule a stock_data DAG that fetches stock data from TIINGO and uploads the data in an AWS S3 bucket every weekday at 5 pm. 
* data/ - folder that stores pickled files of stock data from AWS S3 bucket and data generated from Anaylzing Stock Data.ipynb.
* presentations/ - folder that includes presentations of Airflow and this project.
* ARIMA for Five Stocks.ipynb - performs ARIMA, SARIMA, and Statsmodel time series decomposition on all 5 stocks. This is used to predict stock price.  
* Analyzing Stock Data.ipynb - performs Exploratory Data Analysis on all 5 stocks. Includes technical indicators such as 200-day Moving Average and RSI and stores info as a pickled file. 
* Facebook Prophet on all 5 stocks.ipynb - uses Facebook Prophet to perform time series decomposition on all 5 stocks. This is used to predict market capitalization trend and stock price. 

## Methodology

An Airflow instance is instantantiated and hosted on AWS to fetch stock data every weekday. This data spans from Jan 2015-Sept 2019. 

Stock data is analyzed and technical indicators are used to perform analysis on which stock would do well in the long term. This is based on adjusting closing price (and sometimes volume). 

Afterwards, ARIMA, SARIMA, Statsmodels timeseries decomposition, and Facebook Prophet timeseries decomposition were applied to predict stock price. These models were compared using Mean Absolute Percentage Error. Facebook Prophet timeseries decomposition performed the best out of all of them. 

For presentation, Facebook Prophet was used to predict market capitalization trend and stock price. 

## Conclusions

Microsoft is a viable long term investment choice. And Prophet does a decent job predicting stock prices, despite the volatility in the stock market. 

