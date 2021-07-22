#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:58:30 2019

@author: hdeva
"""

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from datetime import date, timedelta, datetime

import pandas as pd
import pandas_datareader as pdr
from pandas_datareader.tiingo import TiingoDailyReader
import pickle
import ast

from io import StringIO
import boto3


def get_stock_data(**kwargs):
    
    start = datetime(2015, 1, 1)
    end = datetime.now()
    
    api_token=""
    
    df = TiingoDailyReader(kwargs["params"]["stock"], start=start, end=end, api_key=api_token)
    
    stock_df = df.read()
    
    stock_df = stock_df.reset_index()
    
    return stock_df.to_json(orient='columns')

def store_arima_model_in_s3(**kwargs):
    
    ti=kwargs['ti']

    train_data_json = ti.xcom_pull(task_ids=kwargs["params"]["stock_ti"])
    train_data_json = ast.literal_eval(train_data_json)
    train_data_df = pd.DataFrame.from_dict(train_data_json, orient='columns')

    # fit model
    model=sm.tsa.ARIMA(endog=train_data_df["adjClose"], order=(1,1,0))
    model_fit = model.fit(disp=False)

    #dump to s3
    stock = train_data_df['symbol'][0]

    key = stock + '_arima_model.pkl'
    bucket='stock-model'
    pickle_byte_obj = pickle.dumps(model_fit) 
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)    

    # make prediction
    #fc, se, conf = model_fit.forecast(len(amzn_test_data), alpha=0.05)    


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 9, 12),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG('stock_data', default_args=default_args, schedule_interval="0 17 * * 1-5") as dag:
    
    start_task = DummyOperator(task_id='start')
    end_task = DummyOperator(task_id='end')

    stock_list = ['NLS', 'FSLY']

    # Extract
    for stock in stock_list:
        stock_data_task_name = 'get_%s_stock_data'%(str.lower(stock))
        get_stock_data_operator = \
            PythonOperator(task_id=stock_data_task_name,
                provide_context=True,
                python_callable=get_stock_data,
                params={"stock": stock},
                dag=dag)
                
        # Load
        s3_task_name = 'store_%s_arima_model_to_s3'%(str.lower(stock))
        store_stock_model_in_s3_operator = \
            PythonOperator(task_id=s3_task_name,
                   provide_context=True,
                   python_callable=store_arima_model_in_s3,
                   params={"stock_ti": stock_data_task_name},
                   dag=dag)
                   
        # Set Upstream/Downstream
        start_task.set_downstream(get_stock_data_operator)
        get_stock_data_operator.set_downstream(store_stock_model_in_s3_operator)
        end_task.set_upstream(store_stock_model_in_s3_operator)
  
