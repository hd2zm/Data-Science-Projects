#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:58:30 2019

@author: hdeva
"""

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from datetime import date, timedelta, datetime

import pandas_datareader as pdr
from pandas_datareader.tiingo import TiingoDailyReader

from io import StringIO
import boto3


def get_stock_data(**kwargs):
    
    start = datetime(2015, 1, 1)
    end = datetime.now()
    
    api_token="0492b847e65cfba1e9abb96cb013c78404d0eb4f"
    
    df = TiingoDailyReader(kwargs["params"]["stock"], start=start, end=end, api_key=api_token)
    
    stock_df = df.read()
    
    stock_df = stock_df.reset_index()
    
    return stock_df
    

def upload_to_s3(**kwargs):
    
    ti=kwargs['ti']
    
    df = ti.xcom_pull(task_ids=kwargs["params"]["stock_ti"])
    stock = df['symbol'][0]
    
    filename = stock + '_stock_df.csv'
    
    print(filename)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object('tech-stock-data', filename).put(Body=csv_buffer.getvalue())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 9, 12),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG('stock_data', default_args=default_args, schedule_interval="0 17 * * 1-5") as dag:
    
    start_task = DummyOperator(task_id='start')

    # Extract
    get_amzn_stock_data = \
        PythonOperator(task_id='get_amzn_stock_data',
                   provide_context=True,
                   python_callable=get_stock_data,
                   params={"stock": "AMZN"},
                   dag=dag)
        
    get_msft_stock_data = \
        PythonOperator(task_id='get_msft_stock_data',
                   provide_context=True,
                   python_callable=get_stock_data,
                   params={"stock": "MSFT"},
                   dag=dag)
        
    get_fb_stock_data = \
        PythonOperator(task_id='get_fb_stock_data',
                   provide_context=True,
                   python_callable=get_stock_data,
                   params={"stock": "FB"},
                   dag=dag)
        
    get_aapl_stock_data = \
        PythonOperator(task_id='get_aapl_stock_data',
                   provide_context=True,
                   python_callable=get_stock_data,
                   params={"stock": "AAPL"},
                   dag=dag)

    get_googl_stock_data = \
        PythonOperator(task_id='get_googl_stock_data',
                   provide_context=True,
                   python_callable=get_stock_data,
                   params={"stock": "GOOGL"},
                   dag=dag)

    # Load
    upload_amzn_to_s3 = \
        PythonOperator(task_id='upload_amzn_to_s3',
                   provide_context=True,
                   python_callable=upload_to_s3,
                   params={"stock_ti": "get_amzn_stock_data"},
                   dag=dag)
        
    upload_msft_to_s3 = \
        PythonOperator(task_id='upload_msft_to_s3',
                   provide_context=True,
                   python_callable=upload_to_s3,
                   params={"stock_ti": "get_msft_stock_data"},
                   dag=dag)
        
    upload_fb_to_s3 = \
        PythonOperator(task_id='upload_fb_to_s3',
                   provide_context=True,
                   python_callable=upload_to_s3,
                   params={"stock_ti": "get_fb_stock_data"},
                   dag=dag)
        
    upload_aapl_to_s3 = \
        PythonOperator(task_id='upload_aapl_to_s3',
                   provide_context=True,
                   python_callable=upload_to_s3,
                   params={"stock_ti": "get_aapl_stock_data"},
                   dag=dag)

    upload_googl_to_s3 = \
        PythonOperator(task_id='upload_googl_to_s3',
                   provide_context=True,
                   python_callable=upload_to_s3,
                   params={"stock_ti": "get_googl_stock_data"},
                   dag=dag)
        
    end_task = DummyOperator(task_id='end')
       
    
    
    start_task.set_downstream([get_amzn_stock_data, get_msft_stock_data,
                               get_fb_stock_data, get_aapl_stock_data,
                               get_googl_stock_data])
    
    get_amzn_stock_data.set_downstream(upload_amzn_to_s3)
    get_msft_stock_data.set_downstream(upload_msft_to_s3)
    get_fb_stock_data.set_downstream(upload_fb_to_s3)
    get_aapl_stock_data.set_downstream(upload_aapl_to_s3)
    get_googl_stock_data.set_downstream(upload_googl_to_s3)
    
    end_task.set_upstream([upload_amzn_to_s3, upload_msft_to_s3,
                           upload_fb_to_s3, upload_aapl_to_s3,
                           upload_googl_to_s3])
  