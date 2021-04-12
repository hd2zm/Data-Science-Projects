import pandas as pd
import json
import boto3
import os

import numpy
import sys


def model_predict_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')                     

    for record in event['Records']:
        
        record_body = json.loads(record['body')

        # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
        response = runtime.invoke_endpoint(EndpointName = 'sagemaker-role',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = record_body)  
        # The response is an HTTP response whose body contains the result of our inference
        result = response['Body'].read().decode('utf-8')

        print(result)

        return {
            'statusCode' : 200,
            'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
            'body' : result
        }

    
