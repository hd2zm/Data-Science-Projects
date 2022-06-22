import json
import os
import boto3
from trp import Document
from datetime import datetime
from pdf2image import convert_from_bytes
from io import BytesIO

def send_from_s3_to_textract(event, context):

        region = ""
        accountId = ""
        
        # Amazon SageMaker client
        sagemaker = boto3.client('sagemaker', region)
        # Amazon Textract client
        textract = boto3.client('textract', region)
        # Amazon S3 Client
        s3_client = boto3.client('s3')

        records = event["Records"]
        for record in records:
                s3_bucket = record["s3"]["bucket"]["name"]
                filename = record["s3"]["object"]["key"]
                filename_without_extension = filename.split("/")[-1][:-4]

                # Convert PDF To Image     
                f = BytesIO()
                s3_client.download_fileobj(s3_bucket, filename, f)
                pages = convert_from_bytes(f.getvalue())

                i = 0
                print(pages)
                text_dict = {}
                form_dict = {}
                table_dict = {}

                for page in pages:
                    page_jpeg_name = "temp_png_files/" + str(filename_without_extension) + "_Page_" + str(i) + ".png"
                    
                    img_byte_arr = BytesIO()
                    page.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    s3_client.upload_fileobj(
                        BytesIO(img_byte_arr),
                        s3_bucket,
                        page_jpeg_name
                    )
            
                    s3_object = {
                        "S3Object": {
                            'Bucket': s3_bucket,
                            'Name': page_jpeg_name
                        }
                    }

                    human_loop_unique_id='human-loop-'+datetime.utcnow().strftime('%d-%m-%yt%H-%M-%S')
            
                    response = textract.analyze_document(
                        Document=s3_object,
                        FeatureTypes=['TABLES', 'FORMS'],
                        HumanLoopConfig={
                            'FlowDefinitionArn': 'arn:aws:sagemaker:' + region + ':' + accountId + ':flow-definition/surgery-form-workflow',
                            'HumanLoopName': human_loop_unique_id,
                            'DataAttributes': {'ContentClassifiers': ['FreeOfPersonallyIdentifiableInformation']}
                        }
                    )
                    
                    doc = Document(response)
                    for page in doc.pages:
                        text_dict[i] = str(page.text)
                        form_dict[i] = {}
                        table_dict[i] = {}
                        
                        # Print fields
                        print("Fields:")
                        for field in page.form.fields:
                            print("Key: {}, Value: {}".format(field.key, field.value))
                            form_dict[i][str(field.key)] = str(field.value)
                        
                        # Print table
                        for table in page.tables:
                            for r, row in enumerate(table.rows):
                                for c, cell in enumerate(row.cells):
                                    print("Table[{}][{}] = {}".format(r, c, cell.text))
                                    key = "Table[{}][{}]".format(r, c)
                                    table_dict[i][key] = str(cell.text)
                    
                    
                    i = i + 1
                    
                # Output dictionary
                    
                output_dict = {
                    "Text": text_dict,
                    "Forms": form_dict,
                    "Tables": table_dict
                }
                
                output_json = json.dumps(output_dict, indent=4)
                
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key="outbox/" + str(filename_without_extension) + ".json",
                    Body=output_json,
                    ContentType="application/json"
                )