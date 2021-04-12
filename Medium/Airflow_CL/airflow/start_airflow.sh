echo 'Starting Airflow webserver.'
airflow webserver -p 8080 -D
echo 'Starting Airflow scheduler.'
airflow scheduler -D