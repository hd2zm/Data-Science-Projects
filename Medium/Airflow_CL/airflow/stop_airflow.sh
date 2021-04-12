echo 'Stopping Airflow webserver.'
kill $(ps -ef | grep "airflow webserver" | grep -v grep | awk '{print $2}')
echo 'Stopping Airflow scheduler.'
kill $(ps -ef | grep "airflow scheduler" | grep -v grep | awk '{print $2}')