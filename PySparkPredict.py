import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

import mlflow
from mlflow.tracking import MlflowClient


def get_last_prod_model(name, client):
    last_models = client.get_registered_model(name).latest_versions
    return last_models[0]
  
  
  
def process(spark, data_path, result):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param result: путь сохранения результата
    """
    data = spark.read.parquet(data_path)
    mlflow.set_tracking_uri("https://mlflow.lab.karpov.courses")
    client = MlflowClient()
    model_version = get_last_prod_model('d-kruglov', client)
    model = mlflow.spark.load_model(f'models:/d-kruglov/{model_version.version}')
    prediction = model.transform(data)
    prediction.write.parquet(result)
    

def main(data, result):
    spark = _spark_session()
    process(spark, data, result)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkPredict').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.parquet', help='Please set datasets path.')
    parser.add_argument('--result', type=str, default='result', help='Please set result path.')
    args = parser.parse_args()
    data = args.data
    result = args.result
    main(data, result)
