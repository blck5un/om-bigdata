'''
Load model from MLFlow and make predictions
'''
import argparse
import os
import sys
import logging
import random
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

# import model definition
# from model import model

FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME
PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
SPARK_UI_PORT = random.choice(range(10000, 10200))

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def parse_args():
    parser = argparse.ArgumentParser(description="HW 5mlb by hethwynq: predict.py")
    parser.add_argument(
        "--test_path_in",
        type=str,
        help="path to test dataset",
    )
    parser.add_argument(
        "--pred_path_out",
        type=str,
        help="path to output predictions",
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        help="sklearn model name in MLFlow backend",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        help="sklearn model version in MLFlow backend",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run_id",
    )
    return parser.parse_args()


def predict(test_path_in, pred_path_out, sklearn_model, model_version, run_id=None):
    # start mlflow run
    mlrun = mlflow.start_run(run_id=run_id, run_name="PREDICT")
    logging.info(f"test_path_in: {test_path_in}")
    logging.info(f"pred_path_out: {pred_path_out}")
    logging.info(f"sklearn_model: {sklearn_model}")
    logging.info(f"model_version: {model_version}")
    # init spark session
    conf = SparkConf()
    conf.set("spark.ui.port", SPARK_UI_PORT)
    spark = SparkSession.builder.config(conf=conf).appName("hethwynq: hw5mlb_predict.py").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    # load dataset
    logging.info("Read dataset...")
    df_test = spark.read.parquet(test_path_in)
    logging.info("Finished.")
    # load model
    logging.info("Load model...")
    spark_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{sklearn_model}/{model_version}")
    logging.info("Finished.")
    # make predictions
    logging.info("Make predictions...")
    df_test = df_test.withColumn("prediction", spark_udf("reviewText", "summary"))
    logging.info(f"Number of records: {df_test.count()}")
    df_test.printSchema()
    logging.info("Finished.")
    # save predictions
    logging.info("Save predictions...")
    df_test[["id", "prediction"]].write.csv(pred_path_out, mode="overwrite", header=False)
    logging.info("Finished.")
    # stop spark session
    spark.stop()
    # end mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse_args()
    # run predict
    predict(args.test_path_in, args.pred_path_out, args.sklearn_model, args.model_version, args.run_id)
