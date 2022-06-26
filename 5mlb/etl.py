'''
Converts JSON dataset to Parquet format with cast "vote" column to float type
and repartition dataset to 1
'''
import argparse
import os
import sys
import logging
import random
import mlflow
import mlflow.sklearn

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

schema = StructType([
    StructField("id", LongType()),
    StructField("label", IntegerType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType())
])


def parse_args():
    parser = argparse.ArgumentParser(description="HW 5mlb by hethwynq: etl.py")
    parser.add_argument(
        "--etl_path_in",
        type=str,
        help="path to train dataset",
    )
    parser.add_argument(
        "--etl_path_out",
        type=str,
        help="path to modified output train dataset",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run_id",
    )
    return parser.parse_args()


def etl(etl_path_in, etl_path_out, run_id=None):
    # start mlflow run
    mlrun = mlflow.start_run(run_id=run_id, run_name="ETL")
    # init spark session
    conf = SparkConf()
    conf.set("spark.ui.port", SPARK_UI_PORT)
    spark = SparkSession.builder.config(conf=conf).appName("hethwynq: hw5mlb_etl.py").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    # load dataset
    logging.info(f"Load {etl_path_in}...")
    df = spark.read.json(etl_path_in, schema=schema)
    logging.info("Finished.")
    # transform dataset
    logging.info("Transform dataframe...")
    df = df.withColumn("vote", regexp_replace("vote", ",", ""))
    df = df.withColumn("vote", df["vote"].cast(FloatType()))
    df = df[["id", "label", "vote", "reviewText", "summary"]]
    logging.info("Finished.")
    # save dataset
    logging.info(f"Save {etl_path_out}...")
    df.coalesce(1).write.parquet(etl_path_out, mode="overwrite")
    logging.info("Finished.")
    # stop spark session
    spark.stop()
    # end mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse_args()
    # run etl
    etl(args.etl_path_in, args.etl_path_out, args.run_id)
