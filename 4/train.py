import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline, schema


# params
train_dataset_path = sys.argv[1]
model_path = sys.argv[2]

# create dataframe
df = spark.read.json(train_dataset_path, schema=schema)

# train
pipeline_model = pipeline.fit(df)

# save model
pipeline_model.write().overwrite().save(model_path)

spark.stop()
