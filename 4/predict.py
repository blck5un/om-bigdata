import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel


# params
model_path = sys.argv[1]
test_dataset_path = sys.argv[2]
predict_path = sys.argv[3]

schema = StructType([
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


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# load model
model = PipelineModel.load(model_path)

# create dataframe
df = spark.read.json(test_dataset_path, schema=schema)

# make predictions
predictions = model.transform(df)

# save predictions
predictions.select('prediction').write.csv(predict_path, mode='overwrite')

spark.stop()
