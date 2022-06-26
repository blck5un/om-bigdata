from pyspark.sql.types import *
from pyspark.sql import functions as f

from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


schema = StructType([
    StructField("overall", FloatType()),
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

stop_words = StopWordsRemover.loadDefaultStopWords("english")

# transformers
sql = SQLTransformer(statement="SELECT *, concat_ws(' ', reviewText, summary) AS text FROM __THIS__")
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
swr = StopWordsRemover(inputCol=regexTokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
hasher = HashingTF(numFeatures=5000, binary=False, inputCol=swr.getOutputCol(), outputCol="word_vector")
normalizer = Normalizer(inputCol=hasher.getOutputCol(), outputCol="norm_features", p=2)

# ml model
linReg = LinearRegression(featuresCol="norm_features",
                          labelCol="overall",
                          loss="squaredError",
                          #elasticNetParam=0,
                          #regParam=0.001,
                          fitIntercept=True,
                          #standardization=True,
                          maxIter=100)

# pipeline
pipeline = Pipeline(stages=[
    sql,
    regexTokenizer,
    swr,
    hasher,
    normalizer,
    linReg
])
