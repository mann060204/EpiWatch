from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("EpidemicAnalysis") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("data/owid-covid-data.csv")

df.printSchema()
df.show(5)