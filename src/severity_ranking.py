from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("SeverityRanking").getOrCreate()

df = spark.read.option("header","true").csv("data/owid-covid-data.csv")

df = df.select(
    "location",
    "total_cases",
    "total_deaths",
    "population"
).dropna()

df = df.withColumn("total_cases", col("total_cases").cast("double"))
df = df.withColumn("total_deaths", col("total_deaths").cast("double"))
df = df.withColumn("population", col("population").cast("double"))

df = df.withColumn(
    "cases_per_population",
    col("total_cases") / col("population")
)

ranking = df.groupBy("location") \
    .max("cases_per_population") \
    .orderBy(col("max(cases_per_population)").desc())

ranking.show(20)