from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

df = spark.read.option("header","true").csv("data/owid-covid-data.csv")

df = df.select(
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "population"
).dropna()

df = df.withColumn("total_cases", col("total_cases").cast("double"))
df = df.withColumn("new_cases", col("new_cases").cast("double"))
df = df.withColumn("total_deaths", col("total_deaths").cast("double"))
df = df.withColumn("population", col("population").cast("double"))

# Mortality rate
df = df.withColumn(
    "mortality_rate",
    when(col("total_cases") > 0, col("total_deaths") / col("total_cases")).otherwise(None)
)

# Infection ratio
df = df.withColumn(
    "infection_ratio",
    when(col("population") > 0, col("total_cases") / col("population")).otherwise(None)
)

df.show(10)