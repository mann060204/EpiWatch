from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, when
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("GrowthRate").getOrCreate()

df = spark.read.option("header","true").csv("data/owid-covid-data.csv")

df = df.select(
    "location",
    "date",
    "total_cases",
    "new_cases"
).dropna()

df = df.withColumn("total_cases", col("total_cases").cast("double"))

# Window by country ordered by date
window = Window.partitionBy("location").orderBy("date")

# Previous day cases
df = df.withColumn(
    "previous_cases",
    lag("total_cases").over(window)
)

# Growth rate
df = df.withColumn(
    "growth_rate",
    when(col("previous_cases") > 0,
         (col("total_cases") - col("previous_cases")) / col("previous_cases"))
)

# Estimate R0
D = 5

df = df.withColumn(
    "R0",
    1 + col("growth_rate") * D
)

df.show(20)