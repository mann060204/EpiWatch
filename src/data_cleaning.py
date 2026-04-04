from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("EpidemicCleaning").getOrCreate()

df = spark.read.option("header","true").csv("data/owid-covid-data.csv")

# Select important columns
df = df.select(
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "population"
)

# Remove rows with missing values
df = df.dropna()

# Convert numeric columns
df = df.withColumn("total_cases", col("total_cases").cast("double"))
df = df.withColumn("new_cases", col("new_cases").cast("double"))
df = df.withColumn("total_deaths", col("total_deaths").cast("double"))

df.show(10)