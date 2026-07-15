import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, trim
from pyspark.sql.types import DoubleType

# Init
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Config
RAW_PATH = "s3://telcom-churn/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "s3://telcom-churn/processed/"

# Read raw CSV
df = spark.read.option("header", "true").option("inferSchema", "true").csv(RAW_PATH)

# 1. Drop customerID (not a predictive feature)
df = df.drop("customerID")

# 2. Clean TotalCharges — some rows have blank strings instead of numbers
df = df.withColumn("TotalCharges", trim(col("TotalCharges")))
df = df.withColumn(
    "TotalCharges",
    when(col("TotalCharges") == "", None).otherwise(col("TotalCharges")).cast(DoubleType())
)
df = df.na.fill({"TotalCharges": 0.0})

# 3. Encode target column: Churn Yes/No -> 1/0
df = df.withColumn("Churn", when(col("Churn") == "Yes", 1).otherwise(0))

# 4. Encode SeniorCitizen already 0/1, leave as is

# 5. Binary Yes/No columns -> 1/0
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for c in binary_cols:
    df = df.withColumn(c, when(col(c) == "Yes", 1).otherwise(0))

# 6. Write cleaned data to processed/ as a single CSV
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(PROCESSED_PATH)

job.commit()