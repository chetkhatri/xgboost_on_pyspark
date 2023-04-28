import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.90.jar,xgboost4j-0.90.jar pyspark-shell'

from pyspark.sql import SparkSession
import pandas as pd
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import *
from pyspark.ml.tuning import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *


# Create a Spark session
spark = SparkSession.builder \
    .appName("XGBoost Example").getOrCreate()

spark.sparkContext.addPyFile("sparkxgb.zip")

from ml.dmlc.xgboost4j.scala.spark import XGBoostClassificationModel, XGBoostClassifier

# Read the CSV file with column names
rawInput = spark.read.csv("iris.csv", header=True)

# Print the schema to verify if the column names are correctly assigned
rawInput.printSchema()

# Convert the columns to appropriate data types
rawInput = rawInput.withColumn("sepal_length", rawInput["sepal_length"].cast("float"))
rawInput = rawInput.withColumn("sepal_width", rawInput["sepal_width"].cast("float"))
rawInput = rawInput.withColumn("petal_length", rawInput["petal_length"].cast("float"))
rawInput = rawInput.withColumn("petal_width", rawInput["petal_width"].cast("float"))

# Encode the species column as numeric labels
labelIndexer = StringIndexer(inputCol="species", outputCol="label")
preprocessingPipeline = Pipeline(stages=[labelIndexer])
preprocessedData = preprocessingPipeline.fit(rawInput).transform(rawInput)

# Combine input features into a single vector column
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
preprocessedData = assembler.transform(preprocessedData)

# Split the data into training and test sets
(trainData, testData) = preprocessedData.randomSplit([0.7, 0.3])

# Define the XGBoost classifier
xgboost = XGBoostClassifier(objective="multi:softprob", maxDepth=3, numClass=3, featuresCol="features", labelCol="label")

# Train the model
pipeline = Pipeline(stages=[xgboost])
model = pipeline.fit(trainData)

# Make predictions on the test set
predictions = model.transform(testData)

# Evaluate the accuracy of the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: ", accuracy)
