import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import col, count, when
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import col, dayofweek, month, year, hour, to_date


from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[*]").appName("Linear-Regression-Flight-Delay").config("spark.driver.memory", "12g").config("spark.executor.memory", "12g").getOrCreate()

sc = spark.sparkContext

sqlContext = SQLContext(spark.sparkContext)

FLIGHTS_DATA = 'archive/flights_sample_3m.csv'

# define the schema corresponding to a line in the csv data file.
schema = StructType([
    StructField("fl_date", StringType(), nullable=True),                # Flight Date (yyyymmdd)
    StructField("airline", StringType(), nullable=True),                # Flight Date (yyyymmdd)
    StructField("airline_dot", StringType(), nullable=True),                # Flight Date (yyyymmdd)
    StructField("airline_code", StringType(), nullable=True),            # Unique Carrier Code
    StructField("dot_code", DoubleType(), nullable=True),                 # DOT identification number
    StructField("fl_number", DoubleType(), nullable=True),                # Flight Number
    StructField("origin", StringType(), nullable=True),                  # Origin Airport
    StructField("origin_city", StringType(), nullable=True),             # Origin Airport City Name
    StructField("dest", StringType(), nullable=True),                    # Destination Airport
    StructField("dest_city", StringType(), nullable=True),               # Destination Airport City Name
    StructField("crs_dep_time", DoubleType(), nullable=True),             # CRS Departure Time (hhmm)
    StructField("dep_time", DoubleType(), nullable=True),                 # Actual Departure Time (hhmm)
    StructField("dep_delay", DoubleType(), nullable=True),                # Departure Delay (minutes)
    StructField("taxi_out", DoubleType(), nullable=True),                 # Taxi Out Time (minutes)
    StructField("wheels_off", DoubleType(), nullable=True),               # Wheels Off Time (hhmm)
    StructField("wheels_on", DoubleType(), nullable=True),                # Wheels On Time (hhmm)
    StructField("taxi_in", DoubleType(), nullable=True),                  # Taxi In Time (minutes)
    StructField("crs_arr_time", DoubleType(), nullable=True),             # CRS Arrival Time (hhmm)
    StructField("arr_time", DoubleType(), nullable=True),                 # Actual Arrival Time (hhmm)
    StructField("arr_delay", DoubleType(), nullable=True),                # Arrival Delay (minutes)
    StructField("cancelled", DoubleType(), nullable=True),                # Cancelled Flight Indicator (1=Yes)
    StructField("cancellation_code", StringType(), nullable=True),       # Reason for Cancellation
    StructField("diverted", DoubleType(), nullable=True),                 # Diverted Flight Indicator (1=Yes)
    StructField("crs_elapsed_time", DoubleType(), nullable=True),         # CRS Elapsed Time (minutes)
    StructField("elapsed_time", DoubleType(), nullable=True),             # Actual Elapsed Time (minutes)
    StructField("air_time", DoubleType(), nullable=True),                 # Flight Time (minutes)
    StructField("distance", DoubleType(), nullable=True),                 # Distance between airports (miles)
    StructField("delay_due_carrier", DoubleType(), nullable=True),        # Carrier Delay (minutes)
    StructField("delay_due_weather", DoubleType(), nullable=True),        # Weather Delay (minutes)
    StructField("delay_due_nas", DoubleType(), nullable=True),            # NAS Delay (minutes)
    StructField("delay_due_security", DoubleType(), nullable=True),       # Security Delay (minutes)
    StructField("delay_due_late_aircraft", DoubleType(), nullable=True)   # Late Aircraft Delay (minutes)
])

flights_df = spark.read.csv(path=FLIGHTS_DATA, schema=schema, header=True).cache()
flights_df.show(5)
print(flights_df.columns)

# EDA

# group by 'airline' and calculate average delays
avg_delays = flights_df.groupBy("airline").agg(
    F.avg("dep_delay").alias("avg_dep_delay"),
    F.avg("arr_delay").alias("avg_arr_delay")
)

# convert to Pandas for easier visualization
avg_delays_pd = avg_delays.toPandas()

# plot the data in a single figure
fig, ax = plt.subplots(figsize=(10, 6))  # Create a single figure and axes
avg_delays_pd.set_index('airline')[['avg_dep_delay', 'avg_arr_delay']].plot(kind='bar', ax=ax, width=0.8, color=['#1f77b4', '#ff7f0e'])

# add titles and labels
ax.set_title("Average Departure and Arrival Delays by Airline")
ax.set_xlabel("Airline")
ax.set_ylabel("Average Delay (Minutes)")
ax.set_xticklabels(avg_delays_pd['airline'], rotation=90)
plt.tight_layout()
plt.show()

# compute summary statistics
(flights_df.describe().select(
                    "summary",
                    F.round("dep_delay", 4).alias("Departure delay (Minutes)"),
                    F.round("arr_delay", 4).alias("Arrival delay (Minutes)"),
                    F.round("crs_elapsed_time", 4).alias("CRS elapsed time (Minutes)"),
                    F.round("elapsed_time", 4).alias("Elapsed time (Minutes)"),
                    F.round("distance", 4).alias("Distance (Miles)"),
                    F.round("taxi_in", 4).alias("Taxi in (Minutes)"),
                    F.round("taxi_out", 4).alias("Taxi out (Minutes)"),
                    F.round("air_time", 4).alias("Air time (Minutes)"),
                    F.round("delay_due_weather", 4).alias("Delay due to weather (Minutes)")).show())

route_delays = (
    flights_df
    .groupBy("origin", "dest")
    .agg(
        F.count("*").alias("count"),  # count the occurrences of each route
        F.avg("arr_delay").alias("avg_delay")  # calculate average delay
    )
    .filter(F.col("count") >= 200)  # filter to include only routes with at least 200 flights
    .filter(F.col("avg_delay").isNotNull())  # ensure no null average delays
    .orderBy(F.desc("avg_delay"))  # sort by count in descending order
)
#
# convert to Pandas for visualization (Optional: Limit top routes for clarity)
route_delays_pd = route_delays.limit(20).toPandas()

# combine origin and destination for easier plotting
route_delays_pd["route"] = route_delays_pd["origin"] + " → " + route_delays_pd["dest"]

# create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=route_delays_pd, x="avg_delay", y="route", palette="coolwarm")
plt.title("Top 20 Average Arrival Delays by Route (With at least 200 occurences)", fontsize=16)
plt.xlabel("Average Delay (minutes)", fontsize=14)
plt.ylabel("Route", fontsize=14)
plt.tight_layout()
plt.show()



# count missing values for each column in the dataframe
missing_counts = flights_df.select(
    [count(when(col(c).isNull(), 1)).alias(c) for c in flights_df.columns]
)
missing_counts.show()


# filter out rows where 'cancelled' is 1
filtered_flights_df = flights_df.filter(flights_df.cancelled != 1)

cleaned_flights_df = filtered_flights_df.drop("dest_city", "origin_city", "airline_dot", "airline", "cancelled", "cancellation_code", "delay_due_carrier", "delay_due_weather", "delay_due_nas", "delay_due_security", "delay_due_late_aircraft", "elapsed_time", "arr_time", "wheels_on", "taxi_in", "crs_dep_time", "fl_number")

# create a new column 'delayed' as a double (1.0 for delay > 15 minutes, 0.0 otherwise)
cleaned_flights_df = cleaned_flights_df.withColumn(
    "delayed", when(cleaned_flights_df["ARR_DELAY"] > 15, 1.0).otherwise(0.0)
)

# show some rows to verify
cleaned_flights_df.groupBy('delayed').count().show()

cleaned_flights_df = cleaned_flights_df.withColumn("FL_DATE", to_date(col("FL_DATE"), "yyyy-MM-dd"))

# extract useful components from FL_DATE 
cleaned_flights_df = cleaned_flights_df.withColumn("year", year(col("FL_DATE")).cast("double"))
cleaned_flights_df = cleaned_flights_df.withColumn("month", month(col("FL_DATE")))
cleaned_flights_df = cleaned_flights_df.withColumn("day_of_week", dayofweek(col("FL_DATE")))  # 1=Sunday, 7=Saturday

#encode categorical variables in numerical vectors for giving them to the ml model
indexer = StringIndexer(inputCol="month", outputCol="month_index")
encoder = OneHotEncoder(inputCol="month_index", outputCol="month_vec")
cleaned_flights_df = indexer.fit(cleaned_flights_df).transform(cleaned_flights_df)
cleaned_flights_df = encoder.fit(cleaned_flights_df).transform(cleaned_flights_df)

indexer = StringIndexer(inputCol="day_of_week", outputCol="day_of_week_index")
encoder = OneHotEncoder(inputCol="day_of_week_index", outputCol="day_of_week_vec")
cleaned_flights_df = indexer.fit(cleaned_flights_df).transform(cleaned_flights_df)
cleaned_flights_df = encoder.fit(cleaned_flights_df).transform(cleaned_flights_df)


indexer = StringIndexer(inputCol="airline_code", outputCol="airline_index")
encoder = OneHotEncoder(inputCol="airline_index", outputCol="airline_vec")
cleaned_flights_df = indexer.fit(cleaned_flights_df).transform(cleaned_flights_df)
cleaned_flights_df = encoder.fit(cleaned_flights_df).transform(cleaned_flights_df)

indexer = StringIndexer(inputCol="origin", outputCol="origin_in")
encoder = OneHotEncoder(inputCol="origin_in", outputCol="origin_vec")
cleaned_flights_df = indexer.fit(cleaned_flights_df).transform(cleaned_flights_df)
cleaned_flights_df = encoder.fit(cleaned_flights_df).transform(cleaned_flights_df)

indexer = StringIndexer(inputCol="dest", outputCol="dest_in")
encoder = OneHotEncoder(inputCol="dest_in", outputCol="dest_vec")
cleaned_flights_df = indexer.fit(cleaned_flights_df).transform(cleaned_flights_df)
cleaned_flights_df = encoder.fit(cleaned_flights_df).transform(cleaned_flights_df)

cleaned_flights_df = cleaned_flights_df.drop("day_of_week_index", "airline_index", "origin_in", "dest_in", "airline_index", "origin_in", "dest_in", "origin", "month", "day_of_week", "fl_date", "airline_code", "dest", "month_index", "arr_delayed")

# select relevant columns
continuous_columns = ["dep_delay", "arr_delay", "crs_elapsed_time", "taxi_in", "taxi_out", "distance"]

# convert to Pandas DataFrame
flights_pd_df = filtered_flights_df.select(continuous_columns).dropna().toPandas()

# compute the correlation matrix
correlation_matrix = flights_pd_df.corr()
print("Correlation matrix:\n", correlation_matrix)

#visualizations

# plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Pairwise Correlation Matrix Including Diverted Flights")
plt.show()

# Show some rows to verify
cleaned_flights_df.show(10)

# Define features column list, including all specified features
feature_columns = [
    "dep_time",
    "dep_delay",        
    "taxi_out",         
    "distance",  
    "airline_vec",    
    "origin_vec",      
    "wheels_off",
    "day_of_week_vec",
    "month_vec",
    "dest_vec",
    "dot_code"
]

# drop rows with any null values in the feature columns
cleaned_flights_df = cleaned_flights_df.dropna(subset=feature_columns)

# assemble feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
cleaned_flights_df = assembler.transform(cleaned_flights_df)

# show a few rows to verify the feature vector
cleaned_flights_df.select("features").show(5, truncate=False)

# split into training and testing sets (e.g., 75% training, 25% testing)
train_df, test_df = cleaned_flights_df.randomSplit([0.75, 0.25])

cleaned_flights_df.show(10, truncate=False)

cleaned_flights_df.printSchema()

# initialize models
log_reg = LogisticRegression(featuresCol="features", labelCol="delayed")
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="delayed")

# define hyperparameter grids

#regParam  controls the amount of regularization applied. Smaller values (e.g., 0.01) allow more flexibility and may overfit, while larger values (e.g., 0.5) prevent overfitting but may underfit the data.
#elasticNetParam is a parameter for ElasticNet regularization, which blends L1 (Lasso) and L2 (Ridge) regularization. Values like 0.0, 0.25, 0.75 give different regularization patterns. A value of 0.0 uses only L2 (Ridge) regularization, while 1.0 uses only L1 (Lasso).
log_reg_param_grid = ParamGridBuilder() \
    .addGrid(log_reg.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(log_reg.elasticNetParam, [0.0, 0.25, 0.75]) \
    .build()

# Random Forest
rf_param_grid = ParamGridBuilder() \
    .addGrid(rf_classifier.numTrees, [10, 15, 20]) \
    .addGrid(rf_classifier.maxDepth, [5, 8, 10]) \
    .build()

# set up evaluators
evaluator = BinaryClassificationEvaluator(labelCol="delayed", metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="delayed", metricName="f1")


# set up 5-fold cross-validation for each model
cv_log_reg = CrossValidator(
    estimator=log_reg,
    estimatorParamMaps=log_reg_param_grid,
    evaluator=evaluator,
    numFolds=5
)

cv_rf = CrossValidator(
    estimator=rf_classifier,
    estimatorParamMaps=rf_param_grid,
    evaluator=evaluator,
    numFolds=5
)

# fit models with cross-validation on the training data
cv_log_reg_model = cv_log_reg.fit(train_df)
cv_rf_model = cv_rf.fit(train_df)

# apply the best models to the test data
log_reg_predictions = cv_log_reg_model.bestModel.transform(test_df)
rf_predictions = cv_rf_model.bestModel.transform(test_df)

rf_predictions.groupBy("delayed", "prediction").count().show()

# evaluate models on test data with AUC
log_reg_auc = evaluator.evaluate(log_reg_predictions)
rf_auc = evaluator.evaluate(rf_predictions)

# additional Performance Metrics
log_reg_accuracy = accuracy_evaluator.evaluate(log_reg_predictions)
log_reg_precision = precision_evaluator.evaluate(log_reg_predictions)
log_reg_recall = recall_evaluator.evaluate(log_reg_predictions)
log_reg_f1 = f1_evaluator.evaluate(log_reg_predictions)

rf_accuracy = accuracy_evaluator.evaluate(rf_predictions)
rf_precision = precision_evaluator.evaluate(rf_predictions)
rf_recall = recall_evaluator.evaluate(rf_predictions)
rf_f1 = f1_evaluator.evaluate(rf_predictions)

# print the results
print(f"Logistic Regression AUC: {log_reg_auc}")
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
print(f"Logistic Regression Precision: {log_reg_precision}")
print(f"Logistic Regression Recall: {log_reg_recall}")
print(f"Logistic Regression F1-score: {log_reg_f1}")

print(f"Random Forest AUC: {rf_auc}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1-score: {rf_f1}")

# Confusion Matrix
print("Logistic Regression Confusion Matrix:")
log_reg_predictions.groupBy("delayed", "prediction").count().show()

print("Random Forest Confusion Matrix:")
rf_predictions.groupBy("delayed", "prediction").count().show()

# seature Importance Plot (Random Forest)
rf_feature_importances = cv_rf_model.bestModel.featureImportances
feature_importances = pd.DataFrame(list(zip(feature_columns, rf_feature_importances)),
                                   columns=["Feature", "Importance"])
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importances from Random Forest")
plt.gca().invert_yaxis()
plt.savefig('features_importance_4.png')  # Save the plot as an image

# extract TPR (True Positive Rate) and FPR (False Positive Rate) for ROC curve
def get_roc_data(predictions, label_col="delayed"):
    # extract the probability for the indexed 1 class and put it in a temporary data frame
    roc_data = predictions.select("probability", label_col).rdd.map(lambda row: (float(row["probability"][1]), row[label_col])).toDF(["probability", label_col])
    
    # add a new column "negative", which is 1 minus the actual label
    roc_data = roc_data.withColumn("negative", 1 - col(label_col))
    
    # create a BinaryClassificationEvaluator to calculate the AUC (Area Under the Curve)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    
    # compute AUC for the predictions
    auc = evaluator.evaluate(predictions)
    
    # convert the DataFrame to a Pandas DataFrame, ordered by probability (descending)
    roc_df = roc_data.orderBy("probability", ascending=False).toPandas()
    
    # calculate the True Positive Rate (TPR) as the cumulative sum of positive labels divided by the total positive labels
    roc_df["TPR"] = roc_df[label_col].cumsum() / roc_df[label_col].sum()
    
    # calculate the False Positive Rate (FPR) as the cumulative sum of negative labels divided by the total negative labels
    roc_df["FPR"] = roc_df["negative"].cumsum() / roc_df["negative"].sum()
    
    # return the FPR, TPR, and the computed AUC
    return roc_df["FPR"], roc_df["TPR"], auc


# plotting the ROC curves
plt.figure(figsize=(10, 6))

# logistic Regression ROC Curve
fpr_log_reg, tpr_log_reg, auc_log_reg = get_roc_data(log_reg_predictions)
plt.plot(fpr_log_reg, tpr_log_reg, label=f"Logistic Regression (AUC = {auc_log_reg:.3f})")

# random Forest ROC Curve
fpr_rf, tpr_rf, auc_rf = get_roc_data(rf_predictions)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")

# plot settings
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Logistic Regression and Random Forest")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("roc_curves_2.png")
