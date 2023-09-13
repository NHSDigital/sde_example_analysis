# Databricks notebook source
# MAGIC %md
# MAGIC # PySpark MLLib Random Forest Classifier Example Notebook - Python
# MAGIC
# MAGIC **Author**: Adam Hollings
# MAGIC
# MAGIC This example covers the following steps:
# MAGIC
# MAGIC 1. Import data from the collaborative database
# MAGIC 2. Visualize the data using the inbuilt display() table rendering and plotting
# MAGIC 3. Creating a train and test dataset
# MAGIC 4. Modelling
# MAGIC - 1. Binarise the continous MPG column into a 2 category column
# MAGIC - 2. Assembling the feature vector
# MAGIC - 3. Creating the hyperparamter grid specifying the various combinations of parameters that will be explored when optimising
# MAGIC - 4. Creating the model
# MAGIC - 5. Setting up the cross validation
# MAGIC - 6. fitting the model
# MAGIC 5. Visualising the decision tree
# MAGIC 6. Applying the best model from the training to the test data and evaluating the predictions
# MAGIC
# MAGIC Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees.
# MAGIC
# MAGIC In this example, we use a Random Forest model to do a classification. This example code could also be used for other models by replacing the lines referencing "RandomForestClassifier" with the desired model from Spark MLlib and the associated parameters.
# MAGIC
# MAGIC Spark MLlib is the machine learning component of pyspark. It is scalable, like pyspark, and can utilise the distributed processing that is a main feature of databricks and pyspark.
# MAGIC
# MAGIC The dataset being used is the well known MT Cars dataset and we will try and predict if a car will have greather than 25 Miles Per Gallon (mpg)

# COMMAND ----------

# MAGIC %run ../../Wrangler-Utilities/Utilities-Python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting the data

# COMMAND ----------

data = spark.sql(f"""
select *
from {get_collab_db()}.toy_mtcars
""")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Visualising the data

# COMMAND ----------

# DBTITLE 1,Preview the MT Cars data
# You can use display() or display(data) to render a table of the data and create visualisations by clicking the plus button below
data.display()

# COMMAND ----------

# MAGIC %md
# MAGIC The data is all numerical. By visualising the data we can see that there is a rough correlation between mpg and wt, and mpg and hp. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Modelling
# MAGIC
# MAGIC As we will be using pyspark we will import the things needed

# COMMAND ----------

from pyspark.ml import Pipeline # this allows us to string together processes 
from pyspark.ml.feature import Binarizer, VectorAssembler # Binarizer turns a continous feature into a binary category based on a threshold. 
                                                          # VectorAssembler puts selected features together into a vector
from pyspark.ml.classification import RandomForestClassifier # imports the model needed
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#specify the train test split ratio
train, test = data.randomSplit([0.6, 0.4], seed=12345)

#Specify the covariates
inputCols = ['wt', 'gear', 'hp']

#Turn mpg into a binary category
binarizer = Binarizer(threshold=25, inputCol="mpg", outputCol="binarized_mpg")

#assemble covariates into a vector
va = VectorAssembler(inputCols=inputCols,outputCol="features")
# specify the model
rfc = RandomForestClassifier(labelCol="binarized_mpg", featuresCol="features", maxDepth=4)
# specify the evaluation metric
evaluator = BinaryClassificationEvaluator(metricName = "areaUnderPR", labelCol="binarized_mpg")

#Build a grid of model hyperparameters that can be searched over to find the best model hyperparameters
grid = ParamGridBuilder().addGrid(rfc.maxDepth, [3, 5, 7, 10]) \
                         .addGrid(rfc.numTrees, [2,3,4]) \
                         .build()

#Now combine the feature vector "va" and the model "dt" together into a pipeline
pipeline = Pipeline(stages=[binarizer, va, rfc])

#We then put the pipeline through cross validation. I chose 3 folds, and chose to keep the submodels (collectSubModels = True). If you only want to keep the best model, leave collectSubModels as the default False
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds = 3, collectSubModels = False)

# Fit the crossvalidator to the train data to produce a CrossValidatorModel
model = cv.fit(train)

#Display the average metrics for each sub model of the paramgrid (in this case, at 3,5,7 and 10 maxDepth)
print("Model avgMetrics:", model.avgMetrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

#test our predictions
predictions = model.transform(test) # model can be used instead of "model.bestModel.stages[-1]" as it chooses the best model automatically.

display(predictions)

# COMMAND ----------

# Evaluate the predictions on the test holdout data

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

f_score = MulticlassClassificationEvaluator(metricName = "f1", labelCol="binarized_mpg").evaluate(predictions)
accuracy = MulticlassClassificationEvaluator(metricName = "accuracy", labelCol="binarized_mpg").evaluate(predictions)
auprc = evaluator.evaluate(predictions, {evaluator.metricName:"areaUnderPR"})

print(f"f1 score: %.3f" %f_score)
print(f"Accuracy: {accuracy*100}%")
print(f"Area Under Precision Recall Curve: %.3f" %auprc)



# COMMAND ----------

from pyspark.sql.functions import *
# You can take a look at feature importance
featureImportance = model.bestModel.stages[-1].featureImportances.toArray()
#featureNames = map(lambda s: s.name, data.schema.fields)
featureImportanceMap = zip(featureImportance, inputCols)

#convert the above to a dataframe
importancesDf = spark.createDataFrame(sc.parallelize(featureImportanceMap).map(lambda r: [r[1], float(r[0])]))
importancesDf = importancesDf.withColumnRenamed("_1", "Feature").withColumnRenamed("_2", "Importance")
display(importancesDf.orderBy(desc("Importance")))
