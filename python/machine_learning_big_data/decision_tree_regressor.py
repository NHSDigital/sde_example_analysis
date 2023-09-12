# Databricks notebook source
# MAGIC %md
# MAGIC # Spark MLLib Decision Tree Model Example Notebook - Python
# MAGIC
# MAGIC This example covers the following steps:
# MAGIC
# MAGIC 1. Import data from the collaborative database
# MAGIC 2. Visualize the data using the inbuilt display() table rendering and plotting
# MAGIC 3. Creating a train and test dataset
# MAGIC 4. Modelling
# MAGIC - 1. Assembling the feature vector
# MAGIC - 2. Creating the hyperparamter grid specifying the various combinations of parameters that will be explored when optimising
# MAGIC - 3. Creating the model
# MAGIC - 4. Setting up the cross validation
# MAGIC - 5. fitting the model
# MAGIC 5. Visualising the decision tree
# MAGIC 6. Applying the best model from the training to the test data and evaluating the predictions
# MAGIC
# MAGIC Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. They are easy to interpret (with the decision process being easily visualised) , require little data preperation and can be used on both categorical and continuous data.
# MAGIC
# MAGIC In this example, we use a decision tree for regression. This example code could also be used for other models by replacing the lines referencing "DecisionTreeRegressor" with the desired model from Spark MLlib and the associated parameters, as well as removing the block on visualising the decision tree.
# MAGIC
# MAGIC Spark MLlib is the machine learning component of pyspark. It is scalable, like pyspark, and can utilise the distributed processing that is a main feature of databricks and pyspark.
# MAGIC
# MAGIC The dataset being used is the well known MT Cars dataset and we will try and predict Miles Per Gallon (mpg) of a car.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Getting the data

# COMMAND ----------

# DBTITLE 1,Import python utility functions
# MAGIC %run ../../Wrangler-Utilities/Utilities-Python

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
from pyspark.ml.feature import VectorAssembler # puts selected features together into a vector
from pyspark.ml.regression import DecisionTreeRegressor # imports the model needed
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator

#specify the train test split ratio
train, test = data.randomSplit([0.9, 0.1], seed=12345)

#Specify the covariates
inputCols = ['wt', 'gear', 'hp']

#assemble covariates into a vector
va = VectorAssembler(inputCols=inputCols,outputCol="features")
# specify the model
dt = DecisionTreeRegressor(labelCol="mpg", featuresCol="features", maxDepth=4)
# specify the evaluation metric
evaluator = RegressionEvaluator(metricName = "rmse", labelCol="mpg")

#Build a grid of model hyperparameters that can be searched over to find the best model hyperparameters
grid = ParamGridBuilder().addGrid(dt.maxDepth, [3, 5, 7, 10]).build()

#Now combine the feature vector "va" and the model "dt" together into a pipeline
pipeline = Pipeline(stages=[va,dt])

#We then put the pipeline through cross validation. I chose 3 folds, and chose to keep the submodels (collectSubModels = True). If you only want to keep the best model, leave collectSubModels as the default False
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds = 3, collectSubModels = True)

# Fit the crossvalidator to the train data to produce a CrossValidatorModel
model = cv.fit(train)

#Display the average metrics for each sub model of the paramgrid (in this case, at 3,5,7 and 10 maxDepth)
print("Model avgMetrics:", model.avgMetrics)

# COMMAND ----------

# We can access the best model pipeline from the cross validation using .bestModel. We can display the actual tree by using stages to access the stages of the pipeline then using using indexing to navigate to the decision tree model and wrapping it all in a display()
# Just calling the model will show the parameters of the best model.

best_model = model.bestModel.stages[-1]
print(best_model)
display(best_model)

# COMMAND ----------

# We can also call the not-best subModels as a list and navigate and look at the sub models, if collectSubModels was set to True in the CrossValidator
model.subModels[0][1].stages

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

#test our predictions
predictions = model.transform(test) # model can be used instead of "model.bestModel.stages[-1]" as it chooses the best model automatically.

display(predictions)

# COMMAND ----------

# Evaluate the predictions on the test holdout data
rmse = evaluator.evaluate(predictions, {evaluator.metricName:'rmse'})
mae = evaluator.evaluate(predictions, {evaluator.metricName:"mae"})
r2 =evaluator.evaluate(predictions,{evaluator.metricName:'r2'})

print("RMSE: %.3f" %rmse)
print("R2: %.3f" %r2)
print("MAE: %.3f" %mae)


# COMMAND ----------

from pyspark.sql.functions import *

# You can take a look at feature importance
featureImportance = best_model.featureImportances.toArray()
#featureNames = map(lambda s: s.name, data.schema.fields)
featureImportanceMap = zip(featureImportance, inputCols)

#convert the above to a dataframe
importancesDf = spark.createDataFrame(sc.parallelize(featureImportanceMap).map(lambda r: [r[1], float(r[0])]))
importancesDf = importancesDf.withColumnRenamed("_1", "Feature").withColumnRenamed("_2", "Importance")
display(importancesDf.orderBy(desc("Importance")))
