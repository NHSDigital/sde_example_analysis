# Databricks notebook source
# MAGIC %md
# MAGIC ## Multi-Variable Linear Regression - SKLearn
# MAGIC
# MAGIC In this notebook we will walk through an example of using linear regression on a multi variable dataset, it also shows off another useful package called statsmodel at the end of the notebook.
# MAGIC
# MAGIC In this example the toy datasets have already created and added to the collab database to mimic an actual workflow, we will use a general function to get the database name however this can be can be replaced with a string.
# MAGIC
# MAGIC The utility functions are imported via the next command which runs the notebook stored in a different location. You can view these functions by navigating to the folder or you can also click the link in the next command. This can also be a useful way to tidy up your code and store frequently used functions in their own notebook to be imported into others.
# MAGIC
# MAGIC #### SKLearn performance in databricks
# MAGIC
# MAGIC While SKLearn can be useful in certain situations, it is not designed to take advantage of cluster computing resources, which arguably is a major downside to using it inside databricks as you are not utilising the full proccessing power available to you.
# MAGIC
# MAGIC This is not us saying do not use sklearn as it may well be appropriate for certain tasks, however if your are performing tasks over large datasets and want to fully exploit the compute resources you have available to complete these tasks.

# COMMAND ----------

# DBTITLE 1,Import python utility functions
# MAGIC %run ../../Wrangler-Utilities/Utilities-Python

# COMMAND ----------

import matplotlib.pyplot as plt

# get the table name
table_name = f"{get_collab_db()}.toy_diabetes"

# retrieve the dataframe
spark_df = spark.table(table_name)

# show the spark dataframe
display( spark_df )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding the data
# MAGIC
# MAGIC Now that we have retrieved our data and it is in an appropriate format we can now move on. At this stage it may be useful to try and visualise the data and get some metrics to get a better appreciation of the data were trying to model.

# COMMAND ----------

# reformat the data to a pandas dataframe
import pandas as pd
pandas_df = spark_df.toPandas()

# We are going to add the target variable to this dataframe to make viewing it correlations easier
# column called progression as target represents quantative measure of disease progression after 1 year
pandas_df['progression'] = diabetes.target

# view the first few rows of the dataset
display(pandas_df.head(10))

# COMMAND ----------

# DBTITLE 1,View some general information about the dataset
display( pandas_df.info() )

# COMMAND ----------

# DBTITLE 1,View the summary statistics
# get some metrics for each the columns
display(pandas_df.describe())

# COMMAND ----------

# DBTITLE 1,Visualise the linear correlation between the variables
import matplotlib.pyplot as plt
import seaborn as sns
correlation = pandas_df.corr()
plt.subplots(figsize=(12,6))
sns.heatmap(correlation, cmap='RdYlGn', annot=True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This plot shows the linear correlation of variables between one another and potentially more importantly how well the linearly correlate to the target column `progression`. 
# MAGIC
# MAGIC At this stage we may chose to do something called **feature selection** where we may remove some of the less useful variables and focus in on a few more useful variables.
# MAGIC
# MAGIC However that would be an example unto itself, so instead we will continue using all of the variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling the data
# MAGIC
# MAGIC Now that we have some insight int the dataset we are now ready to begin using linear regression to model the data

# COMMAND ----------

# 1. create new variables with all independant variables, and target variable
X = pandas_df.drop(labels='progression', axis=1)
y = pandas_df['progression']

# 2. create our test train split, note random state is used to make notebook reproducable in practice dont use
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# verify the output shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 3. fit a linear regression model ot the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # training the algorithm

# 4. generate our predicitions from the test data
y_pred = regressor.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the Model
# MAGIC The final step is to evaluate the performance of our model, this is important to compare how well different algorithms perform on a particular dataset.
# MAGIC
# MAGIC For regression, three evaluation metrics are commonly used:
# MAGIC - **Mean Absolute Error** is the mean of the absolute value of the errors
# MAGIC $$ MAE = \frac{1}{n} \sum^{n}_{j=1}|y_i - y_j| $$
# MAGIC - **Mean Squared Error** is the mean of th esquared errors
# MAGIC $$ MSE = \frac{1}{N} \sum^{n}_{j=1}(y_i - y_j)^2 $$
# MAGIC - **Root Mean Squared Error** is the square root of the mean squared errors
# MAGIC $$ MSE = \sqrt{\frac{1}{N} \sum^{n}_{j=1}(y_i - y_j)^2} $$

# COMMAND ----------

from sklearn import metrics as mt
import numpy as np
print("The model exlains,", np.round(mt.explained_variance_score(y_test, y_pred)*100,2), "% variance of the target w.r.t feature is" )
print('Mean Absolute Error     :', mt.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error      :', mt.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error :', np.sqrt(mt.mean_squared_error(y_test, y_pred)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Statsmodels
# MAGIC
# MAGIC Stats models is another package we can use to create a model, the difference is that using statsmodels gives a better summary of the model parameters.
# MAGIC
# MAGIC To use this model we repeat all of the previous steps however when we create the model instead of using `LinearRegression()` we now use this new model instead.
# MAGIC
# MAGIC There is no train-test split since accuracy is calculated based on the closeness of predicitions.

# COMMAND ----------

import statsmodels.api as sm
pandas_df.head(0)

# COMMAND ----------

# note were still using the df which has the target column (progression) included
# were instead going to use the column names to tell the model ;
# which columns are independant variables and which column is the target
regressor2 = sm.OLS.from_formula("progression ~ age+sex+bmi+s1+s2+s3+s4+s5+s6", data=pandas_df)

# fit the model
regressor2 = regressor2.fit()

# get our predictions
y_pred2 = regressor2.predict(X)

y_pred2.head()

# COMMAND ----------

# DBTITLE 1,We can now view much more insightful summary results
summary_model = regressor2.summary()
print(summary_model)
