# Databricks notebook source
# MAGIC %md
# MAGIC ## 2D Linear Regression
# MAGIC
# MAGIC #### Description
# MAGIC
# MAGIC This notebook is designed to provide a very basic insight into linear regression and how to utilise sklearn to perform it on datsets.
# MAGIC
# MAGIC In this notebook linear regression is performed on a dataset with 2 numeric variables, its aim is to explain the basic principles of linear regression before moving onto the second notebook which demonstrates linear regression on a multi-variable problem.
# MAGIC
# MAGIC Linear regression uses algebra to define the linear relationship between two or more variables. In 2-dimensional space, this linear relationship can be seen as the 'line-of-best-fit', a straight line that best represents the relationship between the 2 variables. This relationship holds as we add more variables though the line exists in higher dimensions and is hard to visualise through standard means.
# MAGIC
# MAGIC This linear relationship can then be used as a method for helping predicitions.
# MAGIC
# MAGIC #### SKLearn performance in databricks
# MAGIC
# MAGIC While SKLearn can be useful in certain situations, it is not designed to take advantage of cluster computing resources, which arguably is a major downside to using it inside databricks as you are not utilising the full proccessing power available to you.
# MAGIC
# MAGIC This is not us saying do not use sklearn as it may well be appropriate for certain tasks, however if your are performing tasks over large datasets and want to fully exploit the compute resources you have available to complete these tasks. Then you should look into the Spark `MLlib` library.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retrieve the data
# MAGIC
# MAGIC In this example the toy datasets have already created and added to the collab database to mimic an actual workflow, we will use a general function to get the database name however this can be can be replaced with a string.
# MAGIC
# MAGIC The utility functions are imported via the next command which runs the notebook stored in a different location. You can view these functions by navigating to the folder or you can also click the link in the next command. This can also be a useful way to tidy up your code and store frequently used functions in their own notebook to be imported into others.

# COMMAND ----------

# DBTITLE 1,Import python utility functions
# MAGIC %run ../../Wrangler-Utilities/Utilities-Python

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# get the table name
table_name = f"{get_collab_db()}.toy_2d_linear_regression"

# retrieve the dataframe
spark_df = spark.table(table_name)

# show the spark dataframe
display( spark_df )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding the Data
# MAGIC
# MAGIC As a first step before we move on to using creating potentially complex models, it may be useful to get some quick insights into the dataset. This way when moving forward we have a general appreciation of the contents of the dataset.
# MAGIC
# MAGIC There is many ways to do this, here we will show the inbuilt describe method and also how to create a simple plot of the data. 
# MAGIC
# MAGIC *note: because this data is 2d in nature, plots are quite straightforward, more complex visualisation methods are needs for multivariable data*

# COMMAND ----------

# using .describe() gives us insight into some basic metrics of a dataframe
# we can also pass in column names e.g. .describe(['feature']), to isolate columns
display(spark_df.describe())

# COMMAND ----------

# to plot the data we must first convert it to a NumPy array or a Pandas dataframe

# Convert from spark dataframe to pandas dataframe
pandas_df = spark_df.toPandas()

# extract the feature and target columns
X = pandas_df['feature']
y = pandas_df['target']

# plot the data
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker='o')

plt.title("Plot of the Random Regression Dataset", fontsize="large")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Utilising SKLearn Linear Regression
# MAGIC In the above plot of the data we can see that there is a clear pattern in the data. But now suppose we want to model the exact linear relationship of this dataset.
# MAGIC
# MAGIC This is where we can utilise the sklearn LinearRegression function to aid us. To utilise the sklearn methods we must have a pandas dataframe not a spark dataframe.

# COMMAND ----------

# convert spark dataframe to pandas dataframe
pandas_df = spark_df.toPandas()

# extract the 2 features we want into seperate variables
X = pandas_df['feature']
y = pandas_df['target']

# split the data into training and test sets
# note the random_state variable is used so split is same every time, in practice ignore
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

# models require 2d Arrays so reformat
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

# fit a linear regression model ot the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # training the algorithm

# generate our predicitions from the test data
y_pred = regressor.predict( X_test )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualising our Predictions
# MAGIC
# MAGIC There is many different ways we can now visualise our predictions:
# MAGIC - We can plot a figure of the scattered test plots and our predicted line
# MAGIC - We can display a table showing the actual test values vs our predicted values
# MAGIC - We can then plot a figure of this table to visualise it
# MAGIC
# MAGIC These are just few examples of course there is many more ways to gain insight.

# COMMAND ----------

# we can extract the exact intercetp and coefficient of the slope
print("Intercept   : ", regressor.intercept_)
print("Coefficient : ", regressor.coef_)


# plot the figure
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.title("Prediction vs Test data", fontsize="large")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# COMMAND ----------

# create table view of actual values vs predictions
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
display(df)

# COMMAND ----------

# visualise above table as a bar chart, note were only visualising the first 20
df1 = df.head(20)
df1.plot(kind='bar', figsize=(10,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluating our Model
# MAGIC
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print('Mean Absolute Error     :', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error      :', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error :', np.sqrt(mean_squared_error(y_test, y_pred)))
