# Databricks notebook source
# MAGIC %md # Introduction

# COMMAND ----------

# DBTITLE 1,Pyspark Code Example
# MAGIC %md This notebook was created as a helpful resource when coding with PySpark. Use the table of contents on the left hand side for quick navigation throughout the notebook.
# MAGIC 
# MAGIC <br>
# MAGIC <ul>
# MAGIC   <li>Import any necessary <strong>libraries</strong> and create a spark session
# MAGIC   <li><strong>List</strong> all tables
# MAGIC   <li><strong>Read</strong> in a table
# MAGIC   <li><strong>Query</strong> data to select specific columns and rows
# MAGIC   <li>Wrangle data by <strong>grouping</strong> and <strong>aggregating</strong>
# MAGIC   <li>Create new features by adding a calculated column
# MAGIC   <li><strong>Convert</strong> the data to a <strong>Pandas</strong> DataFrame for plotting
# MAGIC   <li>Create a bar plot of the new feature
# MAGIC     <li><strong>Filter</strong>
# MAGIC       <ul>
# MAGIC   <li>By string
# MAGIC   <li>By a list of strings
# MAGIC       </ul>
# MAGIC   <li><strong>Extract</strong> characters from a variable
# MAGIC   <li><strong>Combine</strong> values from two columns into single string with a delimiter
# MAGIC   <li>Standard statistical tests
# MAGIC   <li>Dataset <strong>Merging</strong>
# MAGIC     <ul>
# MAGIC   <li>Outer join
# MAGIC   <li>Left join with subsets
# MAGIC   <li>Right join with subsets
# MAGIC     </ul>
# MAGIC   <li><strong>Sorting</strong>
# MAGIC     <ul>
# MAGIC   <li>Sort table by an ascending value
# MAGIC   <li>Sort table by descending value
# MAGIC   <li>Sort table by two values
# MAGIC     </ul>
# MAGIC   <li><strong>Outputting and sharing</strong>
# MAGIC   <li><strong>Useful links</strong>
# MAGIC     </ul>



# COMMAND ----------

# MAGIC %md # Import any necessary libraries and create a spark session

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import avg, min, max, mean, stddev, col, substring, concat, lit

spark = SparkSession.builder.appName("Pyspark Demo").getOrCreate()

# COMMAND ----------

# MAGIC %md Show the databases available in the agreement

# COMMAND ----------

spark.sql("SHOW DATABASES").show(truncate=False)

# COMMAND ----------

# MAGIC %md # List all tables

# COMMAND ----------

# DBTITLE 0,List all tables
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

result = spark.sql(f"SHOW TABLES IN {get_collab_db()}").toPandas()
print(result)


# COMMAND ----------

# MAGIC %md # Read in a table

# COMMAND ----------

# DBTITLE 0,Read in a table
df = spark.table(f"{get_collab_db()}.toy_mtcars")
display(df)

# COMMAND ----------

# MAGIC %md # Query data to select specific columns and rows

# COMMAND ----------

# DBTITLE 0,Query data to select specific columns and rows
df_filtered = df.select("mpg", "disp", "hp", "qsec").filter(df["gear"] > 4)
display(df_filtered)

# COMMAND ----------

# MAGIC %md # Wrangle data by grouping and aggregating

# COMMAND ----------

# DBTITLE 0,Wrangle data by grouping and aggregating
wrangled_data = df_filtered.groupBy("hp").agg(avg("qsec").alias("avg_qsec"), min("disp").alias("min_disp"))
display(wrangled_data)

# COMMAND ----------

# MAGIC %md # Create new features by adding a calculated column

# COMMAND ----------

# DBTITLE 0,Create new features by adding a calculated column
wrangled_data = wrangled_data.withColumn("new_feature", wrangled_data["avg_qsec"] + wrangled_data["min_disp"])
display(wrangled_data)

# COMMAND ----------

# MAGIC %md # Convert the data to a Pandas DataFrame for plotting

# COMMAND ----------

# DBTITLE 0,Convert the data to a Pandas DataFrame for plotting
plot_data = wrangled_data.toPandas()

# COMMAND ----------

# MAGIC %md # Create a bar plot of the new feature

# COMMAND ----------

# DBTITLE 0,Create a bar plot of the new feature
display(plot_data.plot(kind="bar", x='hp', y="new_feature"))

# COMMAND ----------

# MAGIC %md Access this link for a usefull cheat sheet for PySpark for Data Scientists:
# MAGIC https://images.datacamp.com/image/upload/v1676302905/Marketing/Blog/PySpark_SQL_Cheat_Sheet.pdf

# COMMAND ----------

# MAGIC %md # Filter by word

# COMMAND ----------

# DBTITLE 0,Filter by word
df_2 = spark.table(f"{{get_collab_db()}}.toy_iris")

filtered_df_2 = df_2.filter(col("Species").like("%setosa%"))

display(filtered_df_2)

# COMMAND ----------

# MAGIC %md # Filter by a list of strings

# COMMAND ----------

# DBTITLE 0,Filter by a list of strings
search_strings = ['setosa', 'versicolor', 'virginica']
filtered_list_df_2 = df_2.filter(col('Species').isin(search_strings))

display(filtered_list_df_2)

# COMMAND ----------

# MAGIC %md # Extract the first three characters of the "Species" column

# COMMAND ----------

# DBTITLE 0,Extract the first three characters of the "Species" column
substring_df = filtered_df_2.withColumn("first_3_chars", substring(col("Species"), 1, 3))
substring_df.show()

# COMMAND ----------

# MAGIC %md # Combine values from two columns into single string with a delimiter

# COMMAND ----------

# DBTITLE 0,Combine the "Sepal_length" and "Species" columns into a single string with a delimiter of ": "
delimited_df = df_2.withColumn("delimited", concat(col("Sepal_Length"), lit(": "), col("Species")))
delimited_df.show()

# COMMAND ----------

# MAGIC %md # Basic statistical tests

# COMMAND ----------

# DBTITLE 0,Basic statistical tests
# Get the summary statistics for the Sepal_length column
df_2.agg(mean("Sepal_length"), stddev("Sepal_length"), min("Sepal_length"), max("Sepal_length")).show()

# Count the number of occurrences for each species
df_2.groupBy("Species").count().show()


# COMMAND ----------

# MAGIC %md # Outer join (when only one column has matching values in both tables)

# COMMAND ----------

# DBTITLE 0,Outer join (when only one column has matching values in both tables)
# Rename the value so it matches in both columns

df_2 = df_2.withColumnRenamed('Sepal_Length', 'mpg')

# Merge 

merged_df = df.join(df_2, 'mpg', 'outer')

display(merged_df)

# COMMAND ----------

# MAGIC %md # Left join with subsets

# COMMAND ----------

# DBTITLE 0,Left join with subsets
df_subset = df.select('cyl','disp', 'mpg')
df_2_subset = df_2.select('Petal_Length', 'mpg')

joined_df = df_subset.join(df_2_subset, 'mpg', 'left')

display(joined_df)

joined_df.count

# COMMAND ----------

# MAGIC %md # Right join with subsets

# COMMAND ----------

# DBTITLE 0,Right join with subsets
df_subset = df.select('cyl','disp', 'mpg')
df_2_subset = df_2.select('Petal_Length', 'mpg')

joined_df = df_subset.join(df_2_subset, 'mpg', 'right')

display(joined_df)

joined_df.count

# COMMAND ----------

# MAGIC %md # Sort table by an ascending value

# COMMAND ----------

# DBTITLE 0,Sort table by an ascending value
sorted_df = df.orderBy('mpg')

display(sorted_df)

# COMMAND ----------

# MAGIC %md # Sort table by descending value

# COMMAND ----------

# DBTITLE 0,Sort table by descending value
sorted_df = df.orderBy(df['mpg'].desc())

display(sorted_df)

# COMMAND ----------

# MAGIC %md # Sort table by two values

# COMMAND ----------

# DBTITLE 0,Sort table by two values
# True is for ascending, False descending

sorted_df = df.orderBy(['mpg', 'hp'], ascending=[False,True])

display(sorted_df)

# COMMAND ----------

# MAGIC %md # Outputting and sharing

# COMMAND ----------

# MAGIC %md
# MAGIC Guidance on how to work collaboratively and exporting/downloading results can be found on the Secure Data Environment portal here:
# MAGIC https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment/user-guides/using-databricks-in-sde
# MAGIC https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment/user-guides/output-your-results
# MAGIC https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment/user-guides/using-gitlab-in-sde

# COMMAND ----------

# MAGIC %md # Useful links

# COMMAND ----------

# MAGIC %md
# MAGIC Official AWS PySpark cheatsheet for Data Science can be found here:
# MAGIC https://images.datacamp.com/image/upload/v1676302905/Marketing/Blog/PySpark_SQL_Cheat_Sheet.pdf
