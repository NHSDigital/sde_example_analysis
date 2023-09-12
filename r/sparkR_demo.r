# Databricks notebook source
# MAGIC %md <h1>Using R within Databricks</h1>
# MAGIC <h2>Shoaib Ajaib</h2>
# MAGIC <i>Last Updated: 03/08/2023</i>

# COMMAND ----------

# MAGIC %md <h3>Tutorial Objectives</h3>
# MAGIC In this tutorial we will:
# MAGIC <ul>
# MAGIC <li> Understand basic concepts of Spark
# MAGIC <li> Explore data
# MAGIC <li> Perform interactive analysis
# MAGIC <!-- <li> Run machine learning algorithms in SparkR -->
# MAGIC </ul>
# MAGIC
# MAGIC <h3>Intended Audience</h3>
# MAGIC People familiar with R and those new to SparkR
# MAGIC </br></br>
# MAGIC <b>Reference:</b> [SparkR (R on Spark)](https://spark.apache.org/docs/latest/sparkr.html)

# COMMAND ----------

# MAGIC %md <h3>Understand basic concepts of Spark</h3>
# MAGIC * What is Spark?
# MAGIC * Apache Spark Components
# MAGIC * Why Spark?
# MAGIC * Spark supports multiple languages
# MAGIC * Spark Concepts: Driver and Executors, Resilient Distributed Datasets, SparkDataFrame, Operations

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC <h4>What is Spark?</h4>
# MAGIC
# MAGIC Imagine you're hosting a big cooking competition where you need to prepare a huge feast for a lot of hungry guests. You have a massive kitchen with many different cooking stations and chefs. However, the challenge is that there's so much cooking to do that it's impossible for just one chef to handle everything in a reasonable amount of time.
# MAGIC
# MAGIC This is where Spark comes in. Think of Spark as a master chef and organizer. It divides the cooking tasks into smaller, manageable steps and assigns each step to a different chef at their cooking station. These chefs work in parallel, preparing their assigned dishes simultaneously.
# MAGIC
# MAGIC As each chef finishes their part, Spark orchestrates the process. It combines the dishes from different stations to create the entire feast. If a chef finishes early, they can help out other chefs who might be struggling with more complex recipes. This teamwork, coordination, and division of labor orchestrated by Spark result in a delicious and well-cooked banquet prepared in record time.
# MAGIC
# MAGIC In the world of data, Spark functions similarly. Instead of cooking, it helps process and analyze massive amounts of data. It breaks down the data tasks into smaller chunks and assigns them to different computers (called nodes). Just like the chefs, these computers work together to process the data. Once the processing is done, Spark combines the results to give you valuable insights from your data, all done much faster than if you were trying to handle it all on your own.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Why use Spark?</h4>
# MAGIC <br></br>
# MAGIC <ul>
# MAGIC <li><b>Big Data Processing:</b> When you have a massive amount of data that's too large to be processed by a single computer, Spark can distribute the workload across multiple computers. This allows you to process, transform, and analyze the data much faster.
# MAGIC </br></br>
# MAGIC
# MAGIC <li><b>Speed:</b> Spark is designed to be very fast. It keeps data in memory whenever possible, which means that tasks can be performed much more quickly compared to traditional data processing tools that need to read and write from storage.
# MAGIC </br></br>
# MAGIC
# MAGIC <li><b>Versatility:</b> Spark provides a wide range of libraries and tools for various data processing tasks. You can use it for data cleaning, machine learning, graph processing, streaming data analysis, and more, all within a single framework.
# MAGIC </br></br>
# MAGIC
# MAGIC <li><b>Ease of Use:</b> Spark offers user-friendly APIs in multiple programming languages like Python, Java, Scala, and R. This makes it accessible to a wide range of developers and data scientists, regardless of their programming background.
# MAGIC </br></br>
# MAGIC
# MAGIC <li><b>Fault Tolerance:</b> Spark is resilient to failures. If a computer or node fails during a computation, Spark can recover and continue the task from where it left off, ensuring that your data processing jobs are reliable and uninterrupted.
# MAGIC </br></br>
# MAGIC
# MAGIC <li><b>Scalability:</b> As your data grows, you can easily scale Spark by adding more computers to your cluster. This means it can handle increasing amounts of data without sacrificing performance.
# MAGIC </br></br>
# MAGIC
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md 
# MAGIC <h4>Key Spark Concepts</h4>
# MAGIC
# MAGIC You will encounter many concepts and terms when using Spark for the first time which may be confusing and difficult to understand: the good news is that, the key concepts are relativley simple to understand:
# MAGIC <br></br>
# MAGIC
# MAGIC <ol>
# MAGIC <li><b>Driver and Executors:</b>
# MAGIC <br></br>
# MAGIC Think of Spark like a big construction project. The "Driver" is like the project manager who has the blueprint and coordinates everything. The "Executors" are the construction workers who do the actual building. The manager (Driver) plans what needs to be built, and the workers (Executors) follow the plan and construct different parts of the project. The manager oversees and collects updates from the workers to ensure the project is on track.
# MAGIC <br></br>
# MAGIC
# MAGIC <li><b>Resilient Distributed Datasets (RDDs):</b>
# MAGIC <br></br>
# MAGIC Imagine you have a library with many books. You decide to make copies of some books and distribute them to friends in case some books get damaged. These distributed copies of books represent RDDs. Each friend has a portion of the library's books, so even if one friend's copies get damaged, the original books and other copies are safe.
# MAGIC <br></br>
# MAGIC
# MAGIC <li><b>Spark DataFrames:</b>
# MAGIC <br></br>
# MAGIC A DataFrame is like a well-organized collection of spreadsheets. Imagine you're managing a store and you have spreadsheets for different product categories: one for electronics, one for clothing, and so on. Each spreadsheet has rows for items and columns for details like price and quantity. These individual spreadsheets collectively form a DataFrame, making it easier to analyze and compare information.
# MAGIC <br></br>
# MAGIC
# MAGIC <li><b>Operations:</b>
# MAGIC <br></br>
# MAGIC In Spark, operations are like actions you take to manipulate and analyze your data:
# MAGIC
# MAGIC <ul>
# MAGIC <li><b>Transformations:</b> These are like making changes to your data without affecting the original. For example, you have a list of people's ages, and you want to calculate their birth years. You create a new list of birth years without changing the original ages. Similarly, Spark's transformations create new datasets from existing ones based on your instructions.
# MAGIC
# MAGIC <li><b>Actions:</b> These are operations that trigger actual computations and provide results. For instance, if you want to find the average age of a group of people, Spark calculates it based on the data you've transformed and returns the result.
# MAGIC </ul>
# MAGIC </ol>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Putting it all together...** 
# MAGIC
# MAGIC Imagine you're building a city. The project manager (Driver) plans the city layout and coordinates different construction teams (Executors) to build roads, houses, and buildings. Each construction team has a portion of the blueprint (RDDs) to work on, ensuring that even if one team faces challenges, the project progresses. The city's information, like population and infrastructure, is organized into different sections (DataFrames), making it easier to analyze and manage. Transformations are like adding new features to the city without changing the original plan, and actions are like getting specific data insights from the city's development.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <h4>The Spark and R Workflow </h4>
# MAGIC
# MAGIC If you are new to spark and coming here from an R background, it is useful to understand the general workflow, as this is different to how you may have been working in R on the virtual desktop instance (VDI).
# MAGIC
# MAGIC When working with Spark and R you typically will:
# MAGIC <br></br>
# MAGIC <ol>
# MAGIC <li> Load any big data as a SparkDataFrame
# MAGIC
# MAGIC <li> Transform and aggregate data using SparkR native functions
# MAGIC
# MAGIC <li> Use traditional R on transformed/subsetted data for downstream processing/visualisation
# MAGIC </ol>
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC # Before you start
# MAGIC <br>
# MAGIC
# MAGIC * Make sure this notebook is "Attached" to a cluster that is running 
# MAGIC * Use can use `<Ctrl>+<Enter>` to run the code in a shaded cell

# COMMAND ----------

# MAGIC %md
# MAGIC ####Connect to Spark
# MAGIC **Important note:**
# MAGIC If you are looking at documentation online, you may see code which starts by creating a `SparkSession` or initialising a `SparkContext`: you **DO NOT** need to create these in Databricks as they will be created for you.
# MAGIC
# MAGIC We can check confirm this using:

# COMMAND ----------

# check that spark Context has been created
SparkR::sparkR.conf()$spark.databricks.sparkContextId

# COMMAND ----------

# MAGIC %md
# MAGIC We can also look at the Spark driver maxResultSize, which limits the amount of data you can send back to the driver to avoid a driver out of memory (OOM) exception:

# COMMAND ----------

SparkR::sparkR.conf()$spark.driver.maxResultSize

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load R Packages (including SparkR)
# MAGIC
# MAGIC **Important Note:**  Some R packages have the same named functions as SparkR and will end up masking these functions if loaded directly after loading SparkR. In order to combat this, you can use specific functions from packages by using `::` notation, e.g. to use `mutate` from the `dplyr` package you can do `dplyr::mutate()`. You can also opt to invoke SparkR functions directly, rather than loading the entire package with `library(SparkR)` 

# COMMAND ----------

library(SparkR)
library(magrittr)
library(stringr)
library(glue)

# COMMAND ----------

# MAGIC %md <h3>Explore data</h3>
# MAGIC <br>
# MAGIC Objectives: </br>
# MAGIC
# MAGIC * Understand the differences between executing commands against an R data frame and a Spark data frame
# MAGIC * Import user-defined functions stored in a databricks notebook
# MAGIC * Learn how to bring data into SparkR
# MAGIC * Learn how to perform structured data processing</br></br>

# COMMAND ----------

# MAGIC %md
# MAGIC For this tutorial we will be using the "mtcars" dataset which is provided as an in-built dataset in R, and contains infromation relating to motor car trends in the US from 1973-74.

# COMMAND ----------

# R documentation for the mtcars dataset
# the method can also be used to view package documentation
?mtcars

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create a SparkDataFrame
# MAGIC
# MAGIC We previously mentioned how SparkDataFrame’s are spreadsheet-like objects that are optimised to be utilised with Spark in a distributed manner. Whilst, these DataFrames may look like their R equivalents, it is important to note that they are NOT interchangeable, but you can easily convert between the two types, by using:
# MAGIC
# MAGIC `data.frame()` - to convert a SparkDataFrame to its R equivalent  
# MAGIC
# MAGIC `SparkR::CreateDataFrame()` - to convert an R data.frame to its equivalent  
# MAGIC
# MAGIC
# MAGIC When working with both SparkDataFrame’s and R data.frame’s it is often useful to prefix the table names with `sdf` and `rdf` to avoid confusion and keep track of what type of object you are working with.
# MAGIC
# MAGIC

# COMMAND ----------

# create R data.frame
mtcars_rdf <- data.frame(mtcars)

# create SparkDataFrame
mtcars_sdf <- createDataFrame(mtcars_rdf)


# COMMAND ----------

# cache in memory the SparkDataFrame
persist(mtcars_sdf, "MEMORY_ONLY")
### persist() - Persist this SparkDataFrame with the specified storage level. For details of the supported storage levels, refer to http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence.

# COMMAND ----------

# examine the structure of the R data.frame
str(mtcars_rdf)

# COMMAND ----------

object.size(mtcars_rdf)
### object.size() - Report the Space Allocated for an Object [in R]
### Provides an estimate of the memory that is being used to store an R object.

# COMMAND ----------

# examine the structure of the SparkDataFrame
str(mtcars_sdf)

# COMMAND ----------

object.size(mtcars_sdf)
# this is only the memory in R, not in the Spark context
# check 'Storage' in Spark UI for a persisted object

# COMMAND ----------

# if you are a Pythonista, show() doesn't work the same way in SparkR
# in pyspark, show() is an action
show(mtcars_sdf)
### show() - Print the SparkDataFrame column names and types
### similar to str() in R

# COMMAND ----------

# another way to look at the structure of a SparkDataFrame
printSchema(mtcars_sdf)
### printSchema() - Prints out the schema in tree format

# COMMAND ----------

# view the data in R data.frame
mtcars_rdf

# COMMAND ----------

# how do we view the data in a SparkDataFrame?
mtcars_sdf

# COMMAND ----------

# view the data with collect()
collect(mtcars_sdf)
### collect() - Collects all the elements of a SparkDataFrame and coerces them into an R data.frame

# COMMAND ----------

# MAGIC %md 
# MAGIC <b>REMEMBER</b>: 
# MAGIC <br></br>
# MAGIC `collect()` will bring the results of the operation to the driver node.</br>
# MAGIC Large datasets will exceed the memory of the driver node and produce an OOM error.</br>
# MAGIC Use other actions instead to get the results: `first()`, `head()`, `showDF()` or `take()`</br>
# MAGIC
# MAGIC Read more about actions here: <i> http://spark.apache.org/docs/latest/programming-guide.html#actions </i>

# COMMAND ----------

first(mtcars_sdf)
### first() - Return the first row of a SparkDataFrame

# COMMAND ----------

head(mtcars_sdf)
### head() - Return the first NUM rows of a SparkDataFrame as a R data.frame. If NUM is NULL, then head() returns the first 6 rows in keeping with the current data.frame convention in R.

# COMMAND ----------

showDF(mtcars_sdf)
### showDF() - Print the first numRows rows of a SparkDataFrame
### by default returns the first 20 rows

# COMMAND ----------

take(mtcars_sdf, 10)
### take() - Take the first NUM rows of a SparkDataFrame and return a the results as a R data.frame
### no default value for NUM rows

# COMMAND ----------

# another way to view the data
# Databricks feature; not part of Apache SparkR
display(mtcars_sdf)
# very useful for quick visualizations
# change graph type to "Scatter"
# show sample based on first 1000 rows

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import User-defined Functions
# MAGIC When working within databricks notebooks it can be useful to orgainse code and keep frequently used commands in a central notebook that can be called directly from within another databricks notebook. Here we have provided some useful helper functions to which are located in the `Wrangler-Utilities` folder. In order to load these functions, we can use the special databricks `%run` command followed by the path to the notebook where the functions are stored.
# MAGIC
# MAGIC The path given below is stated as the relative path (relative to this notebook), and the `..` denotes that databricks should start by looking on folder up from the current notebook.  

# COMMAND ----------

# MAGIC %run ../Wrangler-Utilities/Utilities-R

# COMMAND ----------

# Uncomment line to use
# get_user() # Gets the current user 
get_all_databases() # Get a vector of all databases
# get_core_db() # Get the core agreement database
# get_collab_db() # Get the agreement collab database
# get_ref_db() # Get the agreement reference database

# COMMAND ----------

# MAGIC %md
# MAGIC The `get_tables()` utility function can be used to return all the tables within a given database. You can view the tables either as a R `data.frame` by setting `return_list = FALSE` and using the `display()` method:

# COMMAND ----------

# Return a table which details all the tables that are present within the core database under your agreement. 
core_tbls = get_tables(db_name = get_core_db(), return_list=F)

display(core_tbls)
# We are returned with the table name, database path and the full table paths

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Alternativley, you can also return the tables names as a list by setting `return_list=TRUE`. For each table in the database, this function will return a list of length = 2, which inturn comprises of two named vectors: `name` and `full_name`, which correspond to the table name and the table path, respectivley. 
# MAGIC
# MAGIC To eaxtract either the table name or the full table path, you can index the returned list using the usual R syntax, with `$`:

# COMMAND ----------

# Get a list of all tables within the main databases
core_tbls = get_tables(db_name = get_core_db(), return_list=T)
collab_tbls = get_tables(db_name = get_collab_db(), return_list=T)
ref_tbls = get_tables(db_name = get_ref_db(), return_list=T)

# COMMAND ----------

# View the table name and full path(including the database) for calendar table in the reference database  
ref_tbls$calendar

# COMMAND ----------

# We can extract either the table name or the full path (needed to load the table)
print(ref_tbls$calendar[["name"]])
print(ref_tbls$calendar[["full_name"]])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Bring Data into SparkR
# MAGIC
# MAGIC The most common workflow you will encounter in databricks will be to load a table from one of the provisioned databases you have access to under your data sharing agreement. In order to do this, we have provided the `load_tbl()` utility function which is a wrapper around the `SparkR::sql()` function that takes a table name element from the `get_tables()` function that we described above. 
# MAGIC
# MAGIC In the below example we will load in the mtcars tables from the collab database:

# COMMAND ----------

example_sdf_tbl = load_tbl(collab_tbls$toy_mtcars)

# COMMAND ----------

# MAGIC %md <h3>Perform interactive analysis</h3>

# COMMAND ----------

# a common R task - looking at summary statistics
summary(mtcars_rdf)

# COMMAND ----------

# calculate summary statistics of the SparkDataFrame
summary(mtcars_sdf)
### summary() - Computes statistics for numeric columns. If no columns are given, this function computes statistics for all numerical columns.
# why didn't it show the statistics?

# COMMAND ----------

# lazy execution
summ_stats <- summary(mtcars_sdf)
collect(summ_stats) # action

# COMMAND ----------

# can also use describe() for summary statistics 
collect(describe(mtcars_sdf))
# Unlike R, summary() and describe() only provide statistics for numerical columns of a SparkDataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC <i>"Just because you can doesn't mean you should"</i> </br>
# MAGIC `summary()` and `describe()` are expensive operations for large datasets.
# MAGIC They calculate 5 different statistics for each one of the numerical columns.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Aggregate functions
# MAGIC Aggregate functions perform a calculation on a set of values and return a single value.
# MAGIC Some of the aggregate functions in SparkR are: 
# MAGIC `approxCountDistinct()`, `avg()`, `count()`, `countDistinct()`, `first()`,
# MAGIC `last()`, `max()`, `mean()`, `min()`, `sum()`, `sumDistinct()`.
# MAGIC </br>
# MAGIC
# MAGIC **Syntax:** 
# MAGIC </br>
# MAGIC `df2 <- agg(df, <column> = "<aggFunction>")` OR </br>
# MAGIC `df2 <- agg(df, newColName = <aggFunction>(df$column))`</br>
# MAGIC </br>
# MAGIC So, let's find the **average wait time between eruptions** and the **average duration of an eruption** using aggregate functions.

# COMMAND ----------

# calculating average 1/4 mile time in seconds for all cars
avg_qsec <- agg(mtcars_sdf, qsec = "mean")
first(avg_qsec)

# COMMAND ----------

# calculating average Gross horsepower for all vechicles
avg_hp <- agg(mtcars_sdf, avg_hp = mean(mtcars_sdf$hp))
first(avg_hp)
# using only `mean(eruptions)` does not work; need to specify SparkDataFrame$column

# COMMAND ----------

# notice that the column name of the result becomes 'avg_duration' as specified above
result <- first(avg_hp)
result$avg_hp
# this value can now be used in other calculations

# COMMAND ----------

# can perform more than one aggregation at the time
# find the minimum and maximum engine displacement in mtcars
displacement <- first(agg(mtcars_sdf, 
                      min_disp = min(mtcars_sdf$disp), 
                      max_disp = max(mtcars_sdf$disp)))
displacement

# COMMAND ----------

# MAGIC %md 
# MAGIC #### SparkDataFrame Operations
# MAGIC To perform structured data processing, we often need to select a subset of the dataset.  Some of the operations that can be used in SparkDataFrames include: `groupBy()`, `select()`, `filter()`, `where()`, and `arrange()`.

# COMMAND ----------

# to find out how many cars we have with the same number of engine cylinders we can group our table by 
# the number of cylinders column and then perform a count

mtcars_sdf_group <- groupBy(mtcars_sdf, "cyl")
### groupBy() - Groups the SparkDataFrame using the specified columns, so we can run aggregation on them

cylinder_counts <- count(mtcars_sdf_group)

head(cylinder_counts)

# COMMAND ----------

# quick visualization
display(cylinder_counts)
# select the `+` sign next to the `Table` dropdown and the `Vizualization` to view a plot

# COMMAND ----------



# COMMAND ----------

# let's do a proper plot using ggplot2

library(ggplot2)

# plot() does not work with SparkDataFrame so, need to collect() to display the results
plot_data = collect(cylinder_counts)
plot_data$cyl = factor(plot_data$cyl)

ggplot(plot_data, aes(x = cyl, y=count, fill = cyl)) +
geom_col() +
theme_classic()


# COMMAND ----------

# sort in descending order to have the highest engine displacement vechicle be the top result

sorted_disp <- arrange(mtcars_sdf, desc(mtcars_sdf$disp))
### arrange() - Sort a SparkDataFrame by the specified column(s) 

head(sorted_disp)

# COMMAND ----------

# Lets look at the distribution of mpg for the cars with the highest number of cylinders in this dataset 

most_common <- first(agg(mtcars_sdf, max_cyl = max(mtcars_sdf$cyl)))

most_common_rows <- filter(mtcars_sdf, mtcars_sdf$cyl == most_common$max_cyl)
### filter(), where() - Filter the rows of a SparkDataFrame according to a given condition

high_cyl_mpg <- select(most_common_rows, "mpg")
### select() - Selects a set of columns with names or Column expressions

# let's visualize the mpg distribution for the cars which have the highest number of cylinders
display(high_cyl_mpg)
# Select `+` and then change chart type to Box plot with mpg on the y axis to display plot

# COMMAND ----------

# finished the analysis, let's release the memory for the SparkDataFrame
unpersist(mtcars_sdf)
### unpersist() - Mark this SparkDataFrame as non-persistent, and remove all blocks for it from memory and disk.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save SparkDataFrame Tables
# MAGIC You can also save any aggregated tables within your collaboration database: these tables are also accesible within RStudio on your virtual desktop instance via the ODBC connection.
# MAGIC
# MAGIC We have provided the `save_tbl` function which is a wrapper around `SparkR::saveAsTable` that takes as input either a SparkDataFrame or R data.frame; a character table name and automatically saves this table within your corresponding collaboration database within databricks. By default the function appends, but this can be controlled by altering the `mode` argument. 
# MAGIC

# COMMAND ----------

# The mode can be one of 'overwrite', 'append', 'error', 'ignore'
save_tbl(mtcars_sdf, "SparkR_example_df")


# COMMAND ----------

# Check if the table was created
collab_tbls = get_tables(db_name= get_collab_db(), return_list=T)
collab_tbls$sparkr_example_df[["full_name"]]

# COMMAND ----------

# MAGIC %sql
# MAGIC -- You can use this satement to drop the table that was created during this tutorial
# MAGIC -- Just update DATABASE_NAME below
# MAGIC DROP TABLE IF EXISTS DATABASE_NAME.sparkr_example_df
# MAGIC
