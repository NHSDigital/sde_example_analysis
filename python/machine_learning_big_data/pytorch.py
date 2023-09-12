# Databricks notebook source
# MAGIC %md 
# MAGIC # Get started using PyTorch
# MAGIC
# MAGIC In this notebook we will walk through an example of using PyTorch models on generated dataset. 
# MAGIC
# MAGIC In this example the toy datasets have already created and added to the collab database to mimic an actual workflow, we will use a general function to get the database name however this can be can be replaced with a string.
# MAGIC
# MAGIC * Example 1: PyTorch Neural Network model using a toy pyspark dataframe
# MAGIC
# MAGIC * Example 2: PyTorch Neural Network model using a generated data
# MAGIC
# MAGIC The utility functions are imported via the next command which runs the notebook stored in a different location. You can view these functions by navigating to the folder or you can also click the link in the next command. This can also be a useful way to tidy up your code and store frequently used functions in their own notebook to be imported into others.
# MAGIC
# MAGIC If you need more info, feel free to check out the online PyTorch documentation - it's packed with tons of useful details that might be helpful!

# COMMAND ----------

# MAGIC %run ../../Wrangler-Utilities/Utilities-Python

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Note: Unable to download certain features from torch due to the secure nature of the environment
# MAGIC As a secure environment the SDE does not have the ability to make connections to the internet. As such, downloading datasets from PyTorch is disabled. We would advise users to instead generate their own data or use the datasets provided in the SDE databases 
# MAGIC Similarly, pip installing certain packages within a notebook is not enabled within the environment 
# MAGIC
# MAGIC For example the following code will not work and returns the error ***"RuntimeError: Error downloading"***
# MAGIC
# MAGIC  
# MAGIC     #Download training data from open datasets.
# MAGIC     training_data = datasets.FashionMNIST(
# MAGIC         root="data",
# MAGIC         train=True,
# MAGIC         download=True,
# MAGIC         transform=ToTensor(),
# MAGIC     )
# MAGIC
# MAGIC     #Download test data from open datasets.
# MAGIC     test_data = datasets.FashionMNIST(
# MAGIC         root="data",
# MAGIC         train=False,
# MAGIC         download=True,
# MAGIC         transform=ToTensor(),
# MAGIC     )

# COMMAND ----------

# MAGIC %md 
# MAGIC #Example 1: PyTorch Neural Network Model using a toy pyspark dataframe 
# MAGIC 1. Import the required modules
# MAGIC 2. Load in the toy data frame 
# MAGIC 3. Convert PySpark DataFrame to PyTorch Dataset
# MAGIC 4. Define PyTorch neural network model
# MAGIC 5. Train PyTorch model using PySpark data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importing required PyTorch Modules 
# MAGIC The folling command imports the required PyTorch modules, including torch, torchvision, torch.nn, torch.optim, and torch.utils.data.
# MAGIC
# MAGIC These modules provide the functionality for creating and training deep neural networks.

# COMMAND ----------

# pytorch neural network example with a  generated pyspark dataframe 
import pyspark.sql.functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pyspark.sql.types import *


# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2. Load in the toy dataframe

# COMMAND ----------

# retrieve the dataframe
df = spark.table(f'{get_collab_db()}.toy_random_values')

# show the spark dataframe
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3. Convert PySpark DataFrame to PyTorch Dataset

# COMMAND ----------

class PySparkDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.count()

    def __getitem__(self, idx):
        row = self.df.select("*").collect()[idx]
        features = torch.tensor(row[:-1], dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.float32)
        return features, label

dataset = PySparkDataset(df)


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Define PyTorch neural network model

# COMMAND ----------

def relu(x):
    return 0.5 * (x + abs(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Train PyTorch model using PySpark data

# COMMAND ----------

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Epoch: {} \t Loss: {:.6f}'.format(epoch+1, loss.item()))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PyTorch Example 2:  PyTorch Neural Network Example using generated data 

# COMMAND ----------

# import relevant modules 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set up the PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate random training data
train_data = torch.randn(1000, 10)
train_targets = torch.randn(1000, 1)

# Generate random test data
test_data = torch.randn(100, 10)
test_targets = torch.randn(100, 1)

# Set up the PyTorch model, optimizer, and loss function
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.MSELoss()

# Convert the training data and targets to PyTorch tensors
train_data = torch.FloatTensor(train_data)
train_targets = torch.FloatTensor(train_targets)

# Create a PyTorch DataLoader for the training data
train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=32, shuffle=True)

# Train the PyTorch model
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate the PyTorch model on the test data
    test_data = torch.FloatTensor(test_data)
    test_targets = torch.FloatTensor(test_targets)
    test_output = model(test_data)
    test_loss = criterion(test_output, test_targets)
    print('Epoch {}: Test set: Average loss: {:.4f}'.format(epoch, test_loss))

