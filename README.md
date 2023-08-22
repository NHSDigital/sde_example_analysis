# Secure Data Environment (SDE) example analysis

Reproducible and flexible example notebooks that may be used by researchers using Databricks within the [NHS England SDE](https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment).

## Contact
**This repository is maintained by [NHS England Data Science Team](datascience@nhs.net)**.
> _To contact us raise an issue on Github or via email._
> 
> See our (and our colleagues') other work here: [NHS England Analytical Services](https://github.com/NHSDigital/data-analytics-services)

## Description

The NHS England SDE is a data and research analysis platform, which is built to uphold the highest standards of privacy and security of NHS health and social care data when used for research and analysis.

Within the SDE, users have access to [Databricks](https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment/user-guides/using-databricks-in-sde). Some users may be new to Databricks, so we provide example notebooks and useful functions for what they can do with Databricks inside the SDE. In addition, the analysis explores machine learning (ML) capabilities in the SDE.

### Available analysis

#### python

pyspark_code_example: PySpark is the Python API for Aparche Spark - it can handle Big Data better than pandas.

##### machine_learning_big_data

This folder contains notebooks exploring the machine learning capabilities of PySpark in the SDE. Examples include:
- decision_tree_regressor
- PyTorch
- random_forest_classifier

##### machine_learning_small_data

This folder contains analysis for smaller data sets, using libraries like pandas and sklearn. Examples include:
- regression_multivariable
- regression_simple
- sklearn_mlflow_saving models: there is the option in Databricks to save the parameters and output of machine learning models - this example will be useful to go through whether you use PySpark or pandas

#### r

SparkR is an R package that provides a light-weight frontend to use Apache Spark from R. The example notebook demonstrates what you can do with SparkR.

#### wrangler_utilities

Useful functions and code:
- create_toy_datasets: saves fake data to tables for example analysis
- utilities_python: useful Python functions for researchers
- utilities_r: useful R functions for researchers

## Licence

The codebase is released under the [MIT License](LICENCE). This covers both the codebase and any sample code in the documentation.

Any HTML or Markdown documentation is [© Crown copyright](https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and available under the terms of the [Open Government 3.0 licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Acknowledgements

[Data Wranglers](https://github.com/orgs/NHSDigital/teams/sde_wranglers), Data & Analytics, NHS England
