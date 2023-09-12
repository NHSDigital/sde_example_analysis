# Secure Data Environment (SDE) example analysis

Reproducible and flexible example notebooks that may be used by researchers using Databricks within the [NHS England SDE](https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment).

## Contact
**This repository is maintained by [NHS England Data Wranglers team](mailto:england.sdeservice@nhs.net?subject=GitHub%20sde_example_notebooks)**.
> _To contact us raise an issue on Github or via email._
> 
> See our (and our colleagues') other work here:
>- [SDE Data Wranglers](https://github.com/orgs/NHSDigital/teams/sde_wranglers/repositories)
>- [NHS England](https://github.com/orgs/NHSDigital/repositories)

## Description

The NHS England SDE is a data and research analysis platform, which is built to uphold the highest standards of privacy and security of NHS health and social care data when used for research and analysis.

Within the SDE, users have access to [Databricks](https://digital.nhs.uk/services/secure-data-environment-service/secure-data-environment/user-guides/using-databricks-in-sde). Some users may be new to Databricks, so we provide example notebooks and useful functions for what they can do with Databricks inside the SDE. In addition, the analysis explores machine learning (ML) capabilities in the SDE.

### Available analysis

#### :snake: python

pyspark_code_example: PySpark is the Python API for Aparche Spark - it can handle Big Data better than pandas.

##### :sparkles: machine_learning_big_data

Usually, if a dataframe has more than one million rows, it is big data. However, a dataset may have less rows than this and still be classed as "big" due to complexity.

This folder contains notebooks exploring the machine learning capabilities of PySpark in the SDE. Examples include:
- decision_tree_regressor
- random_forest_classifier

##### :panda_face: machine_learning_small_data

This folder contains analysis for smaller data sets, using libraries like pandas and sklearn. Examples include:
- regression_multivariable
- regression_simple

#### :pirate_flag:	r

SparkR is an R package that provides a light-weight frontend to use Apache Spark from R. The example notebook demonstrates what you can do with SparkR.

#### :mechanic: wrangler_utilities

Useful functions and code, including code to create and save toy data sets.

## Licence

The codebase is released under the [MIT License](LICENCE). This covers both the codebase and any sample code in the documentation.

Any HTML or Markdown documentation is [© Crown copyright](https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and available under the terms of the [Open Government 3.0 licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Acknowledgements

[Data Wranglers](https://github.com/orgs/NHSDigital/teams/sde_wranglers), Data & Analytics, NHS England

For individuals involved see [CONTRIBUTORS](CONTRIBUTORS.md).
