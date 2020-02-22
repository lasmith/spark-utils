# Spark Utils
This project is a collection of utility classes / functions to work with Spark / SparkML. 


# Project Structure

```
.
├── README.md
├── build.sbt
├── src
│   ├── main
│   │   └── scala                <- All the scala source code
│   │       
│   └── test
|       ├── resources            <- Data / config for the tests
│       └── scala                <- All the unit tests
```
# Components
The following is a high level description of the components available in this library:

* Target Encoder - A spark ML implementation of [target encoding](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)
* Weight Of Evidence - A spark ML implementation of [Weight of evidence](https://documentation.statsoft.com/STATISTICAHelp.aspx?path=WeightofEvidence/WeightofEvidenceWoEIntroductoryOverview) encoding
* Stats Calculator - A utility class to generate additional metrics for classifiers (such as F1 Score / MCC)
* Timestamps Transformer - A sparkML transformer to take an input date and split into the components (year / month etc)