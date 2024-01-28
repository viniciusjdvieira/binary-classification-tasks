# Binary Classification Tasks

## Description

This repository contains a pipeline for binary classification tasks utilizing three classifiers: LDA (Linear Discriminant Analysis), QDA (Quadratic Discriminant Analysis), and SVM (Support Vector Machines). The pipeline is designed to process a configuration file, specifying the type of classifier, classification combinations, the address of the CSV file containing extracted features from the dataset, and the location to store the result file.

## Dependencies

- Python version: 3.10.11
- Examples of built-in libraries:
  - `os`
  - `sys`
  - `multiprocessing`

Ensure you have the following Python libraries installed:

- `pandas==1.4.2`
- `numpy==1.22.4`
- `scikit-learn==1.2.2`


## Classifier Overview

### Linear Discriminant Analysis (LDA)

LDA is a method for dimensionality reduction and classification. It finds the linear combinations of features that best separate two or more classes.

### Quadratic Discriminant Analysis (QDA)

Similar to LDA, QDA is a classification algorithm, but it allows for different covariance matrices for each class, making it more flexible when dealing with non-linear decision boundaries.

### Support Vector Machines (SVM)

SVM is a powerful algorithm for binary classification. It finds the hyperplane that best separates the data into different classes by maximizing the margin.

## Binary Classification

Binary classification involves categorizing instances into one of two classes. In this context, the classifiers aim to distinguish between two groups based on input features.

## k-Fold Cross-Validation

The pipeline uses k-fold cross-validation, a technique where the dataset is split into k subsets, and the model is trained and evaluated k times, with a different subset used as the test set in each iteration.

## Configuration File

Example of `config.yaml`:

```yaml
classifier: 'svm'
KFold:
  n_folds: 10
classification:
  individual: True
  combinations: True
  save_all_results: True
  save_summary_file: True
svm_config:
  fine_tuning_svm: True
  kernel_svm: 'linear' # if fine_tuning = False
  C_value: 1.0         # if fine_tuning = False
filepaths:
  dataset_path: "D:\\programming\\classification\\medidas2.csv"
  results_path: "D:\\programming\\classification"
dataset:
  label_column: "CLASS_NUM"
  number_of_metrics: 10 # last 10 columns of your dataset
  combos: 'all' # alternatively, chose a list of numbers, e.g., [2,3,4,5]
parallelism: False
```

## CSV File Format
The CSV file should contain columns for features and at least a label column specifying the class. Ensure the label column is named as per the configuration file.

Columns for labels first. The last columns must to be related to each metric.

### Example of the csv-based table
| CLASS_STR | CLASS_NUM | MED1     | MED2     | MED3     | MED4     | MED5     | MED6     | MED7     | MED8     | MED9     | MED10    |
|-----------|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| S'        | 0         | 0.046443 | 0.031733 | 0.062577 | 0.662921 | 1.454993 | 2.567878 | 3.973769 | 5.291051 | 7.150222 | 5.19238  |
| S'        | 0         | 0.091757 | 0.066854 | 0.112539 | 0.628205 | 2.648563 | 4.366297 | 6.45142  | 7.92939  | 9.24053  | 9.734327 |
| S'        | 0         | 0.054298 | 0.050737 | 0.107189 | 0.779693 | 1.862557 | 2.785057 | 3.047597 | 2.668143 | 3.671243 | 4.547837 |
| S'        | 0         | 0.034486 | 0.034058 | 0.067857 | 0.347366 | 0.677926 | 1.071743 | 1.391075 | 1.822097 | 2.28103  | 1.475807 |
| P'        | 1         | 0.10106  | 0.026652 | 0.021664 | 0.126798 | 0.325935 | 0.595971 | 0.989863 | 1.283907 | 2.27186  | 1.578473 |
| P'        | 1         | 0.085388 | 0.047023 | 0.065849 | 0.494819 | 1.088841 | 1.878267 | 2.594137 | 2.800037 | 1.636063 | 1.709373 |
| P'        | 1         | 0.126938 | 0.062788 | 0.070992 | 0.903809 | 2.477953 | 4.340351 | 6.919843 | 7.6242   | 3.515227 | 2.550226 |
| P'        | 1         | 0.151652 | 0.0383   | 0.02825  | 0.111576 | 0.707189 | 1.249175 | 1.68092  | 1.945333 | 2.338433 | 2.77214  |

## Running the Pipeline
- Install dependencies by running: pip install -r requirements.txt
- Execute the pipeline with the command: python main.py config.yaml

### Running an example
You can find an example of csv file to test this pipeline.

Change its file path according to your preference.

### Fine-Tuning for SVM
If you choose fine-tuning for SVM, pay attention to the processing time. It could be slower than the other approaches.


## Output terminal
Example of the output terminal:

```bash
Reading config file ...
  ____________  ____              ____  ___          ___   
 /            | \   \            /   / |   \        /   |  
|     _______/   \   \          /   /  |    \      /    |  
|     \______     \   \        /   /   |     \    /     |  
 \            \    \   \      /   /    |      \  /      |  
  \ _______    |    \   \    /   /     |       \/       |  
           |   |     \   \  /   /      |   |\      /|   |  
  ________/    |      \   \/   /       |   | \    / |   |  
 /             |       \      /        |   |  \__/  |   |  
/_____________/         \____/         |___|        |___|  
CLASSIFIER PIPELINE
Loading dataset ...

>>> Running individual classification ...
---> Here we don't have parallelism ...
Results were put in the path:
 D:\programming\classification\svm_20240127 

>>> Running classification with combinations ...
---> Here we don't have parallelism ...
===> Refult dataframe for combinations of  2  metrics:

===> ... more results will appear...

Results were put in the path:
 D:\programming\classification\svm_20240127

The summarized results (THE BEST ONES) were put in the path:
 D:\programming\classification\svm_20240127

```

