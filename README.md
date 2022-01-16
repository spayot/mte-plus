# Testing Mean Target Encoding vs categorical data embeddings
## 1. Goal
**Mean Target Encoding (MTE)** is a technique to transform categorical data into numerical by replacing the categorical value by the mean target value for all observations belonging to that category.  The goal of this project is to benchmark the performance of mean target encoding  against multiple encoding strategies for categorical variables in a structured dataset task.  
The benchmark is run across multiple classification tasks, and considers multiple types of downstream classifiers. Scoring is focused on Accuracy, F1-score and AUC.

## 2. Tasks

### 2.1 Adult Dataset
#### Description
Provides various census features about individuals and aims to predict whether an individual is earning over $50k or not.
[https://archive.ics.uci.edu/ml/datasets/Adult](https://archive.ics.uci.edu/ml/datasets/Adult)

#### Prediction Task
Predict whether an adult's income is higher or lower than $50k, using census information

## 3. Main Findings
**Adult Task** ![adults](figures/adults.png)
**Mushrooms Task** ![](figures/mushrooms.png)
Mean Target Encoding seem to be the most resilient encoding strategy to classifer choices and performs best with 2 out of the 4 classifier choices.

## 4. Install Requirements
`conda install -n <name> -f conda.yaml`

## 5. Reproducing Experiments
### 5.1 through Scripts
Any classification task can be used to evaluate encoder/model pairs as long as the dataset is made available in the `data/` folder, and an identically named python file is added in `src/columnar/loaders`. this file must include 2 functions:
- `_load` describes the steps to load the dataset in memory
- `_select_features(df)` builds a FeatureSelection object describing which features will be fed into the classifier, which ones are categorical, and which column corresponds to the target.

Once this is done, the user can simply run the `python main.py --task <task_name>` command.
this will generate:
- a CSV report in the `runs/` folder
- a summary figure in the `figures/` folder.
Both new artifacts are named after the prediction task used to evaluate the pipelines.

### 5.2 the Notebook way
check out the [notebooks/modeling](notebooks/modeling.ipynb).

## 6. Next Steps

- [x] Add a fourth type of classifiers (LightGBM)
- [ ] Perform comparison on other classification tasks
  - [x] refactor data loading process into a factory pattern
  - [x] create a main function taking as input a dataset
  - [ ] add new datasets and compare performances
- [ ] **Categorical data embeddings** is a potentially more expressive generalization of MTE which represents each categorical value as an embedding. embeddings sizes can be defined based on the cardinality of each feature. An embedding of size 1 should replicate closely the principle of MTE (even though values are learnt more indireclty), but weights are learnt instead of explicitly defined.
