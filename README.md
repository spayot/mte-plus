# Benchmark Mean Target Encoding and Categorical Embeddings against other Categorical Data Encoding Techniques
## 1. Goal
**Mean Target Encoding (MTE)** is a technique to transform categorical data into numerical by replacing the categorical value by the mean target value for all observations belonging to that category.  The goal of this project is to benchmark the performance of mean target encoding  against multiple encoding strategies for categorical variables in a structured dataset task.  
**Categorical feature embeddings** are a potentially more expressive generalization of MTE which represents each categorical value as an embedding. embeddings sizes can be defined based on the cardinality of each feature. An embedding of size 1 should replicate closely the principle of MTE, but weights are learnt instead of explicitly defined.
The benchmark is run across multiple binary classification tasks, and considers multiple types of downstream classifiers. Scoring is focused on Accuracy, F1-score and AUC.

---
## 2. Tasks

### 2.1 Adult Dataset
Predict whether an adult's income is higher or lower than $50k, using census information given 15 census information.
https://archive.ics.uci.edu/ml/datasets/Adult

### 2.2 Mushrooms Dataset
Aims to predict whether a mushroom is poisonous given 23 categorical descriptors.  
https://www.kaggle.com/uciml/mushroom-classification#

### 2.3 Titanic Dataset
Aims to predict whether a Titanic passenger survived given a few descriptors. only some minimal imputing and feature engineering was performed.
https://www.kaggle.com/uciml/mushroom-classification#

### 2.4 HR Analytics Dataset
Aims to predict whether a data scientist is looking for a job change or not. only some minimal imputing and feature engineering was performed.
https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

### 2.5 PetFinder
Aims to predict whether a pet will be adopted within a 100 days.
https://www.kaggle.com/c/petfinder-adoption-prediction/data

---
## 3. Benchmarking strategy
A ML Pipeline is built with each categorical encoder / classifier pairs, and trained on the task at hand. Evaluation is performed through a 5-fold cross validation strategy, to extract mean and std dev values for each metric of interest.   
**Pipeline Overview**![](figures/pipeline.png)
*`*` sklearn component*  *`**` LightGBM component*  
*other components were built specifically for this project (cf [src/columnar](src/columar))*

---
## 4. Main Findings
### 4.1 F1-score comparison
KNeighborsClassifier                          | LGBMClassifier                               | LogisticRegression | RandomForestClassifier
:--------------------------------------------:|:---------------------------------------:|:------------------:|:--------------------------------------------:|
<img src="figures/heatmap_KNeighborsClassifier.png" alt="KNN" height="150"/> | <img src="figures/heatmap_LGBMClassifier.png" alt="LGBM" height="150"/> | <img src="figures/heatmap_LogisticRegression.png" alt="LR" height="150"/> | <img src="figures/heatmap_RandomForestClassifier.png" alt="RF" height="150"/>

#### Description 
* Each heatmap represents the average cross validation F1-score obtained on each task (x-axis) with each encoding technique (y-axis).
* color coding is using the OneHotEncoding + LogisticRegression score as a baseline for each task. Red values indicate performance superior to baseline, while blue values indicate lower performance.

#### Findings
- Some classifiers are more sensitive to the encoding technique than others. LGBM from that perspective offers both the benefits of limited sensitivity and high level performance.
- **Ordinal Encoders** are, understanbly, performing poorly for linear classifiers that rely on topological distance for training and predictions (KNN, LogisticRegression). Both Mean-Target Encoding and unidimensional embeddings therefore allow to significantly improve performance for those models without increasing the input's dimensionality after transformation.
- **One Hot Encoders** tend to work well for LogisticRegressions. On the other hand, it consistently performs poorly when used in conjunction with RandomForests.


### 4.2 All results 
The below charts provide more detailed results at the task level, including standard deviation observed for each metric for each encoder / classifier pairs.
**Adult Task** ![](figures/adults.png)
**Mushrooms Task** ![](figures/mushrooms.png)
**Titanic Task** ![](figures/titanic.png)
**HR Analytics Task** ![](figures/hr_analytics.png)
**PetFinder** ![](figures/petfinder.png)
---
## 5. Install Requirements
`conda install -n <name> -f conda.yaml`

---
## 6. Reproducing Experiments
### 6.1 through Scripts
the user can simply run the following command:  
  
  `python main.py --task <task_name>`  
  
This command will evaluate the performance of various encoder / classifier combinations and generate:
- a CSV report in the `runs/` folder
- a summary figure in the `figures/` folder.
Both new artifacts are named after the prediction task used to evaluate the pipelines.

### 6.2 using a Streamlit App
Allows to run the benchmark while selecting:
- which task to use for benchmarking
- the type of downstream classifiers
- the encoders to benchmark against


to do so, you can simply run the command:  
    `streamlit run streamlit.py`

### 6.3 the Notebook way
check out [notebooks/adult](notebooks/adult.ipynb).

---
## 7. Benchmarking using a new task
Any binary classification task can be used to evaluate encoder/model pairs as long as the dataset is made available in the `data/` folder, and an identically named python file is added in `src/columnar/loaders`. this file must include 2 functions:
- `_load` describes the steps to load the dataset in memory
- `_select_features(df)` builds a FeatureSelection object describing which features will be fed into the classifier, which ones are categorical, and which column corresponds to the target.

Once this is done, the user can simply run the following command:  
  
  `python main.py --task <new_task_name>`  

You can find examples of `_load` and `_select_features(df)` functions here:
[mushrooms.py](src/columnar/loaders/mushrooms.py)

---
## 8. Next Steps

- [x] Add a fourth type of classifiers (LightGBM)
- [x] Perform comparison on other classification tasks
  - [x] refactor data loading process into a factory pattern
  - [x] create a main function taking as input a dataset
  - [x] add mushrooms dataset and compare performances
  - [x] add titnaic dataset and compare performances
  - [x] add larger dataset and compare performances

- [ ] **Categorical feature embeddings** are a potentially more expressive generalization of MTE which represents each categorical value as an embedding. embeddings sizes can be defined based on the cardinality of each feature. An embedding of size 1 should replicate closely the principle of MTE, but weights are learnt instead of explicitly defined.
