import os
import time

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, OrdinalEncoder
import streamlit as st

import src.columnar as col
import src.streamlit_components as stc

ROOT_PATH = './'
LOG_HEIGHT = 100

plt.style.use('fivethirtyeight')

# ------- title -------
st.title("Benchmarking Categorical Encoders")


# ------- sidebar -------
# choose task
task_options = [file for file in os.listdir('data/') if os.path.isdir('data/'+ file)]
task = st.sidebar.selectbox("Select a Task", task_options)
        
run = st.sidebar.button("Run Benchmark")

# choose encoders
encoder_options = {
        'One Hot': lambda f: col.TransformStrategy(f, OneHotEncoder(handle_unknown='ignore')),
        'Ordinal': lambda f: col.TransformStrategy(f, OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        'Embeddings Single': lambda f: col.embeddings.wrapper.TFEmbeddingWrapper(features=f, emb_size_strategy='single'),
        'Embeddings Max2': lambda f: col.embeddings.wrapper.TFEmbeddingWrapper(features=f, emb_size_strategy='max2'),
        'Embeddings Max50': lambda f: col.embeddings.wrapper.TFEmbeddingWrapper(features=f, emb_size_strategy='max50'),
        'MeanTargetEncoder': lambda f: col.MeanTargetEncoder(f),
}

transformers = stc.create_checkbox_list(st.sidebar, encoder_options, 
                               title='Select Benchmark Encoders')

# choose classifiers
clf_options = {
    'random forest': RandomForestClassifier(n_estimators=100, max_depth=5),
    'logistic regression': LogisticRegression(max_iter=500),
    'knn': KNeighborsClassifier(n_neighbors=10),
    'LightGBM': LGBMClassifier()
    }

clfs = stc.create_checkbox_list(st.sidebar, clf_options, 
                               title='Select Classifiers')


# --- first section: information on task ---
info1, info2 = st.columns(2)

info1.write("""**Mean Target Encoding (MTE)** is a technique to transform categorical data into numerical by replacing the categorical value by the mean target value for all observations belonging to that category.  The goal of this project is to benchmark the performance of mean target encoding  against multiple encoding strategies for categorical variables in a structured dataset task.
**Categorical feature embeddings** are a potentially more expressive generalization of MTE which represents each categorical value as an embedding. embeddings sizes can be defined based on the cardinality of each feature. An embedding of size 1 should replicate closely the principle of MTE, but weights are learnt instead of explicitly defined.
The benchmark can be run across multiple **binary classification** tasks, and considers multiple types of downstream classifiers.  
Scoring is focused on Accuracy, F1-score and AUC.
""")

summary = info2.empty() # this will be 

def _reformat_title(title):
    return title.replace('_', ' ').title()

def _task_info(task, 
               n_samples: int = None, 
               n_categorical: int = None, 
               n_numerical: int = None, 
               max_cardinality: int = None, 
               target_name: str = None):
    """outputs a markdown string describing a task"""
    if not n_samples:
        n_samples, n_categorical, n_numerical, max_cardinality, target_name = ["" for _ in range(5)]
    return f"""
    ### Task Summary
    
    | Task                        |                {task}|
    | --------------------------- |:--------------------:| 
    | # of samples                |        {n_samples} |
    | categorical attributes      |      {n_categorical} |
    | max categorical cardinality |  {max_cardinality} |
    | numerical attributes        |      {n_numerical} |
    |target name                  |        {target_name} |
    """

def print_task_info(loc, task: str, df: pd.DataFrame = None, feature_selection = None) -> None:
    if df is None:
        md = _task_info(task=_reformat_title(task))
    else:
        cardinality = {col: df[col].nunique() for col in feature_selection.categoricals}
        max_cardinality = max(cardinality.values())
        md = _task_info(task=task.capitalize(), 
                        n_samples=len(df), 
                        n_categorical=len(feature_selection.categoricals),
                        max_cardinality=max_cardinality,
                        n_numerical=len(feature_selection.numericals),
                        target_name=feature_selection.target
                       )
    loc.markdown(md)
    

print_task_info(summary, task)


# --- define how to run the benchmark ---
    
def run_benchmark(task, classifiers, transformers):
    
    # initialize log
    logger = stc.StreamlitLogger()
    logger.log("Initializing report")
    
    # define scoring metrics of interest
    scorer = col.Scorer(
        acc=lambda x, y: metrics.accuracy_score(x,y>.5),
        f1=lambda x, y: metrics.f1_score(x,y>.5),
        auc=metrics.roc_auc_score,
    )
    
    # # initialize reporter
    # reporter = col.Report(scorer=scorer)
    # reporter.set_columns_to_show(['transformer', 'classifier'] + list(scorer.scoring_fcts.keys()))
    
    
    logger.log("Loading data")
    # load data
    loader = col.DataLoader(root=ROOT_PATH, task=task)
    data = loader.load_data()
    
    # define cross validation strategy
    kf = KFold(n_splits=5)
    
    # define features to select
    feature_selection = col.FeatureSelection(**loader.get_selected_features(data))
    
    # update task details in the second column of the info section
    print_task_info(summary, task, data, feature_selection)
    
    
    runner = col.benchmark.BenchmarkRunner(features=feature_selection,
                                           transformers=[transformer(feature_selection) for transformer in transformers],
                                           classifiers=classifiers,
                                           scorer = scorer,
                                          )
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        
        logger.log(f"\tRunning benchmark on fold number {i+1}")
        start = time.time()
        
        # split train and test data using the CV fold
        cv_train, cv_test = data.iloc[train_idx], data.iloc[test_idx]
        
        X_train, y_train = feature_selection.select_features(cv_train)
        X_test, y_test = feature_selection.select_features(cv_test)
        
        runner.run(X_train, y_train, X_test, y_test)
        logger.log(f"\tRunning benchmark on fold number {i+1} - time = {col.utils.convert_time(time.time() - start)}")
            
                
    logger.log("evaluation completed")
    
    # generate report
    reporter = runner.create_reporter()
    
    # plot output
    logger.log("Plotting output")
    fig = col.plot_model_encoder_pairs(reporter, title=f"{task} dataset".capitalize(), show=False)
    
    st.write("### Summary Plot")
    st.pyplot(fig)
    
    # display top 5 encoder / classifier pairs
    st.write("### Top 5 Encoder / Classifier Pairs (by F1-score)")
    print(reporter.columns_to_show)
    st.dataframe(reporter.show().sort_values('f1', ascending=False).head(5))
    
    logger.log("Benchmark completed")
    


if run:
    run_benchmark(task, clfs, transformers)
    

    
    # st.write("is this working?")
    # reporter.save(f'runs/{task}.csv')
    # figpath = f'figures/{task}.png'
    # fig.savefig(figpath, transparent=False, facecolor='white')
