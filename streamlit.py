import os

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
        'One Hot': OneHotEncoder(handle_unknown='ignore'),
        'Ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}

encoders = stc.create_checkbox_list(st.sidebar, encoder_options, 
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
    
def run_benchmark(task, classifiers, encoders):
    
    # initialize log
    logger = stc.StreamlitLogger()
    logger.log("initializing report")
    
    # define scoring metrics of interest
    scorer = col.Scorer(
        acc=lambda x, y: metrics.accuracy_score(x,y>.5),
        f1=lambda x, y: metrics.f1_score(x,y>.5),
        auc=metrics.roc_auc_score,
    )
    
    # initialize reporter
    reporter = col.Report(scorer=scorer)
    reporter.set_columns_to_show(['model', 'encoder'] + list(scorer.scoring_fcts.keys()))
    
    
    logger.log("loading data")
    # load data
    loader = col.DataLoader(root=ROOT_PATH, task=task)
    df = loader.load_data()
    
    # define cross validation strategy
    kf = KFold(n_splits=5)
    
    # define features to select
    feature_selection = col.FeatureSelection(**loader.get_selected_features(df))
    
    # update task details in the second column of the info section
    print_task_info(summary, task, df, feature_selection)
    
    # ad the MeanTargetEncoder to the list of encoders to evaluate
    encoders += [col.MeanTargetEncoder(feature_selection)]
    
    # evaluate each encoder / model pair
    for model in classifiers:
        logger.log(f"fitting {model}")
        for encoder in encoders:
            pipe = col.CategoricalPipeline(features=feature_selection,
                                           model=clone(model),
                                           scaler=MaxAbsScaler(),
                                           encoder=clone(encoder))

            cv_score = col.cv_score(pipeline=pipe,
                                    data=df, 
                                    kf=kf,
                                    scorer=reporter.scorer)

            reporter.add_to_report(pipe.config, cv_score, show=False)
    
    # plot output
    logger.log("plotting output")
    fig = col.plot_model_encoder_pairs(reporter, title=f"{task} dataset".capitalize(), show=False)
    
    st.write("### Summary Plot")
    st.pyplot(fig)
    
    # display top 5 encoder / classifier pairs
    st.write("### Top 5 Encoder / Classifier Pairs (by F1-score)")
    st.dataframe(reporter.show().sort_values('f1', ascending=False).head(5))
    
    logger.log("benchmark completed")
    


if run:
    run_benchmark(task, clfs, encoders)
    

    
    # st.write("is this working?")
    # reporter.save(f'runs/{task}.csv')
    # figpath = f'figures/{task}.png'
    # fig.savefig(figpath, transparent=False, facecolor='white')
