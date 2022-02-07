"""
Main file: defines steps to evaluate categorical encoders / classifier pairs on a given task.
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold

# transforms
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, OrdinalEncoder

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# abstract classes
from sklearn.base import clone, TransformerMixin, BaseEstimator

# custom library
from src import columnar as col

ROOT_PATH = './'

plt.style.use('fivethirtyeight')

def main(args):
    
    print(f"Comparing (model, encoder) pairs for the {args.task} classification task")
    print(f"\tloading dataset")
    # load data
    loader = col.DataLoader(root=ROOT_PATH, task=args.task)
    data = loader.load_data()
    
    # define scoring metrics of interest
    scorer = col.Scorer(
        acc=lambda x, y: metrics.accuracy_score(x,y>.5),
        f1=lambda x, y: metrics.f1_score(x,y>.5),
        auc=metrics.roc_auc_score,
    )

    # define cross validation strategy
    kf = KFold(n_splits=5)
    
    # define features to select
    feature_selection = col.FeatureSelection(**loader.get_selected_features(data))
    
    reporter = col.Report(scorer=scorer)
    reporter.set_columns_to_show(['classifier', 'transformer'] + list(scorer.scoring_fcts.keys()))

    classifiers = [
        RandomForestClassifier(n_estimators=100, max_depth=5),
        LogisticRegression(max_iter=500),
        KNeighborsClassifier(n_neighbors=10),
        LGBMClassifier(),
    ]

    transformers = [
        col.MeanTargetEncoder(feature_selection),
        col.TransformStrategy(feature_selection, OneHotEncoder(handle_unknown='ignore')),
        col.TransformStrategy(feature_selection, OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        col.embeddings.wrapper.TFEmbeddingWrapper(features=feature_selection, emb_size_strategy='single'),
        col.embeddings.wrapper.TFEmbeddingWrapper(features=feature_selection, emb_size_strategy='max2'),
    ]

    print(f"""\t  5 cross-validation folds
    \tx {len(transformers)} transformers
    \tx {len(classifiers)} models 
    \tcombinations to train and test""")
    
    runner = col.benchmark.BenchmarkRunner(features=feature_selection,
                                           transformers=transformers,
                                           classifiers=classifiers,
                                           scorer = scorer,
                                          )
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        
        print(f"\tRunning benchmark on fold number {i+1}", end='\r')
        start = time.time()
        
        # split train and test data using the CV fold
        cv_train, cv_test = data.iloc[train_idx], data.iloc[test_idx]
        
        X_train, y_train = feature_selection.select_features(cv_train)
        X_test, y_test = feature_selection.select_features(cv_test)
        
        runner.run(X_train, y_train, X_test, y_test)
        print(f"\tRunning benchmark on fold number {i+1} - time = {col.utils.convert_time(time.time() - start)}")
            
                
    print("evaluation completed")
    
    reporter = runner.create_reporter()

    
    # saving report as csv
    save_path = f'runs/{args.task}.csv'
    reporter.save(os.path.join(ROOT_PATH, save_path))
    print(f"results summary saved in {save_path}")
    
    # saving summary plot model
    figpath = f'figures/{args.task}.png'
    fig = col.plot_model_encoder_pairs(reporter, figpath=os.path.join(ROOT_PATH, figpath), title=f"{args.task} dataset".capitalize())
    print(f"visualization saved in {figpath}")
    
    return reporter
            
        

    

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="the dataset and classification task to evaluate categorical encoders and models against",
                        type=str)

    args = parser.parse_args()

    reporter = main(args)