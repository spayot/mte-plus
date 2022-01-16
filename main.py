import argparse
import os

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, OrdinalEncoder




from src import columnar as col

ROOT_PATH = './'

def main(args):
    plt.style.use('fivethirtyeight')
    
    print(f"Comparing (model, encoder) pairs for the {args.task} classification task")
    print(f"\tloading dataset")
    # load data
    loader = col.DataLoader(root=ROOT_PATH, task=args.task)
    df = loader.load_data()
    
    # define scoring metrics of interest
    scorer = col.Scorer(
        acc=metrics.accuracy_score,
        f1=metrics.f1_score,
        auc=metrics.roc_auc_score,
    )

    # cross validation strategy
    kf = KFold(n_splits=5)
    
    # define features to select
    feature_selection = col.FeatureSelection(**loader.get_selected_features(df))
    
    reporter = col.Report(scorer=scorer)
    reporter.set_columns_to_show(['model', 'encoder'] + list(scorer.scoring_fcts.keys()))

    models = [
        RandomForestClassifier(n_estimators=100, max_depth=5),
        LogisticRegression(max_iter=500),
        KNeighborsClassifier(n_neighbors=10),
        LGBMClassifier()
    ]

    encoders = [
        col.MeanTargetEncoder(feature_selection),
        OneHotEncoder(handle_unknown='ignore'),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    ]
    
    
    for model in models:
        print(f"\tevaluating encoders when fitting a {model}")
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
    
    # saving report as csv
    save_path = f'runs/{args.task}.csv'
    reporter.save(os.path.join(ROOT_PATH, save_path))
    print(f"results summary saved in {save_path}")
    
    # saving summary plot model
    figpath = f'figures/{args.task}.png'
    col.plot_model_encoder_pairs(reporter, figpath=figpath, title=f"{args.task} dataset".capitalize())
    print(f"visualization saved in {figpath}")
            
        

    

    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="the dataset and classification task to evaluate categorical encoders and models against",
                        type=str)

    args = parser.parse_args()

    main(args)