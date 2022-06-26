'''
Train model, register model in mlflow
'''
import argparse
import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn

# import model definition
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression


# custom concatenation transformer
class ConcatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_.fillna('', inplace=True)
        new_column_name = '_'.join(X_.columns)
        X_[new_column_name] = X_.agg(' '.join, axis=1)
        return X_[new_column_name]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
])

text_transformer = Pipeline(steps=[
    ('concat', ConcatTransformer()),
    # all-in-one: tokenizer, stop-words remover, hasher, normalizer
    ('hasher', HashingVectorizer(n_features=1000, stop_words='english', binary=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['vote']),
        ('text', text_transformer, ['reviewText', 'summary'])
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('preproc', text_transformer),
    ('logreg', LogisticRegression(max_iter=10000))
])


FORMAT = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser(description="HW 5mlb by hethwynq: train.py")
    parser.add_argument(
        "--train_path",
        type=str,
        help="hdfs path to train dataset",
    )
    parser.add_argument(
        "--sklearn_model",
        type=str,
        help="sklearn model name in MLFlow backend",
    )
    parser.add_argument(
        "--model_param1",
        type=float,
        default=1.0,
        help="inverse of regularization strength (default: 10.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="random state for train",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run_id",
    )
    return parser.parse_args()


def train(train_path, sklearn_model, model_param1=10.0, seed=43, run_id=None):
    # start mlflow run
    mlrun = mlflow.start_run(run_id=run_id, run_name="TRAIN")
    logging.info(f"train_path: {train_path}")
    logging.info(f"sklearn_model: {sklearn_model}")
    logging.info(f"model_param1: {model_param1}")
    logging.info(f"seed: {seed}")
    # read dataset
    logging.info("Read dataset...")
    os.system(f"hdfs dfs -getmerge {train_path} train.parquet")
    df = pd.read_parquet("train.parquet")
    logging.info("Finished.")
    # set model params
    logging.info("Set params...")
    model.set_params(logreg__C=model_param1)
    model.set_params(logreg__random_state=seed)
    logging.info("Finished.")
    # log params
    logging.info("Log params...")
    mlflow.log_param("C", model_param1)
    mlflow.log_param("random_state", seed)
    logging.info("Finished.")
    # train model
    logging.info("Train model...")
    model.fit(df[["reviewText", "summary"]], df["label"])
    logging.info("Finished.")
    # log model
    logging.info("Register model...")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn_model", registered_model_name=sklearn_model)
    logging.info("Finished.")
    # end mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse_args()
    # run train
    train(args.train_path, args.sklearn_model, args.model_param1, args.seed, args.run_id)
