import os
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse
from model_fetcher import get_model
import config
from tqdm import tqdm
from sklearn import metrics
import joblib
import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline


age_groups = [0., 10.0, 15.0, 20.0, 30, 40, 50, 60, np.inf]


def display_scores(scores):
    print(pd.Series(scores).describe())


def get_y(df):
    return df["Survived"]


def get_X(df):
    res = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    res["AgeBucket"] = res["Age"] // 15 * 15
    res["Num_cotraveler"] = res["SibSp"] + res["Parch"]
    res["Travel_alone"] = (res["Num_cotraveler"] == 0).astype(float)
    res["Under_15"] = (res["Age"] <= 15).astype(float)
    return res


def main(args):
    df = pd.read_csv(args.path)
    model = args.model

    X = get_X(df)
    y = get_y(df)
    feature_pipeline = preprocessing.get_preprocessing_pipeline()
    pipeline = make_pipeline(
        feature_pipeline,
        get_model(model)
    )
    scores = cross_val_score(pipeline, X=X, y=y,
                             scoring="accuracy", cv=10)
    display_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=config.TRAINING_STRATIFIED_KFOLD_FILE
    )

    args = parser.parse_args()
    main(args)

