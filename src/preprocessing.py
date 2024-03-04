from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn import impute
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import pandas as pd


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def get_default_num_pipeline():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        StandardScaler()
    )


def get_ratio_feature_name(_, feature_names_in):
    return ["ratio_of_" + "_".join(feature_names_in)]  # feature names out


def pipeline_add_ratio():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=get_ratio_feature_name),
        StandardScaler()
    )


def pipeline_category():
    return Pipeline([
        ("imputer", impute.SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse_output=False)),
    ])


def pipeline_log():
    return make_pipeline(
        impute.SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one")
    )


def get_preprocessing_pipeline():
    transformer = ColumnTransformer([
        ("category", pipeline_category(), make_column_selector(dtype_include=object)),
    ], remainder=get_default_num_pipeline())

    return transformer


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../datasets/titanic/train_stratified_kfold.csv")
    df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    transformer = FunctionTransformer(column_ratio)
    preprocessing = get_preprocessing_pipeline()
    res = preprocessing.fit_transform(df)
    print(res)
    print(res.shape)
    print(preprocessing.get_feature_names_out())
