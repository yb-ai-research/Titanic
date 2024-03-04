from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def get_model(model_name, random_state=42):
    return {
        "logistic_regression": LogisticRegression(),
        "rf": ensemble.RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        "svc": SVC(gamma="auto")
    }[model_name]