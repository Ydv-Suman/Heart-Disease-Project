# Import core libraries
from pathlib import Path
import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression

# Model selection tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

import pickle




def load_dataset(df_path: Path) -> pd.DataFrame:
    if not df_path.exists():
        raise FileNotFoundError("Data file is not found")
    return pd.read_csv(df_path)


def split_dataset(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    log_grid = {
        "C": np.logspace(-4, 4, 20),
        "solver": ["liblinear", "saga"],
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"]
    }

    np.random.seed(42)
    gs_log_model = GridSearchCV(
        LogisticRegression(),
        param_grid = log_grid,
        cv=5,
        verbose=True
    )

    gs_log_model.fit(X_train, y_train)

    return gs_log_model



def save_model(model):
    MODEL_PATH = "model.pkl"
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def main():
    PARENT_DIR = Path(__file__).resolve().parents[1]
    df_path = PARENT_DIR / "Data" / "heart-disease.csv"

    df = load_dataset(df_path)
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    model = train_model(X_train, y_train)

    save = save_model(model)

    print(model.score(X_test, y_test))



if __name__ == "__main__":
    main()