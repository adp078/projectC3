# Script to train machine learning model.
from src.data import process_data
from src.model import train_model
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from joblib import dump

def train_test_model():
    # Add code to load in the data.
    data = pd.read_csv("data/clean/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, lb_test = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
    )
    # Train and save a model.
    trained_model = train_model(X_train, y_train)
    dump(trained_model, "model/model.joblib")
    dump(encoder, "model/encoder.joblib")
    dump(lb, "model/lb.joblib")
