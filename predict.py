import joblib
import pandas as pd


MODEL_PATH = "model/phishing_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"


def load_model():

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, features


def predict_url(feature_values):

    model, feature_columns = load_model()

    input_data = pd.DataFrame(
        [feature_values],
        columns=feature_columns
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0]

    return prediction, probability