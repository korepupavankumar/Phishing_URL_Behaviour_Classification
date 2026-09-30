import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif


TARGET_COLUMN = "label"


def select_features(X, y, number_of_features=20):
    """
    Select the most informative numerical features.
    """

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=number_of_features
    )

    X_selected = selector.fit_transform(X, y)

    selected_columns = X.columns[
        selector.get_support()
    ]

    return X_selected, selected_columns, selector


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
    )

    drop_columns = [
        "FILENAME",
        "URL",
        "Domain",
        "TLD",
        "Title"
    ]

    df = df.drop(columns=drop_columns)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_selected, selected_columns, selector = select_features(
        X,
        y,
        number_of_features=20
    )

    print("\nSelected Features:")
    
    for column in selected_columns:
        print(column)