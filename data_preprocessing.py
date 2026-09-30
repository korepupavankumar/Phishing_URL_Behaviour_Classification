import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "label"

DROP_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title"
]


def load_data(file_path):
    """Load the phishing URL dataset."""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """Remove columns that should not be directly used for ML."""
    
    df = df.copy()

    columns_to_drop = [
        column for column in DROP_COLUMNS
        if column in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def split_data(df):
    """Separate features and target and create train/test sets."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Scale numerical features."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":

    file_path = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"

    df = load_data(file_path)

    print("Dataset shape:", df.shape)

    df = clean_data(df)

    print("After cleaning:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    print("Preprocessing completed successfully.")