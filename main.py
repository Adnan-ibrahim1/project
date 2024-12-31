from scripts.database import fetch_data
from scripts.preprocess import handle_missing_values, encode_target
from scripts.models.logistic_regression import train_logistic_regression
from scripts.evaluation import evaluate_model
from scripts.utils import split_data

def main():
    # Fetch data
    query = "SELECT * FROM dataset"
    df = fetch_data(query)

    # Preprocess
    df = handle_missing_values(df)
    df = encode_target(df, "target")
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model, auc = train_logistic_regression(X_train, y_train, X_test, y_test)
    print(f"Logistic Regression AUC: {auc}")

if __name__ == "__main__":
    main()
