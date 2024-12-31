from scripts.models.logistic_regression import train_logistic_regression
from scripts.models.random_forest import train_random_forest
from scripts.models.boosting import train_boosting
from scripts.models.svm_rbf import train_svm_rbf
from scripts.models.bayesian_model import train_bayesian_model

def main():
    print("Training Logistic Regression...")
    train_logistic_regression()

    print("Training Random Forest...")
    train_random_forest()

    print("Training Boosting models...")
    train_boosting()

    print("Training SVM with RBF Kernel...")
    train_svm_rbf()

    print("Training Naive Bayes...")
    train_bayesian_model()

if __name__ == "__main__":
    main()
