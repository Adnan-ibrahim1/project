from scripts.models.logistic_regression import train_logistic_regression
from scripts.models.random_forest import train_random_forest
from scripts.models.boosting import train_boosting
from scripts.models.svm_rbf import train_svm_rbf
from scripts.models.bayesian_model import train_bayesian_model
from sklearn.metrics import classification_report, accuracy_score
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Function to save classification reports, metrics, and generate plots
def save_classification_reports(y_test, y_pred, model_name):
    # Generate the classification report (human-readable)
    report = classification_report(y_test, y_pred)

    # Print the classification report to the console
    print(f"{model_name} Classification Report:")
    print(report)

    # Save classification report as a text file (human-readable)
    os.makedirs('outputs/reports', exist_ok=True)
    with open(f'outputs/reports/{model_name}_classification_report.txt', 'w') as f:
        f.write(report)

    # Save performance metrics (accuracy, precision, recall, F1-score) to a JSON file
    accuracy = accuracy_score(y_test, y_pred)
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)  # stores detailed metrics as a dictionary
    }

    # Save JSON metrics
    os.makedirs('outputs', exist_ok=True)
    with open(f'outputs/{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plotting the metrics (accuracy, precision, recall, F1-score)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Extract values from the classification report (for the classes or overall)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Accuracy bar chart
    metrics_values = {
        'accuracy': accuracy,
        'precision': report_dict['accuracy'],
        'recall': report_dict['accuracy'],
        'f1_score': report_dict['accuracy']
    }

    # Create a plot for accuracy, precision, recall, f1-score
    plt.figure(figsize=(8, 6))
    metrics = list(metrics_values.values())
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plt.bar(metrics_labels, metrics, color='skyblue')
    plt.ylabel('Scores')
    plt.title(f'{model_name} Performance Metrics')
    plt.savefig(f'outputs/plots/{model_name}_performance_metrics.png')
    plt.close()
    
    # Plot Precision, Recall, and F1-score for each class
    classes = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    precision = [report_dict[class_]['precision'] for class_ in classes]
    recall = [report_dict[class_]['recall'] for class_ in classes]
    f1_score = [report_dict[class_]['f1-score'] for class_ in classes]
    
    # Create a plot for each class
    x = np.arange(len(classes))
    width = 0.25  # width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', color='blue')
    plt.bar(x, recall, width, label='Recall', color='green')
    plt.bar(x + width, f1_score, width, label='F1-Score', color='red')

    plt.ylabel('Scores')
    plt.title(f'{model_name} Class-wise Performance')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{model_name}_classwise_performance.png')
    plt.close()

def main():
    # Logistic Regression
    print("Training Logistic Regression...")
    y_test, y_pred = train_logistic_regression()  # Ensure this function returns y_test and y_pred
    save_classification_reports(y_test, y_pred, "logistic_regression")

    # Random Forest
    print("Training Random Forest...")
    y_test, y_pred = train_random_forest()  # Ensure this function returns y_test and y_pred
    save_classification_reports(y_test, y_pred, "random_forest")

    # Boosting models
    print("Training Boosting models...")
    y_test, y_pred_ada,y_pred_gb  = train_boosting()  # Ensure this function returns y_test and y_pred
    save_classification_reports(y_test, y_pred, "boosting")
    save_classification_reports(y_test, y_pred_ada, "ADABoosting")
    save_classification_reports(y_test, y_pred_gb, "GradientBoosting")


    # SVM with RBF Kernel
    print("Training SVM with RBF Kernel...")
    y_test, y_pred = train_svm_rbf()  # Ensure this function returns y_test and y_pred
    save_classification_reports(y_test, y_pred, "svm_rbf")

    # Naive Bayes
    print("Training Naive Bayes...")
    y_test, y_pred = train_bayesian_model()  # Ensure this function returns y_test and y_pred
    save_classification_reports(y_test, y_pred, "naive_bayes")

if __name__ == "__main__":
    main()


