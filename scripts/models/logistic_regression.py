from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return model, auc
