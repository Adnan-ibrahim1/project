from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

def train_bayesian_model(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return model, auc
