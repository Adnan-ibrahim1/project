from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def train_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return model, auc
