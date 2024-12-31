from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scripts.preprocess import load_data, encode_target,preprocess_data
from scripts.feature_selection import select_important_features

def train_boosting():
    data = load_data()
    data = encode_target(data)
    x = data.drop(columns=['target'])
    Y = data['target']
    # Preprocess the data (Impute missing values and select features)
    X, y = preprocess_data(x, Y)
    
    # Feature selection
    selected_features = select_important_features(X, y)
    X = X[:, selected_features]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # AdaBoost Model
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada_boost.fit(X_train, y_train)
    y_pred_ada = ada_boost.predict(X_test)
    
    # GradientBoosting Model
    gradient_boost = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gradient_boost.fit(X_train, y_train)
    y_pred_gb = gradient_boost.predict(X_test)
    
    # Print classification report
    return y_test, y_pred_ada,y_pred_gb 