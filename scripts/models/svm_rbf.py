from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scripts.preprocess import load_data, encode_target, preprocess_data
from scripts.feature_selection import select_important_features

def train_svm_rbf():
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
    
    # SVM with RBF kernel
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Print classification report
    return y_test,y_pred
