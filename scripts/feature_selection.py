from sklearn.feature_selection import SelectKBest, f_classif

def select_important_features(X, y):
    """
    Select the top K features using ANOVA F-test.
    """
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    return selector.get_support(indices=True)
