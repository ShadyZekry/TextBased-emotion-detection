from imblearn.over_sampling import SMOTE

smote = SMOTE()

def use_smote(x, y):
    x_smote, y_smote = smote.fit_resample(x, y)
    return (x_smote, y_smote)









