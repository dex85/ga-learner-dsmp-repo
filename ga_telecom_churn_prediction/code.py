# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df = pd.read_csv(path)
X = df.drop(['customerID', 'Churn'], axis = 1)
y = df['Churn'].copy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
# Code starts here





# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
def space_nan(val):
    if val.find(' ')>-1:
        return np.nan
    else:
        return float(val)

X_train['TotalCharges'] = X_train['TotalCharges'].apply(space_nan)
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace = True)

X_test['TotalCharges'] = X_test['TotalCharges'].apply(space_nan)
X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(), inplace = True)


any_null = X_train.isnull().sum()

#select columns
numeric_columns = X_train.select_dtypes(include=np.number).columns
categorical_columns = X_train.drop(columns = numeric_columns).columns

lbl_enc = LabelEncoder()
X_train[categorical_columns] = X_train[categorical_columns].apply(lbl_enc.fit_transform)
X_test[categorical_columns] = X_test[categorical_columns].apply(lbl_enc.fit_transform)

y_train.replace({'No':0, 'Yes':1}, inplace = True)
y_test.replace({'No':0, 'Yes':1}, inplace = True)


# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print(X_train.head(3), y_test[:3])

ada_model = AdaBoostClassifier(random_state = 0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test, y_pred)
ada_cm = confusion_matrix(y_test, y_pred)
ada_cr = classification_report(y_test, y_pred)
print("Adaboost Accuracz Score: {}\n\nAdaboost Confusion Matrix: {}\n\nAdaboost Classification Report".format(ada_score, ada_cm, ada_cr))


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state = 0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

xgb_score = accuracy_score(y_test, y_pred)
xgb_cm = confusion_matrix(y_test, y_pred)
xgb_cr = classification_report(y_test, y_pred)
print("XGBoost Accuracy Score: {}\n\nXGBoost Confusion Matrix: {}\n\nXGBoost Classification Report".format(xgb_score, xgb_cm, xgb_cr))

#Gridboost
clf_model = GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test, y_pred)
clf_cm = confusion_matrix(y_test, y_pred)
clf_cr = classification_report(y_test, y_pred)
print("XGBoost Gridsearch Accuracy Score: {}\n\nXGBoost Gridsearch Confusion Matrix: {}\n\nXGBoost Gridsearch Classification Report".format(clf_score, clf_cm, clf_cr))



