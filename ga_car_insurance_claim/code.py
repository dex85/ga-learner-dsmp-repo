# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
#print(df.head(5))
#print(df.info)
cols = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
for col in cols:
    df[col] = df[col].apply(lambda sample: str(sample).replace("$",""))
    df[col] = df[col].apply(lambda sample: sample.replace(",",""))

print(df.head(5))
X = df.drop(['CLAIM_FLAG'], axis = 1)
y = df['CLAIM_FLAG'].copy()
count = y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)
print(count)
# Code ends here


# --------------
# Code starts here
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
for col in cols:
    X_train[col] = X_train[col].apply(lambda x: float(x));
    X_test[col] = X_test[col].apply(lambda x: float(x));

print(X_train.isnull())
# Code ends here


# --------------
# Code starts here
# drop nans

cols = ['AGE','CAR_AGE','INCOME','HOME_VAL']
for col in cols:
    mean = X_train[col].mean()
    X_train[col].fillna(mean, inplace = True)
    
    mean = X_test[col].mean()
    X_test[col].fillna(mean, inplace = True)

X_train.dropna(inplace = True)
X_test.dropna(inplace = True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Logistic Regression with cleaned data:\nAccuracy: {}\nPrecision: {}".format(score, precision))
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)
scaler = StandardScaler()
X_train, y_train = smote.fit_sample(X_train, y_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Oversampled Dataset with logistig regression classifier\nAccuracy:{}".format(score))

# Code ends here


