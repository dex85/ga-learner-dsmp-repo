# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
df.head()
X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)




# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
X_train['bmi'].plot(kind='box')
q_value = X_train['bmi'].quantile(0.95)

print(y_train.value_counts(), q_value, sep='\n')



# Code ends here


# --------------
# Code starts here
relation = X_train.corr()
print(relation)

sns.pairplot(X_train)

# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols = ['children', 'sex', 'region', 'smoker']
fig, axes = plt.subplots(2,2)

for i in range(2):
    for j in range(2):
        col = cols[i*2 + j]
        sns.countplot(x = X_train[col], hue = y_train, ax=axes[i,j])
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=parameters)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here

score = roc_auc_score(y_test, y_pred)
y_pred_proba = grid.predict_proba(X_test)[:,1]
y_pred = grid.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(y_pred_proba.shape)

plt.figure()
plt.plot(fpr, tpr, label = "Logistic model, auc="+str(roc_auc))
plt.show()
# Code ends here


