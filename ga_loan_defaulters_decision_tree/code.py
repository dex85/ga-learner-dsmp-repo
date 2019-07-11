# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)

X = data.drop(['customer.id', 'paid.back.loan'], axis = 1)
y = data['paid.back.loan'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
print(type(fully_paid))

fully_paid.plot(kind='bar')

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
X_train['int.rate'] = X_train['int.rate'].apply(lambda x: float(x.replace('%',''))/100.)
X_test['int.rate'] = X_test['int.rate'].apply(lambda x: float(x.replace('%',''))/100.)

print(X_train.head(4))
num_df = X_train.select_dtypes(include=np.number)
cat_df = X_train.select_dtypes(include=np.object_)
# Code ends here


# --------------
#Importing header files
import seaborn as sns
rows = 9

# Code starts here
cols = num_df.columns
plt.figure()
fig, axes = plt.subplots(nrows=rows, ncols=1)

for i in range(rows):
    sns.boxplot(x=y_train, y = num_df[cols[i]], ax = axes[i])

plt.show()

# Code ends here


# --------------
# Code starts here
cols = cat_df.columns
rows = 2
columns = 2

plt.figure()

fig, axes = plt.subplots(nrows = rows, ncols = columns)

for i in range(rows):
    for j in range(columns):
        sns.countplot(x = X_train[cols[i*2+j]], hue = y_train, ax = axes[i, j])
        
plt.show()
# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
le = LabelEncoder()

X_train.fillna('NA', inplace = True)
X_test.fillna('NA', inplace = True)

cat_cols = cat_df.columns
#print(X_train[cat_cols].head(2))

X_train[cat_cols] = X_train[cat_cols].apply(le.fit_transform)
X_test[cat_cols] = X_test[cat_cols].apply(le.fit_transform)

y_train = y_train.replace('No', 0)
y_train = y_train.replace('Yes', 1)
y_test = y_test.replace('No', 0)
y_test = y_test.replace('Yes', 1)

model = DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)


acc = model.score(X_test, y_test)
tracc = model.score(X_train, y_train)

print("Decisiontree with Labelencoded data:\nTrain acc: {}\nTest acc: {}\n".format(tracc, acc))


print(y_train[:2])

#print(X_train[cat_cols].head(2))
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}
model_2 = DecisionTreeClassifier(random_state = 0)
p_tree = GridSearchCV(estimator = model_2, param_grid = parameter_grid, cv = 5)

p_tree.fit(X_train, y_train)

acc_2 = p_tree.score(X_test, y_test)
tracc_2 = p_tree.score(X_train, y_train)

# Code starts here
print("Gridsearch Decisiontree: \nTraining acc: {} \nTest acc: {}".format(acc_2, tracc_2))


# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here

dot_data = export_graphviz(decision_tree = p_tree.best_estimator_, out_file = None, feature_names = X.columns, filled = True, class_names = ['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


