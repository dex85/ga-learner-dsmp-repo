# --------------
import pandas as pd
from collections import Counter

# Load dataset
data = pd.read_csv(path)
print("Nan-Values in the Dataset:\n{}\n____________________________________________________________________________________________\nDescription of the Dataset:\n{}".format(data.isnull().sum(), data.describe()))



# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data.iloc[:,-1]
sns.countplot(x = label)
plt.xticks(rotation = 90)
plt.show()

# plot the countplot



# --------------
# make the copy of dataset
data_copy = data.copy()

# Create an empty column 
data_copy['duration'] = ''

# Calculate the duration
duration_df = (data_copy.groupby([label[label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

# Sort the values of duration
plot_data = duration_df.reset_index().sort_values('duration', ascending=False)
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs', 'WALKING_DOWNSTAIRS':'Downstairs'})


# Plot the durations for staircase use
plt.figure(figsize=(15,5))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.title('Participants Compared By Their Staircase Walking Duration')
plt.xlabel('Participants')
plt.ylabel('Total Duration [s]')
plt.show()


# --------------
#exclude the Activity column and the subject column
import numpy as np
feature_cols = data.drop(['Activity', 'subject'], axis = 1).columns

#Calculate the correlation values
correlated_values = data[feature_cols].corr()
correlated_values = correlated_values.stack().to_frame().reset_index().rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'})
correlated_values['abs_correlation'] = correlated_values['Correlation_score'].abs()
corr_var_list = correlated_values.sort_values(by = ['abs_correlation'], ascending = False)
mask = (corr_var_list['abs_correlation'] >= 0.8)
top_corr_fields = corr_var_list[mask].copy()

mask2 = top_corr_fields['Feature_1'] != top_corr_fields['Feature_2']
top_corr_fields = top_corr_fields[mask2].copy()
top_corr_fields.reset_index(drop = True, inplace = True)

print(top_corr_fields.head())
#stack the data and convert to a dataframe



#create an abs_correlation column



#Picking most correlated features without having self correlated pairs




# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
data['Activity'] = le.fit_transform(data['Activity'])

X = data.drop(['Activity'], axis = 1)
y = data['Activity'].copy()

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)

# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# metrics
precision, recall, f_score, _ = error_metric(y_test, y_pred, average = 'weighted')
model1_score = accuracy_score(y_test, y_pred)


# --------------
# importing libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


# init and fit SVC
lsvc = LinearSVC(C = 0.01, penalty = 'l1', dual = False, random_state = 42).fit(X_train, y_train)

# init feature selector
model_2 = SelectFromModel(lsvc, prefit = True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)

classfier_2 = SVC()
clf_2 = classfier_2.fit(new_train_features, y_train)
y_pred_new = clf_2.predict(new_test_features)

model2_score = accuracy_score(y_test, y_pred_new)
precision, recall, f_score, _ = error_metric(y_test, y_pred_new, average = 'weighted')
# Feature selection using Linear SVC



# model building on reduced set of features




# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = {'kernel':['linear','rbf'], 'C':[100, 20, 1, 0.1]}


# Usage of grid search to select the best hyperparmeters
selector = GridSearchCV(estimator = SVC(), param_grid = parameters, scoring = 'accuracy')
selector.fit(new_train_features, y_train)

means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
params = selector.cv_results_['params']
best_parameters = selector.best_params_
print(best_parameters)

classifier_3 = SVC(C = best_parameters['C'], kernel = best_parameters['kernel'])
clf_3 = classifier_3.fit(new_train_features, y_train)
y_pred_final = clf_3.predict(new_test_features)

model3_score = accuracy_score(y_test, y_pred_final)
precision, recall, f_score, _ = error_metric(y_test, y_pred_final, average = 'weighted')

# Model building after Hyperparameter tuning





