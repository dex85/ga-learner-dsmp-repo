# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


# Code starts here

# load data
news = pd.read_csv(path)

# subset data
news = news[['TITLE','CATEGORY']]

# distribution of classes
dist = news.CATEGORY.value_counts()

# display class distribution
print(news.head(), dist, sep = '\n \n')


# display data
import matplotlib.pyplot as plt
plt.figure()
dist.plot.bar()

# Code ends here


# --------------
# Code starts here

# stopwords 
stop = set(stopwords.words('english'))

# retain only alphabets
news['TITLE'] = news['TITLE'].apply(lambda x: re.sub("[^a-zA-Z]", " ",x))

# convert to lowercase and tokenize
news['TITLE'] = news['TITLE'].apply(lambda x: x.lower())
news['TITLE'] = news['TITLE'].apply(lambda x: x.split())

# remove stopwords
news['TITLE'] = news['TITLE'].apply(lambda x: [i for i in x if i not in stop])

# join list elements
news['TITLE'] = news['TITLE'].apply(lambda x: ' '.join(x))

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(news.TITLE, news.CATEGORY, test_size = 0.2, random_state = 3)


# Code ends here


# --------------
# Code starts here

# initialize count vectorizer / tfidf vectorizer
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3))

# fit and transform with count vectorizer
X_train_count = count_vectorizer.fit_transform(X_train)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Code ends here


# --------------
# Code starts here

# initialize multinomial naive bayes
nb_1 = MultinomialNB().fit(X_train_count, y_train)
nb_2 = MultinomialNB().fit(X_train_tfidf, y_train)

# accuracy with count vectorizer / tfidf vectorizer
y_pred_count = nb_1.predict(X_test_count)
acc_count_nb = accuracy_score(y_test, y_pred_count)

y_pred_tfidf = nb_2.predict(X_test_tfidf)
acc_tfidf_nb = accuracy_score(y_test, y_pred_tfidf)

# display accuracies
print("Accuracy of Naive Bayes Classifier with Count Vector processed Data: {}\nAccuracy of Naive Bayes Classifier with TF-IDF processed Data: {}".format(acc_count_nb, acc_tfidf_nb))

import matplotlib.pyplot as plt
plt.figure()
plt.bar([1,2], [acc_count_nb, acc_tfidf_nb], tick_label = ["Count", "TF-IDF"])
plt.show()

# Code ends here


# --------------
import warnings
warnings.filterwarnings('ignore')

# initialize logistic regression / fit on training data
logreg_1 = OneVsRestClassifier(LogisticRegression(random_state = 10)).fit(X_train_count, y_train)
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state = 10)).fit(X_train_tfidf, y_train)

# accuracy with count / tfidf
y_pred_count_logreg = logreg_1.predict(X_test_count)
acc_count_logreg = accuracy_score(y_test,y_pred_count_logreg)

y_pred_tfidf_logreg = logreg_2.predict(X_test_tfidf)
acc_tfidf_logreg = accuracy_score(y_test,y_pred_tfidf_logreg)

print("Accuracy of Logistic Regression Classifier with Count Vector processed Data: {}\nAccuracy of Logistic Regression Classifier with TF-IDF processed Data: {}".format(acc_count_logreg, acc_tfidf_logreg))

# display accuracies
import matplotlib.pyplot as plt
plt.figure()
plt.bar([1,2,3,4], [acc_count_nb, acc_tfidf_nb, acc_count_logreg, acc_tfidf_logreg], tick_label = ["NB-Count", "NB-TF-IDF","LogReg-Count", "LogReg-TF-IDF"])
plt.show()


# Code ends here


