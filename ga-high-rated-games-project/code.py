# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].hist()
mask = data['Rating'] <= 5
data = data[mask].copy()
plt.figure()
data['Rating'].hist()

#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())

missing_data = pd.concat([total_null, percent_null], axis = 1, keys = ['Total', 'Percent'])

print(missing_data)

data.dropna(inplace = True)

total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())

missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys = ['Total', 'Percent'])


print(missing_data_1)


# code ends here


# --------------



#Code starts here
sns.catplot(x="Category", y="Rating", data=data, kind="box", height = 10)
plt.title("Rating vs Category [BoxPlot]")
plt.xticks(rotation = 90)

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'] = data.Installs.apply(lambda str_val: int(str_val[:-1].replace(',','')))

print(data.Installs)

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x="Installs", y="Rating", data=data)
plt.title("Rating vs Installs [RegPlot]")

#Code ends here



# --------------
#Code starts here
print(data.Price)


data.Price = data.Price.apply(lambda str_value: float(str_value.replace('$', '')))

sns.regplot(x="Price", y="Rating", data=data)
plt.title("Rating vs Price [RegPlot]")


#Code ends here


# --------------




#Code starts here
#print(data.Genres.unique())

data.Genres = data.Genres.apply(lambda str_val: str_val.split(";")[0])

#print(data.Genres.unique())
gr_mean = data[["Genres", "Rating"]].groupby(["Genres"], as_index=False).mean()
#print(gr_mean.describe())

gr_mean = gr_mean.sort_values("Rating")


print(gr_mean.iloc[0], gr_mean.iloc[-1], sep="\n")
#Code ends here


# --------------

#Code starts here
#print(data["Last Updated"])


data["Last Updated"] = data["Last Updated"].apply(lambda str_val: pd.to_datetime(str_val))

print(data["Last Updated"].head())

max_date = data["Last Updated"].max()

data["Last Updated Days"] = (max_date - data["Last Updated"]).dt.days

print(data["Last Updated Days"].head())

sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title("Rating vs Last Updated [RegPlot]")

#Code ends here


