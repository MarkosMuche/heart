

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.base import DataError

df = pd.read_csv('heart.csv')
df.tail(7)

df.shape
df.isna().sum()

# visualizing the data

df['target'].value_counts()
sns.countplot(df['target'])


# Further visualizing of the data

fig_dims = (15, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='age', hue='target',data=df,palette= 'Accent',ax=ax)


# splitting the data 

from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1].values 
Y = df.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state=1)


# Feature scaling



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


filename='finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)


print(result)
print((X_test[0]))
print((X_test.shape))
print(loaded_model.predict(X_test))