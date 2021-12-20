
import  pandas as pd
from pandas.core.base import DataError
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import  pickle

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



# choose the model


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=1)
forest.fit(X_train, Y_train)
model = forest
score=model.score(X_train, Y_train)
print(score)



# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


