import pandas as pd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns
sns.set()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train.head())


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

# bar_chart('Sex')
# bar_chart('Pclass')
# bar_chart('Embarked')
# bar_chart('Parch')
# bar_chart('SibSp')
# plt.show();

all_data = [train,test]
for data in all_data :
    data['Status'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train.Status.unique()

for dataset in all_data:
    dataset['Status'] = dataset['Status'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Status'] = dataset['Status'].replace('Mlle', 'Miss')
    dataset['Status'] = dataset['Status'].replace('Ms', 'Miss')
    dataset['Status'] = dataset['Status'].replace('Mme', 'Mrs')

train.isnull().sum()

test.head(10)
status_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_data:
    dataset['Status'] = dataset['Status'].map(status_mapping)
    dataset['Status'] = dataset['Status'].fillna(0)
train.Status.unique()

bar_chart('Status')
plt.show()

train['FamilySize'] = train ['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test ['SibSp'] + test['Parch'] + 1

sex_mapping = {"male": 0, "female": 1}
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

for dataset in all_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train['Cabin_category'] = train['Cabin'].astype(str).str[0]
train['Cabin_category'] = train['Cabin_category'].map({'A': 1, 'B': 2, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
train['Cabin_category'] = train['Cabin_category'].fillna(0)
# Cabin Grouping
train['HasCabin'] = train['Cabin'].apply(lambda x: 0 if x is np.nan else 1)

test['Cabin_category'] = test['Cabin'].astype(str).str[0]
test['Cabin_category'] = test['Cabin_category'].map({'A': 1, 'B': 2, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
test['Cabin_category'] = test['Cabin_category'].fillna(0)
# Cabin Grouping
test['HasCabin'] = test['Cabin'].apply(lambda x: 0 if x is np.nan else 1)



train["Age"].fillna(train.groupby("Status")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Status")["Age"].transform("median"), inplace=True)
train['Fare'].fillna(train.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0], inplace = True)
test['Fare'].fillna(test.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0], inplace = True)
train['Embarked'].fillna('S', inplace = True)
test['Embarked'].fillna('S', inplace = True)

train.groupby("Status")["Age"].transform("median")

# print(train.isnull().sum())

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
y_full = train["Survived"]

features = ["Pclass","Sex", "Age","IsAlone", "FamilySize", "Status","Embarked","Fare","Cabin_category","HasCabin"]
X_full = pd.get_dummies(train[features])
X_test_full = pd.get_dummies(test[features])

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, test_size=0.2,random_state=0)
rf_model = RandomForestClassifier(max_depth=3, random_state=3)
rf_model.fit(X_train, y_train)
rf_val_predictions = rf_model.predict(X_valid)

rf_accuracy = accuracy_score(rf_val_predictions,y_valid)
print(rf_accuracy)

rf_model.fit(X_full, y_full)
predictions = rf_model.predict(X_test_full)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('output.csv', index=False)