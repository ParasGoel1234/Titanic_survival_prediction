import pandas as pd

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test["PassengerId"]

def clean(data):
    data = data.drop(columns=['Ticket', "PassengerId", 'Name', 'Cabin'], axis=1)
    cols = ['SibSp', 'Parch', 'Age', 'Fare']
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data['Embarked'].fillna("U", inplace = True)
    return data;
data = clean(data)
test = clean(test)
# print(test.head())
# print(train.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cols = ["Embarked", "Sex"]
for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
#     print(le.classes_)
# print(data.head(5))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = data.drop('Survived', axis=1)
Y = data["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train, Y_train)
predict = model.predict(X_test)
# score = accuracy_score(predict, Y_test)
# print(score)
sub_prediction = model.predict(test)

df = pd.DataFrame({"PassengerId": test_ids, "Survived": sub_prediction})
df.to_csv("Submission", index=False)
print(data.isnull().sum())
