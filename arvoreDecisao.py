import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('credit-g.csv')

X = base.iloc[:, 0:20].values
y = base.iloc[:, 20].values

labelencoder = LabelEncoder()
i = 0
while i < 20:
    X[:, int(i)] = labelencoder.fit_transform(X[:, int(i)])
    i += 1

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size=0.3,
                                                                  random_state=0)


modelo = DecisionTreeClassifier(criterion='entropy',
min_samples_leaf=5,
min_samples_split=20,
max_depth=None)

modelo.fit(X_treinamento, y_treinamento)
export_graphviz(modelo, out_file='modelo.dot')
previsoes = modelo.predict(X_teste)
accuracy_score(y_teste, previsoes)
confusao = ConfusionMatrix(modelo)
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.poof()
