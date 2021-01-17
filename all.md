```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
# load the boston dataset 
dataframe =  pd.read_csv('heart.csv')

# defining feature matrix(X) and response vector(y) 
X = dataframe.iloc[:,:-1] 
y = dataframe.iloc[:,-1:]

#convert categorical to numeric
labelencoder_X=LabelEncoder()
z = X.iloc[:,-1:]
X.iloc[:,-1:] = labelencoder_X.fit_transform(z.values.ravel())

#z score normalization
scaller = StandardScaler()
X = scaller.fit_transform(X)

# splitting X and y into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 

# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset

```


```python
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#tree.plot_tree(clf)
```

    Accuracy: 0.7950819672131147
    

#  Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
rf = clf.fit(X_train,y_train.values.ravel())
```


```python
Y_pred_rf = rf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

    Accuracy: 0.8360655737704918
    


```python
Y_pred_rf.shape
score_rf = round(accuracy_score(Y_pred_rf,y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")
```

    The accuracy score achieved using Decision Tree is: 79.51 %
    

# Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb = nb.fit(X_train,y_train.values.ravel())
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

    Accuracy: 0.8360655737704918
    


```python
Y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(Y_pred_nb,y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
```

    The accuracy score achieved using Naive Bayes is: 85.25 %
    

# Neural Network


```python
from keras.models import Sequential
from keras.layers import Dense
```


```python
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
```


```python
model.fit(X_train,y_train,epochs=300)
```

  


```python
Y_pred_nn = model.predict(X_test)
Y_pred_nn.shape
```




    (122, 1)




```python
rounded = [round(x[0]) for x in Y_pred_nn]

```


```python


Y_pred_nn = rounded

score_nn = round(accuracy_score(Y_pred_nn,y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")
```

    The accuracy score achieved using Neural Network is: 80.33 %
    


```python

```


```python

```


```python

```


```python

```
