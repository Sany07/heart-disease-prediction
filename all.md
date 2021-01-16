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

    Epoch 1/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.9869 - accuracy: 0.2873
    Epoch 2/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.9587 - accuracy: 0.2983
    Epoch 3/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.9321 - accuracy: 0.3370
    Epoch 4/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.9071 - accuracy: 0.3481
    Epoch 5/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.8843 - accuracy: 0.3757
    Epoch 6/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.8620 - accuracy: 0.3923
    Epoch 7/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.8398 - accuracy: 0.3923
    Epoch 8/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.8200 - accuracy: 0.3978
    Epoch 9/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.8013 - accuracy: 0.4144
    Epoch 10/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.7831 - accuracy: 0.4309
    Epoch 11/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.7662 - accuracy: 0.4309
    Epoch 12/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.7502 - accuracy: 0.4696
    Epoch 13/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.7346 - accuracy: 0.4862
    Epoch 14/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.7204 - accuracy: 0.5304
    Epoch 15/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.7067 - accuracy: 0.5304
    Epoch 16/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.6929 - accuracy: 0.5580
    Epoch 17/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.6804 - accuracy: 0.5912
    Epoch 18/300
    6/6 [==============================] - 0s 976us/step - loss: 0.6683 - accuracy: 0.6133
    Epoch 19/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.6563 - accuracy: 0.6464
    Epoch 20/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.6455 - accuracy: 0.6519
    Epoch 21/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.6348 - accuracy: 0.6575
    Epoch 22/300
    6/6 [==============================] - ETA: 0s - loss: 0.5957 - accuracy: 0.68 - 0s 2ms/step - loss: 0.6245 - accuracy: 0.6961
    Epoch 23/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.6146 - accuracy: 0.7182
    Epoch 24/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.6051 - accuracy: 0.7293
    Epoch 25/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5959 - accuracy: 0.7348
    Epoch 26/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5872 - accuracy: 0.7569
    Epoch 27/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5782 - accuracy: 0.7624
    Epoch 28/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5699 - accuracy: 0.7569
    Epoch 29/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5614 - accuracy: 0.7680
    Epoch 30/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5536 - accuracy: 0.7680
    Epoch 31/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5457 - accuracy: 0.7680
    Epoch 32/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5384 - accuracy: 0.7735
    Epoch 33/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5315 - accuracy: 0.7735
    Epoch 34/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5246 - accuracy: 0.7735
    Epoch 35/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.5175 - accuracy: 0.7845
    Epoch 36/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5108 - accuracy: 0.7845
    Epoch 37/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.5048 - accuracy: 0.7845
    Epoch 38/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4986 - accuracy: 0.7901
    Epoch 39/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4932 - accuracy: 0.7901
    Epoch 40/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4872 - accuracy: 0.7901
    Epoch 41/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4821 - accuracy: 0.7901
    Epoch 42/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4765 - accuracy: 0.7901
    Epoch 43/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4711 - accuracy: 0.7956
    Epoch 44/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4662 - accuracy: 0.8011
    Epoch 45/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4617 - accuracy: 0.8011
    Epoch 46/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4569 - accuracy: 0.8177
    Epoch 47/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4523 - accuracy: 0.8177
    Epoch 48/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4478 - accuracy: 0.8122
    Epoch 49/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4437 - accuracy: 0.8066
    Epoch 50/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4393 - accuracy: 0.8066
    Epoch 51/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4351 - accuracy: 0.8122
    Epoch 52/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4315 - accuracy: 0.8122
    Epoch 53/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4278 - accuracy: 0.8122
    Epoch 54/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4241 - accuracy: 0.8122
    Epoch 55/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4208 - accuracy: 0.8066
    Epoch 56/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4174 - accuracy: 0.8066
    Epoch 57/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4141 - accuracy: 0.8122
    Epoch 58/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.4109 - accuracy: 0.8122
    Epoch 59/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.4078 - accuracy: 0.8122
    Epoch 60/300
    6/6 [==============================] - 0s 976us/step - loss: 0.4050 - accuracy: 0.8122
    Epoch 61/300
    6/6 [==============================] - 0s 976us/step - loss: 0.4021 - accuracy: 0.8177
    Epoch 62/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3991 - accuracy: 0.8232
    Epoch 63/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3965 - accuracy: 0.8232
    Epoch 64/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3935 - accuracy: 0.8232
    Epoch 65/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3909 - accuracy: 0.8232
    Epoch 66/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3881 - accuracy: 0.8177
    Epoch 67/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3855 - accuracy: 0.8177
    Epoch 68/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3830 - accuracy: 0.8177
    Epoch 69/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3806 - accuracy: 0.8177
    Epoch 70/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3783 - accuracy: 0.8177
    Epoch 71/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3760 - accuracy: 0.8177
    Epoch 72/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3740 - accuracy: 0.8177
    Epoch 73/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3718 - accuracy: 0.8177
    Epoch 74/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3697 - accuracy: 0.8177
    Epoch 75/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3678 - accuracy: 0.8177
    Epoch 76/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3660 - accuracy: 0.8177
    Epoch 77/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3639 - accuracy: 0.8177
    Epoch 78/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3622 - accuracy: 0.8177
    Epoch 79/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3603 - accuracy: 0.8177
    Epoch 80/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3586 - accuracy: 0.8177
    Epoch 81/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3568 - accuracy: 0.8177
    Epoch 82/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3552 - accuracy: 0.8232
    Epoch 83/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3537 - accuracy: 0.8177
    Epoch 84/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3519 - accuracy: 0.8177
    Epoch 85/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3504 - accuracy: 0.8177
    Epoch 86/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3489 - accuracy: 0.8232
    Epoch 87/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3474 - accuracy: 0.8232
    Epoch 88/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3459 - accuracy: 0.8232
    Epoch 89/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3444 - accuracy: 0.8287
    Epoch 90/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3430 - accuracy: 0.8287
    Epoch 91/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3415 - accuracy: 0.8287
    Epoch 92/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3403 - accuracy: 0.8287
    Epoch 93/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3393 - accuracy: 0.8287
    Epoch 94/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3377 - accuracy: 0.8232
    Epoch 95/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3364 - accuracy: 0.8232
    Epoch 96/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3350 - accuracy: 0.8232
    Epoch 97/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3339 - accuracy: 0.8232
    Epoch 98/300
    6/6 [==============================] - 0s 975us/step - loss: 0.3327 - accuracy: 0.8232
    Epoch 99/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3316 - accuracy: 0.8232
    Epoch 100/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3305 - accuracy: 0.8287
    Epoch 101/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3294 - accuracy: 0.8287
    Epoch 102/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3283 - accuracy: 0.8343
    Epoch 103/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3273 - accuracy: 0.8343
    Epoch 104/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3262 - accuracy: 0.8343
    Epoch 105/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3253 - accuracy: 0.8398
    Epoch 106/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3243 - accuracy: 0.8398
    Epoch 107/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3233 - accuracy: 0.8398
    Epoch 108/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3223 - accuracy: 0.8398
    Epoch 109/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3215 - accuracy: 0.8398
    Epoch 110/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3205 - accuracy: 0.8453
    Epoch 111/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3197 - accuracy: 0.8453
    Epoch 112/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3188 - accuracy: 0.8453
    Epoch 113/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3177 - accuracy: 0.8453
    Epoch 114/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3171 - accuracy: 0.8453
    Epoch 115/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3160 - accuracy: 0.8453
    Epoch 116/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3152 - accuracy: 0.8453
    Epoch 117/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3143 - accuracy: 0.8453
    Epoch 118/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3136 - accuracy: 0.8453
    Epoch 119/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3126 - accuracy: 0.8564
    Epoch 120/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3119 - accuracy: 0.8564
    Epoch 121/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3110 - accuracy: 0.8564
    Epoch 122/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3103 - accuracy: 0.8564
    Epoch 123/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3095 - accuracy: 0.8564
    Epoch 124/300
    6/6 [==============================] - 0s 976us/step - loss: 0.3085 - accuracy: 0.8564
    Epoch 125/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3079 - accuracy: 0.8564
    Epoch 126/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3070 - accuracy: 0.8564
    Epoch 127/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3063 - accuracy: 0.8619
    Epoch 128/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3055 - accuracy: 0.8564
    Epoch 129/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3047 - accuracy: 0.8564
    Epoch 130/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3040 - accuracy: 0.8564
    Epoch 131/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.3033 - accuracy: 0.8564
    Epoch 132/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3023 - accuracy: 0.8564
    Epoch 133/300
    6/6 [==============================] - 0s 977us/step - loss: 0.3015 - accuracy: 0.8564
    Epoch 134/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.3007 - accuracy: 0.8619
    Epoch 135/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2999 - accuracy: 0.8619
    Epoch 136/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2993 - accuracy: 0.8674
    Epoch 137/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2982 - accuracy: 0.8674
    Epoch 138/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2976 - accuracy: 0.8729
    Epoch 139/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2966 - accuracy: 0.8729
    Epoch 140/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2959 - accuracy: 0.8729
    Epoch 141/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2950 - accuracy: 0.8729
    Epoch 142/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2943 - accuracy: 0.8729
    Epoch 143/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2935 - accuracy: 0.8729
    Epoch 144/300
    6/6 [==============================] - 0s 975us/step - loss: 0.2927 - accuracy: 0.8729
    Epoch 145/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2921 - accuracy: 0.8729
    Epoch 146/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2913 - accuracy: 0.8729
    Epoch 147/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2907 - accuracy: 0.8729
    Epoch 148/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2900 - accuracy: 0.8729
    Epoch 149/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2896 - accuracy: 0.8729
    Epoch 150/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2886 - accuracy: 0.8729
    Epoch 151/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2880 - accuracy: 0.8729
    Epoch 152/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2873 - accuracy: 0.8729
    Epoch 153/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2868 - accuracy: 0.8729
    Epoch 154/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2860 - accuracy: 0.8729
    Epoch 155/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2854 - accuracy: 0.8729
    Epoch 156/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2849 - accuracy: 0.8729
    Epoch 157/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2841 - accuracy: 0.8729
    Epoch 158/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2835 - accuracy: 0.8729
    Epoch 159/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2829 - accuracy: 0.8729
    Epoch 160/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2823 - accuracy: 0.8729
    Epoch 161/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2817 - accuracy: 0.8729
    Epoch 162/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2810 - accuracy: 0.8729
    Epoch 163/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2806 - accuracy: 0.8729
    Epoch 164/300
    6/6 [==============================] - 0s 813us/step - loss: 0.2798 - accuracy: 0.8785
    Epoch 165/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2793 - accuracy: 0.8785
    Epoch 166/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2788 - accuracy: 0.8785
    Epoch 167/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2781 - accuracy: 0.8785
    Epoch 168/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2775 - accuracy: 0.8785
    Epoch 169/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2768 - accuracy: 0.8785
    Epoch 170/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2762 - accuracy: 0.8785
    Epoch 171/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2756 - accuracy: 0.8785
    Epoch 172/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2751 - accuracy: 0.8785
    Epoch 173/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2745 - accuracy: 0.8785
    Epoch 174/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2738 - accuracy: 0.8785
    Epoch 175/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2734 - accuracy: 0.8785
    Epoch 176/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2727 - accuracy: 0.8785
    Epoch 177/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2721 - accuracy: 0.8785
    Epoch 178/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2713 - accuracy: 0.8785
    Epoch 179/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2707 - accuracy: 0.8785
    Epoch 180/300
    6/6 [==============================] - 0s 813us/step - loss: 0.2702 - accuracy: 0.8785
    Epoch 181/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2696 - accuracy: 0.8785
    Epoch 182/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2690 - accuracy: 0.8785
    Epoch 183/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2685 - accuracy: 0.8785
    Epoch 184/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2680 - accuracy: 0.8785
    Epoch 185/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2674 - accuracy: 0.8785
    Epoch 186/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2670 - accuracy: 0.8785
    Epoch 187/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2665 - accuracy: 0.8785
    Epoch 188/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2659 - accuracy: 0.8785
    Epoch 189/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2655 - accuracy: 0.8785
    Epoch 190/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2649 - accuracy: 0.8785
    Epoch 191/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2644 - accuracy: 0.8785
    Epoch 192/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2639 - accuracy: 0.8785
    Epoch 193/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2633 - accuracy: 0.8785
    Epoch 194/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2628 - accuracy: 0.8785
    Epoch 195/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2623 - accuracy: 0.8785
    Epoch 196/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2618 - accuracy: 0.8785
    Epoch 197/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2614 - accuracy: 0.8785
    Epoch 198/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2607 - accuracy: 0.8785
    Epoch 199/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2602 - accuracy: 0.8785
    Epoch 200/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2597 - accuracy: 0.8785
    Epoch 201/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2590 - accuracy: 0.8785
    Epoch 202/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2585 - accuracy: 0.8785
    Epoch 203/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2580 - accuracy: 0.8785
    Epoch 204/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2575 - accuracy: 0.8785
    Epoch 205/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2570 - accuracy: 0.8785
    Epoch 206/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2564 - accuracy: 0.8785
    Epoch 207/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2559 - accuracy: 0.8785
    Epoch 208/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2555 - accuracy: 0.8785
    Epoch 209/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2549 - accuracy: 0.8785
    Epoch 210/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2545 - accuracy: 0.8785
    Epoch 211/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2541 - accuracy: 0.8785
    Epoch 212/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2534 - accuracy: 0.8785
    Epoch 213/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2528 - accuracy: 0.8785
    Epoch 214/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2523 - accuracy: 0.8785
    Epoch 215/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2519 - accuracy: 0.8785
    Epoch 216/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2515 - accuracy: 0.8785
    Epoch 217/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2509 - accuracy: 0.8785
    Epoch 218/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2503 - accuracy: 0.8785
    Epoch 219/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2499 - accuracy: 0.8785
    Epoch 220/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2492 - accuracy: 0.8785
    Epoch 221/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2488 - accuracy: 0.8785
    Epoch 222/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2483 - accuracy: 0.8785
    Epoch 223/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2479 - accuracy: 0.8785
    Epoch 224/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2475 - accuracy: 0.8785
    Epoch 225/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2468 - accuracy: 0.8785
    Epoch 226/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2464 - accuracy: 0.8785
    Epoch 227/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2459 - accuracy: 0.8840
    Epoch 228/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2453 - accuracy: 0.8840
    Epoch 229/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2449 - accuracy: 0.8840
    Epoch 230/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2445 - accuracy: 0.8840
    Epoch 231/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2439 - accuracy: 0.8840
    Epoch 232/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2435 - accuracy: 0.8840
    Epoch 233/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2429 - accuracy: 0.8840
    Epoch 234/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2423 - accuracy: 0.8840
    Epoch 235/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2419 - accuracy: 0.8840
    Epoch 236/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2415 - accuracy: 0.8840
    Epoch 237/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2411 - accuracy: 0.8840
    Epoch 238/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2405 - accuracy: 0.8840
    Epoch 239/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2398 - accuracy: 0.8840
    Epoch 240/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2396 - accuracy: 0.8840
    Epoch 241/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2389 - accuracy: 0.8840
    Epoch 242/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2385 - accuracy: 0.8840
    Epoch 243/300
    6/6 [==============================] - 0s 977us/step - loss: 0.2381 - accuracy: 0.8840
    Epoch 244/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2375 - accuracy: 0.8895
    Epoch 245/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2370 - accuracy: 0.8895
    Epoch 246/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2366 - accuracy: 0.8950
    Epoch 247/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2361 - accuracy: 0.8950
    Epoch 248/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2358 - accuracy: 0.8895
    Epoch 249/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2354 - accuracy: 0.8895
    Epoch 250/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2347 - accuracy: 0.8895
    Epoch 251/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2341 - accuracy: 0.8950
    Epoch 252/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2338 - accuracy: 0.8950
    Epoch 253/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2333 - accuracy: 0.8950
    Epoch 254/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2328 - accuracy: 0.8950
    Epoch 255/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2327 - accuracy: 0.8950
    Epoch 256/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2322 - accuracy: 0.8950
    Epoch 257/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2315 - accuracy: 0.8950
    Epoch 258/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2310 - accuracy: 0.9006
    Epoch 259/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2305 - accuracy: 0.9006
    Epoch 260/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2302 - accuracy: 0.9006
    Epoch 261/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2296 - accuracy: 0.9006
    Epoch 262/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2293 - accuracy: 0.9006
    Epoch 263/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2287 - accuracy: 0.9006
    Epoch 264/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2284 - accuracy: 0.9006
    Epoch 265/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2278 - accuracy: 0.9006
    Epoch 266/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2274 - accuracy: 0.9006
    Epoch 267/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2269 - accuracy: 0.9006
    Epoch 268/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2264 - accuracy: 0.9006
    Epoch 269/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2260 - accuracy: 0.9006
    Epoch 270/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2257 - accuracy: 0.9006
    Epoch 271/300
    6/6 [==============================] - 0s 978us/step - loss: 0.2251 - accuracy: 0.9006
    Epoch 272/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2248 - accuracy: 0.9006
    Epoch 273/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2241 - accuracy: 0.9006
    Epoch 274/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2237 - accuracy: 0.9006
    Epoch 275/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2233 - accuracy: 0.9006
    Epoch 276/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2227 - accuracy: 0.9006
    Epoch 277/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2222 - accuracy: 0.9006
    Epoch 278/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2217 - accuracy: 0.9006
    Epoch 279/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2212 - accuracy: 0.9006
    Epoch 280/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2208 - accuracy: 0.9006
    Epoch 281/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2202 - accuracy: 0.9006
    Epoch 282/300
    6/6 [==============================] - 0s 978us/step - loss: 0.2198 - accuracy: 0.9006
    Epoch 283/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2192 - accuracy: 0.9006
    Epoch 284/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2186 - accuracy: 0.9006
    Epoch 285/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2183 - accuracy: 0.9006
    Epoch 286/300
    6/6 [==============================] - 0s 2ms/step - loss: 0.2177 - accuracy: 0.9006
    Epoch 287/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2171 - accuracy: 0.9006
    Epoch 288/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2167 - accuracy: 0.9006
    Epoch 289/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2161 - accuracy: 0.9006
    Epoch 290/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2156 - accuracy: 0.9006
    Epoch 291/300
    6/6 [==============================] - 0s 975us/step - loss: 0.2152 - accuracy: 0.9006
    Epoch 292/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2147 - accuracy: 0.9006
    Epoch 293/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2139 - accuracy: 0.9006
    Epoch 294/300
    6/6 [==============================] - 0s 813us/step - loss: 0.2134 - accuracy: 0.9006
    Epoch 295/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2129 - accuracy: 0.9061
    Epoch 296/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2127 - accuracy: 0.9061
    Epoch 297/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2121 - accuracy: 0.9061
    Epoch 298/300
    6/6 [==============================] - 0s 976us/step - loss: 0.2115 - accuracy: 0.9061
    Epoch 299/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2111 - accuracy: 0.9061
    Epoch 300/300
    6/6 [==============================] - 0s 1ms/step - loss: 0.2105 - accuracy: 0.9061
    




    <tensorflow.python.keras.callbacks.History at 0x1f022e90850>




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
