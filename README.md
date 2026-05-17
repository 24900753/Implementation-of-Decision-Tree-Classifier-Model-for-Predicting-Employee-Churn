# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. Import the dataset and required libraries
   Load the employee churn dataset and import necessary Python libraries for data analysis and model building.

2. Preprocess and split the data
   Select input features and target variable, handle missing values if any, and split the dataset into training and testing sets.

3. Train the Decision Tree Classifier model
   Initialize the Decision Tree Classifier and train the model using the training data.

4. Predict and evaluate the model
   Predict employee churn using the test data and evaluate the model performance using accuracy or classification metrics.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIGNESH V
RegisterNumber: 212224230303 
*/
```

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv("/content/Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)
plt.show()

```

## Output:

<img width="1530" height="751" alt="image" src="https://github.com/user-attachments/assets/84fb8c8f-f353-46e1-85a7-188b84c7d1a9" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
