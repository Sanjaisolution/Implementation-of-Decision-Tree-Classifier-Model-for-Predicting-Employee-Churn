# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load required libraries and dataset.
2. Preview the data (first 5 rows) and check for missing values.
3. Convert categorical 'salary' column into numerical form using label encoding.
4. Select relevant features for training (x) and target variable (y).
5. Split the dataset into training and testing sets (80% train, 20% test).
6. Train a Decision Tree Classifier using the training data.
7. Make predictions on the test data and a new sample input.
8. Evaluate the model accuracy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANJAI.R
RegisterNumber:212223040180
*/
```
```py
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

# data.head()
![Screenshot 2024-10-20 110101](https://github.com/user-attachments/assets/0043342c-5861-4810-a4ff-c39079615082)

# data.info()
![Screenshot 2024-10-20 110108](https://github.com/user-attachments/assets/61057ce7-0fc6-4135-ab71-1b06c0cd96a2)

# data.isnull().sum()
![Screenshot 2024-10-20 110114](https://github.com/user-attachments/assets/0f371aee-411d-4352-8820-2e2c351b3d7a)

# data.value.counts()
![Screenshot 2024-10-20 110119](https://github.com/user-attachments/assets/dbb19960-bbee-4d11-a94e-b9bfab1a2200)

# x.head()
![Screenshot 2024-10-20 110125](https://github.com/user-attachments/assets/a7c9d56b-0b60-41a7-aeb2-c30553e764d4)

# accuracy
![Screenshot 2024-10-20 110130](https://github.com/user-attachments/assets/ce3c2759-bd6d-423f-93a1-d452713ce61e)

# predicion
![Screenshot 2024-10-20 110137](https://github.com/user-attachments/assets/54584948-030c-4cb2-b5d7-5ec9a5d1ac3e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
