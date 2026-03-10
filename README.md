# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the dataset to initiate the analysis.
2. Examine the dataset to identify patterns, distributions, and relationships.
3. Determine the most important features to enhance model accuracy and efficiency.
4. Separate the dataset into training and testing sets for effective validation.
5. Use the training data to build and train the model.
6. Measure the model’s performance on the test data with relevant metrics.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Sujin M L
RegisterNumber: 212225040435

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
print(data.head())
print(data.columns)
X = data.drop(columns=['Class']) 
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("="*50)
print("Name : Sujin M L")
print("Reg. No : 212225040435")
print("="*50)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("="*50)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:
<img width="886" height="761" alt="Screenshot 2026-03-10 133738" src="https://github.com/user-attachments/assets/0aef9137-9073-4db7-9e98-275b65bd09f0" />
<img width="574" height="744" alt="Screenshot 2026-03-10 133755" src="https://github.com/user-attachments/assets/93543da5-b5c9-47a6-8b42-6a4347449485" />



## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
