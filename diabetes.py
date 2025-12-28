import pandas as pd
df=pd.read_csv('pima_diabetes_data_.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)
print(df.isnull().sum())
cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
print((df[cols]==0).sum())
for col in cols:
  df[col]=df[col].replace(0,df[col].mean())
  print(df[col])
cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
print((df[cols]==0).sum())
print(df.dtypes)
print(df.describe())
x=df.drop('Outcome',axis=1)
y=df['Outcome']
print(x.shape)
print(y.shape)
print(x,y)
x=df.drop('Outcome',axis=1)
y=df['Outcome']
print(x.shape)
print(y.shape)
print(x,y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression

# create model
model = LogisticRegression(max_iter=1000)

# Convert y_train and y_test to discrete integer labels for classification
y_train_discrete = y_train.astype(int)
y_test_discrete = y_test.astype(int)

# train model
model.fit(x_train, y_train_discrete)
print(model)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test_discrete, y_pred)
print("Accuracy:", accuracy)
import numpy as np

print("\nEnter patient details:")

Pregnancies = int(input("Pregnancies: "))
Glucose = float(input("Glucose: "))
BloodPressure = float(input("Blood Pressure: "))
SkinThickness = float(input("Skin Thickness: "))
Insulin = float(input("Insulin: "))
BMI = float(input("BMI: "))
DiabetesPedigreeFunction = float(input("Diabetes Pedigree Function: "))
Age = int(input("Age: "))

# Arrange inputs in SAME ORDER as dataset columns (except Outcome)
user_input = np.array([[Pregnancies, Glucose, BloodPressure,
                        SkinThickness, Insulin, BMI,
                        DiabetesPedigreeFunction, Age]])

# Predict
prediction = model.predict(user_input)

if prediction[0] == 1:
    print("\nResult: DIABETIC")
else:
    print("\nResult: NOT DIABETIC")
import matplotlib.pyplot as plt

df['Outcome'].value_counts().plot(kind='bar')
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.title("Diabetes Outcome Distribution")
plt.show()
