
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import io

from google.colab import files
uploaded = files.upload()

df1 = pd.read_csv("insurance.csv")

plt.xlabel("age1")
plt.ylabel("charges")
plt.scatter(df1.age, df1.charges)

reg1 = linear_model.LinearRegression()
reg1.fit(df1[['age']], df1.charges)


plt.xlabel("age")
plt.ylabel("charges")
plt.scatter(df1.age, df1.charges)
plt.plot(df1.age, reg1.predict(df1[['age']]), color='red')

df1 = pd.read_csv("insurance.csv")

plt.xlabel("bmi")
plt.ylabel("charges")
plt.scatter(df1.bmi, df1.charges)

reg1 = linear_model.LinearRegression()
reg1.fit(df1[['bmi']], df1.charges)


plt.xlabel("bmi")
plt.ylabel("charges")
plt.scatter(df1.bmi, df1.charges)
plt.plot(df1.bmi, reg1.predict(df1[['bmi']]), color='red')

reg1.predict([[2021]])