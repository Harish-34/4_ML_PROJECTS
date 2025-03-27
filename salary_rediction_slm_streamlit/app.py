# import liobarries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
df=pd.read_csv(r"Salary_Data.csv")

# split the dataset into independent and dependent variables
x= df.iloc[:,:-1].values
y= df.iloc[:,-1].values

# split the dataset into training and testing sets(80-20%)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state =0)

# train the model
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predict the test set
y_pred = regressor.predict(x_test)

#Comparision from y_test and y_pred
comparision = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
print(comparision)

#plot the model graphs
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color ='blue')
plt.title("salary vs experience (test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

#plot the model graphs
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color ='blue')
plt.title("salary vs experience (test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

# predict the values based on the model outputs
experience = float(input("Enter experience: "))
y_exp = regressor.predict([[experience]])
print(f"Expected salery with mentioned {experience}years is: ${y_exp[0]:,.2f}")

# Check model performance
variance = round(regressor.score(x_test,y_test),3)
bias = round(regressor.score(x_train,y_train),3)
train_mse = mean_squared_error(y_train,regressor.predict(x_train))
test_mse = mean_squared_error(y_test,y_pred) 

print(f"Training score (R^2): {bias}")
print(f"Testing score (R^2): {variance}")
print(f"Training MSE : {train_mse}")
print(f"Testing MSE : {test_mse}")

# Save the training model to disk
import pickle
filename = 'linera_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())