# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data set
dataset = pd.read_csv(r"Salary_Data.csv")

# split the data to dependent and independent variables
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#Split the data into text train splits
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#reshaping te data from data frame to array
x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

#Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

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

#best fit model
print(f"Coefficient:{regressor.coef_}")
print(f"Intercept:{regressor.intercept_}")

comparsion = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparsion)

# predict the values based on the model outputs
experience = float(input("Enter experience: "))
ar = np.array(experience).reshape(-1,1)
predicted_sal = regressor.predict(ar)
print(f"Expected salery with mentioned {experience}years is: {round(predicted_sal[0],0)}")

# bias variance tradeoff
variance = round(regressor.score(x_test,y_test),3)
bias = round(regressor.score(x_train,y_train),3)
print(f"bias: {bias} & variance:{variance}")


# we can implementthe statistics to this data set
#mean
dataset.mean()
dataset.Salary.mean()

#median
dataset.median()
dataset.Salary.median()

#mode
dataset.mode().head(1)
dataset.Salary.mode()[0]

# variance
dataset.var()
dataset.Salary.var()

# standard deviation
dataset.std()
dataset.Salary.std()

# variance
from scipy.stats import variation
variation(dataset.values)
variation(dataset.Salary)

# correlations
dataset.corr()
dataset.Salary.corr(dataset.YearsExperience)

#Skewness
dataset.skew() 
dataset.Salary.skew()
dataset.YearsExperience.skew()

#Standard error
dataset.sem()
dataset.Salary.sem()
dataset.YearsExperience.sem()

#z-score
import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset.Salary)
stats.zscore(dataset.YearsExperience)

#degree of freedom
a=dataset.shape[0]
b=dataset.shape[1]
degree_of_freedom = a-b
print(f"Degree of freedom: {degree_of_freedom}")

#sum of squareregressor SSR
SSR =np.sum((y_pred-np.mean(y))**2)
print(f"SSR: {SSR}")

#sum of error SSE
SSE =np.sum((y[0:6]-y_pred)**2)
print(f"SSE: {SSE}")

#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R^2
r_square = 1-(SSR/SST)
print(f"R2: {r_square}")

