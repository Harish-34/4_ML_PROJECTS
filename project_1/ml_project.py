#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import data sets
dataset = pd.read_csv(r"data.csv")
x =dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

# impute missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# impute categotical value for independent
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0] = labelencoder_x.fit_transform(x[:,0])

# impute categotical value for dependent
labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y)
y = labelencoder_y.fit_transform(y)

#split the train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)