import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./draft_prediction_denormalized.csv')

data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

y = data.T[n-1]
X = data.T[0:n-1].T

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8)
#initialize object
linear = LinearRegression()

#fit x_train and y_train to model
linear.fit(x_train, y_train)

#make predictions using x_test and y_test
linear_predictions = linear.predict(x_test)
print(linear_predictions)
#plot the actual vs predicted
plt.figure(figsize = (14,5))
plt.scatter(linear_predictions, y_test, s = 10) 
plt.title('Linear Model: Predictions vs. Actual')
plt.xlabel('Predictions'); plt.ylabel('Actual')