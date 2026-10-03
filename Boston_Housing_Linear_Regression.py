"""
Author: Sara Khon
Date: 10/02/2026

Simple Linear Regression with Scikit-Learn

Boston has historically had large differences in housing values across different areas.
This project explores whether factors such as crime rate are related to historical home values.

The goal is to build a linear regression model in Python that uses features from the
Boston Housing dataset to predict median home values.

Scikit-learn is used to load the data, train the model, and make predictions.

In simple terms:
We are teaching a model to look at housing information and estimate a home's value.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt                            # makes the graph

# This is the web address where the original Boston Housing dataset is stored
data_url = "http://lib.stat.cmu.edu/datasets/boston"


raw_df = pd.read_csv(data_url,                 # Read the dataset from the website
                     sep=r"\s+",               # sep=r"\s+" means the values are separated by spaces
                     skiprows=22,              # skiprows=22 skips the description text at the top of the file
                     header=None)              # header=None means the file does not already have column names


data = np.hstack([
    raw_df.values[::2, :],
    raw_df.values[1::2, :2]
])

# This pulls out the home-value column.  This is the number we eventually want our model to predict
target = raw_df.values[1::2, 2]

# These are the names of the input features in the Boston Housing dataset
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
    'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]



# Convert the housing data into a Pandas DataFrame so it is easier to read
boston = pd.DataFrame(data, columns=feature_names)
boston['MEDV'] = target


# Print the first 5 rows so we can see what the dataset looks like
print(boston.head())



# input features: crime rate = CRIM # output Target: Median Home Value = MEDV
# Use crime rate data to predict median home value.
X = boston[['CRIM']]
Y = boston['MEDV']


# Split the data into training data and testing data.
# We do this because we need to know if the model really learned a pattern, or just memorised the data it saw.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# Create the linear Regression model... seriously, that's all it takes to create one.
model = LinearRegression()


# Train the model using the training data.
# The model studies the crime-rate values and their matching home values
# so it can learn the relationship between them.
model.fit(X_train, Y_train)


# Use the trained model to predict home values from the test data.
# X_test = the unseen crime-rate data
# y_prediction = the model's predicted home values
y_prediction = model.predict(X_test)



# Show sample predictions compared with the real home values
results = pd.DataFrame({
    'Crime Rate': X_test['CRIM'].values,
    'Predicted Home Value': y_prediction,
    'Actual Home Value': Y_test.values
})

# Print the output
print("\nSample Predictions:")
print(results.head())



# Check how accurate the model was
mse = mean_squared_error(Y_test, y_prediction)
r2 = r2_score(Y_test, y_prediction)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


print("The R-squared value shows how much of the variation in home values is explained by crime rate.")


# Sort all crime-rate values from smallest to largest
# This is only used to draw a clean regression line
X_sorted = X.sort_values(by='CRIM')

# Predict home values for the sorted crime-rate values
# This is only used for the graph
y_sorted_prediction = model.predict(X_sorted)

# Plot the actual testing data
plt.scatter(X_test, Y_test, color='orange')

# Plot the clean regression line
plt.plot(X_sorted, y_sorted_prediction, color='black')

plt.xlabel("Crime Rate")
plt.ylabel("Median Home Value")
plt.title("Crime Rate vs. Boston Home Value")

plt.show()


# Show the graph
plt.show()





