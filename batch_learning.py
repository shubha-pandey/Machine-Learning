import numpy as np
from sklearn.linear_model import LinearRegression                  # ML algorithm from sklearn for fitting a linear regression model
from sklearn.model_selection import train_test_split               # splits the dataset into training and testing subsets
from sklearn.metrics import mean_squared_error                     # computes the error between predicted and actual values


# Generate synthetic data

# x is the feature(input) and y is the target(output)
x = np.random.rand(1000, 1) * 10                                   # random matrix of shape (1000, 1) with values between 0 and 1       
y = 2.5 * x + np.random.randn(1000, 1) * 2


# Split the data into training and testing sets (train --> 80%       test --> 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)           # random_state=42 ensures reproducibility by fixing the random seed


# Create and train the model
model = LinearRegression()                                          # creates a regression model
model.fit(x_train, y_train)


# Make predictions and evaluate the model
y_pred = model.predict(x_test)