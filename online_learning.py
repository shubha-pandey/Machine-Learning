import numpy as np
from sklearn.linear_model import SGDRegressor                    # SGDRegressor is a linear regression algorithm that uses Stochastic Gradient Descent (SGD) for training
from sklearn.metrics import mean_squared_error                   # mean_squared_error computes the regression error


# Generate synthetic data
X = np.random.rand(1000, 1) * 10                                 # feature                  
y = 2.5 * X + np.random.randn(1000, 1) * 2                       # noisy target


# Create the online learning model
model = SGDRegressor()


# Simulate online learning by feeding data in small batches
batch_size = 10                                                  # each batch contains 10 data points
n_batches = len(X) // batch_size                                 # the total number of batches is 1000/10=100

# Loops through the dataset in small batches
for i in range(n_batches):
    # indices of the current batch
    start = i * batch_size
    end = start + batch_size

    # update the model's parameters (slope and intercept) incrementally for each batch
    model.partial_fit(X[start:end], y[start:end].ravel())        # y[start:end].ravel() flattens the target array (required by SGDRegressor)


# Make predictions and evaluate the model on the entire dataset
y_pred = model.predict(X)                                         # computes predictions for the entire dataset
print("Mean Squared Error:", mean_squared_error(y, y_pred))       # computes the average squared error between the predicted and actual values